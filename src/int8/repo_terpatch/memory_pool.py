from loguru import logger
from .utils import singleton
import onnxruntime as ort
import numpy as np
import os
import re
import sys

import psutil
import math

_DECODER_IDX_RE = re.compile(r"decoder-merge-(\d+)")


def _wants_gpu(onnxfile: str) -> bool:
    """Layer mana yang ditaruh di GPU.

    Llama-2-7B fp16 butuh ~13,5GB untuk 32 decoder; VRAM bebas cuma ~6,5GB, jadi
    tidak semua muat. Karena tiap decoder adalah sesi ONNX terpisah, tiap sesi
    bisa diberi execution provider berbeda: N decoder pertama di GPU, sisanya CPU.

    FIDELITY_GPU_LAYERS = jumlah decoder di GPU (default 14 ~ 5,9GB).
    embed/norm/head kecil (~260MB) dan selalu di GPU.
    """
    n_gpu = int(os.environ.get("FIDELITY_GPU_LAYERS", "14"))
    if n_gpu <= 0:
        return False
    m = _DECODER_IDX_RE.search(os.path.basename(onnxfile))
    if not m:
        return True                      # embed/norm/head -> GPU
    return int(m.group(1)) < n_gpu


class OrtWrapper:
    def __init__(self, onnxfile: str, custom_op_lib_path: str = 'onnx_bitflip.so'):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(custom_op_lib_path)
        # onnx_bitflip.so hanya menyediakan custom.bitflip:BitFlip untuk tensor
        # FP16. Graf INT8 menyimpan FP32, sehingga graph.py memilih
        # DirectBitToggleFp32 di domain ai.onnx.contrib milik
        # onnxruntime-extensions. Tanpa pendaftaran ini, SELURUH baris
        # RANDOM_BITFLIP pada INT8 gagal dimuat.
        try:
            from onnxruntime_extensions import get_library_path as _ext_lib
            sess_options.register_custom_ops_library(_ext_lib())
        except Exception as _e:
            print(f"[warn] onnxruntime-extensions tidak terdaftar: {_e}", flush=True)

        nthreads = int(os.environ.get("ORT_THREADS", "6"))
        sess_options.intra_op_num_threads = nthreads
        sess_options.inter_op_num_threads = 1

        # Cap CUDA arena growth and allow shrink-on-idle to reduce OOM during
        # long mass-injection runs that create many temporary sessions.
        gpu_mem_limit = int(float(os.environ.get("ORT_GPU_MEM_LIMIT_GB", "6")) * 1024**3)
        provider_options = [{
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": gpu_mem_limit,
            "cudnn_conv_algo_search": "DEFAULT",
        }]
        self.on_gpu = _wants_gpu(onnxfile)
        providers = ([("CUDAExecutionProvider", provider_options[0]), "CPUExecutionProvider"]
                     if self.on_gpu else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnxfile, sess_options, providers=providers)

        used = self.sess.get_providers()
        OrtWrapper._n_gpu += int(used[0] == "CUDAExecutionProvider")
        OrtWrapper._n_cpu += int(used[0] != "CUDAExecutionProvider")
        if not OrtWrapper._announced:
            OrtWrapper._announced = True
            logger.info("split layer aktif: FIDELITY_GPU_LAYERS={} threads={}".format(
                os.environ.get("FIDELITY_GPU_LAYERS", "14"), nthreads))

        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('{} loaded on {}'.format(onnxfile, used[0]))

    _n_gpu = 0
    _n_cpu = 0
    _announced = False

    @classmethod
    def placement(cls):
        return {"gpu": cls._n_gpu, "cpu": cls._n_cpu}

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output

    def close(self):
        self.sess = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        logger.debug('{} unload'.format(self.onnxfile))


@singleton
class MemoryPoolSimple:
    def __init__(self, maxGB):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))
        
        self.max_size = maxGB * 1024 * 1024 * 1024
        self.wait_map = {}
        self.active_map = {}

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'file_size': os.path.getsize(onnx_filepath)
            }

    def used(self):
        sum_size = 0
        biggest_k = None
        biggest_size = 0
        for k in self.active_map.keys():
            cur_size = self.wait_map[k]['file_size']
            sum_size += cur_size

            if biggest_k is None:
                biggest_k = k
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = k
        
        return sum_size, biggest_k

    def check(self):
        sum_need = 0
        for k in self.wait_map.keys():
            sum_need = sum_need + self.wait_map[k]['file_size']
            
        sum_need /= (1024 * 1024 * 1024)
        
        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        if total > 0 and total < sum_need:
            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))


    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        
        need = self.wait_map[key]['file_size']
        onnx = self.wait_map[key]['onnx']

        # check current memory use
        used_size, biggest_k = self.used()
        while biggest_k is not None and self.max_size - used_size < need:
            # if exceeded once, delete until `max(half_max, file_size)` left
            need = max(need, self.max_size / 2)
            if len(self.active_map) == 0:
                break

            old = self.active_map.pop(biggest_k)
            if hasattr(old, "close"):
                old.close()
            del old
            used_size, biggest_k = self.used()

        self.active_map[key] = OrtWrapper(onnx)
        return self.active_map[key]

    def clear_active(self):
        for k in list(self.active_map.keys()):
            old = self.active_map.pop(k)
            if hasattr(old, "close"):
                old.close()
            del old
