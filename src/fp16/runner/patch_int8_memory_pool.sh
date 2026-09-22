#!/usr/bin/env bash
# ============================================================================
# patch_int8_memory_pool.sh — WAJIB untuk INT8. Tanpa ini, seluruh baris
# RANDOM_BITFLIP gagal (1/6 dataset hilang).
#
# GEJALA
#   [err] .../RANDOM_BITFLIP/bit0: Fail: [ONNXRuntimeError] : 1 : FAIL :
#   Load model ... failed:Fatal error:
#   ai.onnx.contrib:DirectBitToggleFp32(-1) is not a registered function/op
#
# SEBAB
#   graph.py memilih implementasi bitflip berdasarkan tipe tensor sasaran:
#     tensor FP16 -> custom.bitflip:BitFlip      (dari llama/onnx_bitflip.so)
#     tensor FP32 -> ai.onnx.contrib:DirectBitToggleFp32  (PyOp, dari
#                                                  onnxruntime-extensions)
#   Graf INT8 memakai fake-quant dengan penyimpanan FP32, jadi ia selalu jatuh
#   ke jalur kedua. Tapi llama/memory_pool.py hanya mendaftarkan onnx_bitflip.so
#   saat membuat session — pustaka extensions tidak pernah didaftarkan.
#   cnn_inference.py mendaftarkan keduanya; jalur LLM terlewat.
#
#   FP16 tidak terdampak karena tensornya FP16, jadi memakai jalur pertama.
#
# PERBAIKAN butuh DUA hal, keduanya harus ada:
#   1. daftarkan pustaka onnxruntime-extensions di session  <- patch ini
#   2. import inject_ops supaya dekorator @onnx_op menjalankan pendaftaran PyOp
#      <- sudah terjadi sendiri lewat graph.py di pipeline, tidak perlu diubah
#
# Terverifikasi 14 Sep 2026: tanpa patch graf gagal dimuat; dengan patch
# "LOAD OK — provider: CUDAExecutionProvider".
#
# Pakai:  bash patch_int8_memory_pool.sh
# Aman diulang.
# ============================================================================
set -uo pipefail
ROOT=${ROOT:-/workspace/fidelity}
M=$ROOT/running_experiment_7b/repo/FIdelity-ONNX-master/llama/memory_pool.py
PY=$ROOT/rtenv/bin/python

[ -f "$M" ] || { echo "tidak ada: $M" >&2; exit 1; }
[ -f "$M.asli" ] || cp "$M" "$M.asli"

"$PY" - "$M" <<'PY'
import sys
M = sys.argv[1]
s = open(M).read()
if "_ext_lib" in s:
    print("sudah dipatch, dilewati"); raise SystemExit
lama = ("        sess_options = ort.SessionOptions()\n"
        "        sess_options.register_custom_ops_library(custom_op_lib_path)\n")
baru = '''        sess_options = ort.SessionOptions()
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
'''
if lama not in s:
    print("POLA TIDAK DITEMUKAN — periksa manual", file=sys.stderr); raise SystemExit(2)
open(M, "w").write(s.replace(lama, baru, 1))
print("memory_pool.py dipatch")
PY

"$PY" -m py_compile "$M" && echo "sintaks OK"
