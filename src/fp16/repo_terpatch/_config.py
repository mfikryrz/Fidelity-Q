"""Shared paths, constants, and helpers for FIdelity Path B scripts."""
from __future__ import annotations

import ctypes.util
import glob
import importlib.util
import json
import os
import site
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

REPO = os.environ.get("FIDELITY_REPO", os.path.join(_HERE, "repo", "FIdelity-ONNX-master"))
WORK = os.environ.get("FIDELITY_WORK", os.path.join(_HERE, "work"))
# ONNX 7B di SSD (185GB bebas), BUKAN /dev/shm (16GB) — fp32 antara saja ~26GB.
_ONNX_ROOT = os.environ.get("FIDELITY_ONNX_ROOT", os.path.join(_HERE, "onnx"))
PY310_BIN = os.environ.get("PY310_BIN", "")
VAST_SSH_HOST = os.environ.get("VAST_SSH_HOST", os.environ.get("PUBLIC_IPADDR", ""))
VAST_SSH_PORT = os.environ.get("VAST_SSH_PORT", os.environ.get("VAST_TCP_PORT_22", "22"))

MODEL_ID = os.environ.get("MODEL_ID", "NousResearch/Llama-2-7b-hf")  # MHA; meta-llama gated
ACT_SCALES = os.environ.get("ACT_SCALES", "llama-2-7b.pt")
PRECISION = os.environ.get("PRECISION", "float16")
ONNX_RAW_DIR = os.path.join(_ONNX_ROOT, "onnx_fp32")
ONNX_FP16_DIR = os.path.join(_ONNX_ROOT, "onnx_fp16")
ONNX_INT8_DIR = os.path.join(_ONNX_ROOT, "onnx_int8")
ONNX_DIR = ONNX_INT8_DIR if PRECISION == "int8" else ONNX_FP16_DIR
USE_FP16_IO = os.environ.get("USE_FP16_IO", "0") == "1"
PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The sun rises in the",
    "Two plus two equals",
    "The opposite of hot is",
    "Roses are red, violets are",
    "The first president of the United States was",
    "An apple a day keeps the",
    "The chemical symbol for gold is",
    "The largest planet in the solar system is",
]
EXPORT_MAX_NEW_TOKENS = 2
MAX_TOKENS = 8
# Need ~14GB for 32 fp16 decoder sessions; OOM was from faulty-session leak, not pool size.
# 7B fp16 = ~13.5GB, VRAM bebas cuma ~6.5GB -> tidak semua muat.
# Sebagian layer di GPU, sisanya CPU (lihat FIDELITY_GPU_LAYERS di memory_pool.py).
# POOL harus >= total ukuran SEMUA sesi (~13.5GB) supaya MemoryPoolSimple tidak
# meng-evict; eviction di sini fatal karena tiap token memakai ke-32 decoder.
POOL = int(os.environ.get("FIDELITY_POOL_GB", "20"))
NUM_LAYER_CONFIGS = 1
COMPARE_MAX_NEW_TOKENS = MAX_TOKENS
REF_DTYPE = "float32"

PLOTS_DIR = os.path.join(WORK, "plots")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.environ.get("FIDELITY_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "hasil_bench_7b"))

# Mass experiment (night sprint defaults)
MASS_PROMPT_LIMIT = int(os.environ.get("MASS_PROMPT_LIMIT", "5"))
MASS_MAX_TOKENS = int(os.environ.get("MASS_MAX_TOKENS", "64"))
GOLDEN_CACHE = os.environ.get("GOLDEN_CACHE", "1") == "1"
MASS_SEED = int(os.environ.get("MASS_SEED", "0"))
DATASET_ID = os.environ.get("DATASET_ID", "CohereLabs/Global-MMLU")
MMLU_POOL_CSV = os.path.join(WORK, "mmlu_pool.csv")
FAULT_MODELS = ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]
RUNS_PER_COMBO = int(os.environ.get("RUNS_PER_COMBO", "5"))
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
DEMO_DECODER_LIMIT = int(os.environ.get("DEMO_DECODER_LIMIT", "4"))
DEMO_OPERASI_SKIP = [
    int(x.strip())
    for x in os.environ.get("DEMO_OPERASI_SKIP", "4,5").split(",")
    if x.strip()
]
DEMO_NUM_SHARDS = int(os.environ.get("DEMO_NUM_SHARDS", "2"))
# Daftar operasi BAWAAN — dipakai hanya sebagai fallback (lihat operasi_suffixes()
# di bawah) untuk model yang belum punya configs/my_model.json["operation_suffixes"].
# Ini adalah struktur decoder Llama-2-7B tepatnya: 6 proyeksi attention + 2 matmul
# skor/nilai + 3 proyeksi MLP, dalam urutan eksekusi asli di graf ONNX-nya.
OPERASI_SUFFIXES = [
    "self_attn_q_proj_MatMul",
    "self_attn_k_proj_MatMul",
    "self_attn_v_proj_MatMul",
    "self_attn_MatMul",
    "self_attn_MatMul_1",
    "self_attn_o_proj_MatMul",
    "mlp_gate_proj_MatMul",
    "mlp_up_proj_MatMul",
    "mlp_down_proj_MatMul",
]
RAW_CSV = os.path.join(OUTPUT_ROOT, "raw", "results_raw.csv")
SCORED_CSV = os.path.join(OUTPUT_ROOT, "scored", "results_scored.csv")


def decoder_count() -> int:
    """Jumlah decoder dari configs/my_model.json. Jangan hardcode 32."""
    try:
        with open(os.path.join(REPO, "configs", "my_model.json")) as f:
            return int(json.load(f)["decoder_count"])
    except Exception:
        return 32


def hidden_dim() -> int:
    try:
        with open(os.path.join(REPO, "configs", "my_model.json")) as f:
            return int(json.load(f)["hidden_dim"])
    except Exception:
        return 4096


_operasi_suffixes_cache: list | None = None


def operasi_suffixes() -> list:
    """Daftar suffix operasi untuk MODEL AKTIF, dalam urutan eksekusi graf.

    Dibaca dari configs/my_model.json["operation_suffixes"] kalau field itu ada —
    field itu di-generate oleh pipeline/discover_operasi.py dengan menelusuri graf
    ONNX model yang bersangkutan, jadi otomatis benar untuk arsitektur apa pun
    (Llama, Qwen, Gemma, ...) tanpa perlu diedit manual di sini.

    Jatuh ke OPERASI_SUFFIXES (Llama-2-7B) kalau field itu belum ada, supaya run
    yang datanya sudah ada (hasil final FP16/INT8 Llama-2-7B) tetap konsisten
    tanpa perlu configs/my_model.json diperbarui.

    Di-cache per proses: dipanggil ribuan kali sepanjang satu run masal, dan
    configs/my_model.json tidak berubah di tengah run.
    """
    global _operasi_suffixes_cache
    if _operasi_suffixes_cache is not None:
        return _operasi_suffixes_cache
    try:
        with open(os.path.join(REPO, "configs", "my_model.json")) as f:
            suffixes = json.load(f).get("operation_suffixes")
        if suffixes:
            _operasi_suffixes_cache = list(suffixes)
            return _operasi_suffixes_cache
    except (OSError, json.JSONDecodeError):
        pass
    _operasi_suffixes_cache = OPERASI_SUFFIXES
    return _operasi_suffixes_cache


COMBO_TARGET_ROWS = decoder_count() * len(operasi_suffixes()) * len(FAULT_MODELS) * 5  # nominal


def demo_operasi_count() -> int:
    skip = set(DEMO_OPERASI_SKIP)
    return sum(1 for i in range(1, len(operasi_suffixes()) + 1) if i not in skip)


def demo_target_rows() -> int:
    decoders = DEMO_DECODER_LIMIT if DEMO_MODE else decoder_count()
    operasi = demo_operasi_count() if DEMO_MODE else len(operasi_suffixes())
    if DEMO_MODE:
        return MASS_PROMPT_LIMIT * decoders * operasi * len(FAULT_MODELS) * RUNS_PER_COMBO
    return decoders * operasi * len(FAULT_MODELS) * RUNS_PER_COMBO


def mass_target_rows() -> int:
    """Target rows for combo_8640: 1 random prompt per (decoder, operasi, fault, run)."""
    return demo_target_rows()


def filter_demo_configs(configs: list) -> list:
    if not DEMO_MODE:
        return configs
    skip = set(DEMO_OPERASI_SKIP)
    return [
        c for c in configs
        if c["decoder_idx"] < DEMO_DECODER_LIMIT and c["operasi_idx"] not in skip
    ]


def ensure_work_dir() -> None:
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ONNX_RAW_DIR, exist_ok=True)
    os.makedirs(ONNX_FP16_DIR, exist_ok=True)
    for sub in ("raw", "scored", "summary", "figures"):
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)


def ensure_repo() -> None:
    ensure_work_dir()
    assert os.path.isdir(REPO), f"REPO tidak ditemukan: {REPO}"


def ensure_deps() -> None:
    need = {
        "onnx": "onnx==1.17.0",
        "onnxruntime": "onnxruntime-gpu==1.20.1",
        "onnx_graphsurgeon": "onnx-graphsurgeon",
        "onnxconverter_common": "onnxconverter-common>=1.16.0",
        "sentencepiece": "sentencepiece",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "datasets": "datasets",
        "onnxruntime_extensions": "onnxruntime_extensions",
    }
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    for mod in need:
        print(("  [ok] " if importlib.util.find_spec(mod) else "  [--] ") + mod)
    if missing:
        print("\nMenginstal:", " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)
        print("Install selesai.")
    else:
        print("\nSemua paket ADA.")
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        if ort.__version__ != "1.20.1":
            print(f"[INFO] onnxruntime={ort.__version__} — menginstal onnxruntime-gpu==1.20.1...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
                check=False,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.20.1", "onnx==1.17.0"],
                check=True,
            )
    has_cu12 = any(
        glob.glob(os.path.join(sp, "nvidia", "*", "lib", "libcublasLt.so.12"))
        for sp in site.getsitepackages() + [site.getusersitepackages()]
    )
    if not has_cu12 and ctypes.util.find_library("cublasLt") is None:
        print("[INFO] libcublasLt.so.12 belum di PATH — FASE 2 akan menginstal nvidia-cublas-cu12")


def check_gpu() -> None:
    import torch

    print("Python:", sys.version.split()[0])
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            f"| sm_{cc[0]}{cc[1]}",
            "| CUDA:",
            torch.version.cuda,
        )
    else:
        print("[PERINGATAN] GPU tidak terdeteksi — FASE 2/4+ butuh GPU.")


def verify_repo_files() -> None:
    assert os.path.exists(os.path.join(REPO, "llm_inference.py"))
    print("exporter:", os.path.exists(os.path.join(REPO, "int8", "modeling_llama.py")))
    print("custom op:", os.path.exists(os.path.join(REPO, "llama", "onnx_bitflip.so")))


def should_skip_phase1_export() -> bool:
    """Lewati export hanya kalau model DAN presisi yang sama sudah lengkap.

    Versi lama cuma memeriksa model_id + direktori fp16. Akibatnya setelah export
    FP16 sukses, export INT8 akan mengira dirinya sudah selesai lalu melewatkan
    diri — arm INT8 hilang tanpa pesan error. Sekarang tiap presisi diperiksa
    terhadap direktorinya sendiri.
    """
    from _state import load_state

    state = load_state()
    if state.get("model_id") != MODEL_ID:
        return False
    if state.get("precision") != PRECISION:
        return False
    return onnx_fp16_complete(ONNX_DIR, REPO)


def onnx_fp16_complete(fp16_dir: str, repo: str) -> bool:
    cfg_path = os.path.join(repo, "configs", "my_model.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        cfg = json.load(open(cfg_path))
    except (json.JSONDecodeError, OSError):
        return False
    decoder_count = cfg.get("decoder_count")
    if not decoder_count:
        return False
    need = ["embed.onnx", "norm.onnx", "head.onnx"] + [
        f"decoder-merge-{i}.onnx" for i in range(decoder_count)
    ]

    def ok(d: str, f: str) -> bool:
        p = os.path.join(d, f)
        return os.path.exists(p) and os.path.getsize(p) > 0

    return all(ok(fp16_dir, f) for f in need) and os.path.exists(
        os.path.join(fp16_dir, "tokenizer.model")
    )


def find_results_csv(
    repo: str | None = None,
    onnxdir: str | None = None,
    precision: str = "float16",
    csv_name: str = "prompts.csv",
) -> str | None:
    repo = repo or REPO
    onnxdir = onnxdir or ONNX_FP16_DIR
    model_tag = os.path.basename(onnxdir.rstrip("/\\"))
    dataset_tag = os.path.splitext(os.path.basename(csv_name))[0]
    expected = os.path.join(repo, f"results_{model_tag}_{precision}_{dataset_tag}.csv")
    if os.path.isfile(expected):
        return expected
    cands = sorted(glob.glob(os.path.join(repo, "results_*.csv")), key=os.path.getmtime)
    return cands[-1] if cands else None


def ensure_ort_gpu():
    import onnxruntime as ort

    if ort.__version__ == "1.20.1":
        return ort
    print(f"onnxruntime={ort.__version__} — menginstal onnxruntime-gpu==1.20.1...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "onnxruntime-gpu==1.20.1",
            "onnx==1.17.0",
            "onnx-graphsurgeon",
            "onnxruntime_extensions",
            "onnxconverter-common>=1.16.0",
            "sentencepiece",
            "psutil",
            "loguru",
            "nvidia-cudnn-cu12",
        ],
        check=True,
    )
    import importlib

    importlib.reload(__import__("onnxruntime"))
    import onnxruntime as ort

    return ort


def build_run_env(repo: str | None = None) -> dict[str, str]:
    repo = repo or REPO
    nvidia_libs: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_libs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    nvidia_libs = sorted(set(d for d in nvidia_libs if os.path.isdir(d)))
    llama = os.path.join(repo, "llama")
    ld = os.pathsep.join([llama] + nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")])
    return {**os.environ, "LD_LIBRARY_PATH": ld, "TOKENIZERS_PARALLELISM": "false"}


def preload_ort_libs() -> None:
    """Preload ORT shared lib (LD_LIBRARY_PATH set inside Python is too late on Linux)."""
    import ctypes

    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        capi = os.path.join(sp, "onnxruntime", "capi")
        so_ver = os.path.join(capi, "libonnxruntime.so.1.20.1")
        so_link = os.path.join(capi, "libonnxruntime.so.1")
        if os.path.isfile(so_ver):
            if not os.path.exists(so_link):
                os.symlink("libonnxruntime.so.1.20.1", so_link)
            ctypes.CDLL(so_ver, mode=ctypes.RTLD_GLOBAL)
            return


def apply_run_env(repo: str | None = None) -> dict[str, str]:
    """Set process env (incl. ORT capi) so onnx_bitflip.so can load."""
    repo = repo or REPO
    env = build_run_env(repo)
    extra: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        capi = os.path.join(sp, "onnxruntime", "capi")
        if os.path.isdir(capi):
            so_ver = os.path.join(capi, "libonnxruntime.so.1.20.1")
            so_link = os.path.join(capi, "libonnxruntime.so.1")
            if os.path.isfile(so_ver) and not os.path.exists(so_link):
                os.symlink("libonnxruntime.so.1.20.1", so_link)
            extra.append(capi)
            break
    ort_dev = os.path.join(repo, "onnxruntime-dev", "onnxruntime-linux-x64-gpu-1.20.1", "lib")
    if os.path.isdir(ort_dev):
        extra.append(ort_dev)
    if extra:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(extra + [env.get("LD_LIBRARY_PATH", "")])
    os.environ.update(env)
    preload_ort_libs()
    patch_bitflip_lib_path(repo)
    return env


def patch_bitflip_lib_path(repo: str | None = None) -> str:
    """OrtWrapper defaults to relative onnx_bitflip.so; use repo absolute path."""
    repo = repo or REPO
    bitflip_so = os.path.join(repo, "llama", "onnx_bitflip.so")
    if not os.path.isfile(bitflip_so):
        return bitflip_so
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import llama.memory_pool as mp

    if getattr(mp.OrtWrapper, "_bitflip_patched", False):
        return bitflip_so

    _orig_init = mp.OrtWrapper.__init__

    def _init(self, onnxfile: str, custom_op_lib_path: str = bitflip_so) -> None:
        _orig_init(self, onnxfile, custom_op_lib_path)

    mp.OrtWrapper.__init__ = _init  # type: ignore[method-assign]
    mp.OrtWrapper._bitflip_patched = True
    return bitflip_so


def build_golden_env() -> dict[str, str]:
    nvidia_libs: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_libs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    nvidia_libs = sorted(set(d for d in nvidia_libs if os.path.isdir(d)))
    ld = os.pathsep.join(nvidia_libs + [os.environ.get("LD_LIBRARY_PATH", "")])
    return {**os.environ, "LD_LIBRARY_PATH": ld, "TOKENIZERS_PARALLELISM": "false"}


def add_script_dir_to_path() -> None:
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
