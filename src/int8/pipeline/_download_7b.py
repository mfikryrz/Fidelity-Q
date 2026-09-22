#!/usr/bin/env python3
"""Unduh bobot Llama-2-7B secara resumable. SAFETENSORS SAJA.

PENTING: jangan izinkan *.bin. Repo NousResearch/Llama-2-7b-hf punya KEDUA format:
  safetensors  12,55 GB
  .bin         25,10 GB
Kalau keduanya diizinkan, unduhan jadi 37 GB dan tidak akan selesai sebelum pagi.

Jaringan di sini pernah timeout berkali-kali, jadi retry loop wajib.

MODEL LOKAL (21 Sep 2026)
    Kalau MODEL_ID sudah berupa path direktori yang ADA di mesin ini (bukan
    "org/nama-model" dari HF Hub), skrip ini tidak mengunduh apa pun --
    modelnya memang sudah di situ. Ini support "user lain yang sudah punya
    bobot di server lokal/instance lain tinggal arahkan MODEL_ID ke path
    itu", tanpa perlu menyentuh HF Hub sama sekali.

    Konsumen MODEL_ID lain (export_llama.py, export_llama_int8.py) sudah
    otomatis ikut bekerja untuk path lokal -- semuanya lewat
    transformers.from_pretrained(), yang memang menerima path lokal maupun
    ID repo HF. Satu pengecualian sudah ditambal terpisah: fallback
    tokenizer.model di kedua skrip itu (lihat repo_terpatch/README kalau
    ada, atau PATCH_export_llama_local_model.diff).

    Syarat MINIMAL supaya path lokal dianggap valid: ada config.json DAN
    minimal satu berkas *.safetensors langsung di direktori itu (bukan di
    subfolder -- sama seperti struktur snapshot HF Hub setelah diunduh).
"""
from __future__ import annotations

import os
import sys
import time

MODEL_ID = os.environ.get("MODEL_ID", "NousResearch/Llama-2-7b-hf")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ALLOW = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"]
DENY = ["*.bin", "*.pth", "*.msgpack", "*.h5"]
MAX_ATTEMPTS = int(os.environ.get("DL_MAX_ATTEMPTS", "500"))


def is_local_model_dir(path: str) -> bool:
    """True kalau `path` adalah direktori model lokal yang lengkap.

    Sengaja ketat (config.json + minimal 1 *.safetensors) -- lebih baik
    salah menolak dan jatuh ke unduhan HF (aman, cuma boros waktu) daripada
    salah menerima direktori setengah jadi lalu export gagal di tengah
    jalan setelah memuat berjam-jam.
    """
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(f.endswith(".safetensors") for f in os.listdir(path))


def main() -> None:
    if is_local_model_dir(MODEL_ID):
        n_st = sum(1 for f in os.listdir(MODEL_ID) if f.endswith(".safetensors"))
        print(f"[DL] MODEL_ID='{MODEL_ID}' adalah direktori lokal yang valid "
              f"({n_st} berkas .safetensors) -- LEWATI unduhan HF sama sekali.")
        return

    # MODEL_ID terlihat seperti path (mengandung "/" tapi bukan format "org/repo"
    # HF yang wajar) tapi tidak valid sebagai direktori model -- ini kemungkinan
    # typo path lokal, bukan repo HF sungguhan. Peringatkan, jangan diam-diam
    # lanjut mencoba menariknya dari HF (akan gagal dengan pesan membingungkan).
    if os.path.isabs(MODEL_ID) or MODEL_ID.startswith("./") or MODEL_ID.startswith("../"):
        print(f"[DL] PERINGATAN: MODEL_ID='{MODEL_ID}' terlihat seperti path lokal "
              f"tapi tidak valid (butuh config.json + minimal satu *.safetensors "
              f"langsung di direktori itu). Lanjut mencoba sebagai repo HF Hub -- "
              f"kemungkinan besar akan gagal.")

    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(here, "hf_home", "hub")
    os.makedirs(cache, exist_ok=True)
    print(f"[DL] model={MODEL_ID}", flush=True)
    print(f"[DL] cache={cache}", flush=True)
    print(f"[DL] allow={ALLOW} deny={DENY}", flush=True)

    from huggingface_hub import snapshot_download

    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            path = snapshot_download(
                MODEL_ID,
                cache_dir=cache,
                allow_patterns=ALLOW,
                ignore_patterns=DENY,
                resume_download=True,
                max_workers=4,
            )
            print(f"[DL] SELESAI dalam {(time.time()-t0)/60:.1f} menit -> {path}", flush=True)
            tot = 0
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    sz = os.path.getsize(os.path.realpath(fp))
                    tot += sz
                    print(f"       {sz:>14,}  {f}", flush=True)
            print(f"[DL] total {tot/1e9:.2f} GB", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            wait = min(30, 3 * attempt)
            print(f"[DL] percobaan {attempt} gagal: {type(e).__name__}: {str(e)[:160]}", flush=True)
            print(f"[DL] lanjut dalam {wait}s ...", flush=True)
            time.sleep(wait)
    raise SystemExit(f"Gagal setelah {MAX_ATTEMPTS} percobaan")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e, file=sys.stderr)
        raise SystemExit(1) from e
