#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from _config import *  # noqa: F403,F401
from _state import load_state, save_state  # noqa: F401

def main():
    ensure_repo()
    global RUN_ENV
    RUN_ENV = build_run_env()

    # prompts.csv (config sudah ditulis fase 1: configs/my_model.json)
    import csv, os
    csvpath = os.path.join(REPO, "prompts.csv")
    with open(csvpath, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["text"])
        for p in PROMPTS: w.writerow([p])
    import json
    print("config:", json.load(open(os.path.join(REPO, "configs", "my_model.json"))))
    print("prompts.csv:", len(PROMPTS), "prompt")

    # parser.py -> konfigurasi injeksi per-MatMul (decoder)
    import subprocess, os
    INJ = os.path.join(REPO, "injection_llm")
    # Gunakan direktori aktif agar arm INT8 tidak diam-diam mem-parsing FP16.
    r = subprocess.run([sys.executable, "parser.py", os.path.abspath(ONNX_DIR),
                        "--output_dir", INJ, "--ops", "MatMul"],
                       cwd=REPO, env=RUN_ENV, capture_output=True, text=True)
    print(r.stdout[-1500:]); print(r.stderr[-800:])
    assert r.returncode == 0, "parser.py gagal"
    njson = len([x for x in os.listdir(INJ) if x.endswith(".json")])
    print("Total JSON config:", njson)

    # Subset smoke-test: hanya decoder-merge-0
    import os, shutil
    INJ   = os.path.join(REPO, "injection_llm")
    SMOKE = os.path.join(REPO, "injection_llm_smoke")
    shutil.rmtree(SMOKE, ignore_errors=True); os.makedirs(SMOKE)
    allj = sorted(x for x in os.listdir(INJ) if x.endswith(".json"))
    dm = [x for x in allj if x.startswith("decoder-merge-0_")] or [x for x in allj if x.startswith("decoder-merge-")]
    pick = dm[:NUM_LAYER_CONFIGS]
    assert pick, "Tidak ada config decoder-merge; cek output parser.py."
    for x in pick: shutil.copy(os.path.join(INJ, x), os.path.join(SMOKE, x))
    print("Smoke configs:", pick)
    print("03_phase2_parser.py OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise SystemExit(1) from e
