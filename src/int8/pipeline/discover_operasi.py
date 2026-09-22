#!/usr/bin/env python3
"""Turunkan configs/my_model.json["operation_suffixes"] dari graf ONNX sungguhan.

MASALAH YANG DIPERBAIKI
    _config.py dulu menyimpan OPERASI_SUFFIXES sebagai daftar tetap 9 nama
    tensor Llama-2-7B (self_attn_q_proj_MatMul, ...). fidelity_utils.py
    memfilter config injeksi terhadap daftar itu — kalau parser.py menemukan
    operasi dengan nama lain (arsitektur model lain: Qwen, Gemma, dst.),
    config itu DIAM-DIAM DIBUANG, bukan error. Datanya kelihatan lengkap
    padahal sebagian operasi tidak pernah terinjeksi.

CARA KERJA SKRIP INI
    parser.py (FIdelity-ONNX, generik) sudah menghasilkan satu file JSON per
    (decoder, operasi) di injection_llm/ — tapi urutannya cuma urutan abjad
    nama file, bukan urutan eksekusi asli di graf. Skrip ini membaca graf ONNX
    satu decoder representatif, mencocokkan tiap config ke node yang
    bersangkutan, lalu mengurutkan berdasarkan posisi node itu di graf
    (topological order bawaan ONNX). Hasilnya ditulis ke
    configs/my_model.json["operation_suffixes"] supaya operasi_idx dan jumlah
    operasi per decoder konsisten dan tidak perlu diedit tangan tiap ganti
    model.

Dibuktikan identik dengan OPERASI_SUFFIXES lama untuk Llama-2-7B (lihat
tests/test_discover_operasi.py) — jadi run yang sudah ada tidak terpengaruh
kalau configs/my_model.json belum diberi field ini; begitu field ini ditulis,
run BERIKUTNYA memakainya.

Pakai:
    python discover_operasi.py \
        --injection-dir /path/ke/injection_llm \
        --onnx /path/ke/decoder-merge-0.onnx \
        --model-config /path/ke/configs/my_model.json

Butuh: onnx (murni CPU, tidak butuh GPU/onnxruntime).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_CANONICAL_RE = re.compile(r"^decoder-merge-(\d+)__(.+)\.json$")


def discover(injection_dir: str, onnx_path: str, decoder_idx: int = 0) -> list[str]:
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    node_order = {n.name: i for i, n in enumerate(model.graph.node)}
    out_order: dict[str, int] = {}
    for i, n in enumerate(model.graph.node):
        for o in n.output:
            out_order.setdefault(o, i)

    rows: list[tuple[int | None, str, str]] = []
    prefix = f"decoder-merge-{decoder_idx}__"
    paths = sorted(glob.glob(os.path.join(injection_dir, f"{prefix}*.json")))
    if not paths:
        raise SystemExit(
            f"Tidak ada config di {injection_dir} untuk decoder {decoder_idx} "
            f"(pola {prefix}*.json). Jalankan parser.py dulu."
        )

    for path in paths:
        base = os.path.basename(path)
        m = _CANONICAL_RE.match(base)
        if not m or "_injected" in base:
            continue
        cfg = json.load(open(path))
        suffix = m.group(2)
        target = cfg.get("target_layer", "")

        idx = node_order.get(target)
        if idx is None:
            cands = [i for name, i in node_order.items() if name and (target in name or name in target)]
            idx = min(cands) if cands else out_order.get(cfg.get("target_output"))
        rows.append((idx, suffix, path))

    unmatched = [(suffix, path) for idx, suffix, path in rows if idx is None]
    if unmatched:
        detail = "\n".join(f"  {suffix}  ({path})" for suffix, path in unmatched)
        raise SystemExit(
            f"{len(unmatched)} operasi tidak bisa dicocokkan ke node graf — "
            f"periksa manual, jangan tebak urutannya:\n{detail}"
        )

    rows.sort(key=lambda r: r[0])
    ordered = [suffix for _, suffix, _ in rows]

    seen = set()
    deduped = []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--injection-dir", required=True, help="folder injection_llm/ hasil parser.py")
    p.add_argument("--onnx", required=True, help="satu file decoder-merge-N.onnx representatif")
    p.add_argument("--decoder-idx", type=int, default=0, help="decoder acuan urutan (default 0)")
    p.add_argument("--model-config", required=True, help="configs/my_model.json yang akan ditambah field operation_suffixes")
    p.add_argument("--dry-run", action="store_true", help="cetak hasil, jangan tulis file")
    a = p.parse_args()

    suffixes = discover(a.injection_dir, a.onnx, a.decoder_idx)

    print(f"Ditemukan {len(suffixes)} operasi untuk decoder {a.decoder_idx}, urutan eksekusi graf:")
    for i, s in enumerate(suffixes, start=1):
        print(f"  {i}. {s}")

    if a.dry_run:
        return

    if not os.path.isfile(a.model_config):
        sys.exit(
            f"{a.model_config} tidak ada. Skrip ini menambah field ke manifest yang "
            f"sudah dibuat tahap export (decoder_count, hidden_dim, dst.) — jalankan "
            f"tahap itu dulu."
        )
    with open(a.model_config) as f:
        cfg = json.load(f)
    cfg["operation_suffixes"] = suffixes
    with open(a.model_config, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"\nDitulis ke {a.model_config}[\"operation_suffixes\"]")


if __name__ == "__main__":
    main()
