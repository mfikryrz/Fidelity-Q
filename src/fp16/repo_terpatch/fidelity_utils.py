"""Shared helpers for mass fault-injection experiment."""
from __future__ import annotations

import copy
import glob
import os
import re
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs

import _config
from _config import REPO

_CANONICAL_RE = re.compile(r"^decoder-merge-(\d+)__(.+)\.json$")

LETTERS = ("A", "B", "C", "D")
# Ambil huruf jawaban di awal output. Menerima "A", "A.", "(A)", " A)", "B\n..".
#
# Huruf HARUS diikuti pembatas (akhir string, . ) : - , atau newline) — TIDAK
# cukup spasi. Versi sebelumnya memakai (?![A-Za-z0-9]) yang menerima spasi,
# sehingga 'A decrease in the price...' ter-parse sebagai jawaban "A" padahal
# itu kata sandang. Terbukti di data: 1 baris positif-palsu dari 108.
# Bentuk "B four" (huruf + spasi + teks opsi) tetap tertangkap oleh fallback
# pencocokan teks opsi di bawah.
_LETTER_RE = re.compile(r"^\W*([ABCD])(?=$|[.):\-,]|\s*\n)")


def build_mmlu_prompt(row: dict) -> str:
    """Prompt multiple-choice MMLU standar dengan cue 'Answer:'.

    Run lama mengirim `question` telanjang ke model base, sehingga kelanjutan
    greedy paling natural adalah newline -> 22.6% output kosong. Cue eksplisit
    inilah yang memaksa model mengeluarkan token huruf.
    """
    q = str(row.get("question", "")).strip()
    lines = [q]
    for letter in LETTERS:
        opt = str(row.get(f"option_{letter.lower()}", "") or "").strip()
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def parse_answer_letter(text: str, row: dict | None = None) -> str | None:
    """Kembalikan 'A'|'B'|'C'|'D' dari output model, atau None kalau tak terbaca.

    Dua tahap: (1) huruf di awal string, (2) fallback cocokkan teks opsi penuh
    kalau model menjawab dengan isi opsi alih-alih hurufnya.
    """
    s = (text or "").strip()
    if not s:
        return None

    m = _LETTER_RE.match(s)
    if m:
        return m.group(1)

    # `is not None`, BUKAN `if row:` — 16_post_score.py mengirim pandas Series
    # dan truth-value Series itu ambigu (melempar ValueError).
    if row is not None:
        low = s.lower()
        hits = [
            letter for letter in LETTERS
            if (opt := str(row.get(f"option_{letter.lower()}", "") or "").strip())
            and opt.lower() in low
        ]
        if len(hits) == 1:          # ambigu kalau >1 opsi cocok — perlakukan sebagai gagal parse
            return hits[0]
    return None


def operasi_idx_from_suffix(suffix: str) -> int:
    for i, s in enumerate(_config.operasi_suffixes(), start=1):
        if s == suffix:
            return i
    raise ValueError(f"Unknown operasi suffix: {suffix}")


def list_canonical_configs(injection_dir: str | None = None) -> list[dict[str, Any]]:
    """Return sorted injection configs: decoder-merge-{N}__{op}.json only."""
    inj = injection_dir or os.path.join(REPO, "injection_llm")
    known_suffixes = set(_config.operasi_suffixes())
    out: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(inj, "*.json"))):
        base = os.path.basename(path)
        if "_injected" in base:
            continue
        m = _CANONICAL_RE.match(base)
        if not m:
            continue
        decoder_idx = int(m.group(1))
        suffix = m.group(2)
        if suffix not in known_suffixes:
            continue
        cfg = __import__("json").load(open(path))
        out.append({
            "path": path,
            "basename": base,
            "decoder_idx": decoder_idx,
            "operasi_idx": operasi_idx_from_suffix(suffix),
            "operasi_suffix": suffix,
            "config": cfg,
        })
    return sorted(out, key=lambda x: (x["decoder_idx"], x["operasi_idx"]))


def fix_input_tensor(config: dict) -> dict | None:
    """Resolve input_tensor for INPUT/INPUT16 fault injection.

    - Single-operand linear MatMul: use activation operand[0] (Phase 4 fix).
    - Attention Q·Kᵀ (MatMul): prefer Q = Add_1 when it is a variable operand.
    - Attention SKOR·V (MatMul_1): inject on the score operand that actually
      feeds MatMul (Mul_4 = dequantized Round). Parser may still say Round,
      but Round is not a direct MatMul_1 input in this ONNX graph.
    """
    cfg = copy.deepcopy(config)
    g = gs.import_onnx(onnx.load(cfg["model_name"]))
    tgt_node = None
    for n in g.nodes:
        if n.op in {"MatMul", "Gemm"} and (
            cfg["target_layer"] in n.name
            or any(o.name == cfg["target_layer"] for o in n.outputs)
        ):
            tgt_node = n
            break
    if tgt_node is None:
        raise ValueError(f"Target node not found: {cfg['target_layer']}")
    act = [inp for inp in tgt_node.inputs if not isinstance(inp, gs.Constant)]
    act_names = {a.name for a in act}
    if len(act) == 1:
        cfg["input_tensor"] = act[0].name
        return cfg

    # Prefer named attention operands when present as MatMul inputs.
    preferred = (
        "/self_attn/Add_1_output_0",  # Q
        "/self_attn/Mul_4_output_0",  # SKOR (dequantized Round)
        "/self_attn/Round_output_0",  # SKOR (only if directly wired)
    )
    for name in preferred:
        if name in act_names:
            cfg["input_tensor"] = name
            return cfg

    parser_in = cfg.get("input_tensor")
    if parser_in and parser_in in act_names:
        return cfg
    if act:
        cfg["input_tensor"] = act[0].name
        return cfg
    return None


def _stable_u32(*parts: object) -> int:
    """Process-stable 32-bit digest (Python's hash() is salted per process)."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:4], "little")


def pick_prompt_index(
    pool_size: int,
    seed: int,
    decoder_idx: int,
    operasi_idx: int,
    fault_model: str,
    run_id: int,
) -> int:
    """Deterministic random prompt index for one (decoder, operasi, fault, run)."""
    h = _stable_u32(seed, "prompt", decoder_idx, operasi_idx, fault_model, run_id)
    return int(np.random.default_rng(h).integers(0, pool_size))


def derive_injection_params(
    seed: int,
    decoder_idx: int,
    operasi_idx: int,
    fault_model: str,
    run_id: int,
    sample_id: str,
) -> tuple[int, int, int]:
    """Deterministic inject_seed, bit_position (0-15), rand_idx."""
    inject_seed = _stable_u32(seed, decoder_idx, operasi_idx, fault_model, run_id, sample_id)
    rng = np.random.default_rng(inject_seed)
    bit_position = int(rng.integers(0, 16))
    rand_idx = int(rng.integers(0, 1_000_000))
    return inject_seed, bit_position, rand_idx


def resume_key(row: dict) -> tuple:
    # Combo identity is (decoder, operasi, fault, run); sample_id is derived.
    return (
        int(row.get("decoder_idx", -1)),
        int(row.get("operasi_idx", -1)),
        str(row.get("Fault_Model", "")),
        int(row.get("run_id", -1)),
    )


def load_completed_keys(csv_path: str) -> set[tuple]:
    import csv as _csv

    if not os.path.isfile(csv_path):
        return set()
    done: set[tuple] = set()
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                done.add((
                    int(row["decoder_idx"]),
                    int(row["operasi_idx"]),
                    str(row["Fault_Model"]),
                    int(row["run_id"]),
                ))
            except (KeyError, ValueError):
                continue
    return done
