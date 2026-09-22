# -*- coding: utf-8 -*-
"""Eksperimen massal fault injection dengan protokol LOG-LIKELIHOOD.

Menggabungkan dua hal yang selama ini terpisah:
  * mmlu_loglik_bench.py  -> protokol bersih, tapi fault_config dipatok None
  * 15_mass_inference.py  -> bisa fault injection, tapi memakai generasi+parsing

Definisi is_error di sini SEDERHANA dan tidak bisa disalahartikan:

    is_error = (huruf argmax golden) != (huruf argmax faulty)

Tanpa BLEU, tanpa ambang, tanpa regex, tanpa kemungkinan jawaban kosong —
karena model tidak pernah diminta menulis teks. Untuk tiap soal cukup satu
forward pass; argmax diambil dari empat logit token ' A'/' B'/' C'/' D'.

Contoh uji lokal (validasi fisika bit fp16):
    python mass_loglik_inject.py --decoders 6 --operasi self_attn_o_proj_MatMul \\
        --fault-models RANDOM_BITFLIP --bits 0-15 --soal 10 --out hasil/uji_bit.csv

Contoh skala penuh:
    python mass_loglik_inject.py --runs 25 --soal 1 --out hasil/massal.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _config import *          # noqa: F403,F401  (REPO, POOL, FAULT_MODELS, ...)
from _config import ensure_repo, apply_run_env, MMLU_POOL_CSV  # noqa: F401
import _config as _C
from fidelity_utils import fix_input_tensor

# _config.py proyek 1.3B (lama) memakai ONNX_FP16_DIR dan belum punya
# PRECISION/USE_FP16_IO; proyek 7B/int8 (baru) memakai ONNX_DIR. Samakan di sini
# supaya satu skrip jalan di kedua proyek.
ONNX_DIR = getattr(_C, "ONNX_DIR", None) or getattr(_C, "ONNX_FP16_DIR")
PRECISION = getattr(_C, "PRECISION", "float16")
USE_FP16_IO = getattr(_C, "USE_FP16_IO", True)

LETTERS = ("A", "B", "C", "D")

FIELDS = [
    # identitas kombinasi — cukup untuk mereproduksi baris ini persis
    "ts", "decoder_idx", "operasi", "Fault_Model", "run_id", "Bit_Position",
    "rand_idx", "inject_seed",
    # soal
    "idx", "sample_id", "subject", "subject_category", "answer_letter",
    # HASIL INFERENSI MENTAH — tanpa parsing, tanpa penilaian
    "golden_teks", "faulty_teks",
    # keluaran mentah model untuk keempat opsi (bukan penilaian, cuma angka)
    "g_logit_A", "g_logit_B", "g_logit_C", "g_logit_D",
    "f_logit_A", "f_logit_B", "f_logit_C", "f_logit_D",
    # teknis
    "n_prompt_tokens", "batas_token_teks", "n_tok_golden", "n_tok_faulty", "detik",
    # koordinat injeksi: rand_idx yang sama, diurai terhadap bentuk tensor sasaran
    # supaya letak kerusakan terbaca langsung tanpa menghitung ulang.
    "inject_shape", "inject_coords", "inject_w", "inject_h", "inject_c",
    # prompt utuh yang benar-benar masuk ke model, termasuk lima contoh few-shot
    "prompt_lengkap",
]


# Geometri Llama-2-7B. Dikumpulkan di satu tempat; true_int8_smoke.py menyebarkan
# angka-angka ini di delapan lokasi terpisah dan itu mudah salah saat diubah.
N_HEAD = 32
HEAD_DIM = 128
HIDDEN = 4096
FFN = 11008


def lokasi_tensor(operasi: str, fault_model: str, n_token: int) -> tuple[list[int], str]:
    """Bentuk runtime tensor yang benar-benar disuntik, per operasi & fault model.

    Disalin dari true_int8_smoke.py:235-259. Ini yang menggantikan asumsi lama
    bahwa semua sasaran berbentuk [1, n_token, hidden_dim] — asumsi itu benar
    hanya untuk sebagian operasi, dan membuat separuh tensor gate/up/down tidak
    pernah tersentuh sekaligus salah alamat pada dua BMM attention.

    Fault model menentukan tensor mana yang disasar: INPUT* -> operand aktivasi,
    WEIGHT* -> operand bobot, RANDOM/RANDOM_BITFLIP -> keluaran MatMul.
    Pada dua BMM attention, slot "weight" bukan bobot sungguhan melainkan operand
    aktivasi kedua (Kᵀ lalu V), sehingga bentuknya ikut bergantung n_token.
    """
    h, d, m = N_HEAD, HEAD_DIM, HIDDEN
    if operasi in {"self_attn_q_proj_MatMul", "self_attn_k_proj_MatMul",
                   "self_attn_v_proj_MatMul", "self_attn_o_proj_MatMul"}:
        inp, weight, out = [1, n_token, m], [m, m], [1, n_token, m]
    elif operasi in {"mlp_gate_proj_MatMul", "mlp_up_proj_MatMul"}:
        # Bobot fake-INT8 disimpan pra-transpose dalam tata letak torch [out, in].
        inp, weight, out = [1, n_token, m], [FFN, m], [1, n_token, FFN]
    elif operasi == "mlp_down_proj_MatMul":
        inp, weight, out = [1, n_token, FFN], [m, FFN], [1, n_token, m]
    elif operasi == "self_attn_MatMul":
        # Operand kedua BUKAN bobot: pada graf fake-INT8 ia menunjuk hasil act-quant
        # k_proj, yaitu aktivasi K berbentuk [1, n, 4096] SEBELUM reshape ke 4-D.
        # Ini berbeda dari graf true-INT8 (di sana [1, 32, 128, n]).
        inp, weight, out = [1, h, n_token, d], [1, n_token, m], [1, h, n_token, n_token]
    elif operasi == "self_attn_MatMul_1":
        inp, weight, out = [1, h, n_token, n_token], [1, n_token, m], [1, h, n_token, d]
    else:
        raise ValueError(f"operasi tidak dikenal: {operasi}")
    if fault_model.startswith("INPUT"):
        return inp, "input_tensor"
    if fault_model.startswith("WEIGHT"):
        return weight, "weight_tensor"
    # RANDOM/RANDOM_BITFLIP mengait langsung ke keluaran node; config fake-INT8
    # memang tidak punya field target_output.
    return out, None


def indeks_lanjutan(operasi: str, fault_model: str, coords: list[int], past_len: int) -> int:
    """Petakan koordinat prefill ke bentuk tensor pada langkah decode.

    Disalin dari true_int8_smoke.py:273-304, menggantikan `rand_idx % hidden_dim`.
    Modulo lama kebetulan benar selama rand_idx dibangun dengan asumsi 4096; begitu
    koordinat diundi dari seluruh tensor ia salah alamat, dan karena ScatterND/GatherND
    pada graf ini tidak memeriksa batas, indeks di luar rentang berarti CUDA error 700.

    Aturannya: pertahankan batch/head/fitur, sumbu token-yang-sedang-dihitung jadi 0,
    sumbu key/value cache jadi past_len (entri terbaru), bobot 2-D tidak disentuh.
    """
    import numpy as np
    shape, _ = lokasi_tensor(operasi, fault_model, 1)
    mapped = list(coords)
    if operasi in {"self_attn_q_proj_MatMul", "self_attn_k_proj_MatMul",
                   "self_attn_v_proj_MatMul", "self_attn_o_proj_MatMul",
                   "mlp_gate_proj_MatMul", "mlp_up_proj_MatMul",
                   "mlp_down_proj_MatMul"}:
        if fault_model.startswith("WEIGHT"):
            # bentuk bobot tidak bergantung n_token, indeksnya tetap sah
            return int(np.ravel_multi_index(tuple(coords), tuple(shape), order="C"))
        mapped[-2] = 0
    elif operasi == "self_attn_MatMul":
        if fault_model.startswith("INPUT"):          # Q [1,32,n,128]
            mapped[2] = 0
            shape = [1, N_HEAD, 1, HEAD_DIM]
        elif fault_model.startswith("WEIGHT"):       # aktivasi K [1,n,4096]
            mapped[-2] = 0
            shape = [1, 1, HIDDEN]
        else:                                        # skor [1,32,n,n]
            mapped[2], mapped[3] = 0, past_len
            shape = [1, N_HEAD, 1, past_len + 1]
    elif operasi == "self_attn_MatMul_1":
        if fault_model.startswith("INPUT"):          # probabilitas [1,32,n,n]
            mapped[2], mapped[3] = 0, past_len
            shape = [1, N_HEAD, 1, past_len + 1]
        elif fault_model.startswith("WEIGHT"):       # aktivasi V [1,n,4096]
            mapped[-2] = 0
            shape = [1, 1, HIDDEN]
        else:                                        # konteks [1,32,n,128]
            mapped[2] = 0
            shape = [1, N_HEAD, 1, HEAD_DIM]
    return int(np.ravel_multi_index(tuple(mapped), tuple(shape), order="C"))


def koordinat_metadata(rand_idx: int, shape: list[int]) -> tuple[list[int], int, int, int]:
    """Urai rand_idx datar menjadi koordinat pada tensor sasaran.

    Konvensi w/h/c disamakan dengan true_int8_smoke.py: w = sumbu terakhir,
    h = sumbu kedua dari belakang, c = sisa sumbu di depannya yang diratakan.
    Untuk [1, n_token, feat] hasilnya w = fitur, h = token, c = 0 (sumbu depan
    memang berukuran 1). Untuk BMM attention [1, 32, ·, ·] barulah c bermakna:
    c = indeks head 0-31.
    """
    # numpy sengaja diimpor di dalam main() setelah apply_run_env() menetapkan
    # OMP_NUM_THREADS, jadi impor di sini bukan di level modul.
    import numpy as np
    coords = [int(x) for x in np.unravel_index(rand_idx, tuple(shape), order="C")]
    assert int(np.ravel_multi_index(tuple(coords), tuple(shape), order="C")) == rand_idx
    w = coords[-1]
    h = coords[-2] if len(coords) >= 2 else 0
    c = (int(np.ravel_multi_index(tuple(coords[:-2]), tuple(shape[:-2]), order="C"))
         if len(coords) > 2 else 0)
    return coords, w, h, c


# --------------------------------------------------------------------- argumen
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--soal", type=int, default=1,
                   help="jumlah soal per (kombinasi x run x bit)")
    p.add_argument("--runs", type=int, default=1, help="run per kombinasi")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--decoders", default="", help="mis. '0,6,11' (kosong = semua)")
    p.add_argument("--operasi", default="", help="mis. 'self_attn_o_proj_MatMul' (kosong = semua)")
    p.add_argument("--fault-models", default="", help="kosong = semua 6")
    p.add_argument("--bits", default="",
                   help="'0-15' atau '14' untuk memaksa Bit_Position; kosong = acak berseed")
    p.add_argument("--max-configs", type=int, default=0, help="batasi jumlah konfigurasi (uji cepat)")
    p.add_argument("--konfigurasi-langkah", type=int, default=0,
                   help="ambil tiap konfigurasi ke-k (stratified). 288/16=18 memberi "
                        "16 konfigurasi yang tersebar di seluruh decoder dan operasi. "
                        "0 atau 1 = ambil berurutan dari awal")
    p.add_argument("--konfigurasi-dari", type=int, default=0,
                   help="indeks konfigurasi awal (untuk memotong run jadi beberapa proses)")
    p.add_argument("--konfigurasi-sampai", type=int, default=0,
                   help="indeks konfigurasi akhir, eksklusif (0 = sampai habis)")
    p.add_argument("--posisi-token", choices=("penuh", "akhir", "acak"), default="penuh",
                   help="'penuh' = undi rand_idx merata dari SELURUH tensor sasaran, dengan "
                        "bentuk yang benar per operasi/fault model (default, sama dengan "
                        "true_int8_smoke.py); 'akhir' = kunci ke slice token terakhir dengan "
                        "asumsi lama [1,n,hidden_dim]; 'acak' = seluruh prompt, asumsi lama. "
                        "'akhir' dan 'acak' hanya untuk mereproduksi run sebelumnya")
    p.add_argument("--gaya-prompt", choices=("pg", "pg-teks", "bebas"), default="pg",
                   help="'pg'      = pilihan ditampilkan, contoh dijawab HURUF saja "
                        "-> model menjawab huruf; "
                        "'pg-teks' = pilihan ditampilkan, contoh dijawab HURUF + TEKS OPSI "
                        "-> model menjawab huruf disertai kalimat; "
                        "'bebas'   = soal saja tanpa pilihan -> model menjawab prosa")
    p.add_argument("--token-teks", type=int, default=8,
                   help="jumlah token yang dihasilkan untuk golden_teks/faulty_teks "
                        "(0 = tidak menghasilkan teks, hanya logit)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--deadline", default="", help='batas waktu keras "HH:MM"')
    return p.parse_args()


def urai_daftar(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def urai_bits(s: str) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for bagian in s.split(","):
        bagian = bagian.strip()
        if "-" in bagian:
            a, b = bagian.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif bagian:
            out.append(int(bagian))
    max_bit = 7 if PRECISION == "int8" else 15
    bad = [b for b in out if not 0 <= b <= max_bit]
    if bad:
        raise SystemExit(f"Bit di luar 0-{max_bit} untuk {PRECISION}: {bad}")
    return out


def deadline_ts(s: str) -> float | None:
    if not s:
        return None
    jam, menit = (int(x) for x in s.split(":"))
    now = datetime.now()
    t = now.replace(hour=jam, minute=menit, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def soal_ke_teks(r: dict, dengan_jawaban: bool, gaya: str = "pg") -> str:
    """gaya='pg'    -> tampilkan A/B/C/D, jawaban berupa huruf
       gaya='bebas' -> soal saja, jawaban berupa TEKS OPSI yang benar
                       (rancangan asli: 'question saja, gold = option_{answer}')"""
    if gaya == "bebas":
        s = "Question: {}\nAnswer:".format(str(r["question"]).strip())
        if dengan_jawaban:
            kunci = str(r["answer_letter"]).strip().lower()
            s += " {}".format(str(r.get("option_" + kunci, "")).strip())
        return s
    s = "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:".format(
        str(r["question"]).strip(), r["option_a"], r["option_b"], r["option_c"], r["option_d"])
    if dengan_jawaban:
        kunci = str(r["answer_letter"]).strip().upper()
        if gaya == "pg-teks":
            # contoh dijawab "C. <teks opsi>" supaya model meniru pola itu dan
            # menghasilkan kalimat, bukan satu huruf
            s += " {}. {}".format(kunci, str(r.get("option_" + kunci.lower(), "")).strip())
        else:
            s += " {}".format(kunci)
    return s


_RE_CFG = re.compile(r"decoder-merge-(\d+)__(.+)\.json$")


def daftar_konfigurasi(repo: str) -> list[dict]:
    """Baca injection_llm/*.json -> [{path, decoder_idx, operasi}]."""
    out = []
    for p in sorted(glob.glob(os.path.join(repo, "injection_llm", "*.json"))):
        m = _RE_CFG.search(os.path.basename(p))
        if not m:
            continue
        out.append({"path": p, "decoder_idx": int(m.group(1)), "operasi": m.group(2)})
    out.sort(key=lambda c: (c["decoder_idx"], c["operasi"]))
    return out


# CATATAN: dua fungsi di bawah TIDAK dipakai saat ini. Metode pengukuran
# kemiripan belum ditetapkan (menunggu keputusan Mas Gabriel). Skrip ini hanya
# menyimpan hasil inference mentah; penilaian dihitung terpisah dari CSV.
def jarak_levenshtein(a: str, b: str) -> int:
    """Jumlah minimum sisip/hapus/ganti karakter untuk mengubah a menjadi b."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    sebelum = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        kini = [i]
        for j, cb in enumerate(b, 1):
            kini.append(min(sebelum[j] + 1, kini[j - 1] + 1, sebelum[j - 1] + (ca != cb)))
        sebelum = kini
    return sebelum[-1]


def ukur_kemiripan(g: str, f: str) -> dict:
    """Empat metrik, semuanya PEKA URUTAN.

    Cosine/Jaccard sengaja TIDAK dipakai: keduanya mengabaikan urutan, sehingga
    teks yang katanya teracak dinilai identik (terbukti pada uji: urutan dibalik
    -> cosine 1,000 padahal Levenshtein 0,077).
    """
    tg, tf = g.split(), f.split()
    n_tok = max(len(tg), len(tf))
    cocok = sum(1 for i in range(min(len(tg), len(tf))) if tg[i] == tf[i])
    beda = -1
    for i in range(n_tok):
        if i >= len(tg) or i >= len(tf) or tg[i] != tf[i]:
            beda = i
            break
    panjang = max(len(g), len(f))
    return {
        "sama_persis": g == f,
        "kesamaan_karakter": round(1 - jarak_levenshtein(g, f) / panjang, 4) if panjang else 1.0,
        "kesamaan_token": round(cocok / n_tok, 4) if n_tok else 1.0,
        "token_pertama_beda": beda,
    }


def stabil_u32(*bagian) -> int:
    import hashlib
    h = hashlib.sha256("|".join(str(b) for b in bagian).encode()).digest()
    return int.from_bytes(h[:4], "big")


# ------------------------------------------------------------------------ main
def main() -> None:  # noqa: C901
    a = parse_args()
    ensure_repo()
    apply_run_env()

    import numpy as np
    if REPO not in sys.path:                      # noqa: F405
        sys.path.insert(0, REPO)                  # noqa: F405
    from llm_inference import Llama
    from graph import modify_onnx_graph

    bits_paksa = urai_bits(a.bits)

    # ---- pool soal & prompt few-shot (pola sama persis dengan mmlu_loglik_bench)
    rows = list(csv.DictReader(open(MMLU_POOL_CSV, newline="", encoding="utf-8")))
    assert rows and "option_a" in rows[0], f"{MMLU_POOL_CSV} skema lama / kosong"
    rnd = random.Random(a.seed)
    idxs = list(range(len(rows)))
    rnd.shuffle(idxs)
    shot_rows = [rows[i] for i in idxs[-a.shots:]] if a.shots else []
    prefix = ("\n\n".join(soal_ke_teks(r, True, a.gaya_prompt) for r in shot_rows) + "\n\n") if shot_rows else ""
    kolam_uji = idxs[: len(idxs) - a.shots]        # soal yang boleh dipakai untuk diuji

    # ---- konfigurasi injeksi
    konfig = daftar_konfigurasi(REPO)              # noqa: F405
    if not konfig:
        raise SystemExit("injection_llm/*.json kosong — jalankan 03_phase2_parser.py dulu")
    pilih_dec = set(int(x) for x in urai_daftar(a.decoders))
    pilih_op = set(urai_daftar(a.operasi))
    if pilih_dec:
        konfig = [c for c in konfig if c["decoder_idx"] in pilih_dec]
    if pilih_op:
        konfig = [c for c in konfig if c["operasi"] in pilih_op]
    # Cuplik berselang (stratified). konfig terurut (decoder_idx, operasi), jadi
    # mengambil N pertama = decoder 0-1 saja: tidak ada cakupan kedalaman sama
    # sekali. Mengambil tiap langkah ke-k menyebar sampel ke seluruh 32 decoder
    # DAN memutar kesembilan operasi. Dipakai untuk uji asap 16 dari 288.
    if a.konfigurasi_langkah > 1:
        konfig = konfig[:: a.konfigurasi_langkah]
    if a.max_configs:
        konfig = konfig[: a.max_configs]
    # Iris konfigurasi: arena VRAM ORT tidak pernah menyusut setelah ratusan sesi
    # faulty dibuat-dibuang, jadi run panjang harus dipecah jadi beberapa PROSES.
    if a.konfigurasi_dari or a.konfigurasi_sampai:
        akhir = a.konfigurasi_sampai if a.konfigurasi_sampai > 0 else len(konfig)
        konfig = konfig[a.konfigurasi_dari:akhir]
    fault_models = urai_daftar(a.fault_models) or list(FAULT_MODELS)   # noqa: F405
    if not konfig:
        raise SystemExit("Tidak ada konfigurasi yang cocok dengan filter --decoders/--operasi")

    # ---- muat model
    spec = json.load(open(os.path.join(REPO, "configs", "my_model.json")))   # noqa: F405
    cfg = {"temperature": 0.001, "topp": 0.1, "max": 8, "poolsize": POOL,    # noqa: F405
           "fp16": (PRECISION != "int8") or USE_FP16_IO,
           "precision": PRECISION, "onnxdir": ONNX_DIR}
    llama = Llama(onnxdir=ONNX_DIR, config=cfg, model_spec=spec)
    llama.fault_config = None
    llama.seed = 42
    hidden_dim = int(llama.hidden_dim)

    base = llama.tokenizer.encode("Answer:", False, False)
    letter_ids = {}
    for L in LETTERS:
        full = llama.tokenizer.encode("Answer: " + L, False, False)
        ekor = full[len(base):]
        assert len(ekor) == 1, f"lanjutan ' {L}' bukan 1 token: {ekor}"
        letter_ids[L] = ekor[0]
    assert len(set(letter_ids.values())) == 4

    n_bit = len(bits_paksa) if bits_paksa else 1
    total = len(konfig) * len(fault_models) * a.runs * n_bit * a.soal
    print(f"[cfg] precision={PRECISION} onnxdir={ONNX_DIR}", flush=True)
    print(f"[cfg] hidden_dim={hidden_dim}  token jawaban={letter_ids}", flush=True)
    print(f"[cfg] konfigurasi={len(konfig)} fault={fault_models} runs={a.runs} "
          f"bits={bits_paksa or 'acak'} soal/sel={a.soal}", flush=True)
    print(f"[cfg] total baris target = {total:,}", flush=True)

    dl = deadline_ts(a.deadline)
    if dl:
        print(f"[cfg] batas waktu {datetime.fromtimestamp(dl):%H:%M} "
              f"(sisa {(dl-time.time())/3600:.2f} jam)", flush=True)

    # ---- resume
    sudah: set[tuple] = set()
    if a.resume and os.path.isfile(a.out):
        for r in csv.DictReader(open(a.out, newline="", encoding="utf-8")):
            sudah.add((int(r["decoder_idx"]), r["operasi"], r["Fault_Model"],
                       int(r["run_id"]), int(r["Bit_Position"]), int(r["idx"])))
        print(f"[resume] {len(sudah)} baris sudah ada", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    baru = not os.path.isfile(a.out)
    f = open(a.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if baru:
        w.writeheader(); f.flush()

    golden_cache: dict[int, dict] = {}

    def reset_kv() -> None:
        llama.pastkeys = [None] * llama.DECODER_COUNT
        llama.pastvalues = [None] * llama.DECODER_COUNT

    eos_id = getattr(llama.tokenizer, "eos_id", 2)

    def satu_jalan(ids_arr, faulty: bool, n_teks: int,
                   operasi: str = "", fault_model: str = "",
                   coords: list[int] | None = None) -> dict:
        """Satu forward pass -> logit A/B/C/D, LALU lanjutkan n_teks token.

        Logit diambil dari lintasan pertama (posisi token terakhir prompt).
        Token lanjutan dipilih greedy/argmax supaya deterministik dan bisa
        diulang. KV cache SENGAJA dipertahankan antar langkah generasi, lalu
        dibersihkan di akhir.
        """
        reset_kv()
        try:
            logits = llama.decode_faulty(ids_arr) if faulty else llama.decode(ids_arr)
            last = np.asarray(logits[:, -1, :]).astype(np.float64).reshape(-1)
            vals = {L: float(last[letter_ids[L]]) for L in LETTERS}
            urut = sorted(vals.values(), reverse=True)

            # Tensor sasaran mengecil saat decode (KV cache -> seq_len 1 pada sumbu
            # token-sekarang, sementara sumbu key/value tumbuh). Indeks datar hasil
            # prefill karena itu harus dipetakan ulang tiap langkah, bukan sekali.
            # Versi lama memakai `rand_idx % hidden_dim`, yang hanya benar kalau
            # rand_idx dibangun dengan asumsi [1, n, 4096]. Dengan koordinat yang
            # diundi dari seluruh tensor, indeks itu bisa keluar batas -- dan
            # ScatterND/GatherND pada graf ini tidak memeriksa batas, jadi keluar
            # batas berarti CUDA error 700, bukan sekadar salah alamat.
            remap = None
            if faulty and llama.fault_config is not None:
                if coords and operasi:
                    remap = lambda: indeks_lanjutan(
                        operasi, fault_model, coords,
                        int(llama.pastkeys[0].shape[-2]))
                else:
                    tetap = int(llama.fault_config["rand_idx"]) % hidden_dim
                    remap = lambda: tetap

            # Berhenti di baris baru: jawaban berakhir sebelum "\n". Tanpa ini
            # model lanjut menulis soal berikutnya (pola 5-shot), dan token
            # lanjutan yang identik itu MENGENCERKAN skor kemiripan --
            # terbukti di uji lokal: jawaban berubah C->D tapi kemiripan 0,97.
            teks_ids: list[int] = []
            for _ in range(max(0, n_teks)):
                nxt = int(np.argmax(last))
                if nxt == eos_id:
                    break
                teks_ids.append(nxt)
                if "\n" in llama.tokenizer.decode(teks_ids):
                    break
                arr1 = np.array([[nxt]], dtype=np.int64)
                if faulty:
                    # past_len dibaca ulang tiap langkah karena cache bertambah
                    llama.fault_config["rand_idx"] = remap()
                    logits = llama.decode_faulty(arr1)
                else:
                    logits = llama.decode(arr1)
                last = np.asarray(logits[:, -1, :]).astype(np.float64).reshape(-1)

            teks = llama.tokenizer.decode(teks_ids) if teks_ids else ""
            teks = teks.split("\n")[0].strip()   # baris pertama saja
            # n_tok = jumlah token yang BENAR-BENAR dihasilkan, bukan batasnya.
            # Kalau n_tok == n_teks pada banyak baris, berarti jawaban terpotong
            # dan --token-teks harus dinaikkan.
            return {"huruf": max(vals, key=vals.get), "logit": vals,
                    "margin": urut[0] - urut[1], "teks": teks,
                    "n_tok": len(teks_ids)}
        finally:
            reset_kv()

    n_tulis = n_beda_teks = 0
    t_awal = time.time()
    berhenti = False

    for c in konfig:
        if berhenti:
            break
        cfg_json = json.load(open(c["path"]))
        # Parser lama kadang memilih tensor sebelum fake-quant yang bukan operand
        # langsung MatMul (terutama Q dan SKOR). Normalisasi sekali per config.
        cfg_json = fix_input_tensor(cfg_json)
        if cfg_json is None:
            print(f"[skip] {c['decoder_idx']}/{c['operasi']}: input tensor tidak valid", flush=True)
            continue
        for fm in fault_models:
            if berhenti:
                break
            faulty_path = None
            try:
                # satu graf faulty dipakai ulang untuk seluruh run/bit/soal kombinasi ini
                # Tulis graf injeksi ke TMPDIR, BUKAN ke folder ONNX. Dua alasan:
                # (a) folder ONNX bisa read-only (mis. SSD bertanda dirty)
                # (b) sisa *_injected.onnx di folder ONNX terbaca sebagai decoder
                #     oleh memory pool saat startup
                tmp_inj = os.environ.get("TMPDIR") or "/tmp"
                os.makedirs(tmp_inj, exist_ok=True)
                cfg_json = dict(cfg_json)
                cfg_json["output_path"] = os.path.join(
                    tmp_inj, f"inj_d{c['decoder_idx']}_{c['operasi']}_{fm}.onnx")
                faulty_path = modify_onnx_graph(cfg_json, cfg, fm)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {c['decoder_idx']}/{c['operasi']}/{fm}: "
                      f"modify_onnx_graph gagal: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue

            try:
                # Bit_Position adalah loop LUAR, di atas run_id: seluruh pengujian
                # pada bit 0 selesai dulu, baru pindah ke bit 1, dan seterusnya.
                # Sebelumnya run_id yang di luar, sehingga kolom Bit_Position di
                # sheet terbaca 0,1,2,...,15,0,1,2,... — dibaca sebagai acak.
                # Isi barisnya TIDAK berubah: benih dan benih_soal dihitung dari
                # (seed, decoder, operasi, fm, run_id, bit, k), jadi tidak
                # bergantung urutan iterasi. Yang berubah hanya urutan baris,
                # sehingga kunci --resume yang lama tetap cocok.
                daftar_bit = bits_paksa if bits_paksa else [None]
                for bit in daftar_bit:
                    for run_id in range(1, a.runs + 1):
                        for k in range(a.soal):
                            if dl and time.time() >= dl:
                                print("[stop] batas waktu tercapai", flush=True)
                                berhenti = True
                                break

                            # Seed INJEKSI memuat fm & bit — kerusakannya memang
                            # harus bervariasi antar fault model.
                            benih = stabil_u32(a.seed, c["decoder_idx"], c["operasi"],
                                               fm, run_id, bit if bit is not None else -1, k)
                            rng = np.random.default_rng(benih)
                            bit_width = 8 if PRECISION == "int8" else 16
                            bit_pos = int(rng.integers(0, bit_width)) if bit is None else int(bit)
                            # Seed SOAL memuat fault model dan bit, jadi SETIAP BARIS
                            # menarik soal acak sendiri. Versi sebelumnya sengaja
                            # mengecualikan keduanya supaya keenam fault model memakai
                            # soal yang sama: golden cukup dihitung sekali dan
                            # perbandingannya berpasangan. Itu ditinggalkan atas
                            # permintaan — konsekuensinya golden tidak lagi bisa
                            # dipakai bersama, sehingga forward pass naik dari
                            # 50.400 menjadi 86.400 untuk 43.200 baris.
                            benih_soal = stabil_u32(a.seed, "soal", c["decoder_idx"],
                                                    c["operasi"], fm,
                                                    bit if bit is not None else -1,
                                                    run_id, k)
                            i = kolam_uji[benih_soal % len(kolam_uji)]

                            kunci_baris = (c["decoder_idx"], c["operasi"], fm,
                                           run_id, bit_pos, i)
                            if kunci_baris in sudah:
                                continue

                            r = rows[i]
                            prompt = prefix + soal_ke_teks(r, False, a.gaya_prompt)
                            t0 = time.time()
                            try:
                                ids = llama.tokenizer.encode(prompt.strip(), True, False)
                                arr = np.array(ids, dtype=np.int64).reshape((1, len(ids)))
                                # 'penuh': bentuk tensor sasaran diambil per operasi DAN
                                #   per fault model, lalu rand_idx diundi merata dari
                                #   seluruh elemennya. Ini yang membuat w, h, dan c
                                #   benar-benar acak sejauh bentuknya mengizinkan.
                                #   Untuk [1,n,feat] c memang selalu 0 karena sumbu depan
                                #   berukuran 1; c baru bermakna (indeks head 0-31) pada
                                #   dua BMM attention yang tensornya 4-D.
                                # 'akhir'/'acak': asumsi lama [1, n, hidden_dim] untuk semua
                                #   operasi. Dipertahankan hanya agar run lama bisa
                                #   direproduksi; asumsi itu salah untuk gate/up/down dan
                                #   kedua BMM, sehingga sebagian tensor tak pernah tersentuh.
                                _rng2 = np.random.default_rng(benih)
                                if a.posisi_token == "penuh":
                                    inject_shape, _ = lokasi_tensor(c["operasi"], fm, len(ids))
                                    # JANGAN pakai nama 'total' di sini: itu nama
                                    # target jumlah baris dari baris 415, dan menimpanya
                                    # membuat penghitung progres melaporkan jumlah elemen
                                    # tensor (mis. 45.088.768) sebagai penyebut.
                                    n_elemen = 1
                                    for dim in inject_shape:
                                        n_elemen *= dim
                                    rand_idx = int(_rng2.integers(0, n_elemen))
                                else:
                                    inject_shape = [1, len(ids), hidden_dim]
                                    if a.posisi_token == "akhir":
                                        awal = (len(ids) - 1) * hidden_dim
                                        rand_idx = awal + int(_rng2.integers(0, hidden_dim))
                                    else:
                                        rand_idx = int(_rng2.integers(0, max(1, len(ids) * hidden_dim)))
                                inject_coords, inject_w, inject_h, inject_c = \
                                    koordinat_metadata(rand_idx, inject_shape)

                                if i in golden_cache:
                                    g = golden_cache[i]
                                else:
                                    llama.fault_config = None
                                    g = satu_jalan(arr, faulty=False, n_teks=a.token_teks)
                                    golden_cache[i] = g

                                llama.fault_config = {
                                    "target_decoder_idx": c["decoder_idx"],
                                    "target_token_idx": 0,
                                    "faulty_decoder_path": faulty_path,
                                    "bit_position": bit_pos,
                                    "inject_seed": benih,
                                    "rand_idx": rand_idx,
                                }
                                fl = satu_jalan(
                                    arr, faulty=True, n_teks=a.token_teks,
                                    operasi=c["operasi"], fault_model=fm,
                                    coords=(inject_coords if a.posisi_token == "penuh" else None))
                            except Exception as e:  # noqa: BLE001
                                print(f"[err] d{c['decoder_idx']}/{c['operasi']}/{fm}"
                                      f"/bit{bit_pos}: {type(e).__name__}: {str(e)[:110]}",
                                      flush=True)
                                llama.fault_config = None
                                continue
                            finally:
                                llama.fault_config = None

                            kunci = str(r.get("answer_letter", "")).strip().upper()
                            n_tulis += 1
                            n_beda_teks += (g.get("teks", "") != fl.get("teks", ""))
                            w.writerow({
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                "decoder_idx": c["decoder_idx"], "operasi": c["operasi"],
                                "Fault_Model": fm, "run_id": run_id, "Bit_Position": bit_pos,
                                "rand_idx": rand_idx, "inject_seed": benih,
                                "idx": i, "sample_id": r["sample_id"],
                                "subject": r.get("subject", ""),
                                "subject_category": r.get("subject_category", ""),
                                "answer_letter": kunci,
                                "golden_teks": g.get("teks", ""),
                                "faulty_teks": fl.get("teks", ""),
                                **{f"g_logit_{L}": round(g["logit"][L], 4) for L in LETTERS},
                                **{f"f_logit_{L}": round(fl["logit"][L], 4) for L in LETTERS},
                                "n_prompt_tokens": len(ids),
                                "batas_token_teks": a.token_teks,
                                "n_tok_golden": g.get("n_tok", 0),
                                "n_tok_faulty": fl.get("n_tok", 0),
                                "detik": round(time.time() - t0, 3),
                                "inject_shape": json.dumps(inject_shape, separators=(",", ":")),
                                "inject_coords": json.dumps(inject_coords, separators=(",", ":")),
                                "inject_w": inject_w, "inject_h": inject_h, "inject_c": inject_c,
                                # persis string yang ditokenisasi, bukan versi sebelum strip
                                "prompt_lengkap": prompt.strip(),
                            })
                            f.flush()

                            if n_tulis % 20 == 0:
                                laju = n_tulis / max(1e-9, time.time() - t_awal)
                                print(f"  {n_tulis:6}/{total} baris | teks beda "
                                      f"{100*n_beda_teks/n_tulis:5.1f}% | {laju:.2f} baris/dtk",
                                      flush=True)
                        if berhenti:
                            break
                    if berhenti:
                        break
            finally:
                # buang graf faulty + sesi-nya supaya disk & VRAM tidak menumpuk
                try:
                    if hasattr(llama, "faulty_decoders") and faulty_path in llama.faulty_decoders:
                        sesi = llama.faulty_decoders.pop(faulty_path)
                        try:
                            del sesi.sess          # lepas handle sesi ORT
                        except Exception:  # noqa: BLE001
                            pass
                        del sesi
                    import gc
                    gc.collect()                   # dorong pembebasan VRAM
                except Exception:  # noqa: BLE001
                    pass
                if faulty_path and os.path.exists(faulty_path):
                    try:
                        os.remove(faulty_path)
                    except OSError:
                        pass

    f.close()
    dur = time.time() - t_awal
    print(f"\n=== SELESAI: {n_tulis} baris | teks golden != faulty pada "
          f"{n_beda_teks}/{n_tulis} = {100*n_beda_teks/max(1,n_tulis):.2f}% "
          f"(sekadar info, BUKAN metrik resmi) ===", flush=True)
    print(f"waktu {dur/60:.1f} menit ({dur/max(1,n_tulis):.2f} dtk/baris) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
