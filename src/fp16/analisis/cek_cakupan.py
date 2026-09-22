#!/usr/bin/env python3
"""Verifikasi cakupan: setiap decoder x setiap operasi benar-benar dievaluasi.

Permintaan pembimbing, rapat 11 Sep 2026 (0:25:39-0:27:04):
  "nanti pas eksperimen masal semua layer index dan semua operasi diinjeksikan,
   dievaluasi kan... pas smoke test, nanti dicek ya kalau setiap operasinya,
   setiap dekoder itu dievaluasi atau enggak."

Jalan di laptop, tanpa GPU:
    python cek_cakupan.py hasil.csv [--bit 16] [--runs 2]

Keluar dengan kode 1 kalau ada sel yang kosong atau tidak penuh, supaya bisa
dipakai sebagai gerbang di skrip.
"""
import argparse, csv, collections, sys

OPS = ["mlp_down_proj_MatMul", "mlp_gate_proj_MatMul", "mlp_up_proj_MatMul",
       "self_attn_MatMul", "self_attn_MatMul_1", "self_attn_k_proj_MatMul",
       "self_attn_o_proj_MatMul", "self_attn_q_proj_MatMul", "self_attn_v_proj_MatMul"]
FM = ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]
N_DECODER = 32


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--bit", type=int, default=16, help="jumlah bit (FP16=16, INT8=8)")
    p.add_argument("--runs", type=int, default=2, help="injeksi per bit")
    p.add_argument("--penuh", action="store_true",
                   help="wajibkan seluruh 288 konfigurasi (untuk run masal)")
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    if not rows:
        print("CSV kosong"); return 1
    perlu = a.bit * a.runs                       # baris per (decoder, operasi, fault model)

    n = collections.Counter()
    for r in rows:
        n[(int(r["decoder_idx"]), r["operasi"], r["Fault_Model"])] += 1

    dec_ada = sorted({int(r["decoder_idx"]) for r in rows})
    op_ada = sorted({r["operasi"] for r in rows})
    cfg_ada = sorted({(int(r["decoder_idx"]), r["operasi"]) for r in rows})

    print(f"berkas   : {a.csv}")
    print(f"record   : {len(rows)}   kolom: {len(rows[0])}")
    print(f"harapan  : {perlu} baris per (decoder x operasi x fault model)\n")

    print(f"decoder  : {len(dec_ada)}/{N_DECODER}  {dec_ada}")
    kurang_dec = [d for d in range(N_DECODER) if d not in dec_ada]
    if kurang_dec:
        print(f"           TIDAK ADA: {kurang_dec}")
    print(f"operasi  : {len(op_ada)}/{len(OPS)}")
    for o in OPS:
        if o not in op_ada:
            print(f"           TIDAK ADA: {o}")
    print(f"konfigurasi (decoder x operasi): {len(cfg_ada)}"
          f"{f'/{N_DECODER*len(OPS)}' if a.penuh else ''}")

    # sel yang tidak penuh
    cacat = []
    for d, o in cfg_ada:
        for fm in FM:
            ada = n.get((d, o, fm), 0)
            if ada != perlu:
                cacat.append((d, o, fm, ada))
    print(f"\nsel (config x fault model) : {len(cfg_ada)*len(FM)}")
    print(f"  penuh   : {len(cfg_ada)*len(FM) - len(cacat)}")
    print(f"  cacat   : {len(cacat)}")
    for d, o, fm, ada in cacat[:20]:
        print(f"      d{d}/{o} {fm}: {ada}/{perlu}")
    if len(cacat) > 20:
        print(f"      ... dan {len(cacat)-20} lagi")

    # sebaran bit harus rata — inti permintaan "bit tidak boleh acak"
    bit = collections.Counter(int(r["Bit_Position"]) for r in rows)
    hilang_bit = [b for b in range(a.bit) if b not in bit]
    print(f"\nbit 0-{a.bit-1}: {'LENGKAP' if not hilang_bit else f'HILANG {hilang_bit}'}"
          f"  sebaran {min(bit.values())}-{max(bit.values())} per bit")

    gagal = bool(cacat) or bool(hilang_bit)
    penuh_288 = len(cfg_ada) == N_DECODER * len(OPS)
    if a.penuh and not penuh_288:
        print(f"\nGAGAL: run masal wajib {N_DECODER*len(OPS)} konfigurasi, "
              f"ada {len(cfg_ada)}")
        gagal = True

    # Bedakan dua hal yang mudah tertukar: "sel yang ADA sudah penuh" bukan
    # berarti "seluruh 288 konfigurasi tercakup". Tanpa pembedaan ini, keluaran
    # uji asap 2 konfigurasi terbaca seolah cakupannya sudah lengkap.
    print()
    if gagal:
        print("HASIL: ADA YANG BELUM LENGKAP")
    elif penuh_288:
        print("HASIL: CAKUPAN PENUH — seluruh 288 konfigurasi terisi rata")
    else:
        print(f"HASIL: sel yang ada SUDAH PENUH ({len(cfg_ada)*len(FM)}/{len(cfg_ada)*len(FM)}), "
              f"bit rata")
        print(f"       TAPI ini BUKAN cakupan penuh: {len(cfg_ada)}/{N_DECODER*len(OPS)} "
              f"konfigurasi, {len(dec_ada)}/{N_DECODER} decoder, {len(op_ada)}/{len(OPS)} operasi.")
        print(f"       Untuk run masal jalankan dengan --penuh agar ini dianggap gagal.")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
