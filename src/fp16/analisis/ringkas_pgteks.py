# -*- coding: utf-8 -*-
"""Ringkasan CAKUPAN untuk CSV mode pg-teks.

SENGAJA tidak menghitung SDC/is_error. Di mode pg-teks tidak ada kolom penilaian
sama sekali — metodenya belum diputuskan dan dihitung belakangan di laptop lewat
hitung_penilaian.py. Yang diringkas di sini hanya hal-hal yang menjawab
"eksperimennya jalan atau tidak":

  * berapa baris, berapa kombinasi tercakup
  * teks golden/faulty benar-benar terisi (bukan kosong, bukan satu huruf)
  * ada berapa baris yang teksnya berbeda -> injeksi berdampak atau tidak
  * apakah jawaban mentok di batas token (kalau ya, --token-teks kurang)

Pakai:  python ringkas_pgteks.py hasil.csv
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

FM = ("INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    baris = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
    if not baris:
        raise SystemExit("CSV kosong")

    def angka(r: dict, k: str) -> int:
        try:
            return int(r.get(k) or 0)
        except ValueError:
            return 0

    n = len(baris)
    kombinasi = {(r["decoder_idx"], r["operasi"]) for r in baris}
    per_fm = Counter(r["Fault_Model"] for r in baris)
    batas = max(angka(r, "batas_token_teks") for r in baris)

    g_kosong = sum(1 for r in baris if not (r.get("golden_teks") or "").strip())
    f_kosong = sum(1 for r in baris if not (r.get("faulty_teks") or "").strip())
    beda = sum(1 for r in baris if r.get("golden_teks") != r.get("faulty_teks"))
    # Menyentuh batas token DIBEDAKAN golden vs faulty, karena artinya beda:
    #   golden mentok -> jawaban benar ikut terpotong. Ini MASALAH, batas kurang.
    #   faulty mentok -> model rusak jadi mengoceh tanpa berhenti. Ini TEMUAN,
    #                    justru bukti kerusakan; menaikkan batas tidak menolong.
    mentok_g = sum(1 for r in baris if batas and angka(r, "n_tok_golden") >= batas)
    mentok_f = sum(1 for r in baris if batas and angka(r, "n_tok_faulty") >= batas)
    tok = sorted(angka(r, "n_tok_golden") for r in baris)
    tok_median = tok[len(tok) // 2] if tok else 0
    tok_max = tok[-1] if tok else 0
    panjang = sorted(len((r.get("golden_teks") or "").split()) for r in baris)
    median = panjang[len(panjang) // 2] if panjang else 0
    detik = [float(r["detik"]) for r in baris if r.get("detik")]

    print("RINGKASAN CAKUPAN — mode pg-teks (tanpa penilaian)")
    print("=" * 62)
    print(f"Baris           : {n:,}")
    print(f"Kombinasi        : {len(kombinasi)} / 288  (decoder x operasi)")
    print(f"Fault model      : {len(per_fm)} / 6")
    for k in FM:
        print(f"  {k:<16}{per_fm.get(k, 0):>8,}")
    print()
    print("Teks mentah")
    print(f"  golden kosong  : {g_kosong:,}   (harus 0)")
    print(f"  faulty kosong  : {f_kosong:,}")
    print(f"  teks berbeda   : {beda:,} = {beda / n * 100:.2f}%  (harus > 0)")
    print(f"  panjang median : {median} kata")
    print(f"  token golden   : median {tok_median}, maks {tok_max}  (batas {batas})")
    print(f"  golden mentok  : {mentok_g:,} baris  (harus 0 — kalau tidak, "
          f"naikkan --token-teks)")
    print(f"  faulty mentok  : {mentok_f:,} baris = {mentok_f / n * 100:.1f}%  "
          f"(bukan masalah: model rusak mengoceh)")
    if detik:
        print()
        print(f"Kecepatan        : {sum(detik) / len(detik):.3f} detik/baris")
    print()
    print("Penilaian (kesamaan, is_error, dst) dihitung terpisah:")
    print("  python hitung_penilaian.py hasil.csv hasil_dinilai.csv")


if __name__ == "__main__":
    main()
