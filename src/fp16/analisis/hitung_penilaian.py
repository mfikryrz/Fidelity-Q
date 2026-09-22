# -*- coding: utf-8 -*-
"""Menghitung kolom PENILAIAN dari CSV hasil inference mentah.

Sengaja TERPISAH dari mass_loglik_inject.py. Alasannya:

  * Eksperimen (mahal, butuh GPU) hanya menyimpan yang tidak bisa dihitung
    ulang: teks mentah golden & faulty, logit mentah, dan identitas kombinasi.
  * Penilaian (murah, CPU biasa) dihitung dari CSV itu. Kalau metodenya berubah,
    cukup jalankan skrip ini lagi — TIDAK perlu menyewa GPU lagi.

Pakai:  python hitung_penilaian.py masuk.csv keluar.csv
"""
from __future__ import annotations

import csv
import sys

LETTERS = ("A", "B", "C", "D")


def jarak_levenshtein(a: str, b: str) -> int:
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


def dari_logit(r: dict, awalan: str) -> dict:
    """huruf pemenang + margin, dihitung dari logit yang tersimpan."""
    v = {L: float(r[f"{awalan}_logit_{L}"]) for L in LETTERS}
    urut = sorted(v.values(), reverse=True)
    return {"huruf": max(v, key=v.get), "margin": round(urut[0] - urut[1], 4)}


def dari_teks(g: str, f: str) -> dict:
    """metrik kemiripan, dihitung dari teks yang tersimpan."""
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


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    masuk, keluar = sys.argv[1], sys.argv[2]
    baris = list(csv.DictReader(open(masuk, newline="", encoding="utf-8")))
    if not baris:
        raise SystemExit("CSV kosong")

    tambahan = ["golden_letter", "faulty_letter", "is_error",
                "golden_benar", "faulty_benar", "golden_margin", "faulty_margin",
                "sama_persis", "kesamaan_karakter", "kesamaan_token", "token_pertama_beda"]
    kolom_asal = list(baris[0].keys())          # dicatat SEBELUM baris dimutasi
    kolom = kolom_asal + [k for k in tambahan if k not in kolom_asal]

    with open(keluar, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=kolom)
        w.writeheader()
        for r in baris:
            g = dari_logit(r, "g")
            f = dari_logit(r, "f")
            kunci = str(r.get("answer_letter", "")).strip().upper()
            r.update({
                "golden_letter": g["huruf"], "faulty_letter": f["huruf"],
                "is_error": g["huruf"] != f["huruf"],
                "golden_benar": g["huruf"] == kunci,
                "faulty_benar": f["huruf"] == kunci,
                "golden_margin": g["margin"], "faulty_margin": f["margin"],
                **dari_teks(r.get("golden_teks", ""), r.get("faulty_teks", "")),
            })
            w.writerow(r)

    print(f"{len(baris)} baris | {len(kolom_asal)} kolom -> {len(kolom)} kolom")
    print(f"ditulis ke {keluar}")
    print()
    print("Kolom yang ditambahkan (SEMUANYA dihitung dari CSV masukan,")
    print("tanpa menjalankan model lagi):")
    for k in tambahan:
        asal = "dari logit" if k in ("golden_letter", "faulty_letter", "is_error",
                                     "golden_benar", "faulty_benar",
                                     "golden_margin", "faulty_margin") else "dari teks"
        print(f"  {k:<20} <- {asal}")


if __name__ == "__main__":
    main()
