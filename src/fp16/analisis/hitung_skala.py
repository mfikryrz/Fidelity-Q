# -*- coding: utf-8 -*-
"""Menghitung RUNS yang muat dalam sisa saldo, dari angka yang TERUKUR.

Dipakai setelah uji asap. Dua biaya dibedakan, karena keduanya berperilaku beda:

  * biaya TETAP  — membangun graf ONNX yang sudah disuntik. Jumlahnya
    288 konfigurasi x 6 fault model = 1.728 graf, TIDAK peduli RUNS berapa.
  * biaya PER BARIS — inference golden + faulty. Ini yang naik seiring RUNS.

Karena run dipotong per 8 konfigurasi, tenggat yang terlewat berarti konfigurasi
di ujung tidak kebagian baris sama sekali. Cakupan jadi timpang. Maka RUNS
diambil 80% dari hasil hitungan supaya selesai utuh.

Pakai:
  python hitung_skala.py --saldo 6.42 --tarif 0.956 --detik-per-baris 0.52 \
                         --detik-per-graf 2.4 --terpakai 0.9
"""
from __future__ import annotations

import argparse

N_KONFIG = 288
N_FM = 6
N_GRAF = N_KONFIG * N_FM          # 1.728 graf, dibangun sekali per potongan
BARIS_PER_RUN = N_KONFIG * N_FM   # 1.728 baris untuk setiap tambahan 1 run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--saldo", type=float, required=True, help="USD tersisa")
    p.add_argument("--tarif", type=float, required=True, help="USD per jam")
    p.add_argument("--detik-per-baris", type=float, required=True,
                   help="dari uji asap: detik inference golden+faulty per baris")
    p.add_argument("--detik-per-graf", type=float, default=2.4,
                   help="waktu membangun satu graf tersuntik")
    p.add_argument("--detik-muat", type=float, default=0.0,
                   help="waktu memuat model, terjadi SEKALI TIAP POTONGAN")
    p.add_argument("--potongan", type=int, default=8,
                   help="konfigurasi per proses (POTONGAN_CFG)")
    p.add_argument("--terpakai", type=float, default=0.0,
                   help="jam yang sudah terpakai untuk setup + uji asap")
    p.add_argument("--cadangan", type=float, default=0.5, help="jam cadangan")
    p.add_argument("--aman", type=float, default=0.8,
                   help="ambil sekian bagian dari hasil hitungan")
    a = p.parse_args()

    n_potongan = -(-N_KONFIG // a.potongan)   # pembulatan ke atas

    jam_total = a.saldo / a.tarif
    jam_pakai = jam_total - a.terpakai - a.cadangan
    jam_graf = N_GRAF * a.detik_per_graf / 3600.0
    jam_muat = n_potongan * a.detik_muat / 3600.0
    jam_baris = jam_pakai - jam_graf - jam_muat

    print(f"Saldo            : ${a.saldo:.2f} / ${a.tarif:.3f} per jam = {jam_total:.2f} jam")
    print(f"Terpakai         : {a.terpakai:.2f} jam (setup + uji asap)")
    print(f"Cadangan         : {a.cadangan:.2f} jam")
    print(f"Tersedia         : {jam_pakai:.2f} jam")
    print()
    print(f"Biaya tetap graf : {N_GRAF:,} graf x {a.detik_per_graf:.2f} dtk = {jam_graf:.2f} jam")
    print(f"Biaya muat model : {n_potongan} potongan x {a.detik_muat:.0f} dtk = {jam_muat:.2f} jam")
    print(f"Sisa untuk baris : {jam_baris:.2f} jam")

    if jam_baris <= 0:
        print()
        print("TIDAK CUKUP. Membangun graf saja sudah menghabiskan waktu.")
        print("Pilihan: kurangi cakupan konfigurasi, atau isi saldo.")
        raise SystemExit(1)

    baris_muat = int(jam_baris * 3600 / a.detik_per_baris)
    runs_mentah = baris_muat / BARIS_PER_RUN
    runs = max(1, int(runs_mentah * a.aman))

    print()
    print(f"Baris yang muat  : {baris_muat:,}")
    print(f"RUNS mentah      : {runs_mentah:.2f}")
    print(f"RUNS dipakai     : {runs}  (dibatasi {a.aman:.0%} supaya selesai utuh)")
    print()
    baris = runs * BARIS_PER_RUN
    jam_nyata = jam_graf + jam_muat + baris * a.detik_per_baris / 3600.0
    print(f"-> {baris:,} baris  ({N_KONFIG} konfigurasi x {N_FM} fault model x {runs} run)")
    print(f"-> perkiraan {jam_nyata:.2f} jam, biaya ${jam_nyata * a.tarif:.2f}")
    print()
    print("Perintah:")
    print(f"  BUDGET_HOURS={jam_pakai:.1f} RUNS={runs} POTONGAN_CFG=8 bash run_massal.sh")


if __name__ == "__main__":
    main()
