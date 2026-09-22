#!/usr/bin/env bash
# ============================================================================
# finalisasi.sh — mengunduh hasil akhir, menilai, dan mengemas semuanya
#                 ke laptop untuk reproduksi di kemudian hari.
#
# Dijalankan SETELAH instance berhenti (atau sesaat sebelum dihapus).
#
# Pakai:  bash finalisasi.sh <port> <host>
# ============================================================================
set -uo pipefail

PORT="${1:-23676}"
HOST="${2:-104.37.174.34}"
LOKAL="/home/fikry/hasil_pgteks"
METODE="/home/fikry/metode_massal_7b"
R="/workspace/fidelity/running_experiment_7b"
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

mkdir -p "$LOKAL" "$METODE"

say "1 · Mengunduh hasil dari instance"
timeout 600 scp $SSHO -P "$PORT" \
  "root@$HOST:$R/hasil_massal/massal_pgteks.csv" \
  "$LOKAL/massal_pgteks_final.csv" 2>&1 | tail -1
timeout 300 scp $SSHO -P "$PORT" \
  "root@$HOST:$R/hasil_massal/RINGKASAN.txt" \
  "$LOKAL/RINGKASAN_final.txt" 2>/dev/null || true
# log lengkap: berguna untuk melacak kegagalan kalau ada
timeout 300 scp $SSHO -P "$PORT" \
  "root@$HOST/../workspace/fidelity/logs/massal.log" \
  "$LOKAL/log_massal_final.txt" 2>/dev/null || \
timeout 300 scp $SSHO -P "$PORT" \
  "root@$HOST:/workspace/fidelity/logs/massal.log" \
  "$LOKAL/log_massal_final.txt" 2>/dev/null || true

[ -s "$LOKAL/massal_pgteks_final.csv" ] || { echo "GAGAL mengunduh CSV" >&2; exit 1; }

say "2 · Memeriksa kelengkapan"
python3 - "$LOKAL/massal_pgteks_final.csv" <<'PYEOF'
import csv, sys
from collections import Counter
r = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
print(f"  total baris   : {len(r):,}")
print(f"  kombinasi     : {len({(x['decoder_idx'], x['operasi']) for x in r})} / 288")
print(f"  soal unik     : {len({x['sample_id'] for x in r}):,}")
print()
print("  Kelengkapan tiap pengulangan (1.728 = lengkap):")
c = Counter(int(x["run_id"]) for x in r)
penuh = []
for k in sorted(c):
    tanda = "LENGKAP" if c[k] == 1728 else "sebagian"
    if c[k] == 1728:
        penuh.append(k)
    print(f"    run {k}: {c[k]:5,} / 1728  {tanda}")
print()
if penuh:
    n = max(penuh) if list(range(1, max(penuh) + 1)) == penuh else len(penuh)
    print(f"  -> TINGKAT BERSIH TERTINGGI: RUNS={max(penuh)} = {max(penuh)*1728:,} baris")
    print(f"     Untuk analisis yang cakupannya merata, saring run_id <= {max(penuh)}.")
fm = Counter(x["Fault_Model"] for x in r)
print()
print("  Per jenis kerusakan:")
for k, v in sorted(fm.items()):
    print(f"    {k:<16}{v:6,}")
PYEOF

say "3 · Menghitung penilaian (di laptop, tanpa GPU)"
python3 "$METODE/hitung_penilaian.py" \
    "$LOKAL/massal_pgteks_final.csv" "$LOKAL/massal_pgteks_final_dinilai.csv" | head -3

say "4 · Ringkasan cakupan"
python3 "$METODE/ringkas_pgteks.py" "$LOKAL/massal_pgteks_final.csv" \
    | tee "$LOKAL/RINGKASAN_final.txt" | head -30

say "5 · Menyalin kode & konfigurasi terbaru ke paket metode"
cp -v /home/fikry/staging_bootstrap/{mass_loglik_inject.py,hitung_penilaian.py,ringkas_pgteks.py,\
hitung_skala.py,run_massal.sh,bootstrap_vast.sh,finalisasi.sh} "$METODE/" 2>/dev/null | tail -3

say "6 · Isi paket metode"
ls -la "$METODE" | tail -n +2 | awk '{printf "  %8s  %s\n", $5, $9}'
echo "  konfigurasi injeksi: $(ls "$METODE/injection_llm"/*.json 2>/dev/null | wc -l)"
echo "  ukuran total       : $(du -sh "$METODE" | cut -f1)"

say "SELESAI"
echo "Hasil    : $LOKAL/massal_pgteks_final.csv"
echo "Dinilai  : $LOKAL/massal_pgteks_final_dinilai.csv"
echo "Metode   : $METODE/  (siap dipakai reproduksi)"
echo "Panduan  : $LOKAL/PANDUAN_43K.md"
