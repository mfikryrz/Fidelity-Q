#!/usr/bin/env bash
# pantau_int8.sh — catat satu baris kesehatan tiap 10 menit, lewat cron.
#
# HANYA MENCATAT, tidak pernah membunuh atau menghidupkan apa pun. Tugas
# bertindak ada pada penjaga_vram.sh; skrip ini jejak forensik supaya kalau
# ada yang aneh semalam, paginya bisa dilihat KAPAN mulainya.
#
# Menandai tiga mode kegagalan yang sudah benar-benar terjadi di proyek ini:
#
#   MACET    baris tidak bertambah padahal worker hidup. Penjaga tidak
#            mendeteksi ini — ia hanya memeriksa proses hidup dan error alokasi.
#   CPU?     GPU nyaris kosong tapi CPU 100%. Ini yang terjadi saat arena
#            disetel 25 GB: sesi tidak muat di GPU lalu jatuh ke CPU, dan
#            terlihat "sehat" karena proses hidup dan tidak ada error.
#   WORKER=n n bukan 1. 0 = tidak jalan, >1 = dua penulis di satu CSV.
set -uo pipefail

ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=$ROOT/running_experiment_7b/hasil_massal/massal_int8_288.csv
LOG=$ROOT/logs/pantau.log
STATE=$ROOT/logs/.pantau.state
KUNCI=$ROOT/logs/.pantau.lock

exec 9>"$KUNCI"
flock -n 9 || exit 0

N=0
[ -f "$OUT" ] && N=$("$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null || echo 0)
N=${N:-0}

W=$(pgrep -f mass_loglik_inject | wc -l)
P=0; pgrep -f "pengawas_fp16[.]sh" >/dev/null 2>&1 && P=1
ERR=$(cat "$ROOT/logs/fp16.log" 2>/dev/null | awk '/\[err\]/{n++} END{print n+0}')

read -r VRAM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
                      --format=csv,noheader,nounits 2>/dev/null | tr -d ',' )
VRAM=${VRAM:-0}; UTIL=${UTIL:-0}

CPU=0; UMUR=0
PID=$(pgrep -f mass_loglik_inject | head -1)
if [ -n "$PID" ]; then
  CPU=$(top -b -n1 -p "$PID" 2>/dev/null | tail -1 | awk '{printf "%.0f", $9}')
  # umur worker dalam detik — pembeda kunci antara "sibuk wajar" dan "jatuh ke CPU"
  UMUR=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
fi
CPU=${CPU:-0}; UMUR=${UMUR:-0}

CFG=$(ps -eo cmd | grep mass_loglik_inject | grep -v grep \
      | grep -oE "konfigurasi-dari [0-9]+" | awk '{print $2}' | head -1)
CFG=${CFG:-"-"}

SEBELUM=0; DIAM_N=0
[ -f "$STATE" ] && . "$STATE"

TANDA=""
[ "$W" -ne 1 ]                       && TANDA="$TANDA WORKER=$W"
[ "$P" -ne 1 ]                       && TANDA="$TANDA PENGAWAS-MATI"
if [ "$N" -eq "$SEBELUM" ]; then
  DIAM_N=$(( DIAM_N + 1 ))
  # 4 siklus = ~40 menit tanpa baris baru. Satu potongan penuh ~17 menit,
  # dan sapu ulang bisa lama, jadi ambang longgar supaya tidak berteriak palsu.
  [ "$DIAM_N" -ge 4 ] && [ "$W" -ge 1 ] && TANDA="$TANDA MACET(${DIAM_N}x10mnt)"
else
  DIAM_N=0
fi
# GPU hampir kosong + CPU sibuk + worker SUDAH LAMA = eksekusi jatuh ke CPU.
#
# Syarat umur >8 menit itu yang membedakannya dari perilaku normal: saat
# menyapu konfigurasi yang sudah lengkap, tiap worker hidup beberapa detik
# hanya untuk mem-parse CSV resume (CPU tinggi, GPU nol) lalu keluar. Tanpa
# syarat umur, penanda ini menyala terus sepanjang sapu ulang dan jadi tidak
# berarti. Pada kegagalan arena=25 (16 Sep) worker berumur 22 menit dengan
# GPU 21 GB — itu yang harus tertangkap.
[ "$VRAM" -lt 5000 ] && [ "$CPU" -gt 50 ] && [ "$UMUR" -gt 480 ] && TANDA="$TANDA CPU?(umur=${UMUR}s)"

printf 'SEBELUM=%s\nDIAM_N=%s\n' "$N" "$DIAM_N" > "$STATE"

printf '[%s] baris=%-6s cfg=%-4s worker=%s pengawas=%s err=%-4s vram=%-6s util=%-3s cpu=%-3s umur=%-5s%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$N" "$CFG" "$W" "$P" "$ERR" "$VRAM" "$UTIL" "$CPU" "$UMUR" \
  "${TANDA:+  <<<$TANDA}" >> "$LOG"
