#!/usr/bin/env bash
# lanjutkan_int8.sh — jalankan SEKALI setelah instance dihidupkan kembali.
#
# Kenapa ada: saat instance Vast di-STOP lalu di-START, container dimulai ulang.
# Pengawas pasti mati, dan daemon cron BELUM TENTU hidup. Kalau cron mati, tidak
# ada yang menghidupkan pengawas dan mesin diam sambil tetap ditagih.
#
# Skrip ini idempoten: aman dijalankan berkali-kali. Kalau semuanya sudah jalan,
# ia hanya melapor dan keluar.
#
#   bash /workspace/fidelity/lanjutkan_int8.sh
set -uo pipefail

ROOT=/workspace/fidelity
OUT=$ROOT/running_experiment_7b/hasil_massal/massal_int8_288.csv
PY=$ROOT/rtenv/bin/python

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) melanjutkan INT8 ==="

# ---- 1. berkas hasil masih ada?
if [ ! -f "$OUT" ]; then
  echo "GAGAL: $OUT tidak ada. Disk instance mungkin tidak ikut selamat." >&2
  exit 1
fi
N=$("$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))")
echo "  baris tersimpan : $N / 27648"

# ---- 2. setelan yang akan dipakai
if [ -f "$ROOT/logs/.penjaga_vram.state" ]; then
  echo -n "  state penjaga   : "; tr '\n' ' ' < "$ROOT/logs/.penjaga_vram.state"; echo
else
  echo "  state penjaga   : tidak ada, akan memakai bawaan crontab"
fi

# ---- 3. cron
if pgrep -x cron >/dev/null 2>&1; then
  echo "  cron            : sudah hidup"
else
  echo "  cron            : MATI -> menghidupkan"
  service cron start >/dev/null 2>&1 || cron >/dev/null 2>&1 || true
  sleep 2
  pgrep -x cron >/dev/null 2>&1 \
    && echo "  cron            : hidup sekarang" \
    || echo "  cron            : GAGAL dihidupkan — pengawas di bawah tetap jalan, tapi tanpa penjaga"
fi
echo -n "  entri crontab   : "; crontab -l 2>/dev/null | grep -c . || echo 0

# ---- 4. pengawas
if pgrep -f "pengawas_fp16[.]sh" >/dev/null 2>&1; then
  echo "  pengawas        : sudah hidup, tidak menyentuh apa pun"
else
  echo "  pengawas        : mati -> menghidupkan lewat penjaga (memakai state di atas)"
  TARGET=27648 OUT="$OUT" PRECISION=int8 LANGKAH=1 N_CFG=288 MULAI=0 AKHIR=288 \
  BITS=0-7 RUNS=2 PER_PROSES=2 ORT_GPU_MEM_LIMIT_GB=35 FIDELITY_POOL_GB=35 \
  ARENA_MIN=35 MAKS_PUTARAN=12 AMBANG_ERR=50 \
    setsid nohup bash "$ROOT/penjaga_vram.sh" >/dev/null 2>&1 < /dev/null &
  disown
  sleep 45
fi

echo
echo "=== hasil ==="
echo -n "  pengawas : "; pgrep -f "pengawas_fp16[.]sh" >/dev/null && echo HIDUP || echo "MASIH MATI — periksa logs/penjaga_vram.log"
echo -n "  worker   : "; pgrep -f mass_loglik | wc -l
echo -n "  GPU      : "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo
echo "Pantau: bash $ROOT/status.sh"
