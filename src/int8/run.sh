#!/usr/bin/env bash
# ============================================================================
# run.sh — SATU entry point untuk menjalankan INT8, dua preset:
#
#   ./run.sh --preset smoke   uji asap cepat (foreground, beberapa menit)
#   ./run.sh --preset full    run 288 konfigurasi (background, belasan jam)
#   ./run.sh --status         ringkasan satu layar dari run yang jalan
#
# Menjalankan setup.sh dulu kalau belum (patch + pre-flight). Untuk --preset
# full, config/int8.env WAJIB ada -- run 288 konfigurasi terlalu mahal untuk
# dijalankan dengan tebakan default FP16 (lihat config/int8.env.example untuk
# alasan tiap angka).
#
# Mengulang perintah yang sama SETELAH terputus = cara melanjutkan (resume
# sudah otomatis di jalankan_fp16.sh, tidak perlu flag terpisah). Skrip ini
# menolak menjalankan --preset full KEDUA KALINYA kalau satu sudah hidup --
# dua pengawas menulis CSV yang sama adalah jebakan yang sudah pernah
# terjadi (lihat memori proyek: "dua worker menulis satu CSV").
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PRESET=""
MULAI_OPT=""
AKHIR_OPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET=$2; shift 2 ;;
    --status) PRESET=status; shift ;;
    --mulai) MULAI_OPT=$2; shift 2 ;;
    --akhir) AKHIR_OPT=$2; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "argumen tidak dikenal: $1" >&2; exit 2 ;;
  esac
done

if [ -f config/int8.env ]; then
  set -a; source config/int8.env; set +a
elif [ "$PRESET" = "full" ]; then
  echo "GAGAL: config/int8.env tidak ada. 'cp config/int8.env.example config/int8.env' dulu," >&2
  echo "       lalu sesuaikan kalau GPU-mu bukan H200. --preset full terlalu mahal untuk" >&2
  echo "       dijalankan dengan tebakan default." >&2
  exit 1
else
  echo "[peringatan] config/int8.env tidak ada -- smoke test pakai default bawaan skrip." >&2
fi

ROOT=${ROOT:-/workspace/fidelity}
CODE=$ROOT/running_experiment_7b
LOGS=$ROOT/logs
N_CFG=${N_CFG:-288}

pengawas_hidup() { pgrep -f "bash pengawas_fp16.sh" >/dev/null 2>&1; }

case "$PRESET" in
  smoke)
    echo "=== SMOKE TEST INT8: beberapa konfigurasi, foreground, beberapa menit ==="
    OUT="$CODE/hasil_massal/smoke_int8.csv" \
    PRECISION=int8 BITS="${BITS:-0-7}" RUNS="${RUNS:-2}" \
    N_CFG="${N_CFG_SMOKE:-16}" LANGKAH="${LANGKAH_SMOKE:-19}" \
    PER_PROSES="${PER_PROSES:-2}" \
    ORT_GPU_MEM_LIMIT_GB="${ORT_GPU_MEM_LIMIT_GB:-35}" \
    FIDELITY_POOL_GB="${FIDELITY_POOL_GB:-35}" \
    ROOT="$ROOT" MODEL_ID="${MODEL_ID:-}" \
      bash runner/jalankan_fp16.sh
    ec=$?
    if [ $ec -eq 0 ]; then
      echo
      echo "=== periksa hasil smoke test sebelum lanjut ke --preset full ==="
      echo "  - golden_teks/faulty_teks harus berisi kalimat, bukan kosong/satu huruf"
      echo "  - g_logit_* harus beda dari f_logit_* pada sebagian baris (injeksi masuk)"
      echo "  cek cepat: $ROOT/rtenv/bin/python analisis/cek_cakupan.py $CODE/hasil_massal/smoke_int8.csv --bit 8 --runs ${RUNS:-2}"
    fi
    exit $ec
    ;;

  full)
    if pengawas_hidup; then
      echo "GAGAL: pengawas_fp16.sh SUDAH HIDUP. Menjalankan lagi akan membuat DUA worker" >&2
      echo "       menulis CSV yang sama. Kalau memang mau restart, matikan dulu:" >&2
      echo "       pkill -f 'bash pengawas_fp16.sh'; pkill -f jalankan_fp16.sh; pkill -f mass_loglik" >&2
      echo "       lalu pastikan 'pgrep -f mass_loglik | wc -l' = 0 sebelum ./run.sh lagi." >&2
      exit 1
    fi

    MULAI=${MULAI_OPT:-${MULAI:-0}}
    AKHIR=${AKHIR_OPT:-${AKHIR:-$N_CFG}}
    OUT=${OUT:-$CODE/hasil_massal/massal_int8_288.csv}
    mkdir -p "$LOGS" "$(dirname "$OUT")"

    echo "=== RUN PENUH INT8: config $MULAI..$AKHIR dari $N_CFG | bit ${BITS:-0-7} | arena ${ORT_GPU_MEM_LIMIT_GB:-35}GB ==="
    echo "    berjalan di background, survive SSH putus. Pantau: ./run.sh --status"

    PRECISION=int8 LANGKAH=1 N_CFG="$N_CFG" MULAI="$MULAI" AKHIR="$AKHIR" \
    BITS="${BITS:-0-7}" RUNS="${RUNS:-2}" PER_PROSES="${PER_PROSES:-2}" \
    ORT_GPU_MEM_LIMIT_GB="${ORT_GPU_MEM_LIMIT_GB:-35}" FIDELITY_POOL_GB="${FIDELITY_POOL_GB:-35}" \
    OUT="$OUT" MAKS_PUTARAN="${MAKS_PUTARAN:-12}" ROOT="$ROOT" MODEL_ID="${MODEL_ID:-}" \
      setsid nohup bash runner/pengawas_fp16.sh >> "$LOGS/pengawas.log" 2>&1 < /dev/null &
    disown
    sleep 5
    if pengawas_hidup; then
      echo "    pengawas hidup (PID $(pgrep -f 'bash pengawas_fp16.sh' | head -1))"
    else
      echo "GAGAL: pengawas tidak hidup setelah 5 detik -- lihat $LOGS/pengawas.log" >&2
      exit 1
    fi

    TARGET=$(( (AKHIR - MULAI) * 6 * $(echo "${BITS:-0-7}" | awk -F- '{print ($2?$2-$1+1:1)}') * ${RUNS:-2} ))
    BARIS_CRON="*/10 * * * * PRECISION=int8 OUT=$OUT TARGET=$TARGET ARENA_MIN=${ARENA_MIN:-35} AMBANG_ERR=${AMBANG_ERR:-150} MULAI=$MULAI AKHIR=$AKHIR N_CFG=$N_CFG BITS=${BITS:-0-7} RUNS=${RUNS:-2} MAKS_PUTARAN=${MAKS_PUTARAN:-12} bash $HERE/runner/penjaga_vram.sh"
    if crontab -l 2>/dev/null | grep -qF "penjaga_vram.sh"; then
      echo "    entri crontab penjaga_vram.sh sudah ada, tidak diubah (periksa manual kalau TARGET/OUT berbeda)"
    else
      (crontab -l 2>/dev/null; echo "$BARIS_CRON") | crontab -
      echo "    entri crontab penjaga_vram.sh dipasang (tiap 10 menit)"
    fi
    ;;

  status)
    OUT=${OUT:-$CODE/hasil_massal/massal_int8_288.csv} \
    TARGET=${TARGET:-27648} ROOT="$ROOT" \
      bash runner/status.sh
    ;;

  *)
    echo "pakai: $0 --preset smoke | --preset full [--mulai N --akhir N] | --status" >&2
    exit 2
    ;;
esac
