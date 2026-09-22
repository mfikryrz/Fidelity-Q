#!/usr/bin/env bash
# ============================================================================
# pengawas_fp16.sh — menjalankan jalankan_fp16.sh berulang sampai target penuh.
#
# Dua masalah yang diselesaikan:
#
# 1. RUN MATI SAAT SSH PUTUS.
#    `ssh host 'bash jalankan_fp16.sh'` membuat rantai proses menempel ke sshd:
#      python <- jalankan_fp16.sh <- bash <- sshd
#    Begitu koneksi putus, semuanya kena SIGHUP. Untuk run 50 jam itu rapuh.
#    Pengawas ini diluncurkan dengan setsid + nohup + semua fd dialihkan,
#    sehingga induknya menjadi init (PID 1), bukan sshd.
#
# 2. TAHAP PERBAIKAN TIDAK KEBAGIAN JALAN.
#    Tahap 2 di jalankan_fp16.sh (menambal pasangan config x fault model yang
#    berlubang) hanya berjalan SETELAH seluruh 288 konfigurasi selesai. Kalau
#    run terputus di tengah, lubangnya menetap. Pengawas mengulang seluruh
#    skrip sampai jumlah baris mencapai target, jadi tahap 2 selalu kebagian.
#
# Pakai:
#   setsid nohup bash pengawas_fp16.sh > /workspace/fidelity/logs/pengawas.log 2>&1 < /dev/null &
# ============================================================================
set -uo pipefail

ROOT=${ROOT:-/workspace/fidelity}
PY=$ROOT/rtenv/bin/python
OUT=${OUT:-$ROOT/running_experiment_7b/hasil_massal/massal_fp16_288.csv}
export OUT
LOGS=$ROOT/logs; mkdir -p "$LOGS"

export LANGKAH=${LANGKAH:-1} N_CFG=${N_CFG:-288}
export BITS=${BITS:-0-15} RUNS=${RUNS:-2}
export MULAI=${MULAI:-0} AKHIR=${AKHIR:-$N_CFG}
export PER_PROSES=${PER_PROSES:-1}
# PRECISION menentukan ONNX mana yang dipakai (_config.py): kosong/float16 -> onnx_fp16,
# 'int8' -> onnx_int8 dan lebar bit otomatis jadi 8.
export PRECISION=${PRECISION:-float16}
# 45 -> 60 GB. Kartu 80 GB dan pemakaian puncak terukur 56 GB, jadi masih ada
# ruang. Batas 45 membuat fault model kelima dan keenam (RANDOM, RANDOM_BITFLIP)
# kehabisan arena di konfigurasi berat seperti d2/mlp_gate_proj. Nilai ini tidak
# memengaruhi angka hasil sama sekali — hanya menentukan baris berhasil atau
# gagal, jadi mengubahnya di tengah run aman.
export ORT_GPU_MEM_LIMIT_GB=${ORT_GPU_MEM_LIMIT_GB:-60}
export FIDELITY_POOL_GB=${FIDELITY_POOL_GB:-60}

N_BIT=$(echo "$BITS" | awk -F- '{print ($2?$2-$1+1:1)}')
TARGET=$(( (AKHIR - MULAI) * 6 * N_BIT * RUNS ))
MAKS_PUTARAN=${MAKS_PUTARAN:-8}

hitung() {
  [ -f "$OUT" ] || { echo 0; return; }
  "$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null || echo 0
}

echo "=== PENGAWAS mulai $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "target $TARGET baris | mem ${ORT_GPU_MEM_LIMIT_GB}GB | maks $MAKS_PUTARAN putaran"

SEBELUM_PUTARAN=-1
for (( putaran=1; putaran<=MAKS_PUTARAN; putaran++ )); do
  N=$(hitung)
  if [ "$N" -ge "$TARGET" ]; then
    echo "=== TARGET TERCAPAI: $N/$TARGET ==="; break
  fi
  # Berhenti kalau satu putaran penuh tidak menambah baris sama sekali —
  # tanpa ini, sisa yang memang tidak bisa dijalankan (mis. soal terlalu
  # panjang sehingga attention meledak) akan diulang tanpa henti.
  if [ "$N" -eq "$SEBELUM_PUTARAN" ]; then
    echo "=== BERHENTI: putaran $((putaran-1)) tidak menambah baris apa pun ($N) ==="
    echo "    Sisa $((TARGET-N)) baris kemungkinan gagal permanen, bukan sekadar terputus."
    break
  fi
  SEBELUM_PUTARAN=$N
  echo ""
  echo "=== PUTARAN $putaran  ($(date -u +%H:%M:%S) UTC, $N/$TARGET baris) ==="
  bash "$ROOT/jalankan_fp16.sh"
  echo "=== putaran $putaran selesai: $(hitung)/$TARGET ==="
done

AKHIR_N=$(hitung)
echo ""
echo "=== PENGAWAS SELESAI $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "hasil akhir: $AKHIR_N/$TARGET ($(awk "BEGIN{printf \"%.2f\", 100*$AKHIR_N/$TARGET}")%)"
"$PY" "$ROOT/running_experiment_7b/cek_cakupan.py" "$OUT" --bit "$N_BIT" --runs "$RUNS" \
  $( [ "$((AKHIR-MULAI))" -eq 288 ] && echo --penuh ) 2>/dev/null || true
echo "PENGAWAS_TUNTAS"
