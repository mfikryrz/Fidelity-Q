#!/usr/bin/env bash
# ============================================================================
# jalankan_fp16.sh — runner FP16 tunggal. Menggantikan run_massal.sh untuk
#                    protokol baru (revisi Mas Gabriel, 10 Sep 2026).
#
# Pakai:
#   bash jalankan_fp16.sh                 # 16 layer stratified (uji asap)
#   LANGKAH=1 N_CFG=288 bash jalankan_fp16.sh   # seluruh 288 konfigurasi
#
# Aman diulang: --resume melewati baris yang sudah ada. Kalau run terputus,
# jalankan lagi perintah yang sama.
#
# ---------------------------------------------------------------------------
# TIGA PELAJARAN YANG DIBAYAR MAHAL — jangan diubah tanpa alasan kuat
# ---------------------------------------------------------------------------
# 1. SATU konfigurasi per proses (bukan 8 seperti run_massal.sh).
#    Arena BFC di ONNX Runtime tidak pernah menyusut. Beban protokol baru jauh
#    lebih berat dari run lama: 32 baris per (config x fault model) alih-alih
#    2-4, dan setiap baris menghitung golden sendiri sejak soal diacak per
#    baris — dua sesi per baris, bukan satu bersama enam fault model.
#    Dengan 8 config per proses arena habis di config kedua: 595 baris [err]
#    lalu segfault. Diukur 11 Sep 2026.
#
# 2. ORT_GPU_MEM_LIMIT_GB = 45 (bukan 30).
#    Error muncul di node Max/ReduceMax/Slice, yaitu BMM attention 4-D yang
#    tensornya paling besar. Kartu 80 GB, jadi 45 GB masih jauh di bawah
#    VRAM-4 yang dulu menyebabkan segfault di konfigurasi ke-33.
#
# 3. JANGAN hitung baris dengan `wc -l`.
#    Kolom prompt_lengkap memuat baris baru, jadi baris fisik != jumlah record.
#    Pakai modul csv (fungsi hitung() di bawah).
# ============================================================================
set -uo pipefail

ROOT=${ROOT:-/workspace/fidelity}
CODE=$ROOT/running_experiment_7b
PY=$ROOT/rtenv/bin/python
OUT=${OUT:-$CODE/hasil_massal/massal_fp16.csv}
LOGS=$ROOT/logs
mkdir -p "$LOGS" "$(dirname "$OUT")"

LANGKAH=${LANGKAH:-19}   # cuplik stratified tiap ke-19; 1 = semua berurutan
N_CFG=${N_CFG:-16}       # jumlah konfigurasi
BITS=${BITS:-0-15}       # FP16 = 16 bit. INT8 pakai 0-7
RUNS=${RUNS:-2}          # injeksi per bit
DEADLINE=${DEADLINE:-}   # "HH:MM" UTC, kosong = tanpa batas

# Sharding: bagi 288 konfigurasi ke beberapa MESIN yang jalan paralel. Total
# GPU-hour sama persis, yang berubah cuma waktu tunggu. Contoh 9 shard:
#   mesin ke-k:  MULAI=$((k*32)) AKHIR=$(((k+1)*32))   untuk k = 0..8
# Tiap shard menulis CSV sendiri, lalu digabung di laptop. JANGAN dua shard
# menulis ke berkas yang sama.
MULAI=${MULAI:-0}
AKHIR=${AKHIR:-$N_CFG}

# Berapa konfigurasi per PROSES. Default 1: arena BFC di ONNX Runtime tidak
# pernah menyusut, jadi proses baru = arena direset. Untuk FP16 ini murah
# (muat model ~26 detik).
#
# Untuk INT8 JANGAN pakai 1. Grafnya FP32 berukuran 26 GB dan memuatnya makan
# ~9,6 menit (terukur 14 Sep 2026). Dengan 288 konfigurasi itu 46 jam hanya
# untuk memuat model berulang-ulang. Pakai 3-6, dan biarkan tahap 2 menambal
# lubang yang muncul karena arena penuh di ekor proses.
PER_PROSES=${PER_PROSES:-1}

# LANGKAH harus koprima dengan 9 (ada 9 operasi per decoder). Langkah 18 —
# yang tampak wajar karena 288/16=18 — membuat setiap sampel mendarat di
# operasi yang sama: 16 decoder tapi hanya 1 dari 9 operasi.
if [ "$LANGKAH" -gt 1 ] && [ $(( LANGKAH % 3 )) -eq 0 ]; then
  echo "PERINGATAN: LANGKAH=$LANGKAH habis dibagi 3, beresiko hanya mencuplik" >&2
  echo "  sebagian operasi. Pakai angka koprima dengan 9, mis. 19." >&2
fi

export FIDELITY_REPO="$CODE/repo/FIdelity-ONNX-master" FIDELITY_WORK="$CODE/work"
export FIDELITY_ONNX_ROOT="$CODE/onnx" HF_HOME="$CODE/hf_home" TMPDIR="$CODE/tmp"
export MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}" FIDELITY_GPU_LAYERS=32
export ORT_GPU_MEM_LIMIT_GB=${ORT_GPU_MEM_LIMIT_GB:-45}
export FIDELITY_POOL_GB=${FIDELITY_POOL_GB:-45}
# nproc melaporkan core HOST (mis. 152), bukan jatah container (mis. 18).
# Memakai angka host mencekik mesin. Baca kuota cgroup v2 lalu v1.
deteksi_core() {
  local q p n
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null
    if [ "${q:-max}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
      n=$(( q / p )); [ "$n" -ge 1 ] && { echo "$n"; return; }
    fi
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
    if [ "${q:-0}" -gt 0 ] 2>/dev/null && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
      n=$(( q / p )); [ "$n" -ge 1 ] && { echo "$n"; return; }
    fi
  fi
  nproc
}
CORES=$(deteksi_core)
export ORT_THREADS=${ORT_THREADS:-$CORES} OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CORES}
echo "jatah CPU: $CORES core (nproc melaporkan $(nproc))"
SP="$ROOT/rtenv/lib/python3.10/site-packages"
LD="$SP/onnxruntime/capi"; for d in "$SP"/nvidia/*/lib; do [ -d "$d" ] && LD="$LD:$d"; done
export LD_LIBRARY_PATH="$LD:$FIDELITY_REPO/llama"

hitung() {
  [ -f "$OUT" ] || { echo 0; return; }
  "$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))"
}
lewat_tenggat() {
  [ -z "$DEADLINE" ] && return 1
  [ "$(date -u +%s)" -ge "$(date -u -d "$DEADLINE" +%s)" ]
}
jalan() {   # $1=indeks config awal  $2=fault model (kosong = semua)  $3=indeks akhir
  local i=$1 fm=${2:-} sampai=${3:-$(( $1 + 1 ))}
  local arg=(); [ -n "$fm" ] && arg=(--fault-models "$fm")
  "$PY" "$CODE/mass_loglik_inject.py" \
      --konfigurasi-langkah "$LANGKAH" --max-configs "$N_CFG" \
      --konfigurasi-dari "$i" --konfigurasi-sampai "$sampai" \
      "${arg[@]}" --bits "$BITS" --runs "$RUNS" --soal 1 --shots 5 \
      --gaya-prompt pg-teks --token-teks 48 --posisi-token penuh \
      ${DEADLINE:+--deadline "$DEADLINE"} \
      --out "$OUT" --resume 2>&1 \
    | grep -vE "W:onnxruntime|RegisterCustomOps|Custom op domain|Modified model saved|memory_pool" \
    | tee -a "$LOGS/fp16.log" | tail -2
}

# Target dihitung dari rentang shard ini, bukan N_CFG penuh — kalau tidak,
# shard yang mengerjakan 32 dari 288 konfigurasi akan melapor "11% selesai"
# padahal bagiannya sudah tuntas.
N_SHARD=$(( AKHIR - MULAI ))
TARGET=$(( N_SHARD * 6 * $(echo "$BITS" | awk -F- '{print ($2?$2-$1+1:1)}') * RUNS ))
echo "target $TARGET baris | konfigurasi $MULAI..$AKHIR dari $N_CFG | bit $BITS | $RUNS injeksi"
echo "mulai dari $(hitung) baris"

# --- Tahap 1: satu proses per konfigurasi -----------------------------------
for (( i=MULAI; i<AKHIR; i+=PER_PROSES )); do
  lewat_tenggat && { echo "!! tenggat, berhenti sebelum config $i"; break; }
  j=$(( i + PER_PROSES )); [ "$j" -gt "$AKHIR" ] && j=$AKHIR
  A=$(hitung)
  echo "=== [1/2] config $i..$j  ($(date -u +%H:%M:%S) UTC, $A baris) ==="
  jalan "$i" "" "$j"
  echo "    -> +$(( $(hitung) - A ))"
done

# --- Tahap 2: sisa lubang, satu proses per (config x fault model) ------------
# Sebagian config gagal di fault model terakhir karena arena sudah penuh di
# ekor proses. Mengisolasi per fault model memberi arena segar untuk tiap 32
# baris. Pada run 11 Sep ini memulihkan 31 dari 34 baris yang hilang.
echo "=== [2/2] memeriksa lubang ==="
mapfile -t LUBANG < <("$PY" - "$OUT" "$LANGKAH" "$N_CFG" "$BITS" "$RUNS" "$MULAI" "$AKHIR" <<'PY'
import csv, sys, collections, glob, os, re
out, langkah, ncfg, bits, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
mulai, akhir = int(sys.argv[6]), int(sys.argv[7])
a, _, b = bits.partition("-"); daftar_bit = list(range(int(a), int(b) + 1)) if b else [int(a)]
FM = ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]
repo = os.environ["FIDELITY_REPO"]
RE = re.compile(r"decoder-merge-(\d+)__(.+)\.json$")
cfg = []
for p in sorted(glob.glob(os.path.join(repo, "injection_llm", "*.json"))):
    m = RE.search(os.path.basename(p))
    if m: cfg.append((int(m.group(1)), m.group(2)))
cfg.sort()
if langkah > 1: cfg = cfg[::langkah]
cfg = cfg[:ncfg]
ada = collections.Counter()
if os.path.exists(out):
    for x in csv.DictReader(open(out, newline="", encoding="utf-8")):
        ada[(int(x["decoder_idx"]), x["operasi"], x["Fault_Model"])] += 1
perlu = len(daftar_bit) * runs
for i, (d, op) in enumerate(cfg):
    if not (mulai <= i < akhir):     # shard lain, bukan urusan proses ini
        continue
    for fm in FM:
        if ada.get((d, op, fm), 0) < perlu:
            print(f"{i} {fm}")
PY
)
if [ ${#LUBANG[@]} -eq 0 ]; then
  echo "    tidak ada lubang"
else
  echo "    ${#LUBANG[@]} pasangan (config x fault model) belum penuh"
  for spec in "${LUBANG[@]}"; do
    lewat_tenggat && { echo "!! tenggat"; break; }
    set -- $spec
    A=$(hitung); echo "=== config $1 / $2  ($(date -u +%H:%M:%S) UTC) ==="
    jalan "$1" "$2"
    echo "    -> +$(( $(hitung) - A ))"
  done
fi

AKHIR=$(hitung)
echo "=== SELESAI: $AKHIR/$TARGET baris ($(awk "BEGIN{printf \"%.2f\", 100*$AKHIR/$TARGET}")%) ==="
"$PY" "$CODE/ringkas_pgteks.py" "$OUT" 2>/dev/null || true
