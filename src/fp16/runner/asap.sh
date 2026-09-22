#!/usr/bin/env bash
# ============================================================================
# asap.sh — uji asap mode pg-teks sebelum run penuh.
#
# Menjawab tiga hal:
#   1. golden_teks & faulty_teks berisi kalimat (bukan satu huruf, bukan kosong)
#   2. logit golden != faulty  -> injeksi memang masuk
#   3. BIAYA, dipecah tiga -> ini yang menentukan RUNS
#
# Kenapa DUA titik ukur, bukan satu:
#   Run penuh dipotong per 8 konfigurasi = 36 PROSES, jadi model dimuat 36 kali.
#   Kalau waktu muat model dan waktu bangun graf dijadikan satu angka, ekstrapolasi
#   ke 36 proses salah besar. Dua ukuran memisahkannya:
#
#     W1 = muat + 6 graf  + 6 baris     (1 konfigurasi)
#     W3 = muat + 18 graf + 36 baris    (3 konfigurasi)
#     t_baris  <- rata-rata kolom detik (diukur langsung, di dalam loop terdalam)
#     t_graf   = (W3 - W1 - 30 t_baris) / 12
#     t_muat   = W1 - 6 t_graf - 6 t_baris
#
# Env-nya disalin persis dari run_massal.sh supaya angkanya berlaku untuk run penuh.
# ============================================================================
set -uo pipefail

ROOT="${ROOT:-/workspace/fidelity}"
CODE="$ROOT/running_experiment_7b"
HASIL="$CODE/hasil_massal"

PY="$ROOT/rtenv/bin/python"
[ -x "$PY" ] || { echo "rtenv belum ada" >&2; exit 1; }

export FIDELITY_REPO="$CODE/repo/FIdelity-ONNX-master"
export FIDELITY_WORK="$CODE/work"
export FIDELITY_ONNX_ROOT="$CODE/onnx"
export HF_HOME="$CODE/hf_home"
export TMPDIR="$CODE/tmp"
export MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}"
export FIDELITY_GPU_LAYERS=32
export ORT_GPU_MEM_LIMIT_GB="${ORT_GPU_MEM_LIMIT_GB:-30}"
export FIDELITY_POOL_GB="${FIDELITY_POOL_GB:-30}"

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
export ORT_THREADS=$CORES OMP_NUM_THREADS=$CORES
echo "jatah CPU: $CORES core (nproc melaporkan $(nproc))"

_SP="$ROOT/rtenv/lib/python3.10/site-packages"
_LD="$_SP/onnxruntime/capi"; for d in "$_SP"/nvidia/*/lib; do [ -d "$d" ] && _LD="$_LD:$d"; done
export LD_LIBRARY_PATH="$_LD:$FIDELITY_REPO/llama"
mkdir -p "$FIDELITY_WORK" "$TMPDIR" "$HASIL"

jalan() {   # $1 = jumlah konfigurasi, $2 = berkas keluaran
  rm -f "$2"
  "$PY" "$CODE/mass_loglik_inject.py" \
      --max-configs "$1" --runs 2 --soal 1 --shots 5 \
      --gaya-prompt pg-teks --token-teks 48 --posisi-token akhir \
      --out "$2" 2>&1 \
    | grep -vE "W:onnxruntime|RegisterCustomOps|Custom op domain|Modified model saved" \
    | tail -6
}

echo "=== ukuran 1 : 1 konfigurasi (6 graf, 12 baris) ==="
A0=$(date +%s); jalan 1 "$HASIL/asap1.csv"; A1=$(date +%s)
W1=$(( A1 - A0 )); echo ">>> W1 = ${W1} detik"

echo
echo "=== ukuran 2 : 3 konfigurasi (18 graf, 36 baris) ==="
B0=$(date +%s); jalan 3 "$HASIL/asap3.csv"; B1=$(date +%s)
W3=$(( B1 - B0 )); echo ">>> W3 = ${W3} detik"

echo
echo "=============================================================="
"$PY" "$CODE/ringkas_pgteks.py" "$HASIL/asap3.csv"

echo
echo "=== pemisahan biaya ==="
"$PY" - "$HASIL/asap1.csv" "$HASIL/asap3.csv" "$W1" "$W3" <<'PYEOF'
import csv, sys
a1, a3, W1, W3 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
r3 = list(csv.DictReader(open(a3, newline="", encoding="utf-8")))
r1 = list(csv.DictReader(open(a1, newline="", encoding="utf-8")))
d = [float(r["detik"]) for r in r1 + r3 if r.get("detik")]
t_baris = sum(d) / len(d)
# W1 = muat + 6 graf + 12 baris ; W3 = muat + 18 graf + 36 baris
t_graf = (W3 - W1 - 24 * t_baris) / 12.0
t_muat = W1 - 6 * t_graf - 12 * t_baris
print(f"baris terukur : {len(d)}")
print(f"t_baris       : {t_baris:.3f} dtk  (rata-rata kolom detik)")
print(f"t_graf        : {t_graf:.3f} dtk  (membangun 1 graf tersuntik)")
print(f"t_muat        : {t_muat:.1f} dtk  (memuat model, sekali tiap potongan)")
print()
if t_graf < 0 or t_muat < 0:
    print("PERINGATAN: ada angka negatif -> pengukuran terlalu berisik.")
    print("Pakai angka konservatif, jangan hasil hitungan ini.")
print("Untuk hitung_skala.py:")
print(f"  --detik-per-baris {t_baris:.3f} --detik-per-graf {max(t_graf,0):.2f} "
      f"--detik-muat {max(t_muat,0):.0f}")
PYEOF

echo
echo "=== 3 baris pertama, teks mentah ==="
"$PY" - "$HASIL/asap3.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
for r in rows[:3]:
    dg = sum(abs(float(r[f"g_logit_{L}"]) - float(r[f"f_logit_{L}"])) for L in "ABCD")
    print(f"[{r['Fault_Model']}] d{r['decoder_idx']}/{r['operasi']} bit{r['Bit_Position']}")
    print(f"  G ({r['n_tok_golden']} tok): {r['golden_teks'][:150]!r}")
    print(f"  F ({r['n_tok_faulty']} tok): {r['faulty_teks'][:150]!r}")
    print(f"  jumlah |delta logit| = {dg:.4f}   (0.0000 = injeksi TIDAK berdampak)")
    print()
nol = sum(1 for r in rows if all(r[f"g_logit_{L}"] == r[f"f_logit_{L}"] for L in "ABCD"))
print(f"logit golden == faulty persis : {nol}/{len(rows)} baris")
print(f"teks berbeda                  : "
      f"{sum(1 for r in rows if r['golden_teks'] != r['faulty_teks'])}/{len(rows)} baris")
PYEOF
