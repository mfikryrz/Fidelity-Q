#!/usr/bin/env bash
# ============================================================================
# run_massal.sh — eksperimen massal fault injection Llama-2-7B di Vast.ai
#
# Metode: FIdelity asli (fault mengenai token yang sedang dihitung)
#         -> --posisi-token akhir  [disetujui Mas Gabriel, 28 Agu 2026]
#
# Mode  : pg-teks — soal tetap pilihan ganda, TAPI keluaran golden & faulty
#         disimpan sebagai TEKS MENTAH tanpa parsing apa pun. Tidak ada kolom
#         penilaian; metode pengukuran diputuskan belakangan dan dihitung di
#         laptop lewat hitung_penilaian.py (tanpa menyewa GPU lagi).
#
# Skala : 32 decoder x 9 operasi x 6 fault model x RUNS run
#
# Pakai:
#   export HF_TOKEN=...        # boleh token fine-grained yang sama
#   export HF_TOKEN_TULIS=...
#   bash run_massal.sh
#
# Pengaman:
#   * pengunggah LATAR BELAKANG menyimpan hasil ke HuggingFace tiap 3 menit,
#     TIDAK menunggu proses utama selesai. Ini memperbaiki kelemahan run
#     sebelumnya yang menghilangkan ~5.100 baris saat instance dimatikan.
#   * --resume: baris yang sudah ada dilewati, aman diulang kapan saja
#   * --deadline: berhenti sendiri sebelum saldo habis
# ============================================================================
set -uo pipefail

ROOT="${ROOT:-/workspace/fidelity}"
CODE="$ROOT/running_experiment_7b"
HASIL="$CODE/hasil_massal"
LOGS="$ROOT/logs"
HASIL_REPO="${HASIL_REPO:-mfikryrz/llama2-7b-fidelity-hasil}"

RUNS="${RUNS:-25}"
SOAL="${SOAL:-1}"
SHOTS="${SHOTS:-5}"
BUDGET_HOURS="${BUDGET_HOURS:-4}"
UNGGAH_TIAP="${UNGGAH_TIAP:-180}"     # detik

GAYA="${GAYA:-pg-teks}"
TOKEN_TEKS="${TOKEN_TEKS:-48}"

mkdir -p "$HASIL" "$LOGS"
OUT="$HASIL/massal_pgteks.csv"

say() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n\033[1;31m[GAGAL]\033[0m %s\n' "$*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN belum diset"
[ -d "$CODE" ] || die "$CODE tidak ada — jalankan bootstrap_vast.sh dulu"

PY="$ROOT/rtenv/bin/python"
EXPY="$ROOT/expenv/bin/python"
[ -x "$PY" ] || die "rtenv belum dibangun"

# ---- env FIdelity
export FIDELITY_REPO="$CODE/repo/FIdelity-ONNX-master"
export FIDELITY_WORK="$CODE/work"
export FIDELITY_ONNX_ROOT="$CODE/onnx"
export HF_HOME="$CODE/hf_home"
export TMPDIR="$CODE/tmp"
export MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}"
export FIDELITY_GPU_LAYERS=32
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
export ORT_GPU_MEM_LIMIT_GB="${ORT_GPU_MEM_LIMIT_GB:-30}"   # bukan VRAM-4: arena
# ORT tidak menyusut setelah ratusan sesi faulty dibuat-dibuang -> VRAM habis.
export FIDELITY_POOL_GB="${FIDELITY_POOL_GB:-30}"
# Jatah CPU sebenarnya: nproc melaporkan SELURUH core host, bukan jatah kita.
# Menyetel thread sebanyak nproc di jatah yang jauh lebih kecil membuat mesin
# tercekik (terbukti di instance sebelumnya). Cek cgroup v2 lalu v1.
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
echo "  jatah CPU terdeteksi: $CORES core (nproc melaporkan $(nproc))"
_SP="$ROOT/rtenv/lib/python3.10/site-packages"
_LD="$_SP/onnxruntime/capi"; for d in "$_SP"/nvidia/*/lib; do [ -d "$d" ] && _LD="$_LD:$d"; done
export LD_LIBRARY_PATH="$_LD:$FIDELITY_REPO/llama"
mkdir -p "$FIDELITY_WORK" "$TMPDIR"

DEADLINE=$(date -d "+$(python3 -c "print(int($BUDGET_HOURS*60*0.92))") minutes" +%H:%M)

# ============================================ 1. konfigurasi injeksi
say "1 · Konfigurasi injeksi"
N_CFG=$(ls "$FIDELITY_REPO/injection_llm/"*.json 2>/dev/null | wc -l)
if [ "$N_CFG" -lt 288 ]; then
  echo "  baru $N_CFG konfigurasi — menjalankan 03_phase2_parser.py"
  "$PY" "$CODE/03_phase2_parser.py" 2>&1 | tail -5 || die "parser gagal"
  N_CFG=$(ls "$FIDELITY_REPO/injection_llm/"*.json 2>/dev/null | wc -l)
fi
echo "  konfigurasi siap: $N_CFG  (target 288 = 32 decoder x 9 operasi)"
[ "$N_CFG" -ge 288 ] || die "konfigurasi kurang: $N_CFG/288"

# ============================================ 2. pengunggah latar belakang
say "2 · Pengunggah latar belakang (tiap ${UNGGAH_TIAP}s)"
# CATATAN: ringkas_massal.py SENGAJA tidak dipanggil. Ia membaca kolom is_error,
# yang di mode pg-teks tidak ada lagi (penilaian dipindah ke hitung_penilaian.py
# di laptop). Memanggilnya akan menggagalkan pengunggah tanpa suara.
cat > "$ROOT/pengunggah.sh" <<EOF
#!/usr/bin/env bash
# Menyimpan hasil ke HuggingFace secara berkala, TERPISAH dari proses utama.
while true; do
  sleep $UNGGAH_TIAP
  if compgen -G "$HASIL/*.csv" >/dev/null; then
    "$ROOT/rtenv/bin/hf" upload "$HASIL_REPO" "$HASIL" massal \\
        --repo-type=dataset --token "\$HF_TOKEN_TULIS" >/dev/null 2>&1 \\
      && echo "[\$(date +%H:%M:%S)] terkirim: \$(( \$(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 )) baris"
  fi
done
EOF
chmod +x "$ROOT/pengunggah.sh"
if [ -n "${HF_TOKEN_TULIS:-}" ]; then
  pkill -f "pengunggah.sh" 2>/dev/null
  nohup bash "$ROOT/pengunggah.sh" >> "$LOGS/pengunggah.log" 2>&1 &
  echo "  aktif, PID $!  -> $HASIL_REPO/massal"
else
  echo "  DILEWATI: HF_TOKEN_TULIS belum diset. Hasil hanya tersimpan lokal."
fi

# ============================================ 3. eksperimen massal
say "3 · Eksperimen massal — metode FIdelity asli (token yang sedang dihitung)"
echo "  target : 32 x 9 x 6 x $RUNS = $(( 32 * 9 * 6 * RUNS )) baris"
echo "  tenggat: $DEADLINE  (anggaran ${BUDGET_HOURS} jam)"
echo "  keluaran: $OUT"
echo

# Dipotong per POTONGAN_CFG konfigurasi. Tiap potongan proses baru, sehingga
# arena VRAM ORT direset — run sebelumnya segfault di konfigurasi ke-33 karena
# arena tidak pernah menyusut setelah ratusan sesi faulty dibuat-dibuang.
POTONGAN_CFG="${POTONGAN_CFG:-8}"
N_CFG_DEC=288
for (( i=0; i<N_CFG_DEC; i+=POTONGAN_CFG )); do
  if [ "$(date +%H%M)" -ge "${DEADLINE/:/}" ]; then
    echo "[stop] tenggat $DEADLINE tercapai"; break
  fi
  j=$(( i + POTONGAN_CFG )); [ "$j" -gt "$N_CFG_DEC" ] && j=$N_CFG_DEC
  SEBELUM=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))
  echo "--- potongan konfigurasi $i..$((j-1))  (sudah $SEBELUM baris) ---"
  "$PY" "$CODE/mass_loglik_inject.py" \
      --runs "$RUNS" --soal "$SOAL" --shots "$SHOTS" \
      --gaya-prompt "$GAYA" --token-teks "$TOKEN_TEKS" \
      --posisi-token akhir --konfigurasi-dari "$i" --konfigurasi-sampai "$j" \
      --out "$OUT" --resume --deadline "$DEADLINE" 2>&1 \
    | grep -vE "W:onnxruntime|RegisterCustomOps|Custom op domain|Modified model saved" \
    | tee -a "$LOGS/massal.log" | tail -3
  SESUDAH=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))
  echo "    -> $SEBELUM sampai $SESUDAH baris"
done

# ============================================ 4. penutup
say "4 · Ringkasan akhir"
pkill -f "pengunggah.sh" 2>/dev/null
"$PY" "$CODE/ringkas_pgteks.py" "$OUT" | tee "$HASIL/RINGKASAN.txt"
if [ -n "${HF_TOKEN_TULIS:-}" ]; then
  "$ROOT/rtenv/bin/hf" upload "$HASIL_REPO" "$HASIL" massal \
      --repo-type=dataset --token "$HF_TOKEN_TULIS" 2>&1 | tail -2
  echo
  echo "Hasil: https://huggingface.co/datasets/$HASIL_REPO/tree/main/massal"
fi
