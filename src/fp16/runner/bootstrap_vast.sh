#!/usr/bin/env bash
# ============================================================================
# bootstrap_vast.sh — menyiapkan & menjalankan benchmark MMLU Golden
#                     Llama-2-7B (arm FP16 + INT8) di instance Vast.ai.
#
# Pakai:
#   export HF_TOKEN=hf_xxx            # token READ-ONLY dari huggingface.co
#   bash bootstrap_vast.sh            # jalankan semua tahap
#   bash bootstrap_vast.sh 50         # mulai dari tahap 50 saja
#
# Tahap:
#   10 preflight   20 env      30 kode     40 onnx fp16   50 onnx int8
#   60 pool mmlu   70 sanity   80 run      90 kirim hasil
#
# Semua tahap idempoten: aman diulang, yang sudah selesai dilewati.
# ============================================================================
set -uo pipefail

# ---------------------------------------------------------------- konfigurasi
HF_REPO="${HF_REPO:-mfikryrz/llama2-7b-fidelity-onnx-fp16}"      # sumber ONNX (read)
HASIL_REPO="${HASIL_REPO:-mfikryrz/llama2-7b-fidelity-hasil}"   # tujuan hasil (write)
# HF_TOKEN       = token READ-ONLY, untuk menarik ONNX
# HF_TOKEN_TULIS = token FINE-GRAINED, izin tulis HANYA ke $HASIL_REPO
MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}"
N_SOAL="${N_SOAL:-5000}"          # per arm; naikkan kalau saldo memungkinkan
SHOTS="${SHOTS:-5}"
CHUNK="${CHUNK:-250}"             # soal per proses; proses baru = memori direset
BUDGET_HOURS="${BUDGET_HOURS:-9}" # total jam sewa yang kamu anggarkan
SANITY_N="${SANITY_N:-20}"

ROOT="${ROOT:-/workspace/fidelity}"
CODE="$ROOT/running_experiment_7b"
HASIL="$CODE/hasil"
LOGS="$ROOT/logs"

STAGE_FROM="${1:-10}"
STAGE_TO="${2:-99}"          # mis. "bash bootstrap_vast.sh 10 40" = tahap 10..40 saja
mkdir -p "$ROOT" "$LOGS"

# tenggat keras: 90% dari anggaran, dihitung dari sekarang
DEADLINE_EPOCH=$(( $(date +%s) + $(python3 -c "print(int($BUDGET_HOURS*3600*0.9))") ))
DEADLINE_HHMM=$(date -d "@$DEADLINE_EPOCH" +%H:%M)

say()  { printf '\n\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\n\033[1;31m[GAGAL]\033[0m %s\n' "$*" >&2; exit 1; }
skip() { printf '\033[0;33m[lewati]\033[0m %s\n' "$*"; }
run_stage() { [ "$1" -ge "$STAGE_FROM" ] && [ "$1" -le "$STAGE_TO" ]; }

deteksi_core() {
  # nproc melaporkan core HOST, bukan jatah container. Baca kuota cgroup v2 lalu v1.
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


# ============================================================ 10 — preflight
if run_stage 10; then
say "10 · Preflight"

command -v nvidia-smi >/dev/null || die "nvidia-smi tidak ada — instance tanpa GPU?"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CORES=$(deteksi_core)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
DISK_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')

echo "VRAM=${VRAM_MB}MB  core=${CORES}  RAM=${RAM_GB}GB  disk_bebas=${DISK_GB}GB"

[ "$VRAM_MB" -lt 38000 ] && die "VRAM ${VRAM_MB}MB terlalu kecil. Arm INT8 butuh >=40GB."
DISK_MIN_GB="${DISK_MIN_GB:-120}"   # FP16 saja cukup 60; INT8 butuh 150 (fp32 antara ~26GB)
[ "$DISK_GB" -lt "$DISK_MIN_GB" ] && die "Disk ${DISK_GB}GB < minimum ${DISK_MIN_GB}GB. Untuk FP16 saja: DISK_MIN_GB=60."
[ "$RAM_GB"  -lt 48 ]    && echo "PERINGATAN: RAM ${RAM_GB}GB agak sempit untuk memuat ONNX INT8."
[ -n "${HF_TOKEN:-}" ]   || die "HF_TOKEN belum diset. export HF_TOKEN=hf_xxx (token READ-ONLY)."

echo "Tenggat keras run: $DEADLINE_HHMM (anggaran ${BUDGET_HOURS} jam)"
fi

# ================================================================= 20 — env
if run_stage 20; then
say "20 · Membangun environment Python 3.10"

if [ ! -x "$ROOT/rtenv/bin/python" ]; then
  # Ubuntu 24.04 hanya punya Python 3.12; transformers 4.33.3 terbit sebelum 3.12 ada.
  # Pasang CPython 3.10 mandiri lewat uv (sama seperti .toolchain di laptop).
  PY310="$(command -v python3.10 || true)"
  if [ -z "$PY310" ]; then
    say "20a · Python 3.10 tidak ada — memasang lewat uv"
    export UV_INSTALL_DIR="$ROOT/.uv"
    if [ ! -x "$ROOT/.uv/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.uv" sh >/dev/null 2>&1 \
        || die "gagal memasang uv"
    fi
    export UV_PYTHON_INSTALL_DIR="$ROOT/.pythons"
    "$ROOT/.uv/uv" python install 3.10 >/dev/null 2>&1 || die "uv gagal memasang Python 3.10"
    PY310="$("$ROOT/.uv/uv" python find 3.10 2>/dev/null)"
    [ -x "$PY310" ] || die "Python 3.10 terpasang tapi tidak ditemukan"
  fi
  echo "Python 3.10: $PY310 ($("$PY310" -V 2>&1))"

  "$PY310" -m venv "$ROOT/rtenv"  || die "gagal membuat rtenv"
  "$PY310" -m venv "$ROOT/expenv" || die "gagal membuat expenv"

  # expenv = export ONNX (torch CPU + transformers lama). JANGAN pasang accelerate:
  # ia menaikkan huggingface_hub dan merusak transformers 4.33.3.
  "$ROOT/expenv/bin/pip" install -q --upgrade pip
  "$ROOT/expenv/bin/pip" install -q \
      numpy==1.26.4 \
      torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    || die "gagal memasang torch"
  "$ROOT/expenv/bin/pip" install -q \
      transformers==4.33.3 \
      huggingface_hub==0.16.4 \
      sentencepiece protobuf onnx \
    || die "gagal memasang expenv"
else
  skip "venv sudah ada"
fi

# rtenv = runtime inferensi. DI LUAR blok pembuatan venv, supaya bisa diulang
# tanpa membangun ulang venv (pip cepat kalau sudah terpasang).
#
# req_rtenv.txt TIDAK memuat onnxruntime-gpu — daftar beku itu hanya menutup
# dependensi tersembunyi yang dulu ketahuan hilang satu per satu saat eksperimen
# sudah berjalan (loguru, datasets, sacrebleu). ORT harus dipasang terpisah.
# Urutan penting: ORT dulu (menarik wheel nvidia-*), lalu daftar terpin,
# supaya numpy==1.26.4 dan protobuf==3.20.3 yang menang — persis kombinasi yang
# terbukti menjalankan 43.120 injeksi.
if ! "$ROOT/rtenv/bin/python" -c "import onnxruntime" >/dev/null 2>&1; then
  say "20a2 · Memasang paket rtenv"
  "$ROOT/rtenv/bin/pip" install -q --upgrade pip
  "$ROOT/rtenv/bin/pip" install -q \
      onnxruntime-gpu==1.20.2 \
      onnxruntime-extensions==0.15.2 \
    || die "gagal memasang onnxruntime-gpu"
  if [ -f "$ROOT/req_rtenv.txt" ]; then
    echo "  daftar terpin: $(wc -l < "$ROOT/req_rtenv.txt") paket"
    "$ROOT/rtenv/bin/pip" install -q -r "$ROOT/req_rtenv.txt" \
      || die "gagal memasang req_rtenv.txt"
  else
    echo "  req_rtenv.txt tidak ada — daftar minimum (rawan paket hilang saat run)"
    "$ROOT/rtenv/bin/pip" install -q \
        numpy==1.26.4 sentencepiece protobuf onnx "huggingface_hub>=1.0" \
      || die "gagal memasang rtenv"
  fi
else
  skip "paket rtenv sudah ada"
fi

PY="$ROOT/rtenv/bin/python"
EXPY="$ROOT/expenv/bin/python"

say "20b · Verifikasi CUDA Execution Provider"
_SP="$ROOT/rtenv/lib/python3.10/site-packages"
_LD="$_SP/onnxruntime/capi"
for _d in "$_SP"/nvidia/*/lib; do [ -d "$_d" ] && _LD="$_LD:$_d"; done
export LD_LIBRARY_PATH="$_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" - <<'PYEOF' || die "CUDAExecutionProvider tidak tersedia — cek versi driver vs CUDA."
import onnxruntime as ort
provs = ort.get_available_providers()
print("onnxruntime", ort.__version__, "->", provs)
assert "CUDAExecutionProvider" in provs, "CUDA EP TIDAK ADA"
print("CUDA EP OK")
PYEOF
fi

PY="$ROOT/rtenv/bin/python"; EXPY="$ROOT/expenv/bin/python"
_SP="$ROOT/rtenv/lib/python3.10/site-packages"
_LD="$_SP/onnxruntime/capi"; for _d in "$_SP"/nvidia/*/lib; do [ -d "$_d" ] && _LD="$_LD:$_d"; done

# ================================================================ 30 — kode
if run_stage 30; then
say "30 · Mengambil kode proyek"
if [ -f "$CODE/mmlu_loglik_bench.py" ]; then
  skip "kode sudah ada di $CODE"
elif [ -f "$ROOT/code.tar.gz" ]; then
  tar xzf "$ROOT/code.tar.gz" -C "$ROOT" && echo "kode diekstrak dari code.tar.gz"
else
  "$ROOT/rtenv/bin/hf" download "$HF_REPO" --include 'code/**' \
      --local-dir "$ROOT/_dl" >/dev/null 2>&1 \
    && tar xzf "$ROOT/_dl/code/code.tar.gz" -C "$ROOT" \
    || die "kode tidak ditemukan. Unggah code.tar.gz ke $ROOT lewat scp, atau ke repo HF di folder code/."
fi
[ -x "$CODE/repo/FIdelity-ONNX-master/llama/onnx_bitflip.so" ] \
  || chmod +x "$CODE/repo/FIdelity-ONNX-master/llama/onnx_bitflip.so" 2>/dev/null || true
fi

# ============================================== env FIdelity (GPU besar!)
export FIDELITY_REPO="$CODE/repo/FIdelity-ONNX-master"
export FIDELITY_WORK="$CODE/work"
export FIDELITY_ONNX_ROOT="$CODE/onnx"
export HF_HOME="$CODE/hf_home"
export HF_HUB_CACHE="$CODE/hf_home/hub"
export TMPDIR="$CODE/tmp"
export KEEP_HF_WEIGHTS=1
export MODEL_ID
export LD_LIBRARY_PATH="$_LD:$FIDELITY_REPO/llama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CORES=$(deteksi_core)
# Beda besar dari mesin lokal: SEMUA 32 decoder masuk GPU, tidak ada split ke CPU.
export FIDELITY_GPU_LAYERS="${FIDELITY_GPU_LAYERS:-32}"
export ORT_GPU_MEM_LIMIT_GB="${ORT_GPU_MEM_LIMIT_GB:-$(( VRAM_MB / 1024 - 4 ))}"
export FIDELITY_POOL_GB="${FIDELITY_POOL_GB:-$(( VRAM_MB / 1024 - 4 ))}"
export ORT_THREADS="${ORT_THREADS:-$CORES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CORES}"
mkdir -p "$FIDELITY_WORK" "$TMPDIR" "$HF_HOME" "$HASIL"
echo "[env] gpu_layers=$FIDELITY_GPU_LAYERS mem_limit=${ORT_GPU_MEM_LIMIT_GB}GB threads=$ORT_THREADS"

# =========================================================== 40 — ONNX FP16
if run_stage 40; then
say "40 · ONNX FP16"
FP16D="$CODE/onnx/onnx_fp16"
N_ADA=$(ls "$FP16D"/*.onnx 2>/dev/null | wc -l)

if [ "$N_ADA" -eq 35 ]; then
  skip "ONNX FP16 sudah lengkap (35 file)"
else
  say "40a · Mencoba menarik dari repo HF: $HF_REPO"
  mkdir -p "$FP16D"
  "$ROOT/rtenv/bin/hf" download "$HF_REPO" --local-dir "$FP16D" \
      --exclude 'code/**' 2>&1 | tail -3
  N_ADA=$(ls "$FP16D"/*.onnx 2>/dev/null | wc -l)

  if [ "$N_ADA" -ne 35 ]; then
    say "40b · Repo belum lengkap ($N_ADA/35). Beralih ke export lokal."
    "$EXPY" "$CODE/_download_7b.py"      2>&1 | tail -5
    "$EXPY" "$CODE/01_phase1_export.py"  2>&1 | tail -8 || die "export FP16 gagal"
    N_ADA=$(ls "$FP16D"/*.onnx 2>/dev/null | wc -l)
  fi
  [ "$N_ADA" -eq 35 ] || die "ONNX FP16 tetap tidak lengkap ($N_ADA/35)"
fi

say "40c · Verifikasi 35 file bisa di-parse"
"$PY" - "$FP16D" <<'PYEOF' || die "ada file ONNX rusak"
import sys, os, onnx
d = sys.argv[1]; bad = []
fs = sorted(f for f in os.listdir(d) if f.endswith(".onnx"))
for f in fs:
    try: onnx.load(os.path.join(d, f), load_external_data=False)
    except Exception as e: bad.append((f, str(e)[:60]))
print(f"{len(fs)-len(bad)}/{len(fs)} file ONNX sehat")
for f, e in bad: print("  RUSAK:", f, e)
sys.exit(1 if bad else 0)
PYEOF
du -sh "$FP16D"
fi

# =========================================================== 50 — ONNX INT8
if run_stage 50; then
say "50 · ONNX INT8 (SmoothQuant fake-quant)"
INT8D="$CODE/onnx/onnx_int8"
N_ADA=$(ls "$INT8D"/*.onnx 2>/dev/null | wc -l)

if [ "$N_ADA" -eq 35 ]; then
  skip "ONNX INT8 sudah lengkap"
else
  [ -f "$FIDELITY_REPO/act_scales/llama-2-7b.pt" ] || die "act_scales/llama-2-7b.pt tidak ada"
  # bobot HF diperlukan SmoothQuant (bekerja di model torch, bukan ONNX)
  "$EXPY" "$CODE/_download_7b.py" 2>&1 | tail -3
  PRECISION=int8 "$EXPY" "$CODE/01_phase1_export_int8.py" 2>&1 | tail -10 || die "export INT8 gagal"
fi

# --- INI YANG PENTING: ukur, karena 25GB tadi cuma proyeksi dari model 1.3B ---
say "50b · Ukuran nyata ONNX INT8 vs kapasitas VRAM"
INT8_GB=$(du -sb "$INT8D" 2>/dev/null | awk '{printf "%.1f", $1/1073741824}')
VRAM_GB=$(( VRAM_MB / 1024 ))
echo "ONNX INT8 : ${INT8_GB} GB"
echo "VRAM      : ${VRAM_GB} GB   (butuh ~1,25x ukuran bobot untuk arena+KV cache)"
"$PY" -c "
i=float('$INT8_GB'); v=$VRAM_GB; butuh=i*1.25
print(f'perkiraan pemakaian: {butuh:.1f} GB dari {v} GB')
if butuh > v:
    print('>>> TIDAK MUAT SELURUHNYA DI GPU.')
    print('>>> Turunkan FIDELITY_GPU_LAYERS (mis. 24) supaya sebagian layer ke CPU,')
    print('>>> atau sewa kartu 80GB. Arm FP16 tetap bisa jalan penuh.')
else:
    print('>>> MUAT. Lanjut.')
"
fi

# ============================================================= 60 — pool MMLU
if run_stage 60; then
say "60 · Menyiapkan pool soal MMLU"
if [ -f "$CODE/work/mmlu_pool.csv" ]; then
  skip "mmlu_pool.csv sudah ada ($(wc -l < "$CODE/work/mmlu_pool.csv") baris)"
else
  "$EXPY" "$CODE/14_prepare_mmlu.py" 2>&1 | tail -5 || die "gagal menyiapkan pool MMLU"
fi
fi

# ============================================================== 70 — sanity
if run_stage 70; then
say "70 · Sanity $SANITY_N soal per arm (jangan buang jam kalau protokolnya salah)"
# arm dipilih lewat PRECISION; _config.py:29 -> ONNX_DIR = int8 ? onnx_int8 : onnx_fp16
for PAIR in "fp16:float16" "int8:int8"; do
  ARM="${PAIR%%:*}"; PREC="${PAIR##*:}"
  D="$CODE/onnx/onnx_$ARM"
  [ "$(ls "$D"/*.onnx 2>/dev/null | wc -l)" -eq 35 ] || { skip "arm $ARM belum siap"; continue; }
  OUT="$HASIL/sanity_${ARM}.csv"
  PRECISION="$PREC" "$PY" "$CODE/mmlu_loglik_bench.py" \
      --n "$SANITY_N" --shots "$SHOTS" --out "$OUT" --resume 2>&1 | tail -4
  "$PY" - "$OUT" "$ARM" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows: print(f"[{sys.argv[2]}] TIDAK ADA BARIS"); sys.exit(0)
benar = sum(int(r["benar"]) for r in rows)
dtk   = [float(r["detik"]) for r in rows if r.get("detik")]
dtk_s = sorted(dtk)[len(dtk)//2] if dtk else 0
akur  = 100*benar/len(rows)
print(f"[{sys.argv[2]}] {benar}/{len(rows)} = {akur:.1f}%  |  {dtk_s:.1f} dtk/soal (median)")
if akur <= 30:
    print(f"[{sys.argv[2]}] >>> PERINGATAN: setara tebakan acak. Periksa protokol SEBELUM run penuh.")
PYEOF
done
echo
echo "Kalau akurasi ~25%, HENTIKAN dan periksa. Jangan lanjut ke tahap 80."
fi

# ================================================================= 80 — run
if run_stage 80; then
say "80 · Run penuh — $N_SOAL soal/arm, potongan $CHUNK, tenggat $DEADLINE_HHMM"
for PAIR in "fp16:float16" "int8:int8"; do
  ARM="${PAIR%%:*}"; PREC="${PAIR##*:}"
  D="$CODE/onnx/onnx_$ARM"
  [ "$(ls "$D"/*.onnx 2>/dev/null | wc -l)" -eq 35 ] || { skip "arm $ARM belum siap"; continue; }
  OUT="$HASIL/bench_${ARM}_${SHOTS}shot.csv"
  say "80 · arm $ARM (PRECISION=$PREC) -> $(basename "$OUT")"

  while :; do
    [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ] && { echo "tenggat tercapai, berhenti."; break 2; }
    SUDAH=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))
    [ "$SUDAH" -ge "$N_SOAL" ] && { echo "arm $ARM selesai: $SUDAH soal"; break; }

    # proses BARU tiap potongan -> memori direset, crash biayanya satu potongan
    PRECISION="$PREC" "$PY" "$CODE/mmlu_loglik_bench.py" \
        --n "$N_SOAL" --shots "$SHOTS" --out "$OUT" --resume \
        --deadline "$DEADLINE_HHMM" 2>&1 | tail -2

    BARU=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))
    echo "  progres: $BARU/$N_SOAL soal"
    [ "$BARU" -le "$SUDAH" ] && { echo "  tidak ada kemajuan — berhenti agar tidak berputar."; break; }
    bash "$0" 90 >/dev/null 2>&1 || true    # simpan hasil keluar tiap potongan
  done
done
fi

# ========================================================== 90 — kirim hasil
if run_stage 90; then
say "90 · Mengirim hasil ke HuggingFace (instance bisa dihapus kapan saja)"
if ! compgen -G "$HASIL/*.csv" >/dev/null; then
  skip "belum ada hasil"
elif [ -z "${HF_TOKEN_TULIS:-}" ]; then
  echo "HF_TOKEN_TULIS belum diset — hasil hanya tersimpan lokal di $HASIL"
  echo "Set token fine-grained (tulis ke $HASIL_REPO saja) lalu jalankan: bash $0 90"
  ls -la "$HASIL"/*.csv | awk '{print $5, $9}'
else
  # --- ringkasan yang bisa dibaca langsung di halaman repo HF ---
  say "90a · Menyusun README.md"
  GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  {
    echo "# Golden MMLU — Llama-2-7B lewat FIdelity-ONNX"
    echo
    echo "Benchmark **tanpa fault injection**, protokol log-likelihood atas token \` A\`/\` B\`/\` C\`/\` D\`."
    echo "Referensi paper Llama 2: **45,3%** (5-shot). Tebakan acak: 25,0%."
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| Diperbarui | $(date -u '+%Y-%m-%d %H:%M UTC') |"
    echo "| GPU | $GPU |"
    echo "| Protokol | ${SHOTS}-shot |"
    echo "| Target soal/arm | $N_SOAL |"
    echo "| Model | $MODEL_ID |"
    echo
    echo "## Hasil"
    echo
    echo '```'
    "$PY" "$CODE/ringkas_vast.py" "$HASIL" 2>/dev/null || echo "(ringkasan belum tersedia)"
    echo '```'
    echo
    echo "## Progres per arm"
    echo
    echo "| Arm | Berkas | Soal selesai |"
    echo "|---|---|---|"
    for f in "$HASIL"/bench_*.csv; do
      [ -f "$f" ] || continue
      n=$(( $(wc -l < "$f") - 1 ))
      echo "| $(basename "$f" | sed -E 's/bench_([a-z0-9]+)_.*/\1/') | \`$(basename "$f")\` | $n |"
    done
    echo
    echo "## Isi kolom CSV"
    echo
    echo "\`sample_id\`, \`subject\`, \`answer_letter\`, \`pred_letter\`, \`benar\`,"
    echo "\`logit_A..D\`, \`n_prompt_tokens\`, \`detik\`"
  } > "$HASIL/README.md"

  say "90b · Mengunggah ke $HASIL_REPO"
  "$ROOT/rtenv/bin/hf" upload "$HASIL_REPO" "$HASIL" . \
      --repo-type=dataset 2>&1 | tail -3
  RC=$?
  if [ $RC -eq 0 ]; then
    echo
    echo "Hasil bisa dilihat tanpa menyentuh server lagi:"
    echo "  https://huggingface.co/datasets/$HASIL_REPO"
  else
    echo "GAGAL mengunggah. Periksa izin HF_TOKEN_TULIS terhadap $HASIL_REPO."
    echo "Cadangan manual: scp -r $HASIL ..."
  fi
  ls -la "$HASIL"/*.csv | awk '{print $5, $9}'
fi
fi

say "SELESAI. Ringkasan:"
[ -f "$CODE/ringkas_vast.py" ] && "$PY" "$CODE/ringkas_vast.py" "$HASIL" 2>/dev/null | tail -30
