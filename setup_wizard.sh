#!/usr/bin/env bash
# ============================================================================
# setup_wizard.sh — wizard interaktif untuk menyiapkan framework fault
# injection (FP16/INT8) di server mana pun, tanpa hardcode path.
#
# TIDAK menduplikasi logika bootstrap/setup/export — skrip ini memanggil
# runner/bootstrap_vast.sh, setup.sh, export_llama*.py, run.sh yang sudah ada
# di src/fp16 dan src/int8 (satu sumber kebenaran, satu tempat perbaikan).
#
# Jalankan LANGSUNG DI SERVER TARGET lewat sesi SSH interaktif (bukan dari
# laptop) -- skrip ini butuh TTY untuk menu & prompt-nya:
#   cd /path/ke/clone/Fidelity-Q && bash setup_wizard.sh
#
# Seluruh prompt, jawaban, dan output/error tiap perintah dicatat ke
# simulation_run.log (satu berkas, di folder yang sama dengan skrip ini).
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$HERE/simulation_run.log"
printf '\n===== setup_wizard.sh mulai %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# ---------------------------------------------------------------- helper log
log() {   # tampil di layar + tercatat di log, dengan jam
  printf '%s\n' "$*"
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >> "$LOG_FILE"
}

# tanya bebas (boleh kosong -> pakai default)
#
# Semua variabel lokal di bawah ini SENGAJA diberi awalan "__ask_" -- ask_required()
# memanggil ask() dengan nama variabel TUJUAN AKHIR bisa apa saja (ROOT, MODEL_ID,
# dst.), dan bash me-resolve `local`/`printf -v` secara dynamic-scoped, bukan
# lexical. Kalau nama internal di sini kebetulan sama dengan nama variabel yang
# dipakai fungsi pemanggil (mis. keduanya pakai "value"), tulisan lewat `printf -v`
# akan mendarat di `local` TERDEKAT (milik fungsi ini), bukan punya pemanggil --
# nilainya hilang diam-diam. Sudah kejadian sungguhan saat dites (lihat histori).
ask() {   # ask "teks pertanyaan" NAMA_VARIABEL [default]
  local __ask_prompt="$1" __ask_var="$2" __ask_default="${3:-}" __ask_value
  if [ -n "$__ask_default" ]; then
    read -rp "$__ask_prompt [$__ask_default]: " __ask_value
  else
    read -rp "$__ask_prompt: " __ask_value
  fi
  __ask_value="${__ask_value:-$__ask_default}"
  printf '[INPUT] %s => %s\n' "$__ask_prompt" "$__ask_value" >> "$LOG_FILE"
  printf -v "$__ask_var" '%s' "$__ask_value"
}

# tanya wajib diisi (ulang kalau kosong) -- pakai `${!__var}` (indirect
# expansion) utk baca balik nilainya, BUKAN local baru, supaya tidak jatuh ke
# jebakan penamaan yang sama seperti catatan di ask() di atas.
ask_required() {   # ask_required "teks pertanyaan" NAMA_VARIABEL [default]
  local __ar_prompt="$1" __ar_var="$2" __ar_default="${3:-}"
  while true; do
    ask "$__ar_prompt" "$__ar_var" "$__ar_default"
    [ -n "${!__ar_var}" ] && break
    echo "  Tidak boleh kosong, coba lagi."
  done
}

# tanya y/n (ulang sampai valid)
ask_yn() {   # ask_yn "teks pertanyaan (y/n)" NAMA_VARIABEL
  local __ay_prompt="$1" __ay_var="$2" __ay_value
  while true; do
    read -rp "$__ay_prompt " __ay_value
    case "$__ay_value" in
      [Yy]|[Yy][Ee][Ss]) __ay_value=y; break ;;
      [Nn]|[Nn][Oo])     __ay_value=n; break ;;
      *) echo "  Jawab y atau n." ;;
    esac
  done
  printf '[INPUT] %s => %s\n' "$__ay_prompt" "$__ay_value" >> "$LOG_FILE"
  printf -v "$__ay_var" '%s' "$__ay_value"
}

# jalankan satu perintah, catat perintah+output/error+exit code ke log,
# TETAP tampil live di layar (bukan cuma sesudahnya -- proses ini bisa lama)
run_logged() {   # run_logged "deskripsi" perintah arg1 arg2...
  local desc="$1"; shift
  { echo; echo "=== $desc ($(date '+%Y-%m-%d %H:%M:%S')) ==="; echo "+ $*"; } >> "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status="${PIPESTATUS[0]}"
  echo "[exit=$status] $desc" >> "$LOG_FILE"
  return "$status"
}

# set/tambah KEY=VALUE di berkas config/*.env supaya pilihan wizard tersimpan
# permanen (dipakai lagi kalau user run.sh manual belakangan, bukan cuma
# selama proses wizard ini berjalan)
set_env_var() {   # set_env_var berkas KEY VALUE
  local file="$1" key="$2" val="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$file"
  else
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
}

siapkan_mmlu_pool() {
  mkdir -p "$CODE/work"
  if [ -f "$CODE/work/mmlu_pool.csv" ]; then
    log "[ok] mmlu_pool.csv sudah ada di \$CODE/work/, dilewati"
    return 0
  fi
  if [[ "$MMLU_POOL_SRC" == *.gz ]]; then
    [ -f "$MMLU_POOL_SRC" ] || { log "[GAGAL] $MMLU_POOL_SRC tidak ditemukan"; return 1; }
    run_logged "gunzip mmlu_pool.csv" bash -c "gunzip -c '$MMLU_POOL_SRC' > '$CODE/work/mmlu_pool.csv'"
  else
    [ -f "$MMLU_POOL_SRC" ] || { log "[GAGAL] $MMLU_POOL_SRC tidak ditemukan"; return 1; }
    run_logged "salin mmlu_pool.csv" cp "$MMLU_POOL_SRC" "$CODE/work/mmlu_pool.csv"
  fi
}

echo "============================================================"
echo " Setup Wizard -- Framework Fault Injection (FP16 / INT8)"
echo "============================================================"

# =========================================================== 1. path dasar
# Tidak ada satu pun path di bawah ini yang hardcode -- semua dari input user.
ask_required "Direktori ROOT di server ini (tempat semua data eksperimen)" ROOT "/workspace/fidelity"
ask_required "Path folder hasil git clone repo framework ini" REPO_CLONE "$HOME/Fidelity-Q"
export ROOT

if [ ! -d "$REPO_CLONE" ]; then
  log "[GAGAL] $REPO_CLONE tidak ditemukan -- clone dulu repo framework-nya sebelum lanjut:"
  log "        git clone https://github.com/mfikryrz/Fidelity-Q.git \"$REPO_CLONE\""
  exit 1
fi

CODE="$ROOT/running_experiment_7b"
mkdir -p "$ROOT" "$CODE"
log "ROOT       = $ROOT"
log "CODE       = $CODE"
log "REPO_CLONE = $REPO_CLONE"

# ============================================================= 2. presisi
echo
echo "Pilih presisi:"
echo "  1) FP16"
echo "  2) INT8"
while true; do
  ask "Pilihan (1/2)" PRECISION_CHOICE "1"
  case "$PRECISION_CHOICE" in
    1) PRECISION=float16; PRECISION_DIR=fp16; ONNX_SUBDIR=onnx_fp16; DISK_MIN_GB=60;  break ;;
    2) PRECISION=int8;    PRECISION_DIR=int8; ONNX_SUBDIR=onnx_int8; DISK_MIN_GB=150; break ;;
    *) echo "  Pilih 1 atau 2." ;;
  esac
done
# bootstrap_vast.sh sendiri default DISK_MIN_GB=120 (ukuran INT8) apa pun
# presisi yang dipilih -- FP16 dites GAGAL PADAHAL DISKNYA CUKUP (100GB > 60GB
# yang sebenarnya dibutuhkan FP16) sebelum baris ini ditambahkan. Export
# eksplisit sesuai presisi, angka dari komentar bootstrap_vast.sh sendiri.
export DISK_MIN_GB
SRC_DIR="$REPO_CLONE/src/$PRECISION_DIR"
[ -d "$SRC_DIR" ] || { log "[GAGAL] $SRC_DIR tidak ada di repo clone."; exit 1; }
log "Presisi    = $PRECISION ($SRC_DIR)"

# INT8 TIDAK punya bootstrap_vast.sh sendiri -- numpang milik FP16 (sama
# persis dipakai di kedua arm, cuma env PRECISION yang beda). Pakai variabel
# ini di kedua skenario, JANGAN "$SRC_DIR/runner/bootstrap_vast.sh" -- itu
# 404 untuk INT8.
BOOTSTRAP_SH="$REPO_CLONE/src/fp16/runner/bootstrap_vast.sh"
[ -f "$BOOTSTRAP_SH" ] || { log "[GAGAL] $BOOTSTRAP_SH tidak ada di repo clone."; exit 1; }

# ===================================================== 3. menu sumber model
echo
echo "Sumber model:"
echo "  1) Load model langsung dari Hugging Face"
echo "  2) Load model dari hasil konversi ONNX lokal"
while true; do
  ask "Pilihan (1/2)" MODEL_SOURCE_CHOICE "1"
  case "$MODEL_SOURCE_CHOICE" in
    1) MODEL_SOURCE=huggingface; break ;;
    2) MODEL_SOURCE=onnx_local;  break ;;
    *) echo "  Pilih 1 atau 2." ;;
  esac
done
log "Sumber model = $MODEL_SOURCE"

MODEL_ID=""
ONNX_DIR=""
ONNX_READY=n
NEEDS_EXPORT=0
IS_CUSTOM_ARCH=n

if [ "$MODEL_SOURCE" = huggingface ]; then
  # ---------------------------------------------------- 3a. dari HuggingFace
  ask_required "HF_TOKEN (token READ-ONLY dari huggingface.co/settings/tokens)" HF_TOKEN
  ask_required "MODEL_ID (ID repo HuggingFace)" MODEL_ID "NousResearch/Llama-2-7b-hf"
  export HF_TOKEN MODEL_ID

else
  # ----------------------------------------------------- 3b. dari ONNX lokal
  ask_yn "Apakah model Anda sudah dikonversi ke ONNX? (y/n)" ONNX_READY

  if [ "$ONNX_READY" = y ]; then
    ask_required "Path folder ONNX (berisi decoder-merge-*.onnx, embed.onnx, norm.onnx, head.onnx, tokenizer.model)" ONNX_DIR
    [ -d "$ONNX_DIR" ] || { log "[GAGAL] $ONNX_DIR tidak ditemukan."; exit 1; }
    export HF_TOKEN="local_saja"   # cuma supaya preflight bootstrap_vast.sh tidak menolak; tidak pernah dipakai memanggil HF
  else
    echo
    echo "Jenis model:"
    echo "  1) Llama-2-7B (arsitektur standar/MHA -- default eksperimen ini)"
    echo "  2) Model lain (Llama-3, Mistral, Gemma-2, atau arsitektur GQA lainnya)"
    while true; do
      ask "Pilihan (1/2)" MODEL_TYPE_CHOICE "1"
      case "$MODEL_TYPE_CHOICE" in
        1) MODEL_NAME="Llama-2-7B"; IS_CUSTOM_ARCH=n; break ;;
        2) ask_required "Nama/jenis model (label bebas)" MODEL_NAME; IS_CUSTOM_ARCH=y; break ;;
        *) echo "  Pilih 1 atau 2." ;;
      esac
    done
    ask_required "Path model LLM asal di server ini (asumsi sudah diunduh sebelumnya -- folder berisi config.json + *.safetensors)" MODEL_ID
    log "Model dipilih : $MODEL_NAME"
    log "Path sumber   : $MODEL_ID"   # << path model LLM asal ditampilkan ke layar + log
    [ -d "$MODEL_ID" ] || { log "[GAGAL] Path model '$MODEL_ID' tidak ditemukan di server ini."; exit 1; }
    [ -f "$MODEL_ID/config.json" ] || log "[PERINGATAN] $MODEL_ID/config.json tidak ada -- pastikan ini folder snapshot HF yang valid."
    NEEDS_EXPORT=1
    export HF_TOKEN="local_saja" MODEL_ID
  fi
fi

# ========================================= 4. mmlu_pool.csv -- path dinamis
ask_required "Path mmlu_pool.csv (langsung .csv, atau .csv.gz hasil bootstrap tahap 30)" \
  MMLU_POOL_SRC "$ROOT/_dl/code/mmlu_pool.csv.gz"

# =============================================================== skenario A
# Sumber model = Hugging Face
scenario_huggingface() {
  log "### Skenario eksekusi: Hugging Face ###"
  local stage_to
  [ "$PRECISION" = int8 ] && stage_to=50 || stage_to=40   # INT8: skip tahap 40 (ONNX FP16 tidak perlu), lanjut ke 50 (ONNX INT8 + SmoothQuant)

  log "Memeriksa & memasang dependensi (Python 3.10, torch, transformers, onnxruntime) kalau belum ada, lalu mengunduh model+ONNX dari HuggingFace..."
  run_logged "bootstrap_vast.sh 10 $stage_to" \
    bash "$BOOTSTRAP_SH" 10 "$stage_to" \
    || { log "[GAGAL] bootstrap_vast.sh"; return 1; }

  siapkan_mmlu_pool || return 1

  cp -n "$SRC_DIR/config/${PRECISION_DIR}.env.example" "$SRC_DIR/config/${PRECISION_DIR}.env" 2>/dev/null || true
  set_env_var "$SRC_DIR/config/${PRECISION_DIR}.env" MODEL_ID "$MODEL_ID"

  ( cd "$SRC_DIR" && run_logged "setup.sh" ./setup.sh ) || return 1
  ( cd "$SRC_DIR" && run_logged "run.sh --preset smoke" ./run.sh --preset smoke ) || return 1
}

# =============================================================== skenario B
# Sumber model = unduhan manual / konversi ONNX lokal
scenario_onnx_local() {
  log "### Skenario eksekusi: ONNX lokal / manual ###"

  log "Memeriksa & memasang dependensi (Python 3.10, torch, transformers, onnxruntime) kalau belum ada, dan menyiapkan kode dasar (TANPA menarik ONNX dari HuggingFace)..."
  run_logged "bootstrap_vast.sh 10 30" \
    bash "$BOOTSTRAP_SH" 10 30 \
    || { log "[GAGAL] bootstrap_vast.sh 10-30"; return 1; }

  cp -n "$SRC_DIR/config/${PRECISION_DIR}.env.example" "$SRC_DIR/config/${PRECISION_DIR}.env" 2>/dev/null || true
  [ -n "$MODEL_ID" ] && set_env_var "$SRC_DIR/config/${PRECISION_DIR}.env" MODEL_ID "$MODEL_ID"

  ( cd "$SRC_DIR" && run_logged "setup.sh (sinkron patch awal)" ./setup.sh )   # boleh gagal di sini (injection_llm belum ada) -- itu normal, ditangani di setup.sh final nanti

  local REPO="$CODE/repo/FIdelity-ONNX-master"

  if [ "$NEEDS_EXPORT" = 1 ]; then
    log "Menjalankan konversi ke ONNX secara otomatis..."
    if [ "$PRECISION" = float16 ]; then
      run_logged "export_llama.py (konversi ke ONNX FP16)" \
        "$ROOT/expenv/bin/python" "$REPO/export_llama.py" \
          --repo "$REPO" --model_id "$MODEL_ID" \
          --out "$CODE/onnx/_onnx_raw_fp32" --fp16 "$CODE/onnx/onnx_fp16" \
        || { log "[GAGAL] export_llama.py"; return 1; }
    else
      if [ "$IS_CUSTOM_ARCH" = y ]; then
        log "[PERINGATAN] INT8 utk arsitektur non-Llama-2-7B BELUM didukung penuh -- export_llama_int8.py memakai kalibrasi SmoothQuant (act_scales) Llama-2-7B, belum ada kalibrasi khusus model lain. Lanjut tetap dicoba, tapi hasilnya belum tentu valid -- lihat PANDUAN_CEPAT.md bagian 3."
      fi
      run_logged "export_llama_int8.py (konversi ke ONNX INT8)" \
        "$ROOT/expenv/bin/python" "$REPO/export_llama_int8.py" \
          --repo "$REPO" --model_id "$MODEL_ID" --out "$CODE/onnx/onnx_int8" \
        || { log "[GAGAL] export_llama_int8.py"; return 1; }
    fi

    if [ "$IS_CUSTOM_ARCH" = y ]; then
      ( cd "$SRC_DIR" && run_logged "setup.sh (generate injection_llm/*.json awal)" ./setup.sh )
      run_logged "discover_operasi.py (deteksi nama operasi utk arsitektur baru)" \
        "$ROOT/rtenv/bin/python" "$SRC_DIR/pipeline/discover_operasi.py" \
          --injection-dir "$REPO/injection_llm" \
          --onnx "$CODE/onnx/$ONNX_SUBDIR/decoder-merge-0.onnx" \
          --model-config "$REPO/configs/my_model.json" \
        || log "[PERINGATAN] discover_operasi.py gagal -- operasi mungkin masih pakai nama Llama-2-7B."
    fi
  else
    log "Menyalin ONNX yang sudah ada dari $ONNX_DIR ..."
    mkdir -p "$CODE/onnx/$ONNX_SUBDIR"
    run_logged "salin ONNX" rsync -a "$ONNX_DIR"/ "$CODE/onnx/$ONNX_SUBDIR/" \
      || { log "[GAGAL] gagal menyalin ONNX dari $ONNX_DIR"; return 1; }
  fi

  siapkan_mmlu_pool || return 1

  ( cd "$SRC_DIR" && run_logged "setup.sh (final -- lengkapi injection_llm/*.json)" ./setup.sh ) || return 1
  ( cd "$SRC_DIR" && run_logged "run.sh --preset smoke" ./run.sh --preset smoke ) || return 1
}

# ==================================================================== main
echo
log "Mulai eksekusi skenario: $MODEL_SOURCE"
if [ "$MODEL_SOURCE" = huggingface ]; then
  scenario_huggingface
else
  scenario_onnx_local
fi
STATUS=$?

echo
if [ "$STATUS" -eq 0 ]; then
  log "===== SELESAI -- setup OK, ./run.sh --preset smoke sudah dijalankan. ====="
  log "Periksa manual hasil smoke test (golden_teks/faulty_teks berisi kalimat, ada baris g_logit_* != f_logit_*)."
  log "Kalau OK, lanjutkan manual (run penuh, background, bisa berjam-jam/berhari-hari -- SENGAJA tidak diotomasi di wizard ini):"
  log "  cd $SRC_DIR && ROOT=$ROOT ./run.sh --preset full"
else
  log "===== GAGAL (exit=$STATUS) -- lihat $LOG_FILE untuk detail lengkap. ====="
fi
log "Log lengkap: $LOG_FILE"
exit "$STATUS"
