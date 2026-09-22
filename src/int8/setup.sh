#!/usr/bin/env bash
# ============================================================================
# setup.sh — SATU entry point untuk menyiapkan INT8 di instance yang SUDAH
# di-bootstrap (ONNX + injection_llm sudah ada). Menjalankan kedua patch
# wajib dan memeriksa prasyarat SEBELUM run mahal, bukan sesudah gagal.
#
# TIDAK termasuk di sini: mengunduh ONNX (26 GB) dan 289 config injeksi dari
# awal. Itu proses terpisah, lebih manual, didokumentasikan di
# ../../docs/README_INT8.md bagian "Langkah 1-4" (atau pakai bundel portable
# di experiments/portable_int8_instance_49721366/bootstrap_from_hf.sh kalau
# tersedia) -- ditulis ulang sebagai skrip otomatis di sini beresiko diam-diam
# salah di satu dari empat langkah manualnya (path rewrite, dekompresi zst,
# dll.) tanpa pernah teruji di instance sungguhan. Lebih jujur dibiarkan
# eksplisit sampai ada kesempatan menguji versi otomatisnya di instance nyata.
#
# Pakai (dari instance, setelah bootstrap):
#   cd src/int8 && ./setup.sh
#
# Aman diulang. Baca config/int8.env kalau ada; kalau tidak ada, tetap jalan
# pakai default skrip masing-masing (peringatan dicetak).
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -f config/int8.env ]; then
  echo "[config] membaca config/int8.env"
  set -a; source config/int8.env; set +a
else
  echo "[peringatan] config/int8.env tidak ada -- pakai 'cp config/int8.env.example config/int8.env' dulu." >&2
  echo "             lanjut pakai default bawaan skrip (KEMUNGKINAN BESAR salah untuk INT8, mis. BITS=0-15)." >&2
fi

ROOT=${ROOT:-/workspace/fidelity}
CODE=$ROOT/running_experiment_7b
REPO=$CODE/repo/FIdelity-ONNX-master
PY=$ROOT/rtenv/bin/python
# WAJIB di-export: _config.py membaca FIDELITY_REPO untuk tahu di mana
# configs/my_model.json miliknya sendiri ada. Tanpa ini, decoder_count()/
# operasi_suffixes() diam-diam jatuh ke fallback (32 decoder, 9 operasi
# Llama) alih-alih memvalidasi angka sungguhan -- pre-flight jadi tidak
# pernah bisa menangkap ketidakcocokan yang justru ia dibuat untuk mendeteksi.
export FIDELITY_REPO="$REPO"
# Sama alasannya: konsisten dengan jalankan_fp16.sh (FIDELITY_ONNX_ROOT=$CODE/onnx),
# supaya pre-flight memeriksa lokasi ONNX yang SAMA dengan yang dipakai run
# sungguhan -- bukan menebak lewat dirname(REPO) yang keliru satu tingkat.
export FIDELITY_ONNX_ROOT="$CODE/onnx"
export PRECISION=int8

gagal=0
langkah() { echo; echo "=== $1 ==="; }

# Salin satu berkas sumber -> tujuan, cadangkan dulu kalau isinya beda,
# lewati kalau sudah sinkron.
sinkronkan_berkas() {   # $1=deskripsi  $2=sumber  $3=tujuan
  local desk="$1" src="$2" dst="$3"
  if [ ! -f "$src" ]; then
    echo "  [GAGAL] sumber tidak ada ($desk): $src"; gagal=1; return
  fi
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    echo "  [ok] $dst sudah sinkron, dilewati"; return
  fi
  [ -f "$dst" ] && cp -a "$dst" "$dst.sebelum_sinkron.$(date -u +%Y%m%d%H%M%S)"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "  [diterapkan] $dst"
}

# ---- 1. sinkronkan mass_loglik_inject.py & _download_7b.py ke $CODE --------
# mass_loglik_inject.py WAJIB, ditemukan lewat uji nyata di arm FP16 (21 Sep
# 2026, pola sama berlaku di sini): bootstrap yang menarik code/** dari repo
# HF bisa jadi SNAPSHOT LAMA (sebelum revisi protokol 10 Sep): tidak kenal
# --posisi-token penuh atau --konfigurasi-dari sama sekali. jalankan_fp16.sh
# memakai keduanya -> gagal di argumen pertama kalau tidak disinkronkan dulu.
#
# _download_7b.py: versi di sini mendukung MODEL_ID berupa path lokal (lihat
# komentar di pipeline/_download_7b.py) -- versi bundel HF tidak.
langkah "1/6 sinkronkan mass_loglik_inject.py & _download_7b.py"
sinkronkan_berkas "mass_loglik_inject.py" "$HERE/pipeline/mass_loglik_inject.py" "$CODE/mass_loglik_inject.py"
sinkronkan_berkas "_download_7b.py" "$HERE/pipeline/_download_7b.py" "$CODE/_download_7b.py"

# ---- 2. pastikan req_rtenv.txt TERPIN yang dipakai, bukan daftar minimum ---
langkah "2/6 pastikan req_rtenv.txt terpin terpasang"
if [ -x "$PY" ]; then
  cp -n "$HERE/req_rtenv.txt" "$ROOT/req_rtenv.txt" 2>/dev/null || true
  echo "  memasang $(wc -l < "$HERE/req_rtenv.txt") paket terpin (idempoten, tidak menyentuh yg sudah cocok)..."
  "$ROOT/rtenv/bin/pip" install -q -r "$HERE/req_rtenv.txt" \
    && echo "  [ok] req_rtenv.txt terpasang" \
    || { echo "  [GAGAL] pip install req_rtenv.txt"; gagal=1; }
else
  echo "  [lewati] rtenv python tidak ada ($PY)"
fi

# ---- 3. patch memory_pool.py (RANDOM_BITFLIP) ------------------------------
langkah "3/6 patch memory_pool.py (RANDOM_BITFLIP)"
ROOT="$ROOT" bash runner/patch_int8_memory_pool.sh || gagal=1

# ---- 4. sinkronkan _config.py / fidelity_utils.py --------------------------
langkah "4/6 sinkronkan _config.py, fidelity_utils.py, export_llama_int8.py (daftar operasi dinamis + dukungan model lokal)"
ROOT="$ROOT" CODE="$CODE" bash runner/terapkan_config_dinamis.sh || gagal=1

# ---- 5. pastikan injection_llm/*.json lengkap (generate kalau belum) ------
# WAJIB, ditemukan lewat uji nyata (21 Sep 2026): bootstrap ONNX INT8 (tahap
# 50 di bootstrap_vast.sh milik FP16 -- INT8 belum punya bootstrap sendiri,
# lihat catatan di kepala berkas ini) TIDAK PERNAH men-generate injection_llm/
# sama sekali -- itu tahap terpisah (parser.py generik dari toolkit
# FIdelity-ONNX, dipanggil lewat 03_phase2_parser.py). Sebelumnya ini langkah
# manual yang gampang terlupa; sekarang setup.sh mendeteksi & menjalankannya
# sendiri kalau memang belum lengkap, supaya satu perintah ./setup.sh
# sungguhan cukup -- ASAL ONNX INT8 & configs/my_model.json sudah ada
# (itu prasyarat yang TIDAK diotomasi di sini, lihat catatan di atas).
langkah "5/6 pastikan injection_llm/*.json lengkap"
sinkronkan_berkas "03_phase2_parser.py" "$HERE/pipeline/03_phase2_parser.py" "$CODE/03_phase2_parser.py"

if [ -x "$PY" ] && [ -d "$REPO" ]; then
  N_ADA=$("$PY" -c "
import glob, os, sys
sys.path.insert(0, '$CODE'); sys.path.insert(0, '$REPO')
import _config
decoders = _config.decoder_count(); ops = _config.operasi_suffixes()
found = len(glob.glob(os.path.join('$REPO', 'injection_llm', 'decoder-merge-*__*.json')))
print(found); sys.exit(0 if found == decoders * len(ops) else 1)
" 2>/dev/null)
  ADA_STATUS=$?
  if [ "$ADA_STATUS" -eq 0 ]; then
    echo "  [ok] injection_llm/: $N_ADA config, sudah lengkap, dilewati"
  else
    echo "  [generate] injection_llm/ belum lengkap ($N_ADA config) -- menjalankan 03_phase2_parser.py..."
    mkdir -p "$ROOT/logs"
    if ( cd "$CODE" \
      && FIDELITY_REPO="$REPO" FIDELITY_WORK="$CODE/work" FIDELITY_ONNX_ROOT="$CODE/onnx" \
         HF_HOME="$CODE/hf_home" TMPDIR="$CODE/tmp" PRECISION="$PRECISION" \
         MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}" PYTHONUNBUFFERED=1 \
         "$PY" 03_phase2_parser.py > "$ROOT/logs/03_phase2_parser.log" 2>&1 ); then
      echo "  [ok] 03_phase2_parser.py selesai (log: $ROOT/logs/03_phase2_parser.log)"
    else
      echo "  [GAGAL] 03_phase2_parser.py gagal -- lihat $ROOT/logs/03_phase2_parser.log"
      gagal=1
    fi
  fi
else
  echo "  [lewati] python/repo tidak ditemukan"
fi

# ---- 6. pre-flight: periksa prasyarat sebelum run mahal --------------------
langkah "6/6 pre-flight"

cek() {   # $1=deskripsi  $2=kondisi shell
  if eval "$2"; then echo "  [ok] $1"; else echo "  [GAGAL] $1"; gagal=1; fi
}

cek "rtenv python ada ($PY)" "[ -x \"$PY\" ]"
cek "repo FIdelity-ONNX-master ada ($REPO)" "[ -d \"$REPO\" ]"
cek "mass_loglik_inject.py ada di \$CODE" "[ -f \"$CODE/mass_loglik_inject.py\" ]"

if [ -x "$PY" ] && [ -d "$REPO" ]; then
  "$PY" - "$REPO" "$CODE" <<'EOF'
import glob, json, os, sys
repo, code = sys.argv[1], sys.argv[2]
# _config.py DIIMPOR dari $CODE (persis seperti mass_loglik_inject.py lewat
# HERE) -- BUKAN dari $REPO. Sebagian bundel juga punya salinan kedua di
# $REPO, sebagian tidak (ditemukan lewat uji nyata 21 Sep 2026); $CODE
# adalah satu-satunya lokasi yang dijamin ada di semua bundel.
sys.path.insert(0, code)
sys.path.insert(0, repo)
try:
    import _config
except Exception as e:
    print(f"  [GAGAL] import _config.py dari {code}: {e}")
    raise SystemExit(1)

decoders = _config.decoder_count()
ops = _config.operasi_suffixes()
expected = decoders * len(ops)
found = len(glob.glob(os.path.join(repo, "injection_llm", "decoder-merge-*__*.json")))
status = "ok" if found == expected else "GAGAL"
print(f"  [{status}] injection_llm/: {found} config (harap {decoders} decoder x "
      f"{len(ops)} operasi = {expected})")
if found != expected:
    raise SystemExit(1)

# _config.ONNX_DIR sudah menangani percabangan onnx_fp16/ vs onnx_int8/ lewat
# PRECISION -- JANGAN rekonstruksi path manual di sini (dua bug beda pernah
# ditemukan dari cara itu: salah tingkat dirname, dan lupa subfolder presisi).
onnx_dir = _config.ONNX_DIR
n_onnx = len(glob.glob(os.path.join(onnx_dir, "decoder-merge-*.onnx")))
status = "ok" if n_onnx == decoders else "GAGAL"
print(f"  [{status}] ONNX decoder: {n_onnx} berkas di {onnx_dir} (harap {decoders})")
if n_onnx != decoders:
    raise SystemExit(1)

mp = os.path.join(repo, "llama", "memory_pool.py")
patched = os.path.isfile(mp) and "_ext_lib" in open(mp).read()
print(f"  [{'ok' if patched else 'GAGAL'}] memory_pool.py sudah dipatch (_ext_lib terdaftar)")
if not patched:
    raise SystemExit(1)
EOF
  [ $? -ne 0 ] && gagal=1
else
  echo "  [lewati] pemeriksaan injection_llm/ONNX/memory_pool -- python/repo tidak ditemukan"
fi

echo
if [ "$gagal" -eq 0 ]; then
  echo "=== SETUP OK -- siap ./run.sh --preset smoke ==="
else
  echo "=== ADA YANG GAGAL -- perbaiki dulu di atas sebelum run.sh ===" >&2
fi
exit "$gagal"
