#!/usr/bin/env bash
# ============================================================================
# terapkan_config_dinamis.sh — WAJIB dijalankan sekali per instance/mesin
# supaya perbaikan "daftar operasi tidak lagi hardcode" (repo_terpatch/_config.py
# + fidelity_utils.py, 21 Sep 2026) benar-benar berlaku, bukan cuma tersimpan
# di src/. Berlaku untuk FP16 maupun INT8 — _config.py sudah presisi-agnostik
# (baca env PRECISION), jadi berkas sumbernya sama persis di kedua folder.
#
# GEJALA KALAU TIDAK DIJALANKAN
#   Mesin tetap memakai _config.py/fidelity_utils.py versi lama yang sudah
#   ada di $CODE (biasanya dari code.tar.gz/HF code/**) — OPERASI_SUFFIXES
#   tetap daftar tetap 9 nama Llama, dan operasi model lain tetap DIAM-DIAM
#   DIBUANG oleh list_canonical_configs(). Tidak ada error, datanya cuma
#   hilang. Untuk Llama-2-7B sendiri tidak berdampak (datanya toh cocok),
#   tapi begitu configs/my_model.json dapat field operation_suffixes untuk
#   model lain (lewat pipeline/discover_operasi.py), mesin ini HARUS sudah
#   memakai versi ini supaya field itu benar-benar dibaca.
#
# SEBAB
#   repo_terpatch/ di sini adalah SALINAN DOKUMENTASI/rujukan -- bukan
#   sesuatu yang otomatis disalin balik ke mesin baru. Beda dari
#   memory_pool.py milik INT8 (yang punya patch_int8_memory_pool.sh
#   sendiri dan TIDAK berlaku untuk FP16), _config.py dan fidelity_utils.py
#   belum punya mekanisme serupa sebelum skrip ini ada.
#
#   Topologi nyata di instance (dikonfirmasi dari jalankan_fp16.sh dan
#   struktur eksperimen yang pernah berjalan):
#     $CODE/mass_loglik_inject.py     <- yang dijalankan; HERE=dirname(diri
#                                         sendiri) masuk sys.path[0] duluan
#     $CODE/_config.py                <- DIPAKAI mass_loglik_inject.py (lewat HERE)
#     $CODE/fidelity_utils.py         <- DIPAKAI mass_loglik_inject.py (lewat HERE)
#     $CODE/repo/FIdelity-ONNX-master/_config.py   <- salinan kedua, dipakai
#                                         skrip fase lain yang REPO-nya
#                                         di sys.path sendiri; disinkronkan
#                                         juga di sini supaya tidak bercabang
#   fidelity_utils.py TIDAK punya salinan kedua di repo/FIdelity-ONNX-master/
#   (logikanya spesifik proyek ini, bukan bagian toolkit FIdelity-ONNX asli).
#
# Pakai:  bash terapkan_config_dinamis.sh
# Aman diulang -- kalau isi sudah sama, dilewati tanpa mengubah apa pun.
# ============================================================================
set -uo pipefail
ROOT=${ROOT:-/workspace/fidelity}
CODE=${CODE:-$ROOT/running_experiment_7b}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../repo_terpatch" && pwd)"

targets_config_py=(
  "$CODE/_config.py"
  "$CODE/repo/FIdelity-ONNX-master/_config.py"
)
targets_fidelity_utils_py=(
  "$CODE/fidelity_utils.py"
)
# export_llama.py: dipakai bootstrap_vast.sh tahap 40 (jalur export lokal,
# fallback kalau ONNX FP16 tidak tersedia siap-pakai di HF). Versi di sini
# menambah dukungan MODEL_ID berupa path lokal (21 Sep 2026) -- lihat
# komentar "DIPATCH" di repo_terpatch/export_llama.py.
targets_export_llama_py=(
  "$CODE/repo/FIdelity-ONNX-master/export_llama.py"
)
# modeling_llama.py: exporter hooks yang DITIMPA export_llama.py/export_llama_int8.py
# ke atas transformers.models.llama.modeling_llama sebelum export. Versi di
# sini menambah dukungan GQA (repeat_kv, num_key_value_heads) -- 21 Sep 2026,
# diverifikasi numerik lokal (CPU) terhadap referensi naif, belum diuji
# terhadap model GQA sungguhan di GPU. Lihat repo_terpatch/modeling_llama.py.
targets_modeling_llama_py=(
  "$CODE/repo/FIdelity-ONNX-master/int8/modeling_llama.py"
)

terapkan() {  # $1=berkas sumber di repo_terpatch/  $2...=target
  local src="$SRC/$1"; shift
  local nama; nama=$(basename "$src")
  [ -f "$src" ] || { echo "[skip] sumber tidak ada: $src" >&2; return 1; }

  local ada=0 diubah=0
  for target in "$@"; do
    if [ ! -f "$target" ]; then
      echo "[peringatan] target tidak ada, dilewati: $target" >&2
      continue
    fi
    ada=1
    if cmp -s "$src" "$target"; then
      echo "[ok] $target sudah sinkron, dilewati"
      continue
    fi
    local cadangan="$target.sebelum_dinamis.$(date -u +%Y%m%d%H%M%S)"
    cp -a "$target" "$cadangan"
    cp "$src" "$target"
    echo "[diterapkan] $target  (cadangan lama: $cadangan)"
    diubah=1
  done

  if [ "$ada" -eq 0 ]; then
    echo "[error] tidak ada satu pun target valid untuk $nama -- periksa CODE=$CODE" >&2
    return 1
  fi
  [ "$diubah" -eq 1 ] && PY_CHECK+=("$@")
  return 0
}

PY_CHECK=()
gagal=0
terapkan "_config.py" "${targets_config_py[@]}" || gagal=1
terapkan "fidelity_utils.py" "${targets_fidelity_utils_py[@]}" || gagal=1
# TIDAK ikut menggagalkan skrip kalau tidak ada -- export_llama.py cuma
# dipakai jalur fallback (export lokal), tidak relevan kalau ONNX didapat
# lewat unduhan langsung dari HF (jalur utama).
terapkan "export_llama.py" "${targets_export_llama_py[@]}" \
  || echo "[info] export_llama.py tidak disinkronkan -- tidak masalah kalau tidak dipakai jalur export lokal"
terapkan "modeling_llama.py" "${targets_modeling_llama_py[@]}" \
  || echo "[info] modeling_llama.py tidak disinkronkan -- tidak masalah kalau tidak dipakai jalur export lokal"

if [ "${#PY_CHECK[@]}" -gt 0 ]; then
  PY=${PY:-$ROOT/rtenv/bin/python}
  if [ -x "$PY" ]; then
    for f in "${PY_CHECK[@]}"; do
      [ -f "$f" ] && "$PY" -m py_compile "$f" && echo "[sintaks ok] $f"
    done
  else
    echo "[info] $PY tidak ada -- lewati py_compile, cukup periksa manual" >&2
  fi
fi

exit "$gagal"
