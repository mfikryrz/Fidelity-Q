# Kode FP16

Panduan lengkap (protokol, biaya, jebakan): [`../../docs/README_FP16.md`](../../docs/README_FP16.md).
Bagian di bawah ini adalah **quickstart operasional** — cara menjalankan,
bukan cerita eksperimennya.

Padanan folder ini untuk INT8: [`../int8/`](../int8/) — struktur sama persis
(`setup.sh`, `run.sh`, `config/`, `runner/`, `pipeline/`, `analisis/`,
`repo_terpatch/`, `tests/`), tapi banyak angka di `config/` **berbeda nilai**
karena model INT8 26 GB vs FP16 13 GB. Jangan salin config satu ke yang lain.

---

## Quickstart

Prasyarat: instance dengan `rtenv` (Python 3.10 + `req_rtenv.txt`) dan ONNX
**sudah di-bootstrap** — pakai `runner/bootstrap_vast.sh` (satu-satunya dari
kedua arm yang punya bootstrap otomatis; INT8 belum, lihat `../int8/setup.sh`).
`bootstrap_vast.sh` menyiapkan ONNX + `configs/my_model.json`, **bukan**
`injection_llm/*.json` — itu digenerate otomatis oleh `setup.sh` di langkah 2
kalau belum lengkap (lihat komentar langkah "4/5" di `setup.sh`), jadi tidak
ada langkah manual terpisah yang perlu diingat.

```bash
# 1. bootstrap instance dari nol (unduh model, siapkan rtenv, dst.)
cd src/fp16
bash runner/bootstrap_vast.sh   # baca dulu isinya -- perlu HF_TOKEN

# 2. sekali per instance: sinkronkan config dinamis + lengkapi injection_llm/*.json + pre-flight
cp config/fp16.env.example config/fp16.env   # sesuaikan kalau GPU-mu bukan A100 80GB
./setup.sh

# 3. WAJIB sebelum run penuh: uji asap beberapa menit, foreground
./run.sh --preset smoke
#   periksa manual: golden_teks/faulty_teks berisi kalimat (bukan kosong),
#   g_logit_* != f_logit_* pada sebagian baris (bukti injeksi masuk)

# 4. run penuh: 288 konfigurasi, background, survive SSH putus, ~2,5 hari
./run.sh --preset full

# kapan saja:
./run.sh --status
```

`--preset full` menolak jalan kalau pengawas sebelumnya masih hidup (cegah
dua worker menulis CSV yang sama — insiden yang sudah pernah terjadi dan
menghilangkan 42% baris diam-diam), dan memasang watchdog VRAM
(`penjaga_vram.sh`, **bukan** `penjaga_fp16.sh` versi lama — lihat
`../../docs/README_FP16.md` bagian "Lapis 3" untuk alasannya) ke crontab
otomatis kalau belum ada. Mengulang `./run.sh --preset full` yang sama
setelah run terputus itulah cara "resume".

Semua angka (arena GPU, PER_PROSES, ambang watchdog) ada di
`config/fp16.env.example` dengan alasan tiap nilai — jangan tebak, baca
komentarnya kalau GPU-mu bukan A100 80GB.

---

## Isi

```
setup.sh                         entry point: sinkronkan config + pre-flight
run.sh                            entry point: --preset smoke|full, --status
config/fp16.env.example          SEMUA angka yang beda per mesin/GPU
req_rtenv.txt                    dependency terkunci (==)
pipeline/mass_loglik_inject.py   inti eksperimen (identik dengan INT8 --
                                 presisi ditentukan env PRECISION=float16)
pipeline/discover_operasi.py     generate operation_suffixes utk model baru
pipeline/_download_7b.py         unduh bobot -- mendukung MODEL_ID path lokal
analisis/cek_cakupan.py          verifikasi cakupan, pakai --bit 16 untuk FP16
analisis/hitung_penilaian.py     metrik, jalan di laptop tanpa GPU
runner/bootstrap_vast.sh          siapkan instance dari nol
runner/jalankan_fp16.sh          runner dua tahap
runner/pengawas_fp16.sh          mengulang sampai target tercapai
runner/penjaga_vram.sh           watchdog: proses mati + kegagalan VRAM (pakai INI, bukan penjaga_fp16.sh, untuk run baru)
runner/penjaga_fp16.sh           watchdog LAMA -- dipakai run produksi yang sudah selesai, jangan pakai untuk run baru
runner/terapkan_config_dinamis.sh PATCH WAJIB -- sinkronkan _config.py/
                                 fidelity_utils.py/export_llama.py/
                                 modeling_llama.py instance dgn repo_terpatch/
runner/status.sh                 ringkasan satu layar
repo_terpatch/_config.py, fidelity_utils.py   daftar operasi dinamis (model lain)
repo_terpatch/export_llama.py    exporter -- sekarang terima model lokal
repo_terpatch/modeling_llama.py  exporter hooks -- sekarang dukung GQA
tests/test_discover_operasi.py   regresi + bukti anti "silent drop"
```

FP16 **tidak** butuh patch `memory_pool.py` seperti INT8 — tensor FP16 murni
sudah memakai jalur `custom.bitflip:BitFlip` yang terdaftar dari awal. Lihat
komentar di `setup.sh` untuk alasannya.

---

## Model lain (GQA, model lokal) — 21 Sep 2026

Dua hal yang dulu jadi penghalang model non-Llama-2 sudah ditutup — detail
lengkap (termasuk bukti verifikasi GPU dan peringatan penting soal Qwen2/Gemma
**asli** belum ikut terbantu) ada di bagian "Model lain (GQA, model lokal)"
di [`../int8/README.md`](../int8/README.md) (mekanismenya identik untuk kedua arm):

- **GQA** (`num_key_value_heads < num_attention_heads` — Llama-3, Mistral, dan
  model lain bertag arsitektur `"llama"`) didukung lewat `repeat_kv()` di
  `repo_terpatch/modeling_llama.py`, diverifikasi di GPU nyata (selisih ONNX
  vs PyTorch level floating-point noise).
- **Model lokal**: `MODEL_ID` di `config/fp16.env` boleh path direktori lokal,
  bukan cuma ID repo HF — lihat `pipeline/_download_7b.py`.
