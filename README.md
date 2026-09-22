# hasil_pgteks — Fault Injection Llama-2-7B (FP16 / INT8)

Eksperimen fault injection pada Llama-2-7B lewat FIdelity-ONNX: satu bit
dirusak dalam operasi MatMul tertentu, lalu keluaran model dibandingkan
dengan keluaran tanpa kerusakan (golden vs faulty) pada soal MMLU yang sama.
Dijalankan dalam dua varian presisi — **FP16** dan **INT8** (fake-INT8:
SmoothQuant, disimpan & dieksekusi sebagai FP32 — lihat
[`docs/README_INT8.md`](docs/README_INT8.md) bagian 1).

Direktori ini dirapikan 20 September 2026 dari kumpulan folder/zip yang
menumpuk selama eksperimen berjalan, menjadi struktur berikut. **Kalau kamu
membaca ini lewat GitHub**: cuma `docs/`, `src/`, dan `README.md` ini yang
ikut ter-push (lihat `.gitignore`) — `results/`, `experiments/`, `zip_files/`,
`rahasia/` cuma ada di salinan lokal pembuat repo, terlalu besar (`results/`
punya berkas 149MB, melebihi batas GitHub) atau memang tidak untuk dibagikan
(`rahasia/`). Deskripsinya tetap disertakan di bawah supaya konteksnya jelas.

```
docs/          <- MULAI DI SINI. README_FP16.md dan README_INT8.md adalah
                  dokumen acuan tunggal untuk tiap varian (protokol, cara
                  menjalankan, biaya, jebakan). STATUS_BERJALAN.md dan
                  PROGRES_INT8.txt adalah snapshot serah-terima terakhir.
                  dokumen_protokol_lama/ dan rapat/ adalah arsip pendukung.

src/           <- Kode SUMBER TERBARU, siap dipakai ulang di komputer/server lain.
  fp16/          kode FP16 -- entry point: setup.sh lalu run.sh (lihat
                 src/fp16/README.md untuk quickstart lengkap)
  int8/          kode INT8 -- struktur & entry point sama (setup.sh, run.sh),
                 lihat src/int8/README.md

results/       <- (lokal saja, lihat catatan di atas) Hasil FINAL dari run
                  masal 288 konfigurasi, sudah diekstrak.
  fp16/final/       massal_fp16_288.csv (55.245/55.296 baris, 99,91%)
  fp16/pelengkap/   benchmark tambahan + mmlu_pool.csv + req_rtenv.txt
  int8/final/       massal_int8_288.csv
  int8/pelengkap/   uji asap + referensi pendukung INT8

experiments/   <- (lokal saja) Percobaan, smoke test, dan backup HISTORIS
                  (bukan hasil final). Tiap subfolder diberi nama
                  instance/tanggal. legacy_massal_pgteks/ adalah data dari
                  sebelum protokol fp16/int8 dipisah (10 Sep 2026) — konteks
                  lama, bukan bagian dari eksperimen saat ini.

zip_files/     <- (lokal saja) Semua arsip .zip/.tar.gz (termasuk salinan
                  terkompresi dari src/ dan results/ di atas, plus kiriman
                  untuk kolega). Folder yang sudah diekstrak ada di results/,
                  experiments/, atau src/ — arsip ini salinan cadangan.

rahasia/       <- (lokal saja, dan memang tidak untuk dibagikan) Token
                  HuggingFace dkk (mode 600).
```

## Tutorial: instalasi & menjalankan eksperimen

Ini panduan langkah-demi-langkah yang sesungguhnya (bukan cuma link) untuk
menjalankan eksperimen dari nol, di **komputer/server sendiri (Linux + GPU
NVIDIA)** maupun di **instance cloud** (mis. Vast.ai — begitu cara aslinya
dijalankan). Kedua varian butuh GPU dengan VRAM besar (lihat tabel di bawah);
laptop/PC tanpa GPU semacam itu tidak bisa menjalankan tahap eksperimennya,
tapi tetap bisa menjalankan tahap analisis (langkah 3) di CPU biasa.

### 0. Prasyarat

| | FP16 | INT8 |
|---|---|---|
| VRAM GPU | ≥ 80 GB (A100 80GB terbukti) | ≥ 80 GB, **140 GB (H200) lebih aman** |
| Disk kosong | ≥ 60 GB | ≥ 120–150 GB |
| Kecepatan disk | **≥ 1,5 GB/detik** (penentu biaya, bukan GPU) | ≥ 1 GB/detik cukup |
| OS | Linux, akses root/sudo | sama |

Selain itu:

- **Token HuggingFace read-only** (`huggingface.co/settings/tokens`) — cuma
  perlu kalau bobot model diunduh dari HF. Kalau bobotnya sudah ada di
  server lain (volume bersama/NFS), lewati ini dan arahkan `MODEL_ID` ke
  path lokal (lihat bagian "Dukungan model lain" di bawah) — token tidak
  dibutuhkan sama sekali.
- Python 3.10 (kedua arm membuat virtualenv sendiri, `rtenv`, jadi tidak
  perlu disiapkan manual — `bootstrap_vast.sh`/`./setup.sh` yang mengurus).
- Salin/`git clone` repo ini ke instance/server yang mau dipakai.

### 1. Varian FP16 — bootstrap penuh otomatis

```bash
cd src/fp16
export HF_TOKEN=hf_xxxxxxxxxxxxx     # token read-only, lewati kalau MODEL_ID path lokal
bash runner/bootstrap_vast.sh        # unduh model, export ONNX, siapkan rtenv (baca isinya dulu)

cp config/fp16.env.example config/fp16.env   # sesuaikan kalau GPU-mu bukan A100 80GB
./setup.sh                           # sinkron config dinamis + lengkapi injection_llm/*.json + pre-flight

./run.sh --preset smoke              # WAJIB dulu: uji asap beberapa menit, foreground
#   periksa manual: golden_teks/faulty_teks berisi kalimat (bukan kosong),
#   g_logit_* != f_logit_* pada sebagian baris (bukti injeksi masuk)

./run.sh --preset full               # run penuh: 288 konfigurasi, background, survive SSH putus, ~2,5 hari
./run.sh --status                    # kapan saja, cek progres
```

Kalau run terputus (SSH putus, instance restart, dll.), ulangi
`./run.sh --preset full` yang sama persis — itulah cara resume-nya, otomatis
melanjutkan dari baris terakhir.

Detail tiap langkah (kenapa nilainya begitu, angka apa yang wajib diubah
kalau GPU-mu bukan A100 80GB, cara pakai beberapa mesin sekaligus untuk
mempercepat): [`src/fp16/README.md`](src/fp16/README.md) (quickstart
operasional) dan [`docs/README_FP16.md`](docs/README_FP16.md) (protokol
lengkap + semua jebakan yang sudah ditemukan).

### 2. Varian INT8 — bootstrap sebagian manual

INT8 belum punya bootstrap satu-perintah seperti FP16 (kenapa: lihat komentar
di kepala `src/int8/setup.sh`) — 4 langkah persiapan instance dulu (sewa,
kirim token, `bootstrap_vast.sh 10 30`, lalu tarik ONNX INT8 + 289 konfigurasi
injeksi dari repo dataset terpisah) yang didokumentasikan penuh di
[`docs/README_INT8.md` bagian 8 "Menambah instance baru"](docs/README_INT8.md).
Setelah instance siap (ONNX INT8 + `configs/my_model.json` ada):

```bash
cd src/int8
cp config/int8.env.example config/int8.env   # sesuaikan kalau GPU-mu bukan H200
./setup.sh                           # dua patch wajib + lengkapi injection_llm/*.json + pre-flight

./run.sh --preset smoke              # WAJIB dulu, sama seperti FP16
./run.sh --preset full               # run penuh: 288 konfigurasi, background, ~12-18 jam di H200
./run.sh --status
```

Detail lengkap: [`src/int8/README.md`](src/int8/README.md) dan
[`docs/README_INT8.md`](docs/README_INT8.md) (termasuk kenapa angka arena/
`PER_PROSES`/`BITS` **tidak boleh disalin mentah** dari FP16).

### 3. Setelah run selesai — analisis (tidak butuh GPU)

Hasil mentah (`golden_teks`, `faulty_teks`, logit) ada di CSV output
`run.sh`. Dua skrip ini jalan di laptop/CPU biasa:

```bash
# verifikasi cakupan (semua baris yang diharapkan sudah ada, tidak ada yang hilang diam-diam)
python analisis/cek_cakupan.py hasil.csv --bit 16 --runs 2   # --bit 8 untuk INT8

# hitung kolom PENILAIAN (skor akurasi golden vs faulty) dari CSV mentah
python analisis/hitung_penilaian.py hasil.csv hasil_dinilai.csv
```

Dipisah sengaja dari tahap GPU: kalau metode penilaian berubah, cukup
jalankan `hitung_penilaian.py` lagi tanpa menyewa GPU sekali lagi.

### 4. Mau paham protokol/konteks eksperimennya lebih dalam dulu?

1. [`docs/STATUS_BERJALAN.md`](docs/STATUS_BERJALAN.md) — status eksperimen
   terakhir (FP16 selesai; cek dokumen ini untuk status INT8 terbaru, karena
   file ini terakhir diperbarui 15 Sep sedangkan
   `results/int8/final/RINGKASAN.txt` bisa jadi lebih baru).
2. [`docs/README_FP16.md`](docs/README_FP16.md) — protokol lengkap, berlaku
   untuk kedua varian.
3. [`docs/README_INT8.md`](docs/README_INT8.md) — bagian yang berbeda untuk
   varian INT8 (sumber ONNX, konfigurasi injeksi, batas memori).

## Dukungan model lain (GQA, model lokal)

### Alur data: bobot mentah → ONNX → hasil eksperimen

Penting dipahami dulu supaya tidak salah menaruh berkas — ada **dua tahap
terpisah**, dan yang dikonsumsi tiap tahap berbeda:

```
bobot mentah (config.json + *.safetensors, format HF)
        │
        │  export_llama.py (FP16) / export_llama_int8.py (INT8, + SmoothQuant)
        │  -- MODEL_ID dibaca DI SINI, lewat transformers.from_pretrained()
        ▼
berkas .onnx (di $CODE/onnx/onnx_fp16/ atau onnx_int8/)
        │
        │  mass_loglik_inject.py (dijalankan lewat run.sh)
        │  -- HANYA baca .onnx lewat onnxruntime, TIDAK PERNAH sentuh
        │     bobot mentah / MODEL_ID lagi di tahap ini
        ▼
hasil CSV (golden_teks, faulty_teks, logit, dst.)
```

Jadi: **`MODEL_ID` selalu berarti bobot mentah format HuggingFace** (`config.json`
+ minimal satu `*.safetensors` langsung di direktori itu — BUKAN berkas
`.onnx`). Model itu baru dikonversi ke ONNX satu kali di tahap export;
`run.sh`/`mass_loglik_inject.py` sama sekali tidak butuh bobot mentah atau
`MODEL_ID` lagi setelah itu, cuma butuh `.onnx` yang sudah jadi.
**Kalau kamu sudah punya `.onnx` hasil export sebelumnya** (dari instance/
server lain), kamu bisa lewati seluruh tahap unduh+export: taruh langsung
35 berkas itu di `$ROOT/running_experiment_7b/onnx/onnx_fp16/` (atau
`onnx_int8/`), lalu langsung `./setup.sh` (yang akan melengkapi
`injection_llm/*.json` dari ONNX itu) → `./run.sh --preset smoke`.

### Model lokal — cara pakai

`MODEL_ID` di `config/fp16.env`/`config/int8.env` boleh diisi **path
direktori lokal**, bukan cuma ID repo HuggingFace:

```bash
# config/fp16.env atau config/int8.env
MODEL_ID=/workspace/shared_models/llama2-7b-hf
```

Syarat direktori itu (dicek oleh `pipeline/_download_7b.py`): ada
`config.json` **dan** minimal satu `*.safetensors` **langsung** di
dalamnya (bukan di subfolder) — struktur yang sama seperti snapshot HF Hub
yang sudah diunduh. Kalau path ini terdeteksi valid, unduhan HF dilewati
sama sekali (tidak butuh `HF_TOKEN`), dan `export_llama.py`/
`export_llama_int8.py` otomatis ikut memakainya juga — keduanya memanggil
`transformers.from_pretrained(MODEL_ID)`, yang menerima path lokal secara
native.

### ⚠️ Kalau modelnya BUKAN Llama-2-7B: jangan pakai `bootstrap_vast.sh` untuk tahap ONNX

Ini gotcha nyata yang ditemukan saat menulis panduan ini (22 Sep 2026),
**belum pernah terdokumentasi sebelumnya** — `runner/bootstrap_vast.sh`
tahap 40/50 (`40 · ONNX FP16` / `50 · ONNX INT8`) **tidak memeriksa
`MODEL_ID` sama sekali**:

1. Tahap 40a **selalu** mencoba menarik ONNX siap-pakai dari `$HF_REPO`
   (`mfikryrz/llama2-7b-fidelity-onnx-fp16` secara default — itu ONNX
   Llama-2-7B milik pembuat repo ini). Kalau tarikan itu berhasil dapat 35
   berkas (biasanya berhasil, karena repo itu memang lengkap), skrip
   **berhenti di situ dan menganggap sudah selesai** — walaupun
   `MODEL_ID`-mu model yang sama sekali berbeda. Tidak ada perbandingan
   MODEL_ID vs isi repo; cuma dihitung jumlah berkasnya.
2. Bahkan kalau tahap 40a gagal (mis. `HF_REPO` sengaja diarahkan ke repo
   kosong) dan jatuh ke `40b` (export lokal via `01_phase1_export.py`),
   skrip itu **menimpa `export_llama.py` dengan template bawaannya sendiri**
   sebelum menjalankannya — bukan versi GQA/model-lokal yang sudah dipatch
   di `repo_terpatch/export_llama.py`. Sinkronisasi patch lewat `setup.sh`
   baru terjadi SETELAH bootstrap, jadi terlambat untuk mempengaruhi export
   yang sudah kadung jalan.

**Cara aman untuk model lain (termasuk model GQA/lokal):** siapkan
environment lewat bootstrap tanpa tahap ONNX-nya, lalu jalankan exporter
langsung (yang sudah dipatch) secara manual:

```bash
cd src/fp16   # atau src/int8
export HF_TOKEN=hf_xxx        # lewati kalau MODEL_ID path lokal
bash runner/bootstrap_vast.sh 10 30   # preflight + env + kode SAJA -- BUKAN 40/50

cp config/fp16.env.example config/fp16.env
# isi MODEL_ID=... (path lokal atau ID repo HF model lain) lalu:
./setup.sh   # sinkronkan export_llama.py/modeling_llama.py YANG SUDAH DIPATCH ke $REPO dulu

ROOT=/workspace/fidelity
CODE=$ROOT/running_experiment_7b
REPO=$CODE/repo/FIdelity-ONNX-master

# FP16:
"$ROOT/expenv/bin/python" "$REPO/export_llama.py" \
  --repo "$REPO" --model_id "$MODEL_ID" \
  --out "$CODE/onnx/_onnx_raw_fp32" --fp16 "$CODE/onnx/onnx_fp16"

# INT8 (butuh act_scales/llama-2-7b.pt di $REPO -- lihat docs/README_INT8.md kalau model beda):
"$ROOT/expenv/bin/python" "$REPO/export_llama_int8.py" \
  --repo "$REPO" --model_id "$MODEL_ID" --out "$CODE/onnx/onnx_int8"

./setup.sh   # jalan lagi -- kali ini melengkapi injection_llm/*.json dari ONNX yang baru
./run.sh --preset smoke
```

(Untuk **Llama-2-7B asli dengan `MODEL_ID` default**, gotcha ini tidak
berdampak — tahap 40a memang menarik ONNX yang benar. Ini cuma relevan
begitu `MODEL_ID` diganti.)

### GQA (Grouped-Query Attention)

Selain Llama-2-7B (subjek eksperimen aslinya), exporter juga bisa
meng-export model GQA (dipakai Llama-3, Mistral, Gemma-2, dan model lain
bertag arsitektur `"llama"` di `config.json`). Diverifikasi di GPU nyata:
selisih ONNX vs PyTorch level floating-point noise (~2×10⁻⁷). **Qwen2/Gemma
asli belum ikut terbantu** — keduanya punya file exporter sendiri di
`transformers` yang belum disentuh. Detail lengkap di
[`src/int8/README.md`](src/int8/README.md) bagian "Model lain (GQA, model
lokal)".

## Isi eksperimen historis (`experiments/`)

| Folder | Keterangan |
|---|---|
| `int8_instance_49721366_20260903/` | percobaan awal INT8 |
| `int8_instance_49721366_smoke_20260904/` | uji asap INT8 |
| `int8_instance_49970225_true_smoke_20260905/` | uji INT8 sungguhan (gagal, lihat README_INT8.md §1) |
| `int8_instance_49970225_true_smoke_v2_20260906/` | lanjutan uji INT8 sungguhan |
| `int8_instance_49970225_fake_smoke_v2_20260906/` | uji fake-INT8 v2 |
| `int8_instance_50018492_fake_smoke_v3_20260906/` | uji fake-INT8 v3 |
| `ops_int8_instance_49721366/`, `ops_int8_instance_49970225_true/` | skrip operasional per instance |
| `portable_int8_instance_49721366/` | bundel portable untuk migrasi instance |
| `fp16_smoke_v4_20260911/` | uji asap FP16 16 layer (3.071/3.072 baris) |
| `fp16_smoke960_20260912/` | uji asap FP16 lain |
| `int8_smoke1728_20260914/` | uji asap INT8 (1.646/1.728 baris) |
| `backup_int8_20260915_malam/`, `backup_int8_migrasi_20260915/` | backup saat migrasi A100→H200 |
| `progres_int8_20260916/` | snapshot progres INT8 57,7% |
| `hf_roundtrip_instance_49970225/`, `hf_fast_source_trial/` | percobaan transfer via HuggingFace |
| `test_implementasi_perbaikan/` | uji 42 baris untuk validasi perbaikan, dikirim ke pembimbing |
| `legacy_massal_pgteks/` | data dari protokol lama (sebelum 10 Sep 2026) |

Catatan jebakan-jebakan yang sudah ditemukan (migrasi instance, perbedaan
FP16/INT8, watchdog VRAM, dll.) ada di catatan kerja internal pembuat repo
ini, tidak ikut ter-push ke GitHub — sebagian besar sudah diringkas ulang
sebagai komentar langsung di kode (`setup.sh`, `runner/*.sh`,
`repo_terpatch/*.py`) dan di `docs/*.md`, jadi tetap bisa ditemukan lewat
kode/dokumen di repo ini saja.
