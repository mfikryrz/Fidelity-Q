# INT8 — Dokumen Acuan Tunggal

**Diperbarui 14 September 2026.** Pelengkap [`README_FP16.md`](README_FP16.md);
baca itu dulu untuk protokol, format 34 kolom, dan cara kerja umum — semuanya sama.
Dokumen ini memuat yang **berbeda untuk INT8**.

Kode yang benar-benar berjalan tersimpan di [`kode_int8/`](kode_int8/) —
termasuk `memory_pool.py` yang sudah dipatch, diff-nya, dan `graph.py`/
`inject_ops.py` yang menentukan mekanisme tiap fault model.

Menggantikan `LAPORAN_FILTER_FAKE_INT8.md`, `LAPORAN_V2.md`, `PORTABLE_README.md`,
dan `VALIDATION_SUMMARY.md` — semuanya sudah dihapus, isinya diserap ke sini.

---

## 1. INT8 di sini bukan INT8 sungguhan

> SmoothQuant fake-INT8 alpha=0.85; Div/Round/Mul; **FP32 storage and execution**

Bobot dibulatkan seperti INT8 tapi **disimpan dan dieksekusi sebagai FP32**.
Modelnya justru **lebih besar** (26 GB vs 13 GB) dan **lebih lambat** dari FP16.

### INT8 sungguhan sudah dicoba dan gagal — dua sebabnya

**(a) Gerbang 18/20 tidak mungkin dicapai model kuantisasi mana pun dari induk ini.**
Graf FP16 SmoothQuant yang **belum dikuantisasi sama sekali** hanya cocok **17/20**
dengan oracle fake-INT8. Kuantisasi hanya bisa menjauhkan prediksi dari FP16, jadi
tidak ada W8A8 dari induk sama yang bisa 18/20 kecuali kebetulan.

Ketiga soal yang berbeda semuanya nyaris seri — selisih logit teratas di oracle
0,305 / 0,391 / 0,304. Set sanity-nya memang didominasi soal nyaris seri, sehingga
kesepakatan prediksi di atasnya rapuh terhadap perubahan presisi apa pun.
**Gerbangnya yang cacat, bukan modelnya.**

**(b) Kualitas W8A8 memang rusak, penyebabnya spesifik.** Kandidat terbaik hanya
sepakat 9/20 dengan induk FP16-nya sendiri. Sumbernya input `down_proj` — hasil
perkalian SwiGLU yang membawa *massive activation* khas Llama-2 dan **tidak
disentuh** `smooth_lm`. SmoothQuant hanya memindahkan outlier pada jalur
`input_layernorm → q/k/v` dan `post_attention_layernorm → gate/up`.

Kalau pembimbing mengharapkan INT8 sejati, ini perlu dibicarakan dulu — bukan
masalah yang selesai dengan menyewa mesin lebih besar.

---

## 2. Angka terukur

| | FP16 | **INT8** |
|---|---|---|
| Ukuran ONNX | 13 GB | **26 GB** |
| Muat model | ~26 detik | **~9,6 menit** |
| `ORT_GPU_MEM_LIMIT_GB` | 60 | **35** |
| `PER_PROSES` | 1 | **3** |
| Lebar bit | 16 | **8** (`--bits 0-7`) |

### Laju sangat bergantung amortisasi graf — bukan satu angka

Graf faulty dibangun **sekali per (decoder × operasi × fault model)**, lalu dipakai
ulang untuk seluruh bit/run kombinasi itu. Jadi detik-per-baris turun drastis
seiring banyaknya baris yang menanggung satu graf:

| Setelan | Baris per graf | detik/baris |
|---|---|---|
| `--runs 1`, 24 layer GPU, TMPDIR di disk | 1 | 21,09 |
| `--runs 1`, 32 layer GPU, TMPDIR di RAM | 1 | 16,60 |
| `--bits 0-7 --runs 2` (uji asap kita) | 16 | **~9,1** |
| `--runs 25`, soal dibagi 6 fault model | 25 | 4,16 |
| `--runs 25`, soal acak tiap baris | 25 | 6,76 |

> **Jangan memakai angka smoke untuk memperkirakan run penuh.** Laporan lama
> sempat mencatat "20,4 detik/baris" — benar untuk cara smoke dijalankan,
> menyesatkan untuk skala penuh.

Perkiraan run masal 288 konfigurasi, 27.648 baris, `--bits 0-7 --runs 2`:
**~70 jam ≈ 2,9 hari ≈ $75** di satu mesin. Sharding 2 mesin ~35 jam, 4 mesin
~18 jam — biaya total sama.

---

## 3. Setup yang berbeda dari FP16

### ONNX dari repo DATASET, bukan repo model

FP16 menarik dari repo **model** `llama2-7b-fidelity-onnx-fp16`. INT8 ada di repo
**dataset** `llama2-7b-fidelity-hasil`, di `int8/portable/instance_49721366_v1/onnx/`
— 36 berkas `.zst`, 12 GB terkompresi, mengembang jadi **26 GB** (35 `.onnx` +
tokenizer).

```bash
# setelah bootstrap_vast.sh tahap 10-30 + dua tambalan (README_FP16 bagian 4).
# Tahap 40 (ONNX FP16) TIDAK diperlukan.
hf download mfikryrz/llama2-7b-fidelity-hasil --repo-type dataset \
  --include 'int8/portable/instance_49721366_v1/onnx/**' --local-dir _dl_int8
for z in _dl_int8/.../onnx/*.onnx.zst; do
  zstd -q -d -f "$z" -o "$INT8D/$(basename ${z%.zst})"
done   # hasilnya harus 35 berkas .onnx
```

Jangan membangun ulang lewat tahap 50 — itu butuh unduh bobot HF ~13 GB lalu
export SmoothQuant, jauh lebih lama.

### Konfigurasi injeksi BERBEDA, bukan cuma path

Jebakan paling berbahaya: gagalnya **tidak terlihat sebagai error**.

| | FP16 | INT8 |
|---|---|---|
| `input_tensor` | `/self_attn/Round_output_0` | `/mlp/down_proj/Round_output_0` |
| `weight_tensor` | `onnx::MatMul_357` (initializer mentah) | `/mlp/down_proj/Round_1_output_0` |

INT8 menyuntik ke tensor **hasil fake-quant**. Kalau path config FP16 sekadar
diganti `onnx_fp16`→`onnx_int8`, injeksinya mendarat di tempat salah dan datanya
terlihat wajar tapi keliru.

Ambil 289 config dari `code_data_results.tar.zst` di bundle portable, tulis ulang
`model_name`-nya dari `/workspace/fidelity_int8/running_experiment_7b_int8/onnx/onnx_int8`
ke tata letak yang dipakai.

### Patch wajib: `RANDOM_BITFLIP`

Tanpa ini **1/6 dataset hilang**. Jalankan
[`kode_fp16/runner/patch_int8_memory_pool.sh`](kode_fp16/runner/patch_int8_memory_pool.sh).

`graph.py` memilih implementasi dari tipe tensor: FP16 → `custom.bitflip:BitFlip`
(dari `onnx_bitflip.so`), FP32 → `ai.onnx.contrib:DirectBitToggleFp32` (PyOp dari
onnxruntime-extensions). INT8 menyimpan FP32 jadi selalu jalur kedua, tapi
`llama/memory_pool.py` hanya mendaftarkan `onnx_bitflip.so`. `cnn_inference.py`
mendaftarkan keduanya; jalur LLM terlewat.

```
Fatal error: ai.onnx.contrib:DirectBitToggleFp32(-1) is not a registered function/op
```

Perbaikannya butuh **dua** hal: daftarkan pustaka extensions (patch), **dan**
import `inject_ops` supaya dekorator `@onnx_op` berjalan — yang kedua sudah terjadi
lewat `graph.py`. Mendaftarkan pustaka saja tidak cukup.

---

## 4. Batas memori: 35 GB — dan kenapa bukan 45 atau 60

```
model 26 GB + arena 60 GB = 86 GB  >  kartu 80 GB   GAGAL total
model 26 GB + arena 35 GB = 61 GB  <  kartu 80 GB   BERJALAN
```

**Ini yang benar-benar terjadi di lapangan, 14 Sep 2026** — bukan hasil hitungan
di atas kertas. Penjaga otomatis menurunkannya empat kali, dan tiap penurunan
memberi data:

| Arena | Laju error alokasi | Catatan |
|---|---|---|
| 60 GB | **14%** (56 error / 392 baris) | diwarisi dari setelan FP16 |
| 45 GB | ~3% | masih gagal setelah ~30 menit |
| 40 GB | ~3% | masih gagal setelah ~80 menit |
| 35 GB | **0%** | berjalan bersih, 6,4 dtk/baris |
| 30 GB | 0% | dipakai sebentar, tapi lebih lambat tanpa manfaat |

**Pelajaran yang lebih penting dari angkanya:** menurunkan arena **tidak
menyembuhkan**, hanya memperlambat laju error. Akar masalahnya fragmentasi arena
BFC yang tidak pernah menyusut. Errornya terpusat di `mlp_up_proj`, `k_proj`, dan
`mlp_gate_proj` — operasi bertensor terbesar, yang pertama gagal saat arena
terfragmentasi.

Dan satu hal yang sempat saya salah pahami: **`gpu_mem_limit` berlaku PER SESSION,
bukan global.** Pipeline membuat banyak session (32 decoder + embed/head/norm +
graf faulty), masing-masing dengan arena sendiri. Jadi total pemakaian VRAM bisa
jauh melampaui angka yang diset — teramati 78–81 GB dengan batas 35 GB. Itu normal
dan tidak berarti gagal; yang menentukan adalah ada/tidaknya error alokasi.

> Laju error 2–3% **bukan alasan menurunkan setelan**. Baris yang gagal ditambal
> tahap 2 dengan biaya ~2,5 menit, sedangkan restart membayar 10 menit memuat
> model. Ambang penjaga **50**, bukan 10 — ambang ketat warisan FP16 membuat
> penjaga merestart tiga kali dan membuang ~30 menit untuk menghindari ~36 baris
> yang akan ditambal sendiri.

### `PER_PROSES` = 3, bukan 1

FP16 memakai 1 konfigurasi per proses agar arena sering direset — murah karena
muat model 26 detik. INT8 butuh **9,6 menit**, jadi 288 konfigurasi berarti
**46 jam hanya memuat model**. Pakai 3–6; tahap 2 di `jalankan_fp16.sh` menambal
lubang yang muncul.

---

## 5. Menjalankan

```bash
PRECISION=int8 LANGKAH=1 N_CFG=288 BITS=0-7 RUNS=2 PER_PROSES=3 \
  ORT_GPU_MEM_LIMIT_GB=35 FIDELITY_POOL_GB=35 OUT=.../massal_int8.csv \
  setsid nohup bash pengawas_fp16.sh > logs/pengawas.log 2>&1 < /dev/null &
```

`PRECISION=int8` mengalihkan `_config.py` ke `onnx_int8` **dan** membuat lebar bit
otomatis 8 — `--bits 0-15` akan ditolak.

Penjaga cron wajib diberi `TARGET` dan `OUT`; tanpa itu ia memakai angka FP16
(55.296) dan mengira run belum selesai:

```
*/10 * * * * TARGET=27648 OUT=.../massal_int8.csv PRECISION=int8 \
  ORT_GPU_MEM_LIMIT_GB=35 FIDELITY_POOL_GB=35 PER_PROSES=3 \
  LANGKAH=1 N_CFG=288 BITS=0-7 RUNS=2 bash /workspace/fidelity/penjaga_fp16.sh
```

Verifikasi cakupan: `python cek_cakupan.py hasil.csv --bit 8 --runs 2 --penuh`

---

## 6. Filter sewa mesin — berbeda dari FP16

**Panduan FP16 menyimpulkan "disk yang menentukan biaya total". Untuk fake-INT8
itu TIDAK berlaku.** Model FP32 26 GB sekali dimuat menetap di page cache, dan
graf injeksi hanya ditulis sekali per belasan baris. Terukur: memindah `TMPDIR`
ke RAM hanya menghemat **21%** pada `--runs 1`, dan jauh lebih kecil pada
amortisasi tinggi.

| Filter | Setel ke | Alasan |
|---|---|---|
| GPU Count | **1** | GPU kedua tidak menolong satu job |
| **Per GPU RAM** | **80 GB** | model 26 GB + arena 35 GB = 61 GB, tapi arena berlaku PER SESSION sehingga pemakaian nyata teramati 78-81 GB. Kartu 48 GB tidak cukup |
| **Container Size** | **≥ 100 GB** | ONNX 26 GB + cache unduh 12 GB + venv |
| **Min Cuda Version** | **12.8** | |
| **Disk Bandwidth** | **≥ 1 GB/s cukup** | bukan penentu untuk INT8 — jangan bayar mahal di sini |
| CPU Cores | ≥ 16 | rebuild graf **satu-utas**; di atas ~4 core tidak terpakai |
| CPU RAM | ≥ 64 GB | page cache model 26 GB |
| Host Reliability | ≥ 98% | run panjang |
| Max Duration | ≥ 3 hari | |
| Jenis | On-Demand | jangan interruptible |

Dua yang berbeda dari FP16: **Container Size 100 GB** (bukan 60) dan **Min CUDA
12.8** (bukan 12.4). Dan **Disk Bandwidth boleh lebih longgar** — kebalikan dari
FP16.

---

## 7. Menjalankan di BANYAK MESIN (sharding)

Susunan yang **sedang berjalan** per 14 Sep 2026, 14:36 UTC:

| Mesin | Alamat | Config | Decoder | Target | Berkas |
|---|---|---|---|---|---|
| **A** | `104.37.174.34:25503` | 0–143 | 0–15 | 13.824 | `massal_int8_A_cfg0-143.csv` |
| **B** | `82.66.51.122:31866` | 144–287 | 16–31 | 13.824 | `massal_int8_B_cfg144-287.csv` |

Batasnya jatuh tepat di pergantian decoder: config 143 = `d15/self_attn_v_proj`,
config 144 = `d16/mlp_down_proj`. Itu disengaja — "A punya decoder 0–15, B punya
16–31" tidak mungkin salah dibaca saat verifikasi.

### Tiga syarat mutlak bebas duplikat

**1. Rentang `MULAI`/`AKHIR` disjoint.** Untuk N mesin, tiap mesin dapat 288/N
konfigurasi berurutan.

**2. Berkas CSV terpisah per mesin.** Digabung di laptop setelah selesai.

**3. Urutan konfigurasi identik di semua mesin.** Ini yang paling mudah terlewat.
`daftar_konfigurasi()` membaca `sorted(glob(injection_llm/*.json))` lalu mengurutkan
`(decoder_idx, operasi)`. Kalau satu mesin kekurangan **satu** berkas JSON saja,
seluruh indeksnya bergeser dan rentangnya menimpa mesin lain. Verifikasi di tiap
mesin sebelum meluncurkan — keluarannya harus sama persis: total 288, indeks 0 =
`d0/mlp_down_proj_MatMul`, 143 = `d15/self_attn_v_proj_MatMul`, 144 =
`d16/mlp_down_proj_MatMul`, 287 = `d31/self_attn_v_proj_MatMul`.

### Dua jebakan duplikat yang benar-benar terjadi

**`penjaga_vram.sh` tidak meneruskan `MULAI`/`AKHIR`.** Sudah diperbaiki, tapi
penting diingat kenapa berbahaya: saat penjaga merestart pengawas, versi lama
memakai default `0..N_CFG`. Mesin yang seharusnya hanya mengerjakan konfigurasi
144–287 akan **mulai dari 0 dan menduplikasi seluruh pekerjaan mesin lain** — dan
itu baru ketahuan saat menggabung CSV, ketika sulit membedakan mana yang sah.

**Data lama bisa jatuh ke rentang mesin yang berbeda.** Saat memperluas cakupan
dari uji asap 1.728 ke masal 27.648, konfigurasi 15–17 sudah dikerjakan mesin B
tapi masuk rentang baru mesin A. Tanpa penanganan, A mengulang 288 baris.
Solusinya menyalin baris itu ke berkas A lebih dulu supaya `--resume` melewatinya,
lalu dedup berdasarkan kunci
`(decoder_idx, operasi, Fault_Model, Bit_Position, run_id)`.

### Verifikasi setelah menggabung

Bandingkan himpunan kunci antar mesin: irisannya harus kosong, dan jumlah baris
tiap berkas harus sama dengan jumlah kunci uniknya. Pada penggabungan 14 Sep:
**0 duplikat antar mesin, 0 di dalam masing-masing, 0 konfigurasi beririsan,
18/18 konfigurasi tercakup.**

---

## 8. Menambah instance baru — urutan yang terbukti

Setup memakan **~40 menit**, didominasi unduh ONNX. Dilakukan tiga kali pada
13–14 Sep dengan hasil konsisten.

### Langkah 1 — sewa (filter di bagian 6), lalu verifikasi

Pastikan GPU A100-80GB, disk ≥100 GB, dan `/workspace/fidelity` belum ada.
Catat juga jatah core dari cgroup — `nproc` melaporkan core HOST, bukan jatah
container (pernah terbaca 152 padahal jatahnya 18).

### Langkah 2 — token lewat stdin

Ambil dari `rahasia/hf_tokens.env`, kirim dengan `printf ... | ssh ... 'umask 077;
cat > /root/.hf_token'`. **Jangan** taruh token di baris perintah — terbaca siapa
pun lewat `ps` di mesin sewaan.

### Langkah 3 — bootstrap tahap 10–30 + dua tambalan (~8 menit)

Tarik `bootstrap_vast.sh` dari repo model, jalankan `bash bootstrap_vast.sh 10 30`.

**Tahap 40 (ONNX FP16, 13 GB) TIDAK diperlukan untuk INT8** — melewatinya
menghemat waktu dan 13 GB disk.

Lalu dua tambalan yang bootstrap-nya sendiri tidak lakukan:
(a) pasang `req_rtenv.txt` — tahap 20 memasang paket sebelum tahap 30 mengunduh
daftarnya, jadi tanpa ini ia diam-diam memakai daftar minimum;
(b) tempatkan `mmlu_pool.csv` dan salin `*.py` ke `running_experiment_7b/`.

### Langkah 4 — ONNX INT8 + konfigurasi injeksi INT8 (~25 menit)

Tarik dari repo **dataset** (bukan model): `--include
'int8/portable/instance_49721366_v1/onnx/**'` dan
`--include '.../artifacts/code_data_results.tar.zst'`. Dekompresi `.zst` menjadi
35 berkas `.onnx` (26 GB), salin `tokenizer.model`.

**JANGAN pakai `injection_llm` dari paket FP16** — nama tensornya berbeda (bagian
3). Ambil 289 JSON dari `code_data_results.tar.zst`, lalu tulis ulang
`model_name`-nya dari `/workspace/fidelity_int8/running_experiment_7b_int8/onnx/onnx_int8`
ke tata letak yang dipakai.

Setelah selesai hapus `_dl_int8` — membebaskan 12 GB.

### Langkah 5 — kirim kode dan terapkan PATCH WAJIB

Dari `kode_int8/`: `mass_loglik_inject.py` dan `cek_cakupan.py` ke
`running_experiment_7b/`, seluruh `runner/*.sh` ke `/workspace/fidelity/`.
Lalu jalankan `patch_int8_memory_pool.sh`. Tanpa patch ini seluruh baris
`RANDOM_BITFLIP` gagal — 1/6 dataset hilang tanpa peringatan.

### Langkah 6 — verifikasi SEBELUM meluncurkan

| Yang diperiksa | Nilai benar |
|---|---|
| `injection_llm/*.json` | 289 |
| `mmlu_pool.csv` | 14.042 record |
| ONNX INT8 | 35 berkas, 26 GB |
| rtenv | 71 paket; numpy 1.26.4, protobuf 3.20.3, onnxruntime-gpu 1.20.2 |
| Patch | `grep -c "_ext_lib" .../llama/memory_pool.py` → **2** |
| Urutan konfigurasi | sama dengan mesin lain (bagian 7) |

### Langkah 7 — luncurkan dengan rentang yang belum dipakai

Jalankan `pengawas_fp16.sh` lewat `setsid nohup ... < /dev/null &` supaya lepas
dari sshd, dengan `MULAI`/`AKHIR` sesuai jatah mesin itu dan `OUT` berkas sendiri.
Pasang `penjaga_vram.sh` di cron tiap 10 menit dengan `TARGET`, `OUT`, `MULAI`,
`AKHIR`, dan `AMBANG_ERR=50`.

`TARGET` = `(AKHIR − MULAI) × 6 × 8 × 2`.

### Berapa mesin sebaiknya?

Total GPU-hour **tidak berubah** — yang berubah hanya waktu tunggu. Berdasarkan
**11,36 detik/baris efektif per mesin** (terukur di lapangan, sudah termasuk
pemuatan model):

| Mesin | 27.648 baris | Biaya total |
|---|---|---|
| 1 | ~87 jam (3,6 hari) | ~$91 |
| **2** | **~44 jam** | ~$92 |
| 4 | ~22 jam | ~$92 |
| 8 | ~11 jam | ~$92 |

Batasnya bukan teknis melainkan saldo: tiap mesin menambah ~$1,05/jam ke laju
bakar. Delapan mesin membakar $8,4/jam meski totalnya sama — kalau saldo habis di
tengah, semuanya berhenti sekaligus.

Setup 40 menit per mesin juga masuk hitungan: menambah mesin untuk sisa pekerjaan
di bawah ~2 jam tidak sepadan.
