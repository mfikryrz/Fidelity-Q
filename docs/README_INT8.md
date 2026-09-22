# INT8 — Dokumen Acuan Tunggal

**Diperbarui 14 September 2026.** Pelengkap [`README_FP16.md`](README_FP16.md);
baca itu dulu untuk protokol, format 34 kolom, dan cara kerja umum — semuanya sama.
Dokumen ini memuat yang **berbeda untuk INT8**.

Kode yang benar-benar berjalan tersimpan di [`src/int8/`](../src/int8/) —
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
[`src/fp16/runner/patch_int8_memory_pool.sh`](../src/fp16/runner/patch_int8_memory_pool.sh).

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

### Angka H200 — terukur 15 Sep 2026

Seluruh tabel di atas diambil dari A100 80 GB. Di H200 140 GB angkanya berbeda
cukup jauh:

| | A100 80 GB | **H200 140 GB** |
|---|---|---|
| Muat model | 9,6 menit | **3,8 menit** |
| Komputasi murni | 5,9–7,5 dtk/baris | **3,04 dtk/baris** (stabil di semua jendela) |
| Efektif `PP=2` | — | **4,2 dtk/baris** |
| VRAM puncak | 78–81 / 80 GB (jenuh) | 124–142 / 143,8 GB (**juga jenuh**) |
| Laju error alokasi | ~10% | **~1,6%** |

Dua hal yang perlu dicatat:

1. **Kartu lebih besar menunda kejenuhan, tidak menghapusnya.** VRAM teramati
   memuai 57 → 124 → 142,6 GB sepanjang satu potongan, lalu direset di
   pergantian potongan. Errornya tetap ada, hanya lebih jarang.
2. **Komputasi murni adalah indikator kesehatan terbaik.** Ia bertahan di
   3,03–3,04 dtk/baris di jendela 1.000, 2 jam, dan seluruh run. Kalau angka ini
   naik, ada yang salah; kalau hanya laju *efektif* yang naik, itu cuma
   amortisasi muat model.

### `AMBANG_ERR=50` terlalu ketat untuk H200

Ambang itu dipilih saat muat model 9,6 menit, ketika restart memang mahal. Di
H200 muat model 3,8 menit, jadi perhitungannya berubah — dan ambang lama membuat
penjaga meratchet `PER_PROSES` turun untuk laju error yang dokumen ini sendiri
sebut wajar.

Terjadi dua kali pada 15 Sep:

```
[08:20] 67 error baru  -> PER_PROSES 3 turun ke 2   (GRATIS: laju tetap 4,7 dtk/baris)
[14:40] 60 error baru  -> PER_PROSES 2 turun ke 1   (MAHAL: ~3,8 jam lebih lambat)
```

Penurunan pertama menguntungkan: error anjlok dan laju tidak berubah, karena
lebih sedikit error berarti lebih banyak baris per muat model. Penurunan kedua
merugikan: menyapu satu konfigurasi jadi **5 menit** (dari 7–8 detik), sebab di
`PP=1` tiap konfigurasi berlubang membayar muat model sendirian — sementara laju
error kumulatif hanya 1,6%.

**Pakai `AMBANG_ERR=150` di H200**, dan biarkan `PER_PROSES=2`. Yang dikorbankan
hanya penyetelan otomatis yang terlalu sensitif; kemampuan penjaga menghidupkan
ulang proses mati tidak tersentuh.

### WAJIB: `ARENA_MIN=35` di crontab penjaga

Tabel di atas menyimpulkan 35 GB, tapi **tidak ada yang menegakkannya**.
`penjaga_vram.sh` punya `ARENA_MIN` bawaan **30**, jadi ia akan meratchet
45→40→35→30 dan berhenti di 30 — satu langkah di bawah titik yang terukur
bersih, tepat di baris yang tabel ini sebut "lebih lambat tanpa manfaat".

Itu bukan hipotesis. Pada 14–15 Sep 2026 kedua mesin INT8 masal berakhir di
`ARENA=30 PP=1` dan **70% waktu dindingnya habis memuat model** (terukur:
52 dan 33 jeda ~9,9 menit; laju komputasi sendiri tetap sehat di 5,9 dan
7,5 dtk/baris). Kerugiannya puluhan jam per mesin.

Jadi crontab penjaga harus memuat keduanya:

```
PER_PROSES=3 ORT_GPU_MEM_LIMIT_GB=35 FIDELITY_POOL_GB=35 ARENA_MIN=35
```

Dan kalau Anda mengubah setelan pada run yang sedang berjalan, **tulis ulang
`logs/.penjaga_vram.state`** (`ARENA=35 PP=3 N_UBAH=0 WM=<ukuran fp16.log>`).
Penjaga membaca setelan dari berkas itu, bukan dari crontab, jadi tanpa langkah
ini perubahan Anda dikembalikan pada restart berikutnya.

> Menyapu ulang konfigurasi yang sudah selesai **murah**: potongan yang lengkap
> keluar dalam ~0,7 menit tanpa memuat model. Jadi jangan ragu merestart untuk
> memperbaiki setelan — ongkosnya satu muat model, bukan mengulang dari nol.

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

### Langkah 5 — kirim kode dan terapkan DUA PATCH WAJIB

Dari `src/int8/`: `mass_loglik_inject.py` dan `cek_cakupan.py` ke
`running_experiment_7b/`, seluruh `runner/*.sh` ke `/workspace/fidelity/`,
dan `repo_terpatch/` ikut terkirim (dipakai patch di bawah).

1. `patch_int8_memory_pool.sh` — tanpa ini seluruh baris `RANDOM_BITFLIP`
   gagal, 1/6 dataset hilang tanpa peringatan.
2. `terapkan_config_dinamis.sh` (baru, 21 Sep 2026) — tanpa ini instance
   tetap memakai `_config.py`/`fidelity_utils.py` lama dari paket kode, yang
   masih membuang diam-diam operasi apa pun yang namanya bukan salah satu
   dari 9 nama Llama-2-7B. Untuk Llama-2-7B sendiri tidak berdampak (datanya
   toh cocok), tapi begitu `configs/my_model.json` dapat field
   `operation_suffixes` (lewat `pipeline/discover_operasi.py`, untuk model
   lain), instance HARUS sudah memakai versi ini supaya field itu dibaca.
   Aman diulang, dan bikin cadangan otomatis sebelum menimpa.

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

---

## 11. Otomatisasi di instance — enam skrip

Semuanya ada di `src/int8/runner/`. Yang berjalan **di instance** sengaja
diletakkan di sana, bukan di laptop: kalau koneksi putus atau sesi berakhir,
pekerjaannya tetap jalan.

| Skrip | Di mana | Cron | Tugas |
|---|---|---|---|
| `snapshot_int8.sh` | instance | tiap 15 mnt | segarkan snapshot progres di `/tmp` |
| `pantau_int8.sh` | instance | tiap 10 mnt | catat satu baris kesehatan ke `logs/pantau.log` |
| `siapkan_unduhan_int8.sh` | instance | tiap 10 mnt | deteksi selesai → lucuti penjaga → hentikan → arsipkan |
| `lanjutkan_int8.sh` | instance | manual | jalankan sekali setelah instance START |
| `src/int8/runner/ambil_int8.sh` | laptop | manual | tarik snapshot progres + laporan cakupan |
| `src/int8/runner/ambil_final_int8.sh` | laptop | manual | tarik hasil AKHIR + verifikasi cakupan |

### Urutan yang WAJIB saat run selesai

```
1. lucuti baris penjaga_vram dari crontab
2. baru hentikan pengawas + jalankan_fp16 + worker
```

Terbalik = cron menghidupkan pengawas lagi dalam 10 menit. Ini bukan teori:
pada FP16 (15 Sep) pengawas dihentikan lebih dulu dan `penjaga_fp16.sh`
membangunkannya 7 menit kemudian, karena logikanya cuma `baris < TARGET →
bangunkan`. Run ini **selalu** berakhir di bawah target akibat kegagalan
permanen, jadi kondisi itu tidak pernah berhenti benar. Biayanya ~34 jam mesin
menyapu ulang untuk nol baris, dan penunggu unduhan tidak pernah memicu karena
pengawas terus hidup lagi.

`siapkan_unduhan_int8.sh` sudah menjalankan urutan itu sendiri.

### Cara mendeteksi "selesai" — pakai log pengawas, bukan `pgrep`

```
=== BERHENTI: putaran N tidak menambah baris apa pun (NNNNN) ===
```

Kalimat itu pernyataan pengawas sendiri bahwa satu putaran penuh sudah mencoba
ulang seluruh sisa dan nol berhasil. Itu bukti, bukan dugaan.

**Jangan** menyimpulkan dari "pengawas mati": penjaga dan penunggu sama-sama
jalan tiap 10 menit dan urutannya tidak dijamin, sehingga jendela "mati" bisa
terlewat sepenuhnya.

### Membaca `logs/pantau.log`

```
[waktu] baris=N cfg=N worker=1 pengawas=1 err=N vram=N util=N cpu=N umur=N
```

Baris bertanda `<<<` menandai tiga mode kegagalan yang benar-benar pernah
terjadi:

- **`MACET`** — baris tidak bertambah 4 siklus padahal worker hidup.
  `penjaga_vram.sh` **tidak** mendeteksi ini; ia hanya memeriksa proses hidup
  dan error alokasi.
- **`CPU?`** — VRAM < 5 GB, CPU > 50%, **dan umur worker > 8 menit**. Syarat
  umur itu penting: saat menyapu konfigurasi yang sudah lengkap, tiap worker
  hidup beberapa detik untuk mem-parse CSV resume (CPU tinggi, GPU nol) lalu
  keluar — tanpa syarat umur, penanda ini menyala terus dan jadi tidak berarti.
  Pada kegagalan `arena=25` (16 Sep) worker berumur 22 menit dengan GPU 21 GB.
- **`WORKER=n`** — n bukan 1. Nol berarti berhenti; lebih dari satu berarti dua
  proses menulis satu CSV.

---

## 12. Menghentikan dan melanjutkan (STOP, bukan destroy)

Instance Vast yang di-**STOP** tetap menyimpan disknya; `--resume` melanjutkan
dari baris terakhir. Yang hilang hanya proses yang sedang berjalan.

**Sebelum STOP** — ambil snapshot lengkap ke laptop:

```bash
ssh -p <port> root@<host> 'bash /workspace/fidelity/snapshot_int8.sh'
bash src/int8/runner/ambil_int8.sh      # verifikasi md5 dua sisi + laporan cakupan
```

**Setelah START** — jalankan sekali:

```bash
ssh -p <port> root@<host> 'bash /workspace/fidelity/lanjutkan_int8.sh'
```

> **Kenapa tidak cukup mengandalkan cron.** Saat container dimulai ulang,
> pengawas pasti mati dan **daemon cron belum tentu hidup**. Kalau cron mati,
> tidak ada yang menghidupkan pengawas — mesin diam sambil tetap ditagih.
> `lanjutkan_int8.sh` memeriksa berkas hasil, mencetak state penjaga,
> menghidupkan cron bila perlu, lalu menghidupkan pengawas. Idempoten: aman
> dijalankan berkali-kali.

Verifikasi sesudahnya:

```bash
ssh -p <port> root@<host> 'bash /workspace/fidelity/status.sh'
ssh -p <port> root@<host> 'pgrep -f mass_loglik | wc -l'    # harus 1
```

Angka **1** itu penting: **0** berarti tidak jalan, **2** berarti dua proses
menulis satu CSV. Hitung dengan heredoc (`ssh host 'bash -s' <<'EOF'`), bukan
sebagai argumen ssh — kalau perintahnya jadi argumen, baris perintah shell jarak
jauh memuat string `mass_loglik` dan `pgrep -f` mencocokkan dirinya sendiri,
sehingga melaporkan 2 atau 3 padahal sebenarnya 1.

Ketiga skrip ada di `src/int8/runner/`: `snapshot_int8.sh` (di instance, cron
tiap 15 menit), `src/int8/runner/ambil_int8.sh` (di laptop), `lanjutkan_int8.sh` (di instance,
setelah START).
