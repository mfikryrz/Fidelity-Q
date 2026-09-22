# Kode INT8 — snapshot 14 September 2026

Diambil langsung dari mesin yang menjalankan eksperimen (`82.66.51.122`), jadi
ini **kode yang benar-benar berjalan**, bukan salinan yang mungkin sudah bergeser.

Panduan lengkap (protokol, biaya, jebakan): [`../../docs/README_INT8.md`](../../docs/README_INT8.md).
Bagian di bawah ini adalah **quickstart operasional** — cara menjalankan,
bukan cerita eksperimennya.

---

## Quickstart

Prasyarat: instance dengan `rtenv` (Python 3.10 + `req_rtenv.txt`) dan ONNX +
`configs/my_model.json` **sudah di-bootstrap** — lihat
[`../../docs/README_INT8.md`](../../docs/README_INT8.md) bagian "Langkah 1-4"
untuk itu (di luar cakupan `setup.sh`, sengaja — lihat alasannya di dalam
`setup.sh`). `injection_llm/*.json` **tidak** perlu disiapkan manual —
`setup.sh` men-generate-nya sendiri lewat `03_phase2_parser.py` kalau belum
lengkap (lihat komentar langkah "5/6" di `setup.sh`).

```bash
# 1. sekali per instance: dua patch wajib + lengkapi injection_llm/*.json + pre-flight
cd src/int8
cp config/int8.env.example config/int8.env   # sesuaikan kalau GPU-mu bukan H200
./setup.sh

# 2. WAJIB sebelum run penuh: uji asap beberapa menit, foreground
./run.sh --preset smoke
#   periksa manual: golden_teks/faulty_teks berisi kalimat (bukan kosong),
#   g_logit_* != f_logit_* pada sebagian baris (bukti injeksi masuk)

# 3. run penuh: 288 konfigurasi, background, survive SSH putus, ~12-18 jam
./run.sh --preset full

# kapan saja:
./run.sh --status
```

`--preset full` menolak jalan kalau pengawas sebelumnya masih hidup (cegah
dua worker menulis CSV yang sama), dan memasang watchdog VRAM ke crontab
otomatis kalau belum ada. Mengulang `./run.sh --preset full` yang sama
setelah run terputus itulah cara "resume" — `--resume` sudah otomatis
dipakai `jalankan_fp16.sh` di dalamnya, tidak perlu flag terpisah.

Semua angka (arena GPU, PER_PROSES, ambang watchdog) ada di
`config/int8.env.example` dengan alasan tiap nilai — jangan tebak, baca
komentarnya kalau GPU-mu bukan H200 140GB.

---

## Isi

```
setup.sh                         entry point: dua patch wajib + pre-flight
run.sh                            entry point: --preset smoke|full, --status
config/int8.env.example          SEMUA angka yang beda per mesin/GPU
req_rtenv.txt                    dependency terkunci (==, disalin dari FP16 —
                                 rtenv sama, cuma PRECISION yang beda)
pipeline/mass_loglik_inject.py   inti eksperimen (identik dengan FP16 —
                                 presisi ditentukan env PRECISION=int8)
analisis/cek_cakupan.py          verifikasi cakupan, pakai --bit 8 untuk INT8
runner/jalankan_fp16.sh          runner dua tahap (namanya fp16, dipakai keduanya)
runner/pengawas_fp16.sh          mengulang sampai target tercapai
runner/penjaga_vram.sh           watchdog: proses mati + kegagalan VRAM
runner/status.sh                 ringkasan satu layar
runner/patch_int8_memory_pool.sh PATCH WAJIB untuk INT8
runner/terapkan_config_dinamis.sh PATCH WAJIB kedua — sinkronkan _config.py/
                                 fidelity_utils.py instance dgn repo_terpatch/
pipeline/discover_operasi.py     generate operation_suffixes utk model baru
tests/test_discover_operasi.py   regresi + bukti anti "silent drop"
repo_terpatch/                   berkas FIdelity-ONNX yang menentukan perilaku
```

`mass_loglik_inject.py`, `jalankan_fp16.sh`, `pengawas_fp16.sh`, `status.sh`, dan
`cek_cakupan.py` **identik byte-per-byte** dengan yang ada di `../kode_fp16/`.
Tidak ada cabang kode terpisah untuk INT8 — yang membedakan hanya variabel
lingkungan. Disalin ke sini supaya folder ini berdiri sendiri.

---

## `repo_terpatch/` — kenapa berkas ini disimpan

| Berkas | Kenapa penting |
|---|---|
| `memory_pool.py` | **sudah dipatch.** Tanpa patch, seluruh baris `RANDOM_BITFLIP` gagal di INT8 |
| `memory_pool.py.asli` | versi asli, untuk pembanding |
| `PATCH_memory_pool.diff` | 10 baris yang berubah |
| `graph.py` | memilih implementasi injeksi dari tipe tensor; memuat `delta_init` |
| `inject_ops.py` | merangkai node injeksi (`ScatterND`, `BitFlip`, `DirectBitToggleFp32`) |
| `fidelity_utils.py` | `fix_input_tensor` — menormalkan config parser lama |
| `_config.py` | `PRECISION` → memilih `onnx_fp16` atau `onnx_int8`, dan lebar bit |

### Patch `memory_pool.py`

`graph.py` memilih implementasi bitflip dari tipe tensor sasaran:

- tensor **FP16** → `custom.bitflip:BitFlip` (dari `llama/onnx_bitflip.so`)
- tensor **FP32** → `ai.onnx.contrib:DirectBitToggleFp32` (PyOp dari onnxruntime-extensions)

Graf INT8 memakai fake-quant dengan penyimpanan FP32, jadi **selalu** jalur kedua.
Tapi `memory_pool.py` asli hanya mendaftarkan `onnx_bitflip.so`. Akibatnya:

```
Fatal error: ai.onnx.contrib:DirectBitToggleFp32(-1) is not a registered function/op
```

dan **1/6 dataset hilang tanpa peringatan**. `cnn_inference.py` mendaftarkan kedua
pustaka; jalur LLM yang terlewat.

Perbaikannya butuh dua hal: mendaftarkan pustaka extensions (patch ini) **dan**
meng-import `inject_ops` agar dekorator `@onnx_op` berjalan — yang kedua sudah
terjadi sendiri lewat `graph.py`. Mendaftarkan pustaka saja tidak cukup.

FP16 tidak terdampak karena tensornya FP16.

---

## Mekanisme fault model `RANDOM`

`delta_init` di `graph.py` **tidak mengacak nilai**, melainkan mengacak **pola bit**
lalu menafsirkannya sebagai float:

```python
for _ in range(32): one_bin += str(np.random.randint(0,2))
return bin2fp32(one_bin)
```

Karena eksponen IEEE-754 ikut teracak, sebarannya ekstrem (simulasi 100.000 nilai):
50% negatif, 36,6% di bawah 1e-10, 37,4% di atas 1e10, 11,1% di atas 1e30.
NaN dipetakan ke 0 oleh `bin2fp32`; Inf tetap lolos.

**Batasan yang perlu disadari:** `delta_init` dipanggil **sekali per pembangunan
graf**, dan grafnya dipakai ulang untuk seluruh bit × run kombinasi itu. Jadi dalam
satu (layer × operasi × fault model), **nilai rusaknya sama untuk semua baris** —
yang berubah hanya lokasinya lewat `rand_idx_inject`.

Berbeda dari `RANDOM_BITFLIP`, yang mengirim bit position **dan** lokasi sebagai
input runtime, sehingga tiap baris benar-benar berbeda.

Kalau nilai juga harus bervariasi per baris, `delta_init` perlu dipindah ke dalam
loop dan nilainya dijadikan input graf seperti `bit_pos_inject`, bukan dibakar
sebagai `Constant`.

---

## Menjalankan

```bash
PRECISION=int8 LANGKAH=1 N_CFG=288 BITS=0-7 RUNS=2 PER_PROSES=3 \
  ORT_GPU_MEM_LIMIT_GB=35 FIDELITY_POOL_GB=35 OUT=.../massal_int8.csv \
  setsid nohup bash pengawas_fp16.sh > logs/pengawas.log 2>&1 < /dev/null &
```

Empat angka yang berbeda dari FP16 dan **tidak boleh disalin mentah**:

| | FP16 | INT8 | kenapa |
|---|---|---|---|
| `BITS` | `0-15` | **`0-7`** | INT8 = 8 bit |
| `ORT_GPU_MEM_LIMIT_GB` | 60 | **35** | model 26 GB, bukan 13 GB |
| `PER_PROSES` | 1 | **3** | muat model 9,6 menit, bukan 26 detik |
| `--bit` di `cek_cakupan` | 16 | **8** | |

Sharding antar mesin wajib memakai `MULAI`/`AKHIR` yang **disjoint**, dan berkas
CSV terpisah per mesin. `penjaga_vram.sh` meneruskan keduanya saat restart — tanpa
itu mesin shard akan mulai dari config 0 dan menduplikasi pekerjaan mesin lain.

---

## Daftar operasi per decoder — sekarang per-model, bukan hardcode (20 Sep 2026)

`_config.py` dulu punya `OPERASI_SUFFIXES`, daftar tetap 9 nama tensor
Llama-2-7B. `fidelity_utils.list_canonical_configs()` membuang diam-diam
config apa pun yang nama operasinya tidak ada di daftar itu — kalau parser
dijalankan terhadap ONNX model lain (Qwen, Gemma, dll.) dengan operasi bernama
beda, sebagian datanya hilang tanpa error apa pun.

**Perbaikan:** `_config.operasi_suffixes()` sekarang membaca
`configs/my_model.json["operation_suffixes"]` kalau ada, baru jatuh ke
`OPERASI_SUFFIXES` (Llama-2-7B) kalau tidak ada. `list_canonical_configs()` dan
`operasi_idx_from_suffix()` memakai accessor ini, bukan konstanta langsung.
`COMBO_TARGET_ROWS`/`demo_target_rows()` juga dibetulkan sekalian — keduanya
sempat hardcode `32` padahal `decoder_count()` sudah ada untuk itu.

**Untuk model baru:** jalankan `pipeline/discover_operasi.py` setelah export +
parser (sekali per model, bukan per run). Skrip ini menelusuri graf ONNX satu
decoder representatif, mencocokkan tiap config `injection_llm/*.json` ke node
graf, mengurutkan berdasarkan **urutan eksekusi asli** (bukan abjad nama
file — itu akan salah), lalu menulis hasilnya ke
`configs/my_model.json["operation_suffixes"]`:

```bash
python pipeline/discover_operasi.py \
  --injection-dir /path/ke/injection_llm \
  --onnx /path/ke/decoder-merge-0.onnx \
  --model-config /path/ke/configs/my_model.json
```

Tidak butuh GPU — analisis statis graf ONNX saja, paket `onnx` cukup.
Regresi dan kasus arsitektur-lain dibuktikan di `../tests/test_discover_operasi.py`
(graf ONNX sintetis, tidak butuh data eksperimen).

**Update 21 Sep 2026 — exporter-nya sudah diperbaiki juga, lihat bagian
"Model lain (GQA, model lokal)" di bawah.** Paragraf ini awalnya bilang
exporter masih menolak GQA; itu sudah tidak berlaku lagi, dijaga di sini
supaya riwayatnya jelas.

---

## Model lain (GQA, model lokal) — 21 Sep 2026

Dua keterbatasan yang tadinya menghalangi model non-Llama-2 sudah ditutup:

**1. GQA (Grouped-Query Attention)** — dipakai Llama-3, Mistral, Gemma-2, dan
model lain berarsitektur `"llama"` di `config.json` dengan
`num_key_value_heads < num_attention_heads`. `repo_terpatch/modeling_llama.py`
(exporter hooks yang ditimpa ke atas `transformers.models.llama.modeling_llama`
sebelum export) sekarang punya `repeat_kv()` — implementasi persis sama
dengan punya HuggingFace sendiri — disisipkan tepat sebelum QK^T. Assert
yang dulu menolak GQA sama sekali sekarang cuma mensyaratkan pembagian genap
(`num_attention_heads % num_key_value_heads == 0`), syarat GQA valid yang
sesungguhnya.

**Diverifikasi di GPU nyata** (RTX 4090, 21 Sep 2026): model uji GQA kecil
(`num_attention_heads=8, num_key_value_heads=2`) di-export lewat
`export_llama.py`, tiap sub-graf ONNX (embed/decoder/norm/head) dibandingkan
terhadap PyTorch untuk input yang sama — selisih maksimum 2×10⁻⁷, level
noise floating-point. `smoothquant/{smooth,fake_quant}.py` (dipakai
`export_llama_int8.py`) dibaca tuntas dan terbukti generik penuh terhadap
jumlah head (beroperasi per-Linear-layer, bukan per-head) — belum diuji
langsung di GPU untuk arm INT8, tapi tidak ada perubahan kode yang
dibutuhkan di situ.

**PENTING — bukan berarti Qwen2/Gemma sungguhan langsung bisa dipakai.**
Perbaikan ini menyasar model yang *ditandai* sebagai arsitektur `"llama"`
(termasuk Llama-3, atau model lain yang di-`config.json`-nya memang
`model_type: "llama"`). Qwen2 dan Gemma asli punya file
`modeling_qwen2.py`/`modeling_gemma.py` sendiri di `transformers` — belum
disentuh sama sekali, karena `export_llama.py` cuma menimpa
`modeling_llama.py`. `AutoModelForCausalLM.from_pretrained()` akan memilih
kelas berdasarkan `model_type`, jadi Qwen2/Gemma asli akan lewat jalur yang
sama sekali berbeda dan tidak diuntungkan oleh patch ini.

**2. Model lokal** — `MODEL_ID` (di `config/int8.env`) boleh diisi path
direktori lokal, bukan cuma ID repo HuggingFace. Berguna kalau bobotnya
sudah ada di server/instance lain (NFS bersama, dsb.) — `_download_7b.py`
melewati unduhan HF sama sekali kalau `MODEL_ID` terdeteksi direktori lokal
yang valid (`config.json` + minimal satu `*.safetensors`). Lihat komentar di
`pipeline/_download_7b.py` dan `config/int8.env.example` untuk detail.
