# Reproduksi Eksperimen Fault Injection Llama-2-7B (FIdelity-ONNX)

Paket ini berisi **semua yang tidak bisa dibuat ulang sendiri**. Dengan ini,
menyewa GPU baru tidak perlu mengulang dari nol: tidak perlu meng-compile custom
op, tidak perlu menjalankan parser 288 konfigurasi, tidak perlu menebak versi paket.

Terakhir diperbarui: **31 Agustus 2026** (run mode `pg-teks`).

---

## Isi paket

| Berkas | Kenapa disimpan |
|---|---|
| `code.tar.gz` | **Paling penting.** Repo FIdelity-ONNX: 84 skrip `.py`, custom op `llama/onnx_bitflip.so` (sudah ter-compile), dan `act_scales/llama-2-7b.pt` + `int8/llama-2-7b.pt`. Meng-compile ulang `.so` dan menghitung ulang act_scales itu mahal. |
| `injection_llm/` | 289 konfigurasi injeksi (288 decoder + 1 head). Dihasilkan `03_phase2_parser.py`; menjalankannya ulang butuh model ONNX sudah ada + ~5 menit. |
| `req_rtenv.txt` | 67 paket terpin, kombinasi yang **terbukti** jalan. |
| `LINGKUNGAN.txt` | Catatan perangkat + variabel lingkungan + **biaya terukur**. |
| `mmlu_pool.csv.gz` | Pool 18.741 soal MMLU. Harus sama persis, kalau tidak `sample_id` tidak cocok dengan hasil lama. |
| `bootstrap_vast.sh` | Menyiapkan instance dari kosong sampai siap jalan. |
| `run_massal.sh` | Menjalankan eksperimen massal, dipotong per 8 konfigurasi. |
| `asap.sh` | Uji asap + pengukuran biaya sebelum run penuh. |
| `mass_loglik_inject.py` | Inti eksperimen: injeksi + inference golden/faulty. |
| `hitung_penilaian.py` | Menghitung metrik dari CSV, **di laptop**, tanpa GPU. |
| `ringkas_pgteks.py` | Ringkasan cakupan (bukan penilaian). |
| `hitung_skala.py` | Menghitung RUNS yang muat dari biaya terukur + saldo. |

Yang **tidak** ikut (sengaja, karena besar dan bisa ditarik lagi):
model ONNX FP16 13 GB → `mfikryrz/llama2-7b-fidelity-onnx-fp16`.

---

## Langkah reproduksi

```bash
# 1. Sewa instance. Syarat minimum yang terbukti:
#    GPU >= 40 GB VRAM (dipakai ~35 GB), disk >= 60 GB, RAM >= 48 GB.
#    A100-SXM4-80GB @ ~$0,95/jam sudah cukup; JANGAN sewa multi-GPU,
#    pipeline-nya single-GPU dan GPU kedua hanya membakar saldo.

# 2. Kirim paket ini + token (token lewat stdin, JANGAN di baris perintah)
scp -P <port> code.tar.gz bootstrap_vast.sh run_massal.sh asap.sh \
    mass_loglik_inject.py hitung_penilaian.py ringkas_pgteks.py \
    hitung_skala.py req_rtenv.txt mmlu_pool.csv.gz \
    root@<host>:/workspace/fidelity/
scp -rP <port> injection_llm root@<host>:/workspace/fidelity/
cat ~/.hf_token | ssh -p <port> root@<host> 'umask 077; cat > /root/.hf_token'

# 3. Bootstrap (~6 menit; ONNX 13 GB ditarik ~2 menit berkat dedup Xet)
ssh -p <port> root@<host> 'cd /workspace/fidelity &&
  tar xzf injection_llm.tar.gz; gunzip -kf mmlu_pool.csv.gz
  export HF_TOKEN=$(cat /root/.hf_token)
  DISK_MIN_GB=60 bash bootstrap_vast.sh 10 40'

# 4. Konfigurasi + pool + skrip ke tempat yang benar
#    (code.tar.gz TIDAK memuat injection_llm/ maupun work/)
ssh -p <port> root@<host> '
  R=/workspace/fidelity/running_experiment_7b
  mkdir -p $R/repo/FIdelity-ONNX-master/injection_llm $R/work
  cp /workspace/fidelity/injection_llm/*.json $R/repo/FIdelity-ONNX-master/injection_llm/
  cp /workspace/fidelity/mmlu_pool.csv $R/work/
  cp /workspace/fidelity/*.py $R/
  chmod +x $R/repo/FIdelity-ONNX-master/llama/onnx_bitflip.so'

# 5. Uji asap DULU — mengukur biaya, jangan menebak
ssh -p <port> root@<host> 'cd /workspace/fidelity && bash asap.sh'

# 6. Hitung RUNS dari angka terukur, lalu jalankan
python hitung_skala.py --saldo <USD> --tarif <USD/jam> \
    --detik-per-baris <dari asap> --detik-per-graf <dari asap> --detik-muat <dari asap>
ssh -p <port> root@<host> 'cd /workspace/fidelity &&
  export HF_TOKEN=$(cat /root/.hf_token)
  export HF_TOKEN_TULIS=$(cat /root/.hf_token_tulis)
  BUDGET_HOURS=<jam> RUNS=<hasil> POTONGAN_CFG=8 bash run_massal.sh'

# 7. Penilaian dikerjakan di LAPTOP, bukan di instance
python hitung_penilaian.py massal_pgteks.csv massal_dinilai.csv
```

---

## Jebakan yang sudah memakan waktu — jangan diulang

| Jebakan | Akibat | Pencegahan |
|---|---|---|
| **>1 proses per GPU** | Tanpa MPS, konteks CUDA **bergiliran**, bukan paralel. Terukur: 4 proses = laju sama dengan 1, plus OOM arena membuang **42% baris** tanpa peringatan | **1 proses per kartu.** Multi-GPU boleh, tapi satu proses tiap kartu |
| **Mengukur laju dengan `median(detik)`** | Mengabaikan ekor lambat dan jeda antar-baris → saya melaporkan 1,262 baris/dtk padahal nyatanya 0,25. Anggaran meleset $29 vs $48 | **jam dinding, jendela ≥ 10 menit**, supaya restart potongan ikut terhitung |
| **Token HF di baris perintah** (`hf --token "$X"`) | Terbaca siapa pun lewat `ps` di mesin sewaan | pakai variabel lingkungan `HF_TOKEN` |
| **`pkill -f` / `pgrep -f` dengan pola yang ada di baris perintah sendiri** | Keduanya mencocokkan SELURUH baris perintah, termasuk shell pembungkusnya sendiri. Akibatnya: `pkill -f` membunuh dirinya sendiri (**4 peluncuran gagal diam-diam**), dan `pgrep -f` selalu menemukan "proses" sehingga loop `until ! pgrep ...` **tidak pernah berakhir** — instance menganggur 20 menit tanpa ketahuan | matikan lewat **PID**; untuk mendeteksi proses pakai pola yang tidak cocok dengan perintah sendiri, mis. `ps -eo pid,cmd \| awk '$2 ~ /rtenv\/bin\/python$/ && /mass_loglik/'` |
| `tmux` / `setsid nohup` lewat SSH | proses mati begitu SSH tutup | jalankan lewat koneksi SSH yang dipertahankan sebagai tugas latar belakang |
| `nproc` dipakai sebagai jumlah thread | melaporkan core HOST (128), bukan jatah (30) → mesin tercekik | `deteksi_core()` membaca cgroup v1 **dan** v2 |
| `req_rtenv.txt` tanpa `onnxruntime-gpu` | bootstrap lolos tanpa error, lalu `import onnxruntime` gagal | sudah diperbaiki (67 paket); ORT dipasang **sebelum** daftar terpin |
| Memasang `accelerate` | menaikkan `huggingface_hub`, merusak `transformers 4.33.3` | jangan dipasang |
| `ORT_GPU_MEM_LIMIT_GB` = VRAM−4 | arena ORT **tidak pernah menyusut** setelah ratusan sesi faulty → segfault di konfigurasi ke-33 | pakai 30 GB + potong per 8 konfigurasi (proses baru mereset arena) |
| Graf injeksi ditulis ke folder ONNX | sisa `*_injected.onnx` terbaca sebagai decoder saat startup berikutnya | tulis ke `TMPDIR` |
| Menyewa multi-GPU | pipeline single-GPU; GPU kedua menganggur tapi tetap ditagih | sewa 1 GPU |

---

## Biaya terukur (A100-SXM4-80GB, FP16)

| | mode `pg` | mode `pg-teks` |
|---|---|---|
| Keluaran | logit A/B/C/D saja | **teks mentah** golden & faulty |
| Forward pass per baris | 2 | ~20 |
| Detik per baris | 0,145 | **1,96** (tunak) |
| Pemanasan | — | 22 dtk pada baris pertama tiap proses |
| Muat model | — | ~14 dtk tiap proses |
| Bangun 1 graf | — | 0,33 dtk |
| Overhead tetap sekali lewat 288 konfigurasi | — | ~0,52 jam |

**Mode `pg-teks` ~9x lebih mahal per baris.** Jangan pakai anggaran run mode `pg`
untuk memperkirakan skala mode `pg-teks`.

### Soal `--token-teks 48`

Uji asap (3 konfigurasi, decoder 0) menunjukkan golden maksimum 27 token, sehingga
48 tampak longgar. **Pada pool penuh itu tidak berlaku.** Diukur pada 557 baris
pertama run sebenarnya:

| | |
|---|---|
| token golden | median 10, **maks 48** |
| golden menyentuh 48 | 12 baris = 2,2% (hanya **2 soal unik**) |
| faulty menyentuh 48 | 23 baris = 4,1% |
| faulty kosong | 2 baris |

Yang memicu: mata pelajaran dengan teks opsi sangat panjang —
`high_school_european_history` dan `professional_law`.

Jangan mengubah `--token-teks` di tengah run: dataset jadi terbelah dua dan tidak
sebanding. Golden dan faulty dipotong di titik yang sama, jadi perbandingannya
tetap sah. Baris terpotong bisa **disaring** lewat `n_tok_golden >= batas_token_teks`.

Kalau memulai run baru dari nol dan ingin tidak ada yang terpotong sama sekali,
pakai `--token-teks 96`. Biayanya naik hanya untuk baris yang memang panjang —
baris yang berhenti di 10 token tetap semurah sebelumnya.

**`faulty` kosong bukan bug.** Artinya token pertama yang dihasilkan langsung
baris-baru: model terlalu rusak untuk menjawab. Logitnya ikut anjlok (~5 dibanding
~24 pada golden). Ini justru temuan yang tidak akan pernah terlihat dari kolom
A/B/C/D, karena di sana selalu ada satu huruf "pemenang" seolah model masih menjawab.

---

## Bukti untuk keputusan metrik (belum diputuskan)

`hitung_penilaian.py` menyediakan dua ukuran kemiripan. Keduanya disimpan; mana yang
jadi angka utama adalah keputusan pembimbing. Data run ini menunjukkan keduanya
**tidak setara**, dan salah satunya punya cacat yang bisa menyesatkan.

- **`kesamaan_karakter`** — jarak Levenshtein per karakter. Levenshtein = jumlah
  minimum sisip/hapus/ganti karakter untuk mengubah satu teks jadi teks lain.
  Toleran terhadap pergeseran.
- **`kesamaan_token`** — mencocokkan kata **per posisi**: kata ke-1 lawan kata ke-1,
  ke-2 lawan ke-2, dan seterusnya. Satu kata hilang di depan membuat semua posisi
  berikutnya meleset.

Contoh nyata dari run ini:

```
G: 'D. I, II, and III'
F: 'I, II, and III'
   kesamaan_karakter = 0,8235      kesamaan_token = 0,0000

G: 'A. one hundred seventy-nine thousand twelve'
F: 'one hundred seventy-nine thousand nine hundred twelve'
   kesamaan_karakter = 0,6981      kesamaan_token = 0,0000
```

Teksnya jelas mirip, tapi skor token **nol**. Penyebabnya sama pada kedua kasus:
kerusakan membuat model menjatuhkan awalan huruf `"X. "`, sehingga seluruh kata
bergeser satu posisi.

Terjadi pada **2 dari 44 baris yang berbeda (4,5%)** di 557 baris pertama. Kecil,
tapi **bias-nya searah**: `kesamaan_token` secara sistematis melaporkan kerusakan
lebih parah daripada kenyataan, justru pada kasus di mana jawabannya sebenarnya
masih utuh. Kalau `kesamaan_token` dipakai sebagai angka utama, sebagian kerusakan
ringan akan terbaca sebagai kerusakan total.

Catatan: `kesamaan_token` yang dipakai di sini **bukan** cosinus maupun Jaccard.
Keduanya mengabaikan urutan kata, sehingga tidak cocok untuk membandingkan jawaban
yang urutannya bermakna.

---

## Catatan metode

Mode prompt murni fungsi dari **format jawaban pada contoh few-shot**. Tidak ada
perubahan model, bobot, atau kode inferensi:

| Mode | Contoh few-shot dijawab | Model menghasilkan |
|---|---|---|
| `pg` | `Answer: C` | satu huruf |
| `pg-teks` | `Answer: C. Authentication` | kalimat penuh |
| `bebas` | tanpa pilihan sama sekali | kalimat bebas |

Eksperimen (mahal, GPU) sengaja **dipisah** dari penilaian (murah, CPU).
`mass_loglik_inject.py` hanya menyimpan yang tidak bisa dihitung ulang: teks mentah,
logit mentah, dan identitas kombinasi. Kalau metrik berubah, jalankan ulang
`hitung_penilaian.py` saja — **tidak perlu menyewa GPU lagi**.
