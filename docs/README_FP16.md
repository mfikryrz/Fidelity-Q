# FP16 — Dokumen Acuan Tunggal

**Diperbarui 14 September 2026.**

Ini satu-satunya dokumen yang berlaku untuk FP16. `PANDUAN_43K.md` dan
`LANJUTKAN.md` sudah dihapus (usang). Dokumen lama yang masih memuat analisis
dipindah ke [`dokumen_protokol_lama/`](dokumen_protokol_lama/) dan **tidak
menjelaskan cara kerja sekarang** — beberapa angkanya salah, terutama biaya
membangun graf. Kalau ada yang berbeda, **yang ini yang benar.**

> **Untuk INT8, baca [`README_INT8.md`](README_INT8.md)** — protokolnya sama,
> tapi sumber ONNX, konfigurasi injeksi, batas memori, dan filter sewa mesin
> semuanya berbeda.

> Untuk AI atau orang yang baru masuk: cukup baca dokumen ini. Buka
> `dokumen_protokol_lama/` hanya kalau butuh kamus kolom dataset lama,
> rasional 43.200 baris, atau contoh perubahan golden→faulty.

---

## 1. Apa yang sedang dikerjakan

Fault injection pada **Llama-2-7B FP16** lewat FIdelity-ONNX. Satu bit dirusak di
dalam salah satu operasi MatMul, lalu keluaran model dibandingkan dengan keluaran
tanpa kerusakan (golden vs faulty) pada soal MMLU yang sama.

Protokol saat ini mengikuti revisi pembimbing (Mas Gabriel) tanggal 10 Sep 2026.
Ada **tiga perubahan** dari protokol lama:

| | Lama | Sekarang |
|---|---|---|
| Soal per baris | keenam fault model berbagi satu soal | **tiap baris menarik soal acak sendiri** |
| Koordinat injeksi | tidak dicatat | **dicatat: `inject_coords`, `inject_w/h/c`** |
| `Bit_Position` | acak berseed | **berurutan 0,1,2,…,15** |

Konsekuensi perubahan pertama: golden tidak lagi bisa dipakai bersama enam fault
model, sehingga forward pass naik dari 50.400 ke 86.400 untuk 43.200 baris.
Ini yang membuat beban memori naik drastis — lihat bagian 5.

### Status persetujuan (rapat 11 Sep 2026)

Format dan urutan **sudah disetujui** pembimbing:
*"semua sampai column sudah benar dan juga secara urutan ini sudah benar"*
(`docs/rapat/rapat_20260911/rapat_20260911.txt`, 0:25:39).

Perilaku bit tinggi juga dikonfirmasi sebagai temuan, bukan bug: bit 14 ada di
bagian eksponen FP16, membaliknya melipatgandakan nilai ~65.000 kali.

**Permintaan pembimbing** (0:25:39–0:27:04):

> Pada eksperimen masal, **setiap decoder × setiap operasi** harus diinjeksi dan
> dievaluasi. Dan pada smoke test, **harus dicek eksplisit** apakah ada decoder
> atau operasi yang terlewat.

**Sudah terpenuhi untuk FP16** (15 Sep 2026). `cek_cakupan.py` pada hasil final
melaporkan `decoder 32/32`, `operasi 9/9`, `konfigurasi 288/288` — ditunjukkan,
bukan diasumsikan. Keluaran lengkapnya di bagian 2; cara menjalankannya di
bagian 8.

Yang belum tertutup hanyalah **13 sel** dari 1.728 yang tidak terisi penuh,
semuanya karena kegagalan alokasi VRAM, bukan karena ada decoder atau operasi
yang terlewat. Rinciannya di bagian 2.

Catatan untuk smoke test: uji asap 16 layer stratified mencakup 9/9 operasi tapi
hanya **16 dari 32 decoder** — itu memang sifat stratifikasinya, dan run masal
menutupinya.

---

## 2. Hasil terbaru

### Run masal 288 konfigurasi — **SELESAI** (15 Sep 2026, 11:52 UTC)

**`zip_files/hasil_fp16_final.tar.gz`** di direktori ini — sudah diunduh dan md5-nya
diverifikasi dua sisi (`c15c04ad5ebf8cb8133493f884a7fed3`). Instance sumbernya
sudah dihapus.

| | |
|---|---|
| Baris | **55.245 / 55.296 (99,91%)** |
| Laju | 3,26 dtk/baris, stabil sepanjang ~2,5 hari |
| Konfigurasi | **288 / 288 tersentuh**, 276 penuh |
| Decoder | **seluruh 0–31** |
| Sel (config × fault model) | **1.715 penuh / 1.728** |
| Bit 0–15 | **LENGKAP**, sebaran 3.451–3.454 per bit |
| Duplikat | **0** |
| `golden_teks` kosong | **0** |

Verifikasi cakupan resmi sudah dijalankan (syarat pembimbing, bagian 8):

```bash
python src/fp16/analisis/cek_cakupan.py \
  results/fp16/final/hasil_massal/massal_fp16_288.csv --bit 16 --runs 2 --penuh
```

Hasilnya `ADA YANG BELUM LENGKAP` karena 13 sel cacat. Sebaran per bit yang
rata (3.451–3.454) adalah sinyal mutu penting: tidak ada bit yang sistematis
kekurangan data.

**51 baris yang hilang gagal permanen, bukan terpotong.** Pengawas menjalankan
dua putaran penuh; putaran kedua menambah **nol** baris, lalu berhenti sendiri:

```
=== putaran 1 selesai: 55245/55296 ===
=== putaran 2 selesai: 55245/55296 ===
=== BERHENTI: putaran 2 tidak menambah baris apa pun (55245) ===
```

Dua belas sel kehilangan 1–6 baris — sisa acak dari tekanan VRAM, terparah
`d14/mlp_down_proj_MatMul RANDOM_BITFLIP: 26/32`. Satu sel berbeda polanya:

> **`d28/mlp_up_proj_MatMul INPUT: 0/32`** — sel ini tidak pernah berhasil
> **sama sekali**, bukan gagal sesekali. Sendirian ia menyumbang 32 dari 51
> baris yang hilang. Kalau ada kesempatan menjalankan ulang secara tertarget,
> kombinasi inilah yang paling berharga diselidiki — polanya menunjukkan sebab
> spesifik, bukan nasib buruk.

Kolam soal yang dipakai: `mmlu_pool.csv`,
sha256 `17dbed6f0fc2d90319f0a9698f992c1cbcf3eb60876d66316a5972afe3b38219` —
**identik** dengan yang dipakai arm INT8 (dikonfirmasi `pool_sha256` dari
`validate_int8_gate.py`). Kedua arm karena itu sebanding.

Snapshot parsial dalam perjalanan tersimpan di `experiments/backup_int8_migrasi_20260915/`
(88,3% dan 91,3%) — tidak lagi diperlukan, tapi disimpan sebagai jejak.

### Uji asap 16 layer — SELESAI

**`experiments/fp16_smoke_v4_20260911/massal_fp16_v4.csv`** — 3.071 dari 3.072 baris (99,97%),
34 kolom, md5 `542e89a1f688acd3cea2a47bc7986db0`. Konfigurasi: 16 layer
stratified × 6 fault model × 16 bit × 2 injeksi.

Satu baris tidak terisi: `d16/self_attn_v_proj/WEIGHT/bit12/run2`, gagal konsisten
dengan `RUNTIME_EXCEPTION` di node `Slice` bahkan di proses segar dengan batas
memori 70 GB. Penyebabnya soal yang tertarik untuk baris itu sangat panjang.
Sel yang sama tetap terwakili `run 1`.

### Penilaian dihitung terpisah

Kedua berkas hanya memuat keluaran inferensi mentah. Metrik dihitung di laptop,
tanpa GPU:

```bash
python src/fp16/analisis/hitung_penilaian.py masukan.csv keluaran_dinilai.csv
```

---

## 3. Letak berkas

Direktori proyek dirapikan 20 Sep 2026 menjadi `src/` (kode), `results/`
(hasil final), `experiments/` (percobaan/backup historis), `docs/` (dokumen
ini beserta lampirannya), dan `zip_files/` (semua arsip `.zip`/`.tar.gz`).
Path di bawah relatif terhadap akar proyek, bukan terhadap `docs/`.

```
docs/README_FP16.md                     <- dokumen ini
zip_files/hasil_fp16_final.tar.gz       <- *** HASIL FINAL RUN MASAL 288 *** (55.245 baris)
results/fp16/final/                     <- arsip di atas, sudah diekstrak
zip_files/fp16_pelengkap.tar.gz         <- hasil_vast/ + mmlu_pool.csv + req_rtenv.txt
results/fp16/pelengkap/                 <- arsip di atas, sudah diekstrak
experiments/fp16_smoke_v4_20260911/     <- uji asap 16 layer (3.071 baris), bukan run masal
src/fp16/runner/ambil_fp16.sh           <- pengambil hasil dari instance (sudah tidak dipakai)
src/fp16/
  pipeline/mass_loglik_inject.py  <- inti eksperimen (v4, sudah dipatch)
  runner/jalankan_fp16.sh         <- PAKAI INI untuk menjalankan
  runner/bootstrap_vast.sh        <- menyiapkan instance dari nol
  runner/asap.sh                  <- uji asap + ukur biaya
  runner/pengawas_fp16.sh         <- mengulang sampai target tercapai (bagian 10)
  runner/penjaga_vram.sh          <- watchdog: proses mati + kegagalan VRAM
  runner/status.sh                <- ringkasan satu layar
  runner/run_massal.sh            <- runner LAMA, jangan dipakai (lihat 5 & 6)
  analisis/hitung_penilaian.py    <- metrik, di laptop
  analisis/ringkas_pgteks.py      <- ringkasan cakupan
  dokumen_sumber/                 <- README_REPRODUKSI.md asli dari repo HF
  PERUBAHAN_vs_int8_v3.diff       <- apa yang diubah dari kode INT8 v3
zip_files/kode_fp16.zip                 <- seluruh src/fp16/ dalam satu berkas (snapshot lama)
docs/rapat/rapat_20260910/              <- transkrip rapat + whiteboard
experiments/test_implementasi_perbaikan/ <- uji 42 baris yang dikirim ke pembimbing
docs/dokumen_protokol_lama/             <- arsip, SUDAH TIDAK BERLAKU (ada README-nya)
```

Data lama (run protokol lama, masih sah sebagai pembanding):
`massal_pgteks_RUNS3.csv`, `massal_pgteks_final.csv`, `hasil_instance_*`.

`run_massal.sh` disimpan hanya sebagai rujukan sejarah. Ia mematok
`--posisi-token akhir` dan tidak melewatkan `--bits`, jadi menjalankannya
menghasilkan protokol LAMA tanpa peringatan apa pun. Pakai `jalankan_fp16.sh`.

---

## 4. Menjalankan dari nol

### Sewa mesin

| Syarat | Nilai | Kenapa |
|---|---|---|
| Jumlah GPU | **1** | pipeline single-GPU; GPU kedua terbukti tidak menolong sama sekali |
| VRAM | **80 GB** | kartu 40 GB terlalu mepet |
| **Disk** | **≥ 1,5 GB/detik** | penentu biaya total yang sebenarnya, bukan GPU |
| RAM | ≥ 64 GB | |
| CPU | ≥ 16 inti | di atas itu tidak menambah kecepatan |
| Durasi maks | ≥ 3 hari | |
| Keandalan host | ≥ 98% | |
| Jenis | On-Demand | jangan interruptible |

Harga wajar $1,00–1,20/jam. Mesin yang dipakai 11 Sep: A100-SXM4-80GB Czechia,
disk 4.766 MB/s, $1,050/jam.

> Kesalahan yang mudah terjadi: memilih berdasarkan **tarif per jam**. Yang benar
> adalah **biaya total**, dan itu ditentukan kecepatan disk.

### Setup (~5 menit)

Template Vast.ai: **NVIDIA CUDA** (yang paling ramping). Bootstrap membangun
Python 3.10 + CUDA sendiri lewat pip, jadi image tidak perlu membawa apa pun
selain driver dan SSH.

```bash
# token (lewat stdin, JANGAN di baris perintah — terbaca lewat ps)
cat token.txt | ssh -p PORT root@HOST 'umask 077; cat > /root/.hf_token'
ssh -p PORT root@HOST 'cp /root/.hf_token /root/.hf_token_tulis'

ssh -p PORT root@HOST 'mkdir -p /workspace/fidelity && cd /workspace/fidelity &&
  export HF_TOKEN=$(cat /root/.hf_token) &&
  curl -sfL -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/mfikryrz/llama2-7b-fidelity-onnx-fp16/resolve/main/code/bootstrap_vast.sh" \
    -o bootstrap_vast.sh &&
  DISK_MIN_GB=60 bash bootstrap_vast.sh 10 40'
```

**Dua lubang di `bootstrap_vast.sh` yang harus ditambal manual:**

```bash
ssh -p PORT root@HOST 'bash -s' <<'EOS'
ROOT=/workspace/fidelity; R=$ROOT/running_experiment_7b; D=$ROOT/_dl/code
# (a) tahap 20 memasang paket SEBELUM tahap 30 mengunduh req_rtenv.txt, jadi
#     diam-diam memakai daftar minimum. Pasang 67 paket terpin sekarang.
cp $D/req_rtenv.txt $ROOT/ && $ROOT/rtenv/bin/pip install -q -r $ROOT/req_rtenv.txt
# (b) bootstrap tidak pernah menempatkan injection_llm/ maupun mmlu_pool.csv
mkdir -p $R/repo/FIdelity-ONNX-master/injection_llm $R/work
cp $D/injection_llm/*.json $R/repo/FIdelity-ONNX-master/injection_llm/
gunzip -c $D/mmlu_pool.csv.gz > $R/work/mmlu_pool.csv
cp $D/*.py $R/
chmod +x $R/repo/FIdelity-ONNX-master/llama/onnx_bitflip.so
EOS
```

Verifikasi setup benar: 289 file `injection_llm/*.json`, `mmlu_pool.csv` berisi
**14.042 record** (18.741 baris fisik), 35 file ONNX, rtenv 71 paket dengan
`numpy==1.26.4`, `protobuf==3.20.3`, `onnxruntime-gpu==1.20.2`.

### Jalankan

```bash
# salin jalankan_fp16.sh ke /workspace/fidelity lebih dulu
ssh -p PORT root@HOST 'cd /workspace/fidelity && bash jalankan_fp16.sh'

# seluruh 288 konfigurasi:
ssh -p PORT root@HOST 'cd /workspace/fidelity && LANGKAH=1 N_CFG=288 bash jalankan_fp16.sh'
```

Aman diulang — `--resume` melewati baris yang sudah ada. Skrip punya dua tahap:
satu proses per konfigurasi, lalu tahap penambal yang mendeteksi sendiri pasangan
(config × fault model) yang belum penuh dan mengulanginya terisolasi.

### Pemilihan layer stratified

288 konfigurasi = 32 decoder × 9 operasi. Untuk mencuplik 16, ambil tiap langkah
ke-**19**.

> **Langkah harus koprima dengan 9.** Langkah 18 — yang tampak wajar karena
> 288/16 = 18 — justru rusak: 18 = 2×9, sehingga setiap sampel mendarat di operasi
> yang sama. Hasilnya 16 decoder tapi hanya **1 dari 9 operasi**.

---

## 5. Biaya — angka yang BENAR

Model biaya, diukur dari run 11 Sep 2026:

```
total = baris × 3,11 detik  +  graf × 3,0 detik
graf  = jumlah_konfigurasi × 6 fault model
```

Diukur dari run masal 288 konfigurasi pada 14 Sep 2026, atas 22.885 baris:
jam dinding 3,20 dtk/baris, inferensi murni 3,11 dtk/baris, Pada 33.987 baris dan 1.066 graf angkanya **2,8 detik per graf** —
konsisten.

> **Dua angka lama SALAH, jangan dipakai.** `PANDUAN_43K_v2.md` menulis t_graf
> = 0,33 detik; versi awal dokumen ini menulis 59 detik. Keduanya keliru.
> Angka 0,33 berasal dari `asap.sh` yang hasilnya negatif dan skripnya sendiri
> memperingatkan "terlalu berisik". Angka 59 berasal dari run pertama yang penuh
> error OOM — selisih waktunya dikira biaya graf, padahal itu overhead kegagalan.
> Nilai sebenarnya **3,0 detik**, diukur pada run yang sehat.

Untuk 288 konfigurasi penuh, biaya graf hanya 1.728 × 3,0 dtk = **1,4 jam** —
bukan penentu. Yang menentukan adalah jumlah BARIS.

---

## 6. Jebakan — sudah terbukti memakan waktu dan uang

| Jebakan | Akibat | Pencegahan |
|---|---|---|
| **8 konfigurasi per proses** | Arena BFC ORT tidak pernah menyusut. Beban protokol baru 2× lebih berat, arena habis di config kedua: 595 baris `[err]` lalu **segfault**. Data berlubang tanpa peringatan | **1 konfigurasi per proses** |
| **`ORT_GPU_MEM_LIMIT_GB=30`** | Gagal alokasi di node `Max`/`ReduceMax`/`Slice` (BMM attention 4-D) | **45 GB.** Jangan VRAM−4: itu segfault di config ke-33 |
| **`wc -l` untuk menghitung baris** | `prompt_lengkap` memuat baris baru → baris fisik ≠ record | modul `csv` |
| **t_graf dari dokumen lama** | 0,33 dtk (PANDUAN v2) dan 59 dtk (draf awal) sama-sama salah | **3,0 dtk**, terukur atas 717 graf |
| **Langkah cuplik kelipatan 3** | hanya 1 dari 9 operasi tercakup | koprima dengan 9, mis. 19 |
| **Membunuh hanya proses python** | `jalankan_fp16.sh` yatim menelurkan worker baru -> DUA worker di satu GPU | matikan berurutan: pengawas -> skrip -> python |
| **`grep -c ... \|\| echo 0`** | grep mencetak "0" lalu exit 1, nilainya jadi "0\n0" -> perbandingan gagal -> restart acak | pakai `awk` |
| **Child mewarisi fd `flock`** | penjaga lumpuh permanen setelah restart pertama, tanpa pesan | `9>&-` saat meluncurkan |
| **`MULAI`/`AKHIR` tidak diteruskan saat restart** | mesin shard mulai dari 0 dan menduplikasi mesin lain | teruskan eksplisit |
| **`pkill -f`** | mencocokkan baris perintahnya sendiri → membunuh diri sendiri | matikan lewat **PID**; deteksi dengan `ps -eo pid,etime,cmd \| awk '$3 ~ /rtenv\/bin\/python$/ && /mass_loglik/'` — perhatikan indeks field harus cocok dengan format `ps` |
| **>1 proses per GPU** | konteks CUDA bergiliran, bukan paralel. 4 proses = laju sama dengan 1, plus 42% baris hilang | 1 proses per kartu |
| **Mengukur laju dengan `median(detik)`** | mengabaikan overhead graf → anggaran meleset | jam dinding, jendela ≥ 10 menit |
| **Menyewa multi-GPU** | GPU kedua menganggur tapi ditagih | sewa 1 GPU |
| **Token HF di baris perintah** | terbaca lewat `ps` di mesin sewaan | lewat stdin / env |
| **Memasang `accelerate`** | menaikkan `huggingface_hub`, merusak `transformers 4.33.3` | jangan dipasang |
| **Graf injeksi ditulis ke folder ONNX** | sisa `*_injected.onnx` terbaca sebagai decoder saat startup | tulis ke `TMPDIR` |
| **`nproc` sebagai jumlah thread** | melaporkan core HOST (128), bukan jatah (30) | baca cgroup v1 dan v2 |

---

## 7. Parameter dan artinya

| Flag | Nilai | Alasan |
|---|---|---|
| `--posisi-token` | `penuh` | koordinat diundi merata dari SELURUH tensor sasaran dengan bentuk per operasi. Mode `akhir` mengasumsikan `[1,n,4096]` untuk semua operasi — salah untuk gate/up/down dan kedua BMM, sehingga sebagian tensor tak pernah tersentuh |
| `--bits` | `0-15` | FP16 = 16 bit. INT8 pakai `0-7`. Kosong = acak (jangan) |
| `--runs` | `2` | injeksi per bit, sesuai whiteboard |
| `--gaya-prompt` | `pg-teks` | contoh few-shot dijawab `"C. Authentication"` → model menghasilkan kalimat, bukan satu huruf |
| `--token-teks` | `48` | persentil ke-99 panjang jawaban golden (p99 = 45, median 10). Pada run ini 0,7% golden dan 5,7% faulty menyentuh batas. Golden dan faulty dipotong di titik yang sama, jadi perbandingan berpasangan tetap sah. Baris terpotong disaring lewat `n_tok_golden >= batas_token_teks`. **Jangan diubah di tengah run** — dataset jadi terbelah dan tidak sebanding |

### Kolom koordinat injeksi

`rand_idx` = indeks **datar**: tensor diratakan jadi satu baris, lalu `ScatterND`
menimpa elemen di posisi itu (`inject_ops.py`). Untuk `RANDOM_BITFLIP`, custom op
`BitFlip` menerima `(tensor, bit_position, rand_idx)` — `rand_idx` menentukan
elemen mana, `Bit_Position` menentukan bit ke berapa di dalam elemen itu.

`rand_idx` didaftarkan sebagai **input graf**, bukan konstanta. Karena itu satu
graf melayani 32 baris dengan lokasi berbeda — kalau dibakar ke graf, graf harus
dibangun ulang 3.072 kali (50 jam, bukan 2 jam).

`inject_coords` = `rand_idx` yang sama, diurai terhadap `inject_shape`:

```
rand_idx 2795441  dengan shape [1,602,11008]
2795441 ÷ 11008 = 253 sisa 10417  ->  inject_h=253, inject_w=10417
```

`rand_idx` **tidak bisa dibandingkan antar baris** karena batasnya bergantung
`inject_shape`, dan bentuk itu bergantung panjang prompt yang kini berbeda tiap
baris. `inject_coords` bermakna langsung. Itu sebabnya kolomnya diminta.

`inject_seed` = **masukan** RNG, bukan keluaran. Dihitung deterministik:
`sha256(seed | decoder | operasi | fault_model | run_id | bit | k)`. Dari benih
itu diturunkan `rand_idx`. Benih juga diteruskan ke kernel injeksi
(`llm_inference.py:302`) untuk fault model yang butuh nilai acak. Gunanya:
baris mana pun bisa diputar ulang persis, dan run yang terputus bisa dilanjutkan
tanpa membuat separuh dataset tidak sebanding.

---

## 8. Verifikasi cakupan (wajib sebelum lapor ke pembimbing)

```bash
python src/fp16/analisis/cek_cakupan.py hasil.csv                  # smoke test
python src/fp16/analisis/cek_cakupan.py hasil.csv --penuh          # run masal
python src/fp16/analisis/cek_cakupan.py hasil.csv --bit 8          # INT8
```

Memeriksa tiga hal yang diminta pembimbing: setiap decoder ada, setiap operasi
ada, dan setiap sel (decoder × operasi × fault model) terisi penuh. Juga
memeriksa sebaran `Bit_Position` rata — inti dari permintaan "bit tidak boleh
acak". Keluar dengan kode **1** kalau ada yang kurang, jadi bisa dipakai sebagai
gerbang di skrip.

Hasil pada dataset 11 Sep (smoke test 16 layer):

```
decoder  : 16/32     <- disengaja, stratified
operasi  : 9/9
sel      : 95/96 penuh   (d16/v_proj WEIGHT: 31/32)
bit 0-15 : LENGKAP, 191-192 per bit
```

Untuk run masal, `--penuh` mewajibkan seluruh 288 konfigurasi dan akan gagal
kalau ada satu pun decoder atau operasi yang terlewat.

---

## 9. Menjalankan tanpa dipantau — pengawas, penjaga, sharding

Dibangun 13–14 Sep untuk run masal 55.296 baris yang butuh ~50 jam. Ketiga
lapisnya menjawab kegagalan yang berbeda, dan **masing-masing sudah diuji dengan
sengaja mematikan prosesnya**, bukan diasumsikan bekerja.

### Lapis 1 — lepas dari SSH

`ssh HOST 'bash jalankan_fp16.sh'` membuat rantai proses menempel ke sshd:

```
python <- jalankan_fp16.sh <- bash <- sshd
```

Sekali koneksi putus, semuanya kena SIGHUP. Untuk run 50 jam itu rapuh. Luncurkan
lewat `setsid nohup ... < /dev/null &` sehingga induknya menjadi **init (`ppid=1`)**.
Diuji: putus koneksi, sambung lagi, proses masih hidup.

### Lapis 2 — `pengawas_fp16.sh`

Menjalankan `jalankan_fp16.sh` berulang sampai target tercapai. Perlu karena
**tahap 2 (penambal lubang) hanya berjalan setelah seluruh konfigurasi selesai** —
kalau run terputus di tengah, lubangnya menetap selamanya.

Berhenti sendiri kalau satu putaran penuh tidak menambah satu baris pun, supaya
baris yang gagal permanen tidak diulang tanpa henti sampai saldo habis.

### Lapis 3 — `penjaga_vram.sh` lewat cron tiap 10 menit

Menangani **dua** kegagalan:

| Gejala | Tindakan |
|---|---|
| Pengawas mati (OOM killer, crash) | hidupkan lagi |
| Error alokasi VRAM menumpuk | turunkan arena, baru hidupkan lagi |

Yang kedua perlu penanganan sendiri karena **kegagalan VRAM tidak membunuh
proses** — baris yang gagal dilewati diam-diam dan run terus jalan seolah sehat.
Penjaga yang hanya memeriksa "hidup atau mati" tidak akan pernah melihatnya.

Ambang **50 error baru** sejak restart terakhir (pakai watermark byte pada log,
supaya error lama tidak dihitung berulang). Untuk FP16 laju error normal 0,7%
sehingga ambang ini praktis tidak pernah tersentuh.

> FP16 masih memakai `penjaga_fp16.sh` versi lama yang hanya menangani proses
> mati. Itu memadai karena laju errornya rendah dan stabil — tidak ada alasan
> menyentuh run yang sudah 60% dan sehat hanya demi menyeragamkan skrip.
> Untuk run FP16 **baru**, pakai `penjaga_vram.sh`.

### Empat bug yang ditemukan karena diuji

Keempatnya lolos kalau hanya dibaca. Semuanya sudah diperbaiki.

**1. Rantai tiga tingkat, bukan dua.** Versi pertama penjaga hanya membunuh
`python`. Akibatnya `jalankan_fp16.sh` yang yatim tetap hidup, melanjutkan
loop-nya, dan menelurkan worker baru — lalu pengawas yang baru dihidupkan
menelurkan worker kedua. **Dua worker di satu GPU**, persis bencana yang
terdokumentasi: laju tidak bertambah dan 42% baris hilang diam-diam. Matikan
berurutan: `pengawas` → `jalankan_fp16.sh` → `python`.

**2. `grep -c` mencetak "0" lalu keluar dengan status 1.** Jadi `|| echo 0`
menambahkan nol kedua, nilainya jadi `"0\n0"`, perbandingan `[ -lt ]` gagal, dan
penjaga menyimpulkan ada masalah padahal sehat — **memicu restart acak**. Pakai
`awk` yang selalu keluar status 0 dan selalu mencetak satu angka.

**3. Pengawas mewarisi file descriptor kunci `flock`.** Begitu penjaga melakukan
restart pertama, pengawas baru menahan kunci itu berjam-jam, sehingga setiap
pemanggilan penjaga berikutnya keluar diam-diam. **Penjaga lumpuh permanen setelah
restart pertama, tanpa satu pun pesan.** Tutup dengan `9>&-` saat meluncurkan.

**4. `MULAI`/`AKHIR` tidak ikut diteruskan saat restart.** Pada sharding, mesin
yang seharusnya hanya mengerjakan konfigurasi 144–287 akan mulai dari 0 dan
**menduplikasi seluruh pekerjaan mesin lain** — baru ketahuan saat menggabung CSV.

### Memantau tanpa bertanya

```bash
ssh -p PORT root@HOST 'bash /workspace/fidelity/status.sh'
```

Dua baris yang perlu diperhatikan: **TERAKHIR** menandai sendiri `<-- MACET?`
kalau lebih dari 15 menit tanpa baris baru, dan **PENJAGA** menunjukkan berapa
kali pengawas dihidupkan ulang.

### Sharding ke beberapa mesin

`MULAI`/`AKHIR` membagi 288 konfigurasi ke beberapa mesin. Total GPU-hour tidak
berubah — hanya waktu tunggu. Tiga syarat mutlak bebas duplikat (rentang disjoint,
CSV terpisah, urutan konfigurasi identik) dijelaskan lengkap di
[`README_INT8.md`](README_INT8.md) bagian 7; mekanismenya sama untuk FP16.

Untuk FP16 288 konfigurasi: 1 mesin ~50 jam, 2 mesin ~25 jam, 4 mesin ~13 jam,
biaya total ~$53 di ketiganya.

### `PER_PROSES` — berapa konfigurasi per proses

Default **1** untuk FP16: arena BFC di ONNX Runtime tidak pernah menyusut, jadi
proses baru meresetnya. Murah karena muat model FP16 cuma ~26 detik.

**Jangan disalin ke INT8.** Di sana muat model ~9,6 menit, sehingga 288
konfigurasi berarti 46 jam hanya untuk memuat model berulang. INT8 memakai 3–6.

---

## 10. Sumber data

| | |
|---|---|
| Model ONNX FP16 | `mfikryrz/llama2-7b-fidelity-onnx-fp16` (**model** repo, 13,5 GB, 35 file) |
| Kode + konfigurasi | repo yang sama, folder `code/` |
| Cadangan hasil | `mfikryrz/llama2-7b-fidelity-hasil` (**dataset** repo — `repo_type="dataset"`, bukan model) |
| Model asal | `NousResearch/Llama-2-7b-hf` |

Keduanya privat. Token fine-grained punya izin tulis ke keduanya.
Sambungan dari instance ke laptop hanya ~240 KB/s, jadi artefak besar harus
lewat HuggingFace, bukan `scp`.

---
