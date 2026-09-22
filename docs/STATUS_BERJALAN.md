# Status yang sedang berjalan — 15 September 2026, 13:00 UTC

Dokumen serah-terima untuk sesi berikutnya. Baca ini dulu, lalu
[`README_FP16.md`](README_FP16.md) dan [`README_INT8.md`](README_INT8.md).

> **FP16 SELESAI dan instance-nya dihapus.** Tinggal satu mesin berjalan: INT8
> di H200. Dua mesin INT8 A100 lama juga sudah dihapus pagi ini setelah backup.

---

## Satu mesin berjalan

| | **INT8 (H200)** |
|---|---|
| **SSH** | `-p 39801 root@ssh7.vast.ai` |
| **GPU** | H200 NVL, 143.771 MiB |
| **CPU** | EPYC 9755 (Zen 5) |
| **Baris** | **15.950 / 27.648 (57,7%)** per 18:17 UTC |
| **Config** | 0–287 (satu mesin, tidak di-shard) |
| **Berkas** | `massal_int8_288.csv` |
| **Setelan** | `ARENA=35`, **`PER_PROSES=1`** (diturunkan penjaga, lihat di bawah) |
| **Tarif** | $4,416/jam |
| **Sisa** | 11.698 baris |

```bash
ssh -p 39801 root@ssh7.vast.ai 'bash /workspace/fidelity/status.sh'
```

> **INSTANCE DI-STOP semalam 15→16 Sep.** Prosedur melanjutkannya ada di bagian
> "Menghidupkan lagi besok" di bawah. Backup penuh sudah di laptop.

### Penjaga menurunkan `PER_PROSES` dua kali — dan yang kedua patut ditinjau

```
[08:20:01Z] 67 error alokasi baru di 9148 baris  -> PER_PROSES 3 turun ke 2
[14:40:01Z] 60 error alokasi baru di 14071 baris -> PER_PROSES 2 turun ke 1
```

Penurunan **pertama gratis**: error anjlok (40 per 11 menit → 1 per 2,5 jam) dan
laju tidak berubah sama sekali (4,67 vs 4,74 dtk/baris), karena dengan error
lebih sedikit tiap konfigurasi menghasilkan lebih banyak baris untuk menanggung
satu muat model.

Penurunan **kedua tampaknya merugikan**. Terukur setelahnya: menyapu satu
konfigurasi makan **5 menit**, dibanding **7–8 detik** saat `PP=2` — karena di
`PP=1` tiap konfigurasi berlubang membayar muat model sendirian.

| | muat model | komputasi | total sisa 11.700 baris |
|---|---|---|---|
| `PP=1` (sekarang) | ~122× × 3,8 mnt = 7,7 j | 9,9 j | **~17,6 jam** |
| `PP=2` | ~61× × 3,8 mnt = 3,9 j | 9,9 j | **~13,8 jam** |

Selisihnya **~3,8 jam ≈ $17**. Sementara laju error kumulatif hanya
**131 dari ~8.000 baris = 1,6%** — di dalam pita yang `README_INT8.md` bagian 4
sebut eksplisit *"bukan alasan menurunkan setelan"*.

Pemicunya `AMBANG_ERR=50`, ambang yang dipilih untuk A100 dengan muat model 9,6
menit. Di H200 muat model 3,8 menit, jadi aritmetikanya berbeda dan ambang itu
terlalu ketat.

**Usul untuk besok:** saat menghidupkan lagi, setel `PP=2` di
`logs/.penjaga_vram.state` dan naikkan `AMBANG_ERR` ke 150 di crontab.
Penjagaan terhadap proses mati tetap utuh — yang berubah hanya ambang
penyetelan. `lanjutkan_int8.sh` sudah memakai `PER_PROSES=2` sebagai bawaan.

## Menghidupkan lagi besok

Instance di-**STOP**, bukan destroy, jadi disknya utuh dan `--resume` melanjutkan
dari 15.950 baris.

**Setelah instance START, jalankan sekali:**

```bash
ssh -p 39801 root@ssh7.vast.ai 'bash /workspace/fidelity/lanjutkan_int8.sh'
```

Skrip itu ada justru karena satu hal yang mudah terlewat: **daemon cron belum
tentu ikut hidup saat container dimulai ulang.** Kalau cron mati, tidak ada yang
menghidupkan pengawas, dan mesin diam sambil tetap ditagih $4,416/jam. Skrip ini
memeriksa berkas hasil, mencetak state penjaga, menghidupkan cron kalau perlu,
lalu menghidupkan pengawas — dan **idempoten**, aman dijalankan berkali-kali
(sudah diuji saat semuanya hidup: ia hanya melapor tanpa menyentuh apa pun).

Kalau mau sekalian mengembalikan `PER_PROSES=2` (lihat alasannya di atas),
**sebelum** menjalankan skrip di atas:

```bash
ssh -p 39801 root@ssh7.vast.ai \
  'printf "ARENA=35\nPP=2\nN_UBAH=0\nWM=0\n" > /workspace/fidelity/logs/.penjaga_vram.state'
```

Lalu naikkan ambangnya supaya tidak diratchet turun lagi:

```bash
ssh -p 39801 root@ssh7.vast.ai \
  'crontab -l | sed "s/AMBANG_ERR=50/AMBANG_ERR=150/" | crontab - && crontab -l'
```

Verifikasi sesudahnya: `bash /workspace/fidelity/status.sh`, dan pastikan
`pgrep -f mass_loglik | wc -l` bernilai **1** — bukan 0, bukan 2.

---

## FP16 — SELESAI

| | |
|---|---|
| **Baris** | **55.245 / 55.296 (99,91%)** |
| Konfigurasi | **288/288**, decoder **32/32**, operasi **9/9** |
| Sel penuh | 1.715 / 1.728 |
| Bit 0–15 | LENGKAP, sebaran rata 3.451–3.454 |
| Duplikat / golden kosong | **0 / 0** |
| Hasil | `zip_files/hasil_fp16_final.tar.gz` (md5 terverifikasi dua sisi) |

51 baris hilang **gagal permanen** — dibuktikan putaran kedua penuh yang
menambah nol baris. Rincian dan anomali `d28/mlp_up_proj_MatMul INPUT: 0/32`
ada di [`README_FP16.md`](README_FP16.md) bagian 2.

**Instance FP16 sudah dihapus.** Semua yang ada padanya tersimpan di laptop:
hasil masal, seluruh log, skrip, crontab, plus `zip_files/fp16_pelengkap.tar.gz` berisi
`hasil_vast/` (5 berkas benchmark 26 Agustus), `mmlu_pool.csv`, dan
`req_rtenv.txt`. Dua yang terakhir terbukti identik dengan salinan yang sudah
ada, jadi yang benar-benar baru hanya `hasil_vast/`.

---

## Pindah ke H200 — kenapa, dan apa hasilnya

### Kenapa bukan sekadar menunggu di A100

Dua masalah terukur di A100, keduanya **bukan** soal model GPU:

1. **70% waktu dinding habis memuat model.** Terukur dari jeda antar baris:
   52 dan 33 jeda ~9,9 menit. Penyebabnya `PER_PROSES=1` warisan FP16 plus
   penjaga yang meratchet arena 45→40→35→**30**. Sudah diperbaiki di tempat
   (lihat bagian berikutnya), tapi menyisakan pertanyaan kedua.
2. **~10% baris gagal alokasi.** Permintaan 67–324 MB gagal — sementara
   `nvidia-smi` menunjukkan kartu **terpaku di 78–81 dari 80 GB**. Arena ORT
   berlaku **per session**, dan sesi-sesinya sudah mentok di langit-langit
   kartu. Itu kehabisan kapasitas, bukan fragmentasi murni.

Mengganti instance dengan A100 lain tidak menolong — kartunya sama. Yang
menolong hanya kartu **lebih besar** (untuk masalah 2) dan CPU dengan
performa satu-utas lebih tinggi (untuk masalah 1, karena muat model itu
pekerjaan CPU satu-utas: worker memegang 90,9% dari **satu** core sementara
255 core menganggur).

### Hasil terukur, 450 baris pertama di H200 (1 jam)

| | A100 80 GB | H200 140 GB |
|---|---|---|
| Muat model | 9,6 menit | **3,7 menit** |
| Laju komputasi | 5,9–7,5 dtk/baris | **4,2 dtk/baris** |
| VRAM puncak | 78–81 / 80 GB (99%) | **124,7 / 143,8 GB (87%)** |
| Error alokasi | ~10% baris | **~1,5%** (16 dari 1.020 baris) |

Perbaikan 6–7×, dan hipotesis "kegagalan alokasi = kartu kehabisan ruang"
terdukung. Tapi **bukan nol**, dan cara mengukurnya penting.

> **Laju error INT8 bergerombol, jangan percaya sampel pendek.** Sampel 842
> baris pertama memberi 1 error (0,12%) dan sempat dilaporkan seolah masalahnya
> tuntas. Salah. Begitu run masuk decoder 5, muncul 15 error dalam ~150 baris —
> 8× `self_attn_k_proj_MatMul`, 7× `self_attn_MatMul_1`. Itu tanda tangan yang
> dijelaskan `README_INT8.md` bagian 4: operasi bertensor **terbesar** gagal
> lebih dulu saat arena terfragmentasi. Sampel yang melewati konfigurasi ringan
> akan selalu terlihat bersih. Ukur per-konfigurasi, bukan rata-rata bergulir.

Proyeksi cakupan akhir: **~98,5%**, kemungkinan lebih baik karena percobaan
ulang di H200 terbukti berhasil (konfigurasi penuh naik 30 → 45 → 50 dalam dua
jam). Bandingkan A100: ~90–96%. Jangan umumkan 100%.

> **VRAM: H200 juga jenuh, hanya lebih lama sampainya.** Terpantau memuai terus
> sepanjang run — 57 GB (04:46) → 124,7 GB (05:35) → **142,6 dari 143,8 GB
> (99,2%, 06:27)**. Sama seperti A100 yang terpaku di 78–81/80 GB. Kartu lebih
> besar **menunda** kejenuhan, tidak menghapusnya; yang berubah adalah berapa
> banyak konfigurasi sempat dikerjakan sebelum menyentuh dinding. Itulah
> mengapa laju errornya 1,5% dan bukan nol — dan mengapa ledakan error di
> decoder 5 muncul tepat saat kartu penuh.
>
> Sampel awal 57 GB sempat saya laporkan sebagai "87 GB menganggur". Itu salah;
> ia hanya belum panas.

**Tuas kalau laju error naik** (jangan dipakai tanpa bukti): turunkan
`PER_PROSES` supaya arena lebih sering direset, atau turunkan
`ORT_GPU_MEM_LIMIT_GB` supaya tiap sesi berplafon lebih rendah sehingga total
lebih lambat menyentuh langit-langit. Keduanya menukar kecepatan dengan
cakupan — kebalikan dari tuning kecepatan.

**Konsekuensi langsung: jangan naikkan `PER_PROSES` ke 6.** Untung kecepatannya
cuma ~2 jam (7%), sementara arena akan menumpuk enam konfigurasi alih-alih tiga
di kartu yang sudah terpakai 87%. Menukar laju error 0,2% — yang justru menjadi
alasan seluruh migrasi ini — demi 7% adalah tukaran yang buruk.

### Dua fase, dan kenapa ETA mentah menyesatkan

Laju efektif terukur sekarang **8,6 dtk/baris**, tapi itu diukur saat mesin
sedang **menambal lubang** di konfigurasi 0–77: tiap potongan membayar muat
model penuh hanya untuk beberapa baris yang hilang. Amortisasinya terburuk yang
mungkin.

| Fase | Baris | Laju efektif | Waktu |
|---|---|---|---|
| Menambal lubang (32 cfg) | 825 | ~7,2 dtk/baris | ~1,6 jam |
| Konfigurasi perawan (211 cfg) | 20.256 | **~5,0 dtk/baris** | ~28 jam |
| **Total** | **21.081** | | **~29,7 jam** |

Fase perawan mencakup **93% sisa pekerjaan** — di sana tiap muat model
menanggung 288 baris, bukan ~26. Angka `SISA` di `status.sh` (dan proyeksi
50 jam dari laju mentah) akan turun sendiri begitu mesin melewati config 77.

### Satu variabel sengaja TIDAK diubah

`ORT_GPU_MEM_LIMIT_GB` **tetap 35**, nilai yang terukur di
`README_INT8.md` bagian 4 — meski kartunya sekarang 140 GB. Alasannya: kalau
kartu **dan** arena berubah bersamaan lalu errornya hilang, tidak ada cara tahu
mana penyebabnya. Dengan hanya kartu yang berubah, hasilnya menjawab langsung.

Pertanyaannya sekarang sudah terjawab — dan jawabannya juga berarti **biarkan
saja**. Dengan VRAM puncak 87%, menaikkan arena atau `PER_PROSES` menukar laju
error 0,2% demi ~7% kecepatan. Setelan sekarang (`arena=35`, `PER_PROSES=3`)
adalah yang terbukti; jangan diutak-atik tanpa alasan baru.

---

## Tiga jebakan saat migrasi — semuanya sudah kena dan sudah ditangani

Catat ini kalau harus memindahkan lagi.

**1. Venv Python 3.10 mati di image baru.** `runtime_py310.tar.zst` hanya
berisi `fidelity/rtenv`; CPython 3.10 standalone-nya **tidak ikut**. Image vast
biasanya cuma punya 3.12, jadi `rtenv/bin/python3.10` jadi symlink putus.
Perbaikannya:

```bash
UV_PYTHON_INSTALL_DIR=/workspace/fidelity/.pythons uv python install 3.10
```

Path itu harus persis, karena `pyvenv.cfg` menunjuk ke sana.

**2. `bootstrap_from_hf.sh` gagal di dekompresi.** Ia memanggil
`zstd --decompress --force X --output Y`, tapi zstd v1.5.5 tidak mengenal
`--output` — hanya `-o`. Unduhan dan verifikasi SHA256 (39 berkas) sudah lolos
sebelum titik itu, jadi cukup lanjutkan loop dekompresinya dengan `-o`.

**3. `mass_loglik_inject.py` di bundel portable adalah versi LAMA.** Ia menolak
`--posisi-token penuh` (hanya `akhir`/`acak`). **Jangan sekadar ganti ke
`akhir`** — baris barunya tidak akan sebanding dengan data lama. Versi yang
benar ada di `src/int8/pipeline/mass_loglik_inject.py`
(md5 `840b96704ddd3e55515ac8d31b920141`); buktikan kecocokannya dengan
membandingkan `FIELDS` terhadap header CSV — harus identik persis, 34 kolom.

> **Dan satu kesalahan proses yang hampir mahal:** saat menghentikan run untuk
> menambal, membunuh `pengawas_fp16.sh` + `mass_loglik` **tidak cukup** —
> `jalankan_fp16.sh` selamat dan terus memunculkan worker baru, sehingga
> sempat ada **dua worker menulis satu CSV**. Selalu sertakan
> `jalankan_fp16.sh` dalam daftar yang dimatikan, lalu verifikasi
> `pgrep -f mass_loglik | wc -l` = 1.

---

## Yang sudah terverifikasi benar di H200

- `validate_int8_gate.py --require-configs` →
  `GATE_OK onnx=35 fake_quant=Div/Round/Mul pool_records=14042 configs=289`
- Config injeksi menunjuk tensor **fake-quant INT8** yang benar:
  `input_tensor=/mlp/down_proj/Round_output_0`,
  `weight_tensor=/mlp/down_proj/Round_1_output_0` — bukan initializer FP16.
  Ini jebakan paling berbahaya di `README_INT8.md` bagian 3, dan memakai bundel
  portable membuatnya beres sendiri.
- `memory_pool.py` bundel **sudah** mendaftarkan onnxruntime-extensions, jadi
  patch `RANDOM_BITFLIP` tidak perlu dipasang lagi.
- onnxruntime 1.20.2 + CUDAExecutionProvider, extensions 0.15.2, numpy 1.26.4.
- CSV 6.117 baris hasil gabungan A+B: 0 duplikat, 0 rusak, 34 kolom, header
  identik dengan `FIELDS`.

---

## Yang menjaga H200

Sama seperti sebelumnya, dengan perbaikan pagi ini sudah terpasang:

1. **`setsid nohup`** — pengawas ber-`ppid=1`, selamat dari koneksi putus
2. **`pengawas_fp16.sh`** — `MAKS_PUTARAN=12`, berhenti sendiri kalau satu
   putaran tidak menambah baris
3. **cron tiap 10 menit** — `penjaga_vram.sh` versi **tertambal**, dengan
   `ARENA_MIN=35` supaya arena tidak bisa diratchet ke bawah titik terukur

Perbaikan penjaga pagi ini (penting, jangan hilang): jalur `MENYERAH` dulu
`exit 0` **sebelum** blok restart, sementara watermark hanya maju **di dalam**
blok itu — sehingga sekali menyerah, penjaga berhenti menjaga selamanya tanpa
terlihat. Diganti `menyerah_tapi_jaga()`: pengawas hidup → majukan watermark
dan diam; pengawas mati → **tetap** restart. Detail di
[penjaga-vram-watchdog-bug](README_INT8.md).

---

## Hasil yang sudah aman di lokal

| Berkas | Isi |
|---|---|
| **`zip_files/hasil_fp16_final.tar.gz`** | **FP16 FINAL — 55.245 baris + log + skrip + RINGKASAN** |
| **`zip_files/fp16_pelengkap.tar.gz`** | `hasil_vast/` (5 berkas benchmark), `mmlu_pool.csv`, `req_rtenv.txt` |
| `experiments/backup_int8_migrasi_20260915/backup_int8_A.tar.gz` | INT8-A lama: 3.890 baris + log + skrip |
| `experiments/backup_int8_migrasi_20260915/backup_int8_B.tar.gz` | INT8-B lama: 2.227 baris + log + skrip |
| `experiments/backup_int8_migrasi_20260915/massal_int8_288_gabungan.csv` | 6.117 baris, dasar resume H200 |
| `experiments/fp16_smoke_v4_20260911/` | uji asap FP16 — 3.071/3.072 |
| `experiments/int8_smoke1728_20260914/` | uji asap INT8 — 1.646/1.728 |
| `src/fp16/`, `src/int8/` | seluruh kode, termasuk tambalan 15 Sep |
| `rahasia/hf_tokens.env` | token HuggingFace (mode 600) |

Semua md5 diverifikasi dua sisi saat pengunduhan.

**Ketiga instance A100 (FP16, INT8-A, INT8-B) sudah dihapus.** Tidak ada lagi
data di sana yang belum tersalin.

---

## Setelah selesai

### FP16 — SUDAH SELESAI, tidak ada yang perlu dikerjakan

Hasilnya sudah di laptop dan terverifikasi. Bagian ini disimpan hanya karena
**dua pelajaran mekanismenya berlaku untuk INT8 nanti**:

**1. Penunggu otomatis di instance, bukan di laptop.**
`siapkan_unduhan_fp16.sh` (ada di `src/fp16/runner/`) berjalan lewat cron di
mesin, membuat arsip + md5 + `RINGKASAN.txt` begitu run selesai, lalu menaruh
penanda. Kalau koneksi putus atau sesi berakhir, snapshot tetap terbentuk.
`src/fp16/runner/ambil_fp16.sh` di laptop hanya menarik dan memverifikasi. Pola yang sama layak
dipakai untuk INT8.

**2. "Selesai" tidak boleh berarti "baris = target".** FP16 berhenti di 55.245
dan tidak pernah menyentuh 55.296. Penunggu karena itu punya jalur kedua:
pengawas mati **dan** jumlah baris tidak berubah pada dua pemeriksaan
berturut-turut. Jalur itulah yang akhirnya dipakai.

> **JEBAKAN yang memakan waktu:** `penjaga_fp16.sh` menghidupkan ulang run yang
> sudah selesai, karena logikanya cuma `baris < TARGET → bangunkan`. Ia tidak
> punya pengertian "run menyimpulkan sisanya tidak terjangkau". Dengan
> `MAKS_BANGUN=20` itu berarti ~34 jam mesin menyapu ulang untuk **nol** baris,
> dan selama pengawas terus dihidupkan, penunggu unduhan tidak pernah memicu.
>
> Kalau run INT8 nanti berhenti di bawah 27.648 — dan hampir pasti begitu —
> **buang baris `penjaga_vram.sh` dari crontab lebih dulu**, baru hentikan
> pengawasnya. Kalau tidak, hal yang sama terulang.

### INT8

1. **Unduh terkompresi** — salin ke `/tmp` di instance dulu supaya tidak
   menangkap baris yang sedang ditulis, lalu verifikasi md5 kedua sisi.
2. **Tidak perlu menggabungkan INT8 lagi** — H200 menulis satu berkas untuk
   seluruh 288 konfigurasi. Tetap periksa nol duplikat.
3. **Jalankan verifikasi cakupan** — syarat eksplisit pembimbing:
   ```bash
   python src/fp16/analisis/cek_cakupan.py hasil.csv --bit 16 --runs 2 --penuh   # FP16
   python src/fp16/analisis/cek_cakupan.py hasil.csv --bit 8  --runs 2 --penuh   # INT8
   ```
4. **Hitung metrik di laptop**, tanpa GPU:
   ```bash
   python src/fp16/analisis/hitung_penilaian.py masukan.csv keluaran_dinilai.csv
   ```

---

## Dua hal yang masih perlu dibicarakan dengan pembimbing

**1. Fault model `RANDOM` memakai nilai tetap per graf.** `delta_init` dipanggil
sekali per pembangunan graf, jadi dalam satu (layer × operasi × fault model)
nilai rusaknya sama untuk semua baris — yang berubah hanya lokasinya. Detail
dan cara memperbaikinya ada di `src/int8/README.md`.

**2. INT8 di sini bukan INT8 sungguhan** — fake-INT8 dengan penyimpanan dan
eksekusi FP32. INT8 sejati sudah dicoba dan gagal, dan analisisnya menunjukkan
**gerbang 18/20-nya yang cacat**, bukan modelnya: FP16 yang belum dikuantisasi
sama sekali pun hanya mencapai 17/20. Lihat `README_INT8.md` bagian 1.
Pindah ke kartu lebih besar **tidak** menyentuh masalah ini.
