> **SUDAH DIGANTIKAN.** Dokumen ini menjelaskan protokol LAMA (sebelum revisi 10 Sep 2026).
> Untuk protokol yang berlaku, baca `../README_FP16.md`.
> Disimpan karena memuat analisis yang masih dipakai — bukan instruksi operasional.

# Berapa Lama dan Berapa Biaya untuk Mencapai 43.200 Baris

Disusun 1 September 2026, dari pengukuran nyata — bukan perkiraan di atas kertas.

---

## Ringkasan untuk yang buru-buru

| Pertanyaan | Jawaban |
|---|---|
| Satu baris butuh berapa lama? | **4,3 detik** di mesin yang kami pakai; bisa ~3,2 detik di mesin berdisk lebih cepat |
| 43.200 baris butuh berapa lama? | **34–45 jam**, tergantung kecepatan disk |
| Butuh saldo berapa? | **± $40** di mesin terbaik, **± $55** di mesin biasa |
| Sewa mesin seperti apa? | **1 kartu A100 80 GB, disk minimal 1,5 GB/detik** — makin cepat disknya makin murah totalnya |
| Kandidat terbaik saat ini | **1× A100 Czechia (m:144644)**, disk 4.463 MB/s, $0,956/jam |
| Jangan sewa apa? | **Mesin dengan banyak GPU** — sudah terbukti tidak menolong |
| Kesalahan yang mudah terjadi | memilih berdasarkan **tarif per jam**; yang benar adalah **biaya total**, dan itu ditentukan disk |

---

## 1. Satu baris butuh berapa lama?

**4,3 detik per baris**, atau sekitar 14 baris per menit.

### Bagaimana angka ini didapat

Caranya sederhana: catat jumlah baris, tunggu beberapa menit, catat lagi, lalu bagi
selisihnya dengan waktu yang berlalu. Misalnya:

```
Pukul 08:03  ->  4.255 baris
Pukul 08:25  ->  4.559 baris
Selisih      ->    304 baris dalam 22 menit (1.320 detik)
1.320 detik / 304 baris  =  4,3 detik per baris
```

Pengukuran ini diulang tiga kali dengan lama pengamatan berbeda, dan hasilnya
saling mendekati:

| Lama pengamatan | Baris terkumpul | Hasil |
|---|---|---|
| 8 menit | 112 baris | 4,3 detik/baris |
| 22 menit | 304 baris | 4,3 detik/baris |
| 2 jam 45 menit (mesin lain) | 3.456 baris | 2,9 detik/baris |

Yang 2,9 detik itu di mesin berbeda yang disknya lebih cepat. Untuk perencanaan
dipakai angka yang lebih hati-hati: **4,3 detik per baris**.

---

## 2. Kenapa selambat itu?

Satu "baris" terdengar sedikit, padahal isinya **dua kali menjalankan model dari
awal sampai akhir**: sekali dalam keadaan sehat, sekali dengan satu bit sengaja
dirusak.

### Contoh nyata dari data kita

Soal yang diberikan ke model:

```
Which of these is not a type of primate?
A. baboon
B. marmot
C. orangutan
D. chimpanzee
Answer:
```

Yang dihasilkan:

```
Model sehat (golden) : "A. baboon"          ->  6 kata
Model rusak (faulty) : "D. chimpanzee"      ->  7 kata
```

Sekilas cuma 13 kata. Tapi model bahasa tidak menulis kalimat sekaligus — ia menulis
**satu kata, lalu mengulang seluruh perhitungan untuk menentukan kata berikutnya.**
Dan sekali perhitungan berarti melewati **32 lapisan** model.

```
13 kata  x  32 lapisan  =  416 putaran perhitungan
```

### Dan soalnya sendiri jauh lebih panjang dari yang terlihat

Model tidak cuma diberi satu pertanyaan. Ia diberi **lima contoh soal beserta
jawabannya** lebih dulu, supaya ia paham format jawaban yang diharapkan. Baru
setelah itu soal yang sesungguhnya.

Jadi yang sebenarnya masuk ke model kira-kira seperti ini:

```
Which of the following is not a security exploit?
A. Eavesdropping
B. Cross-site scripting
C. Authentication
D. SQL Injection
Answer: C. Authentication                      <- contoh ke-1, sudah dijawab

The regional lymphatic drainage of the left side of the tip of the tongue is to the
A. left submental lymph node.
B. left and right submental lymph nodes.
C. left submandibular lymph node.
D. left and right submandibular lymph nodes.
Answer: B. left and right submental lymph nodes.    <- contoh ke-2

            ... contoh ke-3, ke-4, dan ke-5 ...

Which of these is not a type of primate?
A. baboon
B. marmot
C. orangutan
D. chimpanzee
Answer:                                        <- DIBIARKAN KOSONG
```

Bagian paling bawah itu yang penting: `Answer:` sengaja dikosongkan, dan di situlah
model melanjutkan menulis.

Seluruh teks di atas panjangnya sekitar **600 kata**, dan model harus membaca
semuanya lebih dulu sebelum menulis kata pertamanya. Itu terjadi **dua kali** untuk
tiap baris — sekali untuk model sehat, sekali untuk model rusak.

Jadi satu baris di tabel hasil = membaca 600 kata dua kali, ditambah ratusan putaran
perhitungan untuk menulis jawabannya.

### Kenapa contohnya dijawab lengkap, bukan cuma hurufnya

Perhatikan contoh di atas ditulis `Answer: C. Authentication`, bukan `Answer: C`.
Ini disengaja. Model meniru format contoh yang diberikan — kalau contohnya dijawab
satu huruf, model juga menjawab satu huruf saja. Dengan menjawab lengkap, model ikut
menulis kalimat penuh, dan **kalimat penuh itulah yang bisa dibandingkan** antara
model sehat dan model rusak.

Ini perubahan pokok dari eksperimen lama, yang hanya menyimpan huruf A/B/C/D.

### Yang membuatnya lebih lambat lagi: disk

Program ini menyimpan tiap lapisan model sebagai **berkas terpisah di harddisk**.
Setiap kali kerusakan disuntikkan ke satu lapisan, berkas lapisan itu ditulis ulang
ke disk, lalu dibaca lagi untuk dijalankan. Ukurannya sekitar 400 MB, dan ini terjadi
berulang kali.

Itu sebabnya **kecepatan disk lebih menentukan daripada kekuatan GPU.**

Buktinya jelas: selama eksperimen berjalan, kartu grafisnya hanya terpakai **12%**.
Delapan puluh delapan persen kekuatannya menganggur — bukan karena kurang kuat,
melainkan karena sedang menunggu disk selesai bekerja.

---

## 3. Kenapa harus 43.200 baris?

Karena eksperimen lama menghasilkan 43.120 baris dengan susunan yang sama persis.
Menyamakan jumlahnya membuat kedua dataset bisa dibandingkan langsung.

### Susunannya dari mana?

```
32 lapisan  x  9 titik per lapisan   =  288 titik kerusakan
288 titik   x  6 jenis kerusakan     =  1.728 kombinasi
1.728       x  25 pengulangan        =  43.200 baris
```

Bayangkan model ini seperti mesin bertingkat 32 lantai. Di tiap lantai ada 9 tempat
yang bisa dirusak. Tiap tempat bisa dirusak dengan 6 cara berbeda. Dan tiap kerusakan
diuji 25 kali dengan soal yang berbeda-beda, supaya hasilnya bukan kebetulan.

### Kenapa jumlahnya harus sama dengan run lama?

Supaya bisa dibandingkan **berpasangan** — istilah statistik untuk membandingkan dua
hal pada kondisi yang benar-benar sama.

Contohnya begini. Ambil satu baris yang sama persis dari kedua dataset:

```
Soal            : "Which of these is not a type of primate?"
Lapisan dirusak : lantai 2, titik self_attn_k_proj
Jenis kerusakan : WEIGHT, bit ke-14

Dataset LAMA (lihat huruf) : golden "A", faulty "D"  ->  BERBEDA, terhitung error
Dataset BARU (lihat teks)  : golden "A. baboon", faulty "D. chimpanzee"
```

Karena soal dan kerusakannya identik, kita bisa bertanya untuk **setiap baris**:
"apakah cara lama melewatkan kerusakan yang cara baru berhasil tangkap?"

Kalau jumlah dan susunannya tidak sama, pertanyaan itu tidak bisa dijawab — kita
hanya bisa membandingkan rata-rata secara kasar, bukan baris per baris. Dan
perbandingan baris per baris itulah bukti terkuat untuk klaim bahwa **cara lama
menghitung kerusakan terlalu sedikit.**

---

## 4. Apa yang sudah didapat sejauh ini

Sebelum membahas biaya melanjutkan, ini posisi datanya sekarang.

Terkumpul **5.997 baris dengan nol kegagalan**, tapi hanya sebagian yang tingkat
pengulangannya lengkap:

| Pengulangan | Baris | Status |
|---|---|---|
| run 1–3 | **5.184** | **lengkap, 288/288 kombinasi merata** |
| run 4 | 708 | sebagian |
| run 5–6 | 105 | sebagian |

Untuk analisis dipakai **5.184 baris yang lengkap** (`massal_pgteks_RUNS3.csv`).
Yang sebagian disimpan terpisah, tidak dicampur — kalau dicampur, sebagian kombinasi
punya lebih banyak sampel daripada yang lain dan rata-ratanya jadi condong.

### Temuan pada 5.184 baris tersebut

| Cara melihat | Terdeteksi berubah |
|---|---|
| Huruf A/B/C/D (cara lama) | 197 = **3,80%** |
| Teks mentah (cara baru) | 366 = **7,06%** |
| **Huruf SAMA tapi teks BEDA** | **185 = 3,57%** |

Akurasi golden **45,25%** (tebakan acak 25%, benchmark terpisah 43,8%) — model
memang menjawab sungguhan.

**Angkanya stabil.** Pada 3.456 baris hasilnya 7,06% dan 3,62%; kini dengan 1,5×
data hasilnya 7,06% dan 3,57%. Kesimpulan tidak bergeser meski datanya bertambah.

| Jenis kerusakan | teks beda (cara baru) | huruf beda (run lama) |
|---|---|---|
| RANDOM | **19,68%** | 8,76% |
| INPUT | 11,11% | 8,39% |
| RANDOM_BITFLIP | 4,98% | 2,49% |
| WEIGHT | 3,24% | 2,16% |
| INPUT16 | 2,55% | 0,75% |
| WEIGHT16 | 0,81% | 0,42% |

Peringkatnya **persis sama** dengan run lama, angkanya konsisten sekitar dua kali
lipat. Run lama tidak salah — ia hanya menghitung terlalu sedikit.

Rincian lengkap ada di `BACA_DULU.md`.

---

## 5. Hitungan waktu dan biaya untuk melanjutkan

Yang sudah ada: **5.184 baris** (3 pengulangan, lengkap dan merata).
Kekurangan menuju 43.200: **38.016 baris**.

```
38.016 baris  x  4,3 detik  =  163.469 detik  =  45 jam
```

| Pos | Waktu | Biaya (@ $1,10/jam) |
|---|---|---|
| Menyiapkan mesin | 10 menit | $0,18 |
| Menjalankan 38.016 baris | 45 jam | $49,50 |
| Cadangan 8% (mesin mati, ulang) | 4 jam | $4,40 |
| **Total** | **± 49 jam** | **± $55** |

### Kalau saldo tidak sampai segitu

Jumlah pengulangan boleh dikurangi. Cakupannya tetap penuh — semua 288 titik dan
semua 6 jenis kerusakan tetap diuji — hanya jumlah ulangannya yang lebih sedikit.

| Pengulangan | Baris | Waktu | Biaya | Ketelitian |
|---|---|---|---|---|
| 3 (**sudah ada**) | 5.184 | — | — | ± 0,7 poin |
| 6 | 10.368 | 6 jam | $7 | ± 0,5 poin |
| 10 | 17.280 | 14 jam | $16 | ± 0,4 poin |
| 15 | 25.920 | 25 jam | $28 | ± 0,3 poin |
| **25** | **43.200** | **45 jam** | **$51** | **± 0,24 poin** |

### Apa arti "ketelitian" itu

#### Mulai dari hal yang sudah biasa: jajak pendapat

Bayangkan ingin tahu berapa persen mahasiswa UGM yang suka kopi.

Kalau bertanya ke **30 orang** dan 40% menjawab suka — angka sebenarnya belum tentu
40%. Bisa jadi 33%, bisa jadi 47%. Kalau besok bertanya ke 30 orang lain, hasilnya
mungkin 36% atau 45%. Angkanya **bergoyang**.

Kalau bertanya ke **3.000 orang** dan tetap 40% — sekarang boleh yakin angka
sebenarnya memang dekat 40%. Bertanya ke 3.000 orang lain besok akan memberi hasil
yang mirip.

Makin banyak yang ditanya, makin kecil goyangannya. **"Ketelitian ± 0,4 poin"
artinya: goyangannya paling banyak 0,4 poin ke atas atau ke bawah.**

Eksperimen ini persis sama. Tiap baris ibarat satu orang yang ditanya. 5.184 baris
= bertanya 5.184 kali. 43.200 baris = bertanya 43.200 kali.

#### Bukti nyata dari data kita sendiri

Kita sudah punya buktinya, karena eksperimen ini sudah dihitung dua kali dengan
jumlah data berbeda:

| | 3.456 baris | 5.184 baris | Bergeser |
|---|---|---|---|
| Teks berbeda | 7,06% | 7,06% | **0,00 poin** |
| Huruf sama tapi teks beda | 3,62% | 3,57% | **0,05 poin** |

Datanya bertambah satu setengah kali lipat, tapi angkanya **hampir tidak bergerak**.
Itu tanda goyangannya memang kecil, dan kesimpulannya sudah bisa dipegang.

#### Yang benar-benar dibeli oleh ketelitian ekstra

Ketelitian tidak dibutuhkan untuk angkanya sendiri — dibutuhkan untuk **membandingkan
dua angka**. Pertanyaannya selalu: kalau A terlihat lebih besar dari B, apakah itu
sungguhan, atau cuma goyangan?

Aturannya sederhana: **kalau jarak antara dua angka lebih besar daripada goyangannya,
perbedaannya nyata. Kalau lebih kecil, kita belum bisa memastikan.**

Mari uji pada tiga pasangan nyata dari hasil kita:

**Pasangan 1 — RANDOM (19,68%) lawan INPUT (11,11%). Jaraknya 8,57 poin.**

| Pengulangan | Goyangan | Kesimpulan |
|---|---|---|
| 3 (yang kita punya) | ± 3,38 | jarak 8,57 jauh lebih besar → **RANDOM jelas lebih merusak** |
| 10 | ± 1,85 | sama saja, sudah pasti |
| 25 | ± 1,17 | sama saja, sudah pasti |

Jaraknya begitu lebar sehingga **data yang sekarang pun sudah cukup**. Menambah data
tidak mengubah apa pun di sini.

**Pasangan 2 — RANDOM_BITFLIP (4,98%) lawan WEIGHT (3,24%). Jaraknya 1,74 poin.**

| Pengulangan | Goyangan | Kesimpulan |
|---|---|---|
| 3 (yang kita punya) | ± 1,87 | goyangan **lebih besar** dari jarak → **belum bisa dipastikan** |
| 10 | ± 1,02 | jarak 1,74 lebih besar → **sudah bisa dipastikan** |
| 25 | ± 0,65 | makin mantap |

Di sini menambah data **benar-benar berguna**. Dengan data sekarang kita belum boleh
bilang RANDOM_BITFLIP lebih merusak daripada WEIGHT; dengan 10 pengulangan, boleh.

**Pasangan 3 — WEIGHT (3,24%) lawan INPUT16 (2,55%). Jaraknya 0,69 poin.**

| Pengulangan | Goyangan | Kesimpulan |
|---|---|---|
| 3 (yang kita punya) | ± 1,58 | belum bisa dipastikan |
| 10 | ± 0,87 | **masih belum** bisa dipastikan |
| 25 | ± 0,55 | jarak 0,69 lebih besar → **baru di sini bisa dipastikan** |

Inilah satu-satunya perbandingan yang **memang membutuhkan 25 pengulangan penuh**.

#### Ringkasnya

| Yang ingin dijawab | Butuh berapa pengulangan | Biaya |
|---|---|---|
| Jenis kerusakan mana yang paling parah (RANDOM) | **3 — sudah punya** | — |
| Membedakan jenis kerusakan yang mirip | **10** | ± $16 |
| Membedakan jenis kerusakan yang sangat mirip | **25** | ± $51 |
| Membandingkan baris per baris dengan run lama | **25** | ± $51 |

Jadi keputusannya bergantung pada seberapa halus pertanyaan yang mau dijawab.
Untuk kesimpulan besar — bahwa melihat teks mentah menemukan dua kali lebih banyak
kerusakan, dan bahwa RANDOM paling merusak — **data yang sekarang sudah cukup**.
Untuk membedakan pasangan yang berdekatan seperti WEIGHT dan INPUT16, barulah
43.200 baris diperlukan.

---

## 6. Mesin seperti apa yang harus disewa

| Syarat | Nilai | Kenapa |
|---|---|---|
| Jumlah GPU | **1** | GPU kedua tidak menolong sama sekali |
| VRAM per GPU | **80 GB** | kartu 40 GB terlalu mepet dan memicu kegagalan |
| **Disk** | **minimal 1,5 GB/detik** | **ini penentu kecepatan sebenarnya** |
| RAM | ≥ 64 GB | terpakai jauh di bawah itu, jangan bayar lebih |
| CPU | ≥ 16 inti | di atas 15 inti tidak menambah kecepatan |
| Durasi maksimum | **≥ 3 hari** | run 45 jam; jangan sampai mesin dibatasi 1 jam |
| Keandalan host | **≥ 98%** | run 2 hari, mesin mati di tengah mahal |
| Jenis sewa | **On-Demand** | jangan interruptible — bisa diputus di tengah |

Perkiraan harga wajar: **$1,00–1,20 per jam**.

### Kenapa jangan mesin banyak GPU

Ini pelajaran paling mahal. Kami menyewa mesin 2 GPU seharga $3,385/jam dengan alasan
yang terdengar masuk akal: karena GPU hanya terpakai 12%, tumpuk saja beberapa
pekerjaan sekaligus.

Hasilnya nol.

| Susunan | Kecepatan |
|---|---|
| 1 pekerjaan, 1 GPU | 2,9 detik/baris |
| 4 pekerjaan, 2 GPU | 3,0 detik/baris |

Empat pekerjaan **tidak lebih cepat dari satu**, dan 42% barisnya hilang karena
kehabisan memori. Sebabnya: dua pekerjaan di kartu yang sama tidak berjalan
bersamaan — mereka **bergiliran**. Kartu melayani satu dulu, baru yang lain. Jadi
masing-masing dapat separuh waktu, dan totalnya sama saja.

Perbandingan biaya untuk pekerjaan yang sama persis:

| Mesin | Tarif | Biaya per 1.000 baris |
|---|---|---|
| 2 GPU | $3,385/jam | $1,88 |
| **1 GPU** | **$1,10/jam** | **$0,87** |

**Satu GPU dua kali lebih murah.**

### Kalau mau lebih cepat, cari disk — bukan GPU

Bukti bahwa disk yang menentukan: dua mesin sama-sama A100 80 GB, tapi hasilnya beda
jauh karena disknya beda.

| Mesin | Disk | Kecepatan |
|---|---|---|
| Mesin lintasan 1 | cepat | **2,9 detik/baris** |
| Mesin Texas | 2.169 MB/detik | 4,3 detik/baris |

Selisihnya **48% lebih lambat**, padahal GPU-nya identik.

### Angka pasti untuk filter di Vast.ai

Di halaman pencarian Vast.ai, cari kotak **"Disk Bandwidth"** di panel filter sebelah
kiri (ada di bawah bagian *Machine Resources*). Geser batas bawahnya ke:

| Setelan | Nilai | Artinya |
|---|---|---|
| **Minimal wajib** | **1,5 GB/s** | di bawah ini terlalu lambat, hindari |
| **Disarankan** | **3 GB/s** | pilihan masih banyak, kecepatan baik |
| **Terbaik** | **10 GB/s ke atas** | biasanya mesin kelas atas |

Perhatian: **satuan di filter adalah GB/s, sedangkan di daftar mesin ditampilkan
MB/s.** 1 GB/s = 1.000 MB/s. Jadi kalau daftar menampilkan `2169 MB/s`, itu berarti
2,1 GB/s — sudah lewat batas minimal, tapi belum ideal.

Setelan filter lain yang perlu diubah:

| Filter | Setel ke |
|---|---|
| GPU Count | **1** |
| Per GPU RAM | **78 GB** (menyaring ke kartu 80 GB) |
| Disk Bandwidth | **minimal 1,5 GB/s** |
| CPU Cores | minimal 16 |
| CPU RAM | minimal 64 GB |
| Container Size | 80 GB |
| Max Instance Duration | **minimal 3 hari** |
| Host Reliability | **98%** |
| Unverified Machines | jangan dicentang |

---

## 7. Contoh mesin nyata dan perkiraan hasilnya

Ini mesin-mesin yang benar-benar muncul saat kami mencari, beserta perkiraan waktu
dan biaya untuk menyelesaikan 38.016 baris yang tersisa.

| Mesin | Disk | Tarif | Perkiraan waktu | Perkiraan biaya | Penilaian |
|---|---|---|---|---|---|
| **1× A100 Czechia** (m:144644) | **4.463 MB/s** | **$0,956/jam** | **~34 jam** | **± $32** | **PILIHAN TERBAIK** |
| 1× A100 Texas (host 639771) | 1.856 MB/s | $1,116/jam | 45 jam | $50 | terbukti dipakai, aman, tapi lebih mahal |
| 1× A100 Jepang (host 458442) | 430 MB/s | $1,048/jam | ~90 jam | $95 | **hindari** — tarif termurah tapi total termahal |
| 2× A100 Georgia (host 195410) | 12.663 MB/s | $3,386/jam | ~30 jam | $102 | disk bagus, tapi bayar 2 kartu untuk pekerjaan 1 kartu |
| 4× A100 (m:144579) | 7.419 MB/s | $6,423/jam | ~30 jam | $193 | jangan — bayar 4 kartu untuk pekerjaan 1 kartu |

### Yang menarik dari tabel ini

**Mesin dengan tarif per jam termurah justru paling mahal totalnya.** Yang Jepang
$1,048/jam terlihat paling hemat, tapi karena disknya 430 MB/detik — sepuluh kali
lebih lambat dari Czechia — pekerjaannya molor jadi 90 jam dan totalnya $95.

Jadi jangan memilih berdasarkan tarif per jam. **Pilih berdasarkan perkiraan biaya
total**, yaitu tarif dikalikan lama pengerjaan, dan lama pengerjaan ditentukan disk.

### Kenapa Czechia yang terbaik

| | Czechia | Texas |
|---|---|---|
| Disk | **4.463 MB/s** | 1.856 MB/s |
| Tarif | **$0,956/jam** | $1,116/jam |
| DLPerf per dolar | **134,8** | 107,4 |
| Durasi maksimum | **4 bulan** | 7 hari |

Disknya **2,4 kali lebih cepat sekaligus lebih murah**. Durasi 4 bulan juga
menghilangkan kekhawatiran mesin dipaksa berhenti di tengah pekerjaan 45 jam.

⚠️ Angka ~34 jam untuk Czechia adalah **perkiraan, belum diukur**. Kecepatan tidak
naik sebanding lurus dengan disk, karena sebagian waktu tetap dipakai menghitung.

**Cara amannya:** sewa, jalankan 15 menit, lalu ukur dengan cara di Bagian 1. Kalau
hasilnya di bawah 3,5 detik per baris, lanjutkan. Kalau ternyata tidak lebih cepat
dari Texas, kerugiannya cuma ± $0,25 dan bisa langsung pindah.

Cara mencari yang serupa: pasang filter di atas, urutkan dengan **DLPerf/$/hr
tertinggi**, lalu **periksa kolom disk pada tiap kandidat sebelum menyewa** — kolom
itu tidak ikut diperhitungkan dalam urutan DLPerf.

⚠️ Contoh mesin di atas adalah kondisi **1 September 2026**. Harga dan ketersediaan
di Vast.ai berubah tiap hari, jadi angkanya dipakai sebagai gambaran, bukan patokan.

---

## 8. Catatan teknis

Setelan program yang harus dipakai sudah didokumentasikan terpisah di
`README_REPRODUKSI.md` di dalam paket metode, lengkap dengan alasan tiap angkanya.
Setelan itu tidak perlu diubah — cukup disalin apa adanya.

---

## 9. Kesimpulan

Untuk 43.200 baris: **sewa mesin dengan 1 kartu A100 80 GB dan disk secepat mungkin.**
Kandidat terbaik saat ini adalah **1× A100 Czechia (m:144644), disk 4.463 MB/detik,
$0,956 per jam** — perkiraan selesai dalam **± 34 jam dengan biaya ± $32**.

Siapkan saldo **± $40** (sudah termasuk cadangan). Kalau memakai mesin berdisk lebih
lambat seperti Texas, siapkan **± $55**.

Kalau anggarannya lebih terbatas, **17.280 baris (10 pengulangan) dengan biaya ± $16**
sudah memberi kesimpulan yang praktis sama kuat untuk menganalisis pola kerusakan.
Yang hilang hanya kemampuan membandingkan baris per baris dengan run lama.