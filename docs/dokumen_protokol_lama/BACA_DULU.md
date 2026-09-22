> **SUDAH DIGANTIKAN.** Dokumen ini menjelaskan protokol LAMA (sebelum revisi 10 Sep 2026).
> Untuk protokol yang berlaku, baca `../README_FP16.md`.
> Disimpan karena memuat analisis yang masih dipakai — bukan instruksi operasional.

# Hasil Eksperimen Fault Injection Llama-2-7B — Keluaran Mentah

Diperbarui 1 September 2026.

Ini menjawab permintaan pada rapat 16 Agustus: **membandingkan hasil mentah golden
dan faulty secara langsung, tanpa parsing A/B/C/D.**

---

## Isi paket

| Berkas | Isi |
|---|---|
| **`massal_pgteks_RUNS3.csv`** | **Data untuk analisis.** 5.184 baris, cakupan merata sempurna. **Pakai yang ini.** |
| `massal_pgteks_final.csv` | Seluruh 5.997 baris, termasuk yang belum lengkap. Data mentah, tanpa penilaian. |
| `massal_pgteks_final_dinilai.csv` | Sama + 11 kolom penilaian **usulan** (39 kolom). |
| `CONTOH_PERUBAHAN.txt` | 12 contoh nyata: soal, keluaran golden, keluaran faulty, lokasi kerusakan. |
| `PANDUAN_43K_v2.md` | Berapa lama dan berapa biaya untuk melanjutkan sampai 43.200 baris. |
| `RINGKASAN_final.txt` | Ringkasan cakupan. |

Kalau hanya sempat membuka satu berkas: **`CONTOH_PERUBAHAN.txt`**.

---

## Istilah

- **golden** — model dijalankan normal, tanpa kerusakan apa pun.
- **faulty** — model yang sama, soal yang sama, tapi satu bit dirusak di dalam
  perhitungan salah satu lapisan. Input-nya identik dengan golden.
- **decoder / lapisan** — Llama-2-7B punya 32 lapisan bertumpuk, bernomor 0–31.
- **bit** — satu angka disimpan sebagai 16 posisi 0/1. Bit 14 bagian eksponen:
  membaliknya mengalikan angka itu ~65.000 kali. Bit 0 hanya menggeser ~0,1%.
- **logit** — skor mentah model untuk suatu kata sebelum diubah jadi persentase.

---

## Bagaimana data ini dibuat

Model diberi **lima contoh soal beserta jawabannya**, lalu satu soal yang dinilai:

```
Which of the following is not a security exploit?
A. Eavesdropping
B. Cross-site scripting
C. Authentication
D. SQL Injection
Answer: C. Authentication          <- contoh, dijawab HURUF + TEKS OPSI

            ... empat contoh lain ...

Which of these is not a type of primate?
A. baboon
B. marmot
C. orangutan
D. chimpanzee
Answer:                            <- dikosongkan, model melanjutkan di sini
```

Karena contohnya dijawab **huruf + teks opsi**, model ikut menulis kalimat penuh,
bukan satu huruf. Itulah yang disimpan apa adanya di kolom `golden_teks` dan
`faulty_teks` — **tanpa dipotong, tanpa diparsing, tanpa dinilai.**

Soal dan prompt untuk golden dan faulty **sama persis**.

---

## Kelengkapan data

Total terkumpul **5.997 baris**, dengan **nol kegagalan**. Tapi tidak semuanya
tingkat pengulangannya lengkap:

| Pengulangan | Baris | Status |
|---|---|---|
| run 1 | 1.728 / 1.728 | **lengkap** |
| run 2 | 1.728 / 1.728 | **lengkap** |
| run 3 | 1.728 / 1.728 | **lengkap** |
| run 4 | 708 / 1.728 | sebagian |
| run 5 | 69 / 1.728 | sebagian |
| run 6 | 36 / 1.728 | sebagian |

**Untuk analisis, gunakan `massal_pgteks_RUNS3.csv`** — isinya hanya run 1–3, yaitu
**5.184 baris dengan cakupan merata sempurna**: setiap satu dari 288 kombinasi
mendapat tepat 3 pengulangan × 6 jenis kerusakan.

Baris run 4–6 tidak dibuang, tapi disimpan terpisah di `massal_pgteks_final.csv`.
Mencampurnya ke dalam analisis akan membuat sebagian kombinasi punya lebih banyak
sampel daripada yang lain, sehingga rata-ratanya condong.

### Cakupan subset bersih

| | |
|---|---|
| Baris | **5.184** |
| Kombinasi lapisan × operasi | **288 / 288** |
| Jenis kerusakan | 6, masing-masing tepat 864 baris |
| Soal unik | 833 |
| Gagal | **0** |

Akurasi golden **45,25%** (tebakan acak 25%, benchmark terpisah 43,8%). Ini
pemeriksaan kewarasan: model memang menjawab sungguhan, bukan mengoceh.

---

## Temuan utama

### 1. Melihat teks mentah menemukan hampir dua kali lipat

| Cara melihat | Terdeteksi berubah |
|---|---|
| Huruf A/B/C/D (cara lama) | 197 = **3,80%** |
| Teks mentah (cara baru) | 366 = **7,06%** |
| **Huruf SAMA tapi teks BEDA** | **185 = 3,57%** |

185 baris itu di dataset lama 43.120 baris seluruhnya terbaca "tidak ada error",
karena huruf pilihannya memang tidak berubah.

**Angka ini stabil.** Pada 3.456 baris hasilnya 7,06% dan 3,62%; kini dengan 1,5×
data lebih banyak hasilnya 7,06% dan 3,57%. Kesimpulannya tidak bergeser — tanda
temuannya kokoh, bukan kebetulan.

### 2. Peringkat jenis kerusakan tidak berubah

| Jenis kerusakan | teks beda (cara baru) | huruf beda (run lama) |
|---|---|---|
| RANDOM | **19,68%** | 8,76% |
| INPUT | 11,11% | 8,39% |
| RANDOM_BITFLIP | 4,98% | 2,49% |
| WEIGHT | 3,24% | 2,16% |
| INPUT16 | 2,55% | 0,75% |
| WEIGHT16 | 0,81% | 0,42% |

Urutannya **persis sama**, angkanya konsisten sekitar dua kali lipat. Jadi run lama
tidak salah — ia hanya **menghitung terlalu sedikit**.

### 3. Kerusakan punya empat wajah

Dari 366 baris yang berubah:

| Jenis | Jumlah | | Terlihat di cara lama? |
|---|---|---|---|
| Model mengoceh sampai batas token | 197 | 53,8% | tidak |
| Jawaban benar-benar berganti | 105 | 28,7% | **ya** |
| Huruf sama, teks beda | 57 | 15,6% | tidak |
| Model diam, tidak menjawab | 7 | 1,9% | tidak |

**Hanya 28,7% yang terlihat lewat huruf A/B/C/D.** Sisanya — 71,3% — adalah kerusakan
yang di cara lama terbaca sebagai "model menjawab normal".

Contoh dari masing-masing jenis ada di `CONTOH_PERUBAHAN.txt`. Yang paling
menggambarkan:

```
golden : 'A. pleasure exists for its own sake.'
faulty : 'A. Group A members would be more likely to persist in the occupation'
```

Model menyalin jawaban dari soal lain — kalimat itu berasal dari salah satu contoh
few-shot di bagian atas prompt, bukan dari soal ini. Hurufnya tetap A, jadi cara lama
melaporkannya sebagai tidak ada error.

---

## Kamus kolom — `massal_pgteks_RUNS3.csv`

### Identitas kerusakan
| Kolom | Arti |
|---|---|
| `decoder_idx` | lapisan yang dirusak, 0–31 |
| `operasi` | operasi perkalian mana di lapisan itu (9 jenis) |
| `Fault_Model` | jenis kerusakan: INPUT, WEIGHT, INPUT16, WEIGHT16, RANDOM, RANDOM_BITFLIP |
| `Bit_Position` | bit ke berapa yang dibalik, 0–15 |
| `rand_idx` | posisi angka mana di dalam tensor yang kena |
| `inject_seed` | benih acak — menjamin percobaan bisa diulang persis |
| `run_id` | pengulangan ke berapa (1–3 pada berkas ini) |

### Soal
| Kolom | Arti |
|---|---|
| `sample_id` | penanda soal, mis. `philosophy/test/253` |
| `subject`, `subject_category` | mata pelajaran dan rumpunnya |
| `answer_letter` | **kunci jawaban** yang benar |

### Keluaran — yang paling penting
| Kolom | Arti |
|---|---|
| **`golden_teks`** | **kalimat yang ditulis model normal, apa adanya** |
| **`faulty_teks`** | **kalimat yang ditulis model rusak, apa adanya** |

### Logit (sudut pandang kedua, gratis dari perhitungan yang sama)
| Kolom | Arti |
|---|---|
| `g_logit_A` … `g_logit_D` | skor golden untuk masing-masing huruf |
| `f_logit_A` … `f_logit_D` | skor faulty untuk masing-masing huruf |

### Kolom penilaian (usulan, belum final)
| Kolom | Arti |
|---|---|
| `golden_letter`, `faulty_letter` | huruf pemenang menurut logit |
| `is_error` | huruf berubah? (cara lama menilai) |
| `golden_benar`, `faulty_benar` | huruf itu cocok dengan kunci jawaban? |
| `golden_margin`, `faulty_margin` | jarak skor juara 1 ke juara 2. Kecil = ragu-ragu |
| `sama_persis` | teks golden dan faulty identik? |
| `kesamaan_karakter` | 0–1, berbasis jarak Levenshtein per karakter |
| `kesamaan_token` | 0–1, mencocokkan kata **per posisi** |
| `token_pertama_beda` | kata ke berapa mulai berbeda. −1 = tidak ada beda |

Kolom penilaian ini **usulan, bukan keputusan.** Semuanya dihitung dari CSV mentah
tanpa menjalankan model lagi, jadi kalau metodenya diganti cukup dihitung ulang —
tidak perlu menyewa GPU lagi.

---

## Yang perlu diputuskan: metrik mana yang jadi angka utama

Datanya menunjukkan dua kandidat itu **tidak setara**, dan salah satunya punya cacat
yang bisa menyesatkan.

- **`kesamaan_karakter`** — jarak Levenshtein per karakter. Levenshtein = jumlah
  minimum sisip/hapus/ganti karakter untuk mengubah satu teks jadi teks lain.
  Toleran terhadap pergeseran.
- **`kesamaan_token`** — mencocokkan kata **per posisi**: kata ke-1 lawan kata ke-1,
  ke-2 lawan ke-2. Satu kata hilang di depan membuat semua posisi berikutnya meleset.

Contoh nyata dari data ini:

```
golden : 'D. I, II, and III'
faulty : 'I, II, and III'
         kesamaan_karakter = 0,8235      kesamaan_token = 0,0000
```

Teksnya jelas mirip, tapi skor token **nol**. Penyebabnya: kerusakan membuat model
menjatuhkan awalan huruf `"D. "`, sehingga seluruh kata bergeser satu posisi.

Terjadi pada sebagian kecil baris, tapi **biasnya searah**: `kesamaan_token` secara
sistematis melaporkan kerusakan lebih parah daripada kenyataan, justru pada kasus
yang jawabannya sebenarnya masih utuh.

Catatan: `kesamaan_token` di sini **bukan** cosinus maupun Jaccard. Keduanya
mengabaikan urutan kata, sehingga tidak cocok untuk membandingkan jawaban.

---

## Catatan jujur tentang batasan

1. **42 baris (0,8%) keluaran golden-nya terpotong** di batas 48 kata — semuanya dari
   mata pelajaran yang teks opsinya sangat panjang (`high_school_european_history`,
   `professional_law`). Golden dan faulty dipotong di titik yang sama, jadi
   perbandingannya tetap sah. Bisa disaring lewat `n_tok_golden >= 48`.

2. **236 baris keluaran faulty menyentuh batas 48** — ini bukan kekurangan batas,
   melainkan gejala kerusakan: model kehilangan kemampuan berhenti.

3. **7 baris keluaran faulty kosong** — bukan bug. Token pertama yang dihasilkan
   langsung baris-baru: model terlalu rusak untuk menjawab. Logitnya ikut anjlok
   (~5 dibanding ~24 pada golden).

4. **Tiap kombinasi baru 3 sampel.** Cukup untuk kesimpulan agregat dan perbandingan
   antar jenis kerusakan, tipis untuk analisis per-lapisan yang rinci. Menambah
   sampai 25 sampel (43.200 baris) butuh ~34–45 jam GPU lagi — rinciannya di
   `PANDUAN_43K_v2.md`.

5. **Angka 7,06% bukan metrik resmi** — itu sekadar "teksnya tidak identik". Angka
   resminya menunggu keputusan metrik.