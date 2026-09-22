# Dokumen Protokol Lama — Arsip

**Semua isi folder ini menjelaskan protokol SEBELUM revisi 10 September 2026.
Untuk cara kerja yang berlaku sekarang, baca [`../README_FP16.md`](../README_FP16.md).**

Folder ini disimpan karena memuat **analisis dan rasional** yang masih dipakai —
bukan karena instruksi operasionalnya masih berlaku. Instruksi operasional di
sini sudah kedaluwarsa dan beberapa angkanya **salah** (lihat bagian bawah).

---

## Isi

| Berkas | Yang masih berguna |
|---|---|
| `BACA_DULU.md` | **Kamus kolom** `massal_pgteks_RUNS3.csv`, tiga temuan utama run lama, catatan batasan, diskusi metrik mana yang jadi angka utama |
| `PANDUAN_43K_v2.md` | Rasional **kenapa 43.200 baris** (32×9×6×25) dan penjelasan perbandingan berpasangan dengan dataset lama |
| `CONTOH_PERUBAHAN.txt` | 12 contoh nyata: soal, keluaran golden, keluaran faulty, lokasi kerusakan |
| `RINGKASAN.txt` | Cakupan lintasan 1 — 3.456 baris |
| `RINGKASAN_final.txt` | Cakupan run lama gabungan — 5.997 baris |

Data yang dirujuk dokumen-dokumen ini masih ada di direktori induk:
`massal_pgteks_RUNS3.csv`, `massal_pgteks_final.csv`, `massal_pgteks*.csv`.

---

## Yang berubah, dan mana yang sekarang salah

**Protokol** — tiga hal berubah pada 10 Sep 2026 atas permintaan pembimbing:
setiap baris kini menarik soal acak sendiri (dulu keenam fault model berbagi satu
soal), koordinat injeksi dicatat sebagai kolom, dan `Bit_Position` dijalankan
berurutan (dulu acak berseed). Akibatnya dataset lama dan baru **tidak sebanding
baris per baris**, meski keduanya sah masing-masing.

**Angka biaya di `PANDUAN_43K_v2.md` SALAH.** Dokumen itu menulis waktu membangun
satu graf injeksi = **0,33 detik**. Nilai sebenarnya, diukur dari run 11 Sep 2026,
adalah **~59 detik** — 180× lebih besar. Angka 0,33 berasal dari `asap.sh` yang
hasilnya keluar negatif dan skripnya sendiri memperingatkan "terlalu berisik",
lalu tetap dipakai. Memakainya membuat perkiraan anggaran meleset lebih dari dua
kali lipat.

**Setelan operasional di dokumen lama juga sudah tidak berlaku:**

| | Dokumen lama | Sekarang |
|---|---|---|
| Konfigurasi per proses | 8 | **1** |
| `ORT_GPU_MEM_LIMIT_GB` | 30 | **45** |
| `--posisi-token` | `akhir` | **`penuh`** |
| `--bits` | kosong (acak) | **`0-15`** |

Memakai setelan lama dengan protokol baru menyebabkan arena memori ONNX Runtime
habis di konfigurasi kedua: 595 baris gagal lalu segfault, dan datanya berlubang
tanpa peringatan. Ini sudah terjadi sekali pada 11 Sep 2026.

Yang **masih berlaku** dari dokumen lama: spesifikasi mesin sewaan (1 GPU,
80 GB VRAM, disk ≥1,5 GB/detik), pelajaran bahwa multi-GPU tidak menolong, dan
bahwa kecepatan disk lebih menentukan biaya total daripada kekuatan GPU.
Semuanya sudah disalin ke `../README_FP16.md`, jadi tidak perlu membaca ke sini
untuk itu.
