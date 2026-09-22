# Screenshot — Whiteboard Excalidraw (screen share Gabriel Kaunang)

**Konteks:** Zoom meeting, Gabriel Kaunang share screen (Firefox → excalidraw.com).
Timestamp di layar Gabriel: **Thu 10 Sep 19.00**. Screenshot diambil **11 Sep 2026, 09:00**.
Peserta di panel Zoom: **Fikry** (kamera off) dan **Gabriel Kaunang** (kamera on).

---

## Isi whiteboard — 3 kolom perbandingan

| | **total buat paper** | **total buat smoke test (INT8)** | **total buat smoke test (INT8)** |
|---|---|---|---|
| **Layer** | 288 layer | 16 layer → mau ambil random, atau 16 pertama gpp | 8 layer → mau ambil random, atau 16 pertama gpp |
| **Fault model** | 6 fault model | 6 fault model | 6 fault model |
| **Bit** | 8 bit yang akan diinject | 8 bit yang akan diinject | 16 bit yang akan diinject |
| **Repetisi** | 2 kali diinject | 2 kali diinject | 2 kali diinject |

### Catatan pembacaan (dikoreksi dari audio rapat)

Tulisan di whiteboard **tidak sepenuhnya sama** dengan kesepakatan lisan. Dua koreksi penting
berdasarkan [transkrip_rapat.txt](transkrip_rapat.txt):

1. **Kolom 3 sebenarnya FP16, bukan INT8.** Judulnya salah tulis (copy-paste dari kolom 2).
   Lihat transkrip `0:51:46–0:52:04`: *"supaya murah begini aja. Ini INT 8 kan? Ini INT 8.
   Nanti buat FP 16."* — Gabriel sedang membuat kolom FP16 saat menulis kolom ketiga.
   Konsisten juga dengan jumlah bit: 8 bit untuk INT8, 16 bit untuk FP16.

2. **Kolom 3 disepakati 16 layer, bukan 8.** Angka 8 di whiteboard sempat ditulis, lalu
   dibatalkan dalam pembicaraan (`0:52:29–0:52:36`):
   Gabriel: *"atau kalau 16 layer disini apa menurut kamu?"* →
   Fikry: *"ya 16 aja biar kelihatan perbandingannya"* → Gabriel: *"Oke, oke."*
   Jadi tulisan "16 pertama" di kolom 3 justru yang benar, dan "8 layer" yang basi.

### Konfigurasi final (setelah koreksi)

| | **paper (full)** | **smoke test INT8** | **smoke test FP16** |
|---|---|---|---|
| Layer | 288 | 16 | 16 |
| Fault model | 6 | 6 | 6 |
| Bit | 8 | 8 | 16 |
| Injeksi per bit | 2 | 2 | 2 |

- **288 layer = 32 layer × 9 operasi** (`0:42:53–0:43:04`). Setiap operasi dihitung sebagai
  satu "layer" tersendiri supaya tidak membingungkan.
- Pemilihan 16 layer: **boleh random**, tapi harus **stratified** (`0:49:38–0:49:52`).
- Untuk paper aslinya 5 injeksi per bit, diturunkan jadi 2 supaya jumlah eksperimen terkendali
  (`0:46:42–0:46:50`).

---

## Konteks lain yang terlihat di screenshot

**Tab Firefox (layar Gabriel):** Calm and Relaxing · The University of ... · Inbox - Gabriel Ka... ·
Google Maps · mobilenet_cifar10 · **Excalidraw Whiteboard** (aktif) · multihead.png (PNG I...) ·
Paper Results - Fi...

**Bookmark bar Gabriel:** Interesting · Grad School · **LLM Fault Injection** · Others · Thinkpad ·
Quantum · 19th IEEE MCSoC 2... · Preface - Hello Algo · ScholarOne Manusc... · uchicago-cmsc143... ·
2026-01-26: Items f...

**Tab Chrome (layar Fikry):** (84) WhatsApp · Gabriel Kaunang's Zoom · Web Conferencing | Uni... ·
Universitas Gadjah Mad...

---

---

**File gambar:** [whiteboard_gabriel_20260910.png](whiteboard_gabriel_20260910.png)
(salinan dari `~/Pictures/Screenshots/Screenshot_2026-09-11_09-00-21.png`, 2560×1440, 526 KB)

**Transkrip audio rapat:** [transkrip_rapat.txt](transkrip_rapat.txt) · [transkrip_rapat.srt](transkrip_rapat.srt)
