#!/usr/bin/env bash
# ambil_int8.sh — tarik snapshot progres INT8 terbaru ke laptop, lalu susun
# laporan yang siap dikirim ke pembimbing.
#
# Snapshot-nya disegarkan cron tiap 15 menit DI INSTANCE
# (`snapshot_int8.sh`), jadi skrip ini selalu mendapat yang terbaru tanpa
# mengganggu run. Aman diulang kapan saja.
#
#   bash ambil_int8.sh
set -uo pipefail

SSH_OPT=(-o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 55554)
HOST=root@93.91.156.87
TUJUAN=/home/fikry/hasil_pgteks/progres_int8
LAPORAN=/home/fikry/hasil_pgteks/PROGRES_INT8.txt

mkdir -p "$TUJUAN"

STAMP=$(timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/int8_progres.stamp 2>/dev/null' 2>/dev/null | tr -d '[:space:]')
if [ -z "$STAMP" ]; then
  echo "Snapshot belum ada di instance. Jalankan dulu:" >&2
  echo "  ssh ${SSH_OPT[*]} $HOST 'bash /workspace/fidelity/snapshot_int8.sh'" >&2
  exit 1
fi
echo "Snapshot di instance bertanggal: $STAMP"

MD5_JAUH=$(timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/int8_progres.tar.gz.md5' 2>/dev/null | tr -d '[:space:]')
timeout 600 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/int8_progres.tar.gz' 2>/dev/null > "$TUJUAN/int8_progres.tar.gz"

MD5_DEKAT=$(md5sum "$TUJUAN/int8_progres.tar.gz" | cut -d' ' -f1)
if [ "$MD5_JAUH" != "$MD5_DEKAT" ]; then
  echo "GAGAL: md5 tidak cocok ($MD5_JAUH vs $MD5_DEKAT) — unduhan rusak." >&2
  exit 2
fi
echo "md5 cocok dua sisi: $MD5_DEKAT"

rm -rf "$TUJUAN/int8_progres"
tar xzf "$TUJUAN/int8_progres.tar.gz" -C "$TUJUAN"
CSV="$TUJUAN/int8_progres/massal_int8_288.csv"

# ---- susun laporan
{
  echo "PROGRES EKSPERIMEN INT8 — Llama-2-7B fault injection"
  echo "Snapshot: $STAMP"
  echo "=================================================================="
  echo
  cat "$TUJUAN/int8_progres/RINGKASAN.txt"
  echo
  echo "=================================================================="
  echo "VERIFIKASI CAKUPAN (cek_cakupan.py --bit 8 --runs 2)"
  echo "=================================================================="
  echo
  python3 /home/fikry/hasil_pgteks/kode_fp16/analisis/cek_cakupan.py \
    "$CSV" --bit 8 --runs 2 2>&1 | head -45
  echo
  echo "=================================================================="
  echo "CARA MEMBACA ANGKA 'cacat' DI ATAS"
  echo "=================================================================="
  python3 - "$CSV" <<'PY'
import csv, sys, collections
r = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
FM = sorted({x["Fault_Model"] for x in r})
sel = collections.Counter((int(x["decoder_idx"]), x["operasi"], x["Fault_Model"]) for x in r)
cfg = {(int(x["decoder_idx"]), x["operasi"]) for x in r}
penuh = sum(1 for v in sel.values() if v == 16)
# sel yang ADA barisnya tapi belum 16, di konfigurasi yang sedang/sudah dikerjakan
sebagian = sum(1 for v in sel.values() if 0 < v < 16)
# sel yang NOL barisnya padahal konfigurasinya sudah tersentuh
kosong = sum(1 for c in cfg for fm in FM if (c[0], c[1], fm) not in sel)
print(f"  sel penuh (16/16)            : {penuh}")
print(f"  sel terisi sebagian (1-15)   : {sebagian}   <- gagal alokasi VRAM, ditambal tahap 2")
print(f"  sel belum tersentuh (0/16)   : {kosong}   <- BELUM DIKERJAKAN, bukan gagal")
print()
print(f"  Angka 'cacat' pada laporan di atas menjumlahkan dua kolom terakhir.")
print(f"  Yang benar-benar perlu diperhatikan hanya {sebagian} sel terisi sebagian.")
PY
  echo
  echo "=================================================================="
  echo "CATATAN"
  echo "=================================================================="
  echo "- Run masih BERJALAN; angka di atas potret sesaat, BUKAN hasil akhir."
  echo "- Target 27.648 baris = 288 konfigurasi x 6 fault model x 8 bit x 2 injeksi."
  echo "- 288 konfigurasi = 32 decoder x 9 operasi; decoder yang belum muncul"
  echo "  hanya berarti belum dijangkau sapuan, bukan terlewat."
  echo "- Baris di decoder 16-19 berasal dari mesin sebelumnya dan sudah digabung;"
  echo "  nol duplikat sudah diverifikasi."
  echo "- Sebaran bit rata (lihat 'sebaran bit 0-7') menandakan tidak ada bit"
  echo "  yang sistematis kekurangan data."
  echo "- INT8 di sini fake-INT8 (penyimpanan & eksekusi FP32); lihat README_INT8.md bagian 1."
} > "$LAPORAN"

echo
echo "=================================================================="
cat "$LAPORAN"
echo "=================================================================="
echo
echo "Laporan  : $LAPORAN"
echo "CSV penuh: $CSV"
