#!/usr/bin/env bash
# ambil_fp16.sh — tarik hasil akhir FP16 dari instance ke laptop.
#
# Aman diulang: kalau arsipnya belum siap ia cuma melapor dan keluar.
# Penunggu di instance (cron `siapkan_unduhan_fp16.sh`, tiap 10 menit) yang
# membuat arsipnya begitu run selesai — skrip ini hanya mengambil dan
# memverifikasi.
#
#   bash ambil_fp16.sh          # cek + ambil kalau sudah siap
#   bash ambil_fp16.sh --tunggu # tunggu sampai siap, cek tiap 5 menit
set -uo pipefail

SSH_OPT=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 18217)
HOST=root@118.163.199.123
TUJUAN=/home/fikry/hasil_pgteks/hasil_fp16_final.tar.gz

siap() {
  timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'test -f /tmp/FP16_SIAP_UNDUH && echo SIAP || echo BELUM' 2>/dev/null | tail -1
}

progres() {
  timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'bash /workspace/fidelity/status.sh 2>/dev/null | head -3' 2>/dev/null | grep -v "^Welcome\|^Have fun\|^AI agents"
}

if [ "${1:-}" = "--tunggu" ]; then
  while [ "$(siap)" != "SIAP" ]; do
    echo "[$(date -u +%H:%M:%SZ)] belum siap:"
    progres | sed 's/^/    /'
    sleep 300
  done
fi

if [ "$(siap)" != "SIAP" ]; then
  echo "Arsip BELUM siap. Keadaan sekarang:"
  progres | sed 's/^/  /'
  echo
  echo "Jalankan lagi nanti, atau: bash ambil_fp16.sh --tunggu"
  exit 1
fi

echo "Arsip siap. Mengunduh..."
MD5_JAUH=$(timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/hasil_fp16_final.tar.gz.md5' 2>/dev/null | tr -d '[:space:]')
timeout 900 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/hasil_fp16_final.tar.gz' 2>/dev/null > "$TUJUAN"

MD5_DEKAT=$(md5sum "$TUJUAN" | cut -d' ' -f1)
echo "  md5 instance : $MD5_JAUH"
echo "  md5 laptop   : $MD5_DEKAT"
if [ "$MD5_JAUH" != "$MD5_DEKAT" ]; then
  echo "  GAGAL: md5 TIDAK COCOK — unduhan rusak, ulangi." >&2
  exit 2
fi
tar tzf "$TUJUAN" >/dev/null 2>&1 || { echo "  GAGAL: arsip tidak bisa dibuka" >&2; exit 2; }
echo "  md5 COCOK, arsip bisa dibuka."
echo
echo "=== ringkasan hasil ==="
tar xzOf "$TUJUAN" hasil_fp16_final/RINGKASAN.txt 2>/dev/null | sed 's/^/  /'
echo
echo "Tersimpan: $TUJUAN"
echo "Ekstrak  : tar xzf $TUJUAN"
echo
echo "Langkah berikutnya (syarat pembimbing):"
echo "  python kode_fp16/analisis/cek_cakupan.py <csv> --bit 16 --runs 2 --penuh"
