#!/usr/bin/env bash
# ambil_final_int8.sh — tarik HASIL AKHIR INT8 dari instance ke laptop.
#
# Arsipnya dibuat otomatis oleh `siapkan_unduhan_int8.sh` (cron tiap 10 menit
# DI INSTANCE) begitu run benar-benar selesai. Skrip ini hanya mengambil,
# memverifikasi md5 dua sisi, lalu menjalankan verifikasi cakupan.
#
# Aman diulang: kalau belum siap ia melapor progres dan keluar.
#
#   bash ambil_final_int8.sh          # cek + ambil kalau siap
#   bash ambil_final_int8.sh --tunggu # tunggu sampai siap, cek tiap 5 menit
set -uo pipefail

SSH_OPT=(-o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 55554)
HOST=root@93.91.156.87
TUJUAN=/home/fikry/hasil_pgteks/hasil_int8_final.tar.gz
EKSTRAK=/home/fikry/hasil_pgteks/hasil_int8_final
CEK=/home/fikry/hasil_pgteks/kode_fp16/analisis/cek_cakupan.py

siap() {
  timeout 60 ssh "${SSH_OPT[@]}" "$HOST" \
    'test -f /tmp/INT8_SIAP_UNDUH && echo SIAP || echo BELUM' 2>/dev/null | tail -1
}
progres() {
  timeout 60 ssh "${SSH_OPT[@]}" "$HOST" \
    'bash /workspace/fidelity/status.sh 2>/dev/null | head -3' 2>/dev/null \
    | grep -v "^Welcome\|^Have fun\|^AI agents"
}

if [ "${1:-}" = "--tunggu" ]; then
  while [ "$(siap)" != "SIAP" ]; do
    echo "[$(date -u +%H:%M:%SZ)] belum siap:"; progres | sed 's/^/    /'; sleep 300
  done
fi

if [ "$(siap)" != "SIAP" ]; then
  echo "Hasil akhir BELUM siap. Keadaan sekarang:"
  progres | sed 's/^/  /'
  echo
  echo "Jalankan lagi nanti, atau: bash ambil_final_int8.sh --tunggu"
  exit 1
fi

echo "Hasil akhir siap. Mengunduh..."
MD5_JAUH=$(timeout 60 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/hasil_int8_final.tar.gz.md5' 2>/dev/null | tr -d '[:space:]')
timeout 900 ssh "${SSH_OPT[@]}" "$HOST" 'cat /tmp/hasil_int8_final.tar.gz' 2>/dev/null > "$TUJUAN"
MD5_DEKAT=$(md5sum "$TUJUAN" | cut -d' ' -f1)
echo "  md5 instance : $MD5_JAUH"
echo "  md5 laptop   : $MD5_DEKAT"
[ "$MD5_JAUH" != "$MD5_DEKAT" ] && { echo "  GAGAL: md5 TIDAK COCOK — ulangi." >&2; exit 2; }
tar tzf "$TUJUAN" >/dev/null 2>&1 || { echo "  GAGAL: arsip tidak bisa dibuka" >&2; exit 2; }
echo "  md5 COCOK, arsip bisa dibuka."

rm -rf "$EKSTRAK"; tar xzf "$TUJUAN" -C /home/fikry/hasil_pgteks
CSV="$EKSTRAK/hasil_massal/massal_int8_288.csv"

echo
echo "=== RINGKASAN ==="
cat "$EKSTRAK/RINGKASAN.txt"
echo
echo "=== VERIFIKASI CAKUPAN (syarat pembimbing) ==="
python3 "$CEK" "$CSV" --bit 8 --runs 2 --penuh 2>&1 | head -40
echo
echo "Tersimpan : $TUJUAN"
echo "Diekstrak : $EKSTRAK"
echo "CSV       : $CSV"
echo
echo "Langkah berikutnya:"
echo "  python3 kode_fp16/analisis/hitung_penilaian.py \"$CSV\" hasil_int8_dinilai.csv"
