#!/usr/bin/env bash
# ============================================================================
# penjaga_fp16.sh — watchdog. Dipanggil cron tiap 10 menit.
#
# pengawas_fp16.sh sudah lepas dari sshd (ppid=1), jadi selamat dari koneksi
# putus. Tapi ia tetap bisa mati sendiri: OOM killer, crash interpreter, atau
# instance di-restart. Kalau itu terjadi tengah malam, tidak ada yang tahu
# sampai paginya. Penjaga ini yang membangunkannya kembali.
#
# Idempoten dan aman dipanggil berulang: kalau pengawas masih hidup, ia keluar
# tanpa berbuat apa-apa.
#
# Pasang:
#   (crontab -l 2>/dev/null; echo "*/10 * * * * bash /workspace/fidelity/penjaga_fp16.sh") | crontab -
# ============================================================================
set -uo pipefail

ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=${OUT:-$ROOT/running_experiment_7b/hasil_massal/massal_fp16_288.csv}
LOGS=$ROOT/logs; mkdir -p "$LOGS"
LOG=$LOGS/penjaga.log
# Target dibaca dari lingkungan supaya penjaga yang sama bisa dipakai untuk
# FP16 (55.296) maupun INT8 (27.648 masal / 1.728 uji asap). Angka tetap di
# sini dulu membuat penjaga mengira run INT8 belum selesai dan menghidupkan
# pengawas terus-menerus.
TARGET=${TARGET:-55296}
MAKS_BANGUN=${MAKS_BANGUN:-20}   # berhenti membangunkan setelah sekian kali
HITUNG_FILE=$LOGS/.penjaga_bangun

catat() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

baris() {
  [ -f "$OUT" ] || { echo 0; return; }
  "$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null || echo 0
}

# Sudah selesai? jangan bangunkan apa-apa.
N=$(baris)
if [ "$N" -ge "$TARGET" ]; then
  grep -q "TUNTAS_DICATAT" "$LOG" 2>/dev/null || catat "TUNTAS_DICATAT target tercapai: $N/$TARGET"
  exit 0
fi

# Masih hidup? tidak perlu campur tangan.
# Pola 'bash pengawas' tidak cocok dengan baris perintah penjaga ini sendiri,
# jadi aman dari jebakan pgrep -f yang menemukan dirinya sendiri.
if pgrep -f "bash pengawas_fp16.sh" >/dev/null 2>&1; then
  exit 0
fi

# Pengawas mati padahal target belum tercapai.
#
# BAHAYA: worker python adalah anak dari pengawas. Kalau pengawas mati sendirian,
# worker jadi yatim tapi TETAP JALAN. Menghidupkan pengawas kedua saat itu
# menghasilkan DUA worker di satu GPU — konteks CUDA bergiliran bukan paralel,
# lajunya tidak bertambah, dan arena bentrok sampai baris hilang diam-diam.
# Jadi yatimnya dimatikan dulu. Tidak ada data hilang: tiap baris di-fsync saat
# ditulis, dan --resume melewati yang sudah ada.
# Rantainya TIGA tingkat, bukan dua:
#     pengawas_fp16.sh -> jalankan_fp16.sh -> python mass_loglik
# Versi pertama penjaga hanya membunuh python. Akibatnya jalankan_fp16.sh yang
# yatim tetap hidup, melanjutkan loop config-nya, dan menelurkan worker BARU —
# lalu pengawas yang baru dihidupkan menelurkan worker keduanya. Terbukti pada
# uji 13 Sep: dua worker di satu GPU sekaligus.
# Urutan penting: skrip dulu (supaya berhenti menelurkan), baru python-nya.
for POLA in "jalankan_fp16.sh" "mass_loglik"; do
  for Y in $(pgrep -f "$POLA" 2>/dev/null); do
    catat "yatim [$POLA] pid=$Y — dimatikan"
    kill -TERM "$Y" 2>/dev/null
  done
  sleep 8
  for Y in $(pgrep -f "$POLA" 2>/dev/null); do
    kill -9 "$Y" 2>/dev/null; catat "  pid=$Y perlu SIGKILL"
  done
  sleep 2
done

BANGUN=$(cat "$HITUNG_FILE" 2>/dev/null || echo 0)
if [ "$BANGUN" -ge "$MAKS_BANGUN" ]; then
  catat "MENYERAH: sudah $BANGUN kali membangunkan, masih $N/$TARGET. Perlu diperiksa manusia."
  exit 0
fi
BANGUN=$(( BANGUN + 1 )); echo "$BANGUN" > "$HITUNG_FILE"
catat "pengawas MATI di $N/$TARGET — membangunkan (ke-$BANGUN dari $MAKS_BANGUN)"

cd "$ROOT" || exit 1
setsid nohup bash pengawas_fp16.sh >> "$LOGS/pengawas.log" 2>&1 < /dev/null &
disown
sleep 5
if pgrep -f "bash pengawas_fp16.sh" >/dev/null 2>&1; then
  catat "pengawas hidup lagi"
else
  catat "GAGAL menghidupkan pengawas"
fi
