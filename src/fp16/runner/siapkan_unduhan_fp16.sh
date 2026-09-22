#!/usr/bin/env bash
# siapkan_unduhan_fp16.sh — dijalankan cron tiap 10 menit di mesin FP16.
#
# Begitu run selesai, ia membuat arsip final di /tmp lalu menaruh penanda.
# Sengaja berjalan DI INSTANCE, bukan dari laptop: kalau koneksi putus atau
# sesi berakhir, snapshot pada saat selesai tetap terbentuk.
#
# "Selesai" ada dua bentuk, keduanya harus ditangani:
#   (a) baris >= TARGET                          -> selesai penuh
#   (b) pengawas MATI dan baris tidak bertambah  -> berhenti dengan sisa gagal
#       permanen (FP16 punya laju error 0,7%, jadi mungkin tidak pernah
#       menyentuh 55.296 persis). Dua kali pemeriksaan berturut-turut supaya
#       pergantian potongan tidak salah dibaca sebagai berhenti.
set -uo pipefail

ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=$ROOT/running_experiment_7b/hasil_massal/massal_fp16_288.csv
TARGET=${TARGET:-55296}

PENANDA=/tmp/FP16_SIAP_UNDUH
ARSIP=/tmp/hasil_fp16_final.tar.gz
STATE=$ROOT/logs/.siap_unduh.state
LOG=$ROOT/logs/siap_unduh.log
KUNCI=$ROOT/logs/.siap_unduh.lock

exec 9>"$KUNCI"
flock -n 9 || exit 0

catat() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

# sudah pernah dibuat -> tidak ada yang perlu dikerjakan
[ -f "$PENANDA" ] && exit 0

baris() {
  [ -f "$OUT" ] || { echo 0; return; }
  local n
  n=$("$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null)
  echo "${n:-0}"
}

N=$(baris)
HIDUP=0
pgrep -f "pengawas_fp16[.]sh" >/dev/null 2>&1 && HIDUP=1

SEBELUM=0; DIAM=0
[ -f "$STATE" ] && . "$STATE"

SELESAI=0
ALASAN=""
if [ "$N" -ge "$TARGET" ]; then
  SELESAI=1; ALASAN="baris $N >= target $TARGET"
elif [ "$HIDUP" -eq 0 ] && [ "$N" -eq "$SEBELUM" ] && [ "$N" -gt 0 ]; then
  DIAM=$(( DIAM + 1 ))
  if [ "$DIAM" -ge 2 ]; then
    SELESAI=1; ALASAN="pengawas mati dan baris tetap $N pada 2 pemeriksaan"
  fi
else
  DIAM=0
fi
printf 'SEBELUM=%s\nDIAM=%s\n' "$N" "$DIAM" > "$STATE"

if [ "$SELESAI" -eq 0 ]; then
  exit 0
fi

catat "SELESAI terdeteksi: $ALASAN — membuat arsip"

# Salin ke /tmp dulu supaya tidak menangkap baris yang sedang ditulis.
D=/tmp/hasil_fp16_final
rm -rf "$D"; mkdir -p "$D/hasil_massal" "$D/logs" "$D/skrip"
cp -a "$ROOT/running_experiment_7b/hasil_massal/." "$D/hasil_massal/" 2>/dev/null
cp -a "$ROOT/logs/." "$D/logs/" 2>/dev/null
cp -a "$ROOT"/*.sh "$D/skrip/" 2>/dev/null
crontab -l > "$D/crontab.txt" 2>/dev/null

"$PY" - "$D/hasil_massal/massal_fp16_288.csv" > "$D/RINGKASAN.txt" 2>&1 <<'PY'
import csv, sys, collections
f = sys.argv[1]
r = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
k = [(x["decoder_idx"], x["operasi"], x["Fault_Model"], x["Bit_Position"], x["run_id"]) for x in r]
cfg = collections.Counter((x["decoder_idx"], x["operasi"]) for x in r)
print(f"baris          : {len(r)} / 55296 ({100*len(r)/55296:.2f}%)")
print(f"duplikat       : {len(k)-len(set(k))}")
print(f"kolom          : {len(r[0])}")
print(f"golden kosong  : {sum(1 for x in r if not x['golden_teks'].strip())}")
print(f"cfg tersentuh  : {len(cfg)} / 288")
print(f"cfg penuh (192): {sum(1 for v in cfg.values() if v == 192)} / 288")
print(f"decoder        : {sorted({int(x['decoder_idx']) for x in r})}")
PY

tar czf "$ARSIP" -C /tmp hasil_fp16_final
md5sum "$ARSIP" | cut -d' ' -f1 > "$ARSIP.md5"
touch "$PENANDA"

catat "arsip siap: $ARSIP ($(du -h "$ARSIP" | cut -f1)) md5=$(cat "$ARSIP.md5")"
catat "$(head -2 "$D/RINGKASAN.txt" | tr '\n' ' ')"
