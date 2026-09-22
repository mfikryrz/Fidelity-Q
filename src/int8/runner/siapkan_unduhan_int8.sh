#!/usr/bin/env bash
# siapkan_unduhan_int8.sh — cron tiap 10 menit di instance.
#
# Begitu run INT8 benar-benar selesai, ia:
#   1. MELUCUTI baris penjaga_vram dari crontab   <- HARUS pertama
#   2. menghentikan pengawas + jalankan + worker
#   3. membuat arsip final + md5 + RINGKASAN
#   4. menaruh penanda /tmp/INT8_SIAP_UNDUH
#
# Urutan 1 sebelum 2 itu inti pelajaran dari FP16 (15 Sep). Di sana pengawas
# dihentikan lebih dulu, lalu cron penjaga menghidupkannya kembali 7 menit
# kemudian — karena logika penjaga cuma `baris < TARGET -> bangunkan`, dan
# run ini SELALU berakhir di bawah target akibat kegagalan permanen. Akibatnya
# ~34 jam mesin menyapu ulang untuk nol baris, dan penunggu unduhan tidak
# pernah memicu karena pengawas terus hidup lagi.
#
# DETEKSI SELESAI — dua sinyal, keduanya otoritatif:
#   A. baris >= TARGET
#   B. pengawas menulis "BERHENTI: putaran N tidak menambah baris apa pun"
#      di pengawas.log. Itu pernyataan pengawas sendiri bahwa satu putaran
#      penuh sudah mencoba ulang sisanya dan nol berhasil.
#
# Sinyal B dipakai alih-alih menebak dari `pgrep`, karena penjaga dan penunggu
# sama-sama jalan tiap 10 menit dan urutannya tidak dijamin — "pengawas mati"
# bisa terlewat total.
set -uo pipefail

ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=$ROOT/running_experiment_7b/hasil_massal/massal_int8_288.csv
TARGET=${TARGET:-27648}

PENANDA=/tmp/INT8_SIAP_UNDUH
ARSIP=/tmp/hasil_int8_final.tar.gz
LOG=$ROOT/logs/siap_unduh_int8.log
KUNCI=$ROOT/logs/.siap_unduh_int8.lock

exec 9>"$KUNCI"
flock -n 9 || exit 0

catat() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

[ -f "$PENANDA" ] && exit 0
[ -f "$OUT" ] || exit 0

N=$("$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null || echo 0)
N=${N:-0}
[ "$N" -eq 0 ] && exit 0

SELESAI=0; ALASAN=""
if [ "$N" -ge "$TARGET" ]; then
  SELESAI=1; ALASAN="baris $N >= target $TARGET"
elif grep -aq "BERHENTI: putaran .* tidak menambah baris" "$ROOT/logs/pengawas.log" 2>/dev/null; then
  SELESAI=1; ALASAN="pengawas mencatat BERHENTI (putaran penuh nol tambahan) di $N baris"
fi
[ "$SELESAI" -eq 0 ] && exit 0

catat "SELESAI: $ALASAN"

# ---- 1. LUCUTI penjaga lebih dulu, supaya tidak menghidupkan ulang
crontab -l > "$ROOT/crontab.sebelum_selesai.bak" 2>/dev/null || true
crontab -l 2>/dev/null | grep -v "penjaga_vram" | crontab - 2>/dev/null || true
catat "penjaga_vram dilucuti dari crontab (cadangan: crontab.sebelum_selesai.bak)"

# ---- 2. hentikan; KETIGA pola, jangan lupa jalankan_fp16.sh
for POLA in "pengawas_fp16[.]sh" "jalankan_fp16[.]sh" "mass_loglik_inject"; do
  for P in $(pgrep -f "$POLA" 2>/dev/null); do kill -TERM "$P" 2>/dev/null; done
done
sleep 10
for POLA in "pengawas_fp16[.]sh" "jalankan_fp16[.]sh" "mass_loglik_inject"; do
  for P in $(pgrep -f "$POLA" 2>/dev/null); do kill -9 "$P" 2>/dev/null; done
done
sleep 4
SISA_PROSES=$(pgrep -f "pengawas_fp16[.]sh|jalankan_fp16[.]sh|mass_loglik_inject" | wc -l)
catat "proses dihentikan, sisa=$SISA_PROSES"

# ---- 3. arsip. Salin ke /tmp dulu supaya tidak menangkap baris separuh ditulis
N=$("$PY" -c "import csv;print(sum(1 for _ in csv.DictReader(open('$OUT',newline='',encoding='utf-8'))))" 2>/dev/null || echo "$N")
D=/tmp/hasil_int8_final
rm -rf "$D"; mkdir -p "$D/hasil_massal" "$D/logs" "$D/skrip"
cp "$OUT" "$D/hasil_massal/"
cp -a "$ROOT/logs/." "$D/logs/" 2>/dev/null || true
cp "$ROOT"/*.sh "$D/skrip/" 2>/dev/null || true
cp "$ROOT/running_experiment_7b/mass_loglik_inject.py" "$D/skrip/" 2>/dev/null || true
cp "$ROOT/crontab.sebelum_selesai.bak" "$D/" 2>/dev/null || true

"$PY" - "$D/hasil_massal/massal_int8_288.csv" "$TARGET" > "$D/RINGKASAN.txt" 2>&1 <<'PY'
import csv, sys, collections
from datetime import datetime, timezone
f, T = sys.argv[1], int(sys.argv[2])
r = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
n = len(r)
k = [(x["decoder_idx"],x["operasi"],x["Fault_Model"],x["run_id"],x["Bit_Position"],x["idx"]) for x in r]
cfg = collections.Counter((int(x["decoder_idx"]), x["operasi"]) for x in r)
sel = collections.Counter((x["decoder_idx"],x["operasi"],x["Fault_Model"]) for x in r)
bit = collections.Counter(int(x["Bit_Position"]) for x in r)
print(f"HASIL AKHIR INT8 — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
print()
print(f"  baris            : {n:,} / {T:,}  ({100*n/T:.2f}%)")
print(f"  duplikat         : {len(k)-len(set(k))}")
print(f"  kolom            : {len(r[0])}")
print(f"  golden kosong    : {sum(1 for x in r if not x['golden_teks'].strip())}")
print(f"  decoder          : {len({int(x['decoder_idx']) for x in r})} / 32")
print(f"  operasi          : {len({x['operasi'] for x in r})} / 9")
print(f"  konfigurasi      : {len(cfg)} / 288 tersentuh, {sum(1 for v in cfg.values() if v==96)} penuh")
print(f"  sel (cfg x fault): {sum(1 for v in sel.values() if v==16)} / 1728 penuh")
print(f"  sebaran bit 0-7  : {[bit.get(b,0) for b in range(8)]}")
cacat = sorted([(kk,v) for kk,v in sel.items() if v!=16], key=lambda x:x[1])
print(f"  sel tidak penuh  : {len(cacat)}")
for kk,v in cacat[:15]:
    print(f"      d{kk[0]}/{kk[1]} {kk[2]}: {v}/16")
if len(cacat) > 15:
    print(f"      ... dan {len(cacat)-15} lagi")
PY
cat "$D/RINGKASAN.txt" >> "$LOG"

tar czf "$ARSIP" -C /tmp hasil_int8_final
md5sum "$ARSIP" | cut -d' ' -f1 > "$ARSIP.md5"
touch "$PENANDA"
catat "arsip siap: $ARSIP ($(du -h "$ARSIP" | cut -f1)) md5=$(cat "$ARSIP.md5")"
