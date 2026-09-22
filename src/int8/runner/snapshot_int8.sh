#!/usr/bin/env bash
# snapshot_int8.sh — menyegarkan snapshot progres INT8 di /tmp, dijalankan cron.
#
# Berjalan DI INSTANCE supaya snapshot tetap terbentuk walau koneksi putus.
# Aman dijalankan kapan saja: CSV disalin ke /tmp dulu, jadi tidak pernah
# menangkap baris yang sedang separuh ditulis.
set -uo pipefail

ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=$ROOT/running_experiment_7b/hasil_massal/massal_int8_288.csv
TARGET=27648

ARSIP=/tmp/int8_progres.tar.gz
KUNCI=$ROOT/logs/.snapshot_int8.lock

exec 9>"$KUNCI"
flock -n 9 || exit 0          # lewati kalau snapshot sebelumnya masih jalan

[ -f "$OUT" ] || exit 0

D=/tmp/int8_progres
rm -rf "$D"; mkdir -p "$D"

# salin dulu — jangan mengarsipkan berkas yang sedang ditulis
cp "$OUT" "$D/massal_int8_288.csv"
cp "$ROOT/logs/penjaga_vram.log" "$D/" 2>/dev/null || true
tail -c 2000000 "$ROOT/logs/fp16.log" > "$D/fp16_ekor.log" 2>/dev/null || true
tail -c 500000 "$ROOT/logs/pengawas.log" > "$D/pengawas_ekor.log" 2>/dev/null || true
cp "$ROOT/logs/.penjaga_vram.state" "$D/penjaga_vram.state" 2>/dev/null || true

"$PY" - "$D/massal_int8_288.csv" "$TARGET" > "$D/RINGKASAN.txt" 2>&1 <<'PY'
import csv, sys, collections
from datetime import datetime, timezone

f, T = sys.argv[1], int(sys.argv[2])
r = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
n = len(r)
ops = sorted({x["operasi"] for x in r})
c = collections.Counter((int(x["decoder_idx"]), x["operasi"]) for x in r)
k = [(x["decoder_idx"], x["operasi"], x["Fault_Model"],
      x["Bit_Position"], x["run_id"], x["idx"]) for x in r]
sel = collections.Counter((x["decoder_idx"], x["operasi"], x["Fault_Model"]) for x in r)

print(f"SNAPSHOT INT8  —  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
print()
print(f"  baris            : {n:,} / {T:,}  ({100*n/T:.2f}%)")
print(f"  duplikat         : {len(k)-len(set(k))}")
print(f"  kolom            : {len(r[0])}")
print(f"  golden kosong    : {sum(1 for x in r if not x['golden_teks'].strip())}")
print(f"  teks berbeda     : {100*sum(1 for x in r if x['golden_teks']!=x['faulty_teks'])/n:.2f}%")
print()
print(f"  decoder tercakup : {len({int(x['decoder_idx']) for x in r})} / 32")
print(f"  operasi tercakup : {len(ops)} / 9")
print(f"  konfigurasi      : {len(c)} / 288 tersentuh, {sum(1 for v in c.values() if v==96)} penuh")
print(f"  sel (cfg x fault): {len(sel)} / 1728 tersentuh, {sum(1 for v in sel.values() if v==16)} penuh")
print()
bit = collections.Counter(int(x["Bit_Position"]) for x in r)
print(f"  sebaran bit 0-7  : {[bit.get(b,0) for b in range(8)]}")
fm = collections.Counter(x["Fault_Model"] for x in r)
print(f"  per fault model  : " + ", ".join(f"{a}={b}" for a, b in sorted(fm.items())))
print()
ts = [datetime.fromisoformat(x["ts"]) for x in r if x.get("ts")]
if len(ts) > 1000:
    laju = (ts[-1] - ts[-1000]).total_seconds() / 1000
    print(f"  laju (1000 akhir): {laju:.2f} dtk/baris")
    print(f"  sisa             : {T-n:,} baris  ~{(T-n)*laju/3600:.1f} jam")
print(f"  baris terakhir   : {ts[-1]:%Y-%m-%d %H:%M} UTC")
PY

tar czf "$ARSIP" -C /tmp int8_progres
md5sum "$ARSIP" | cut -d' ' -f1 > "$ARSIP.md5"
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/int8_progres.stamp
