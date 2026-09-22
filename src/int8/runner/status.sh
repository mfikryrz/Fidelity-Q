#!/usr/bin/env bash
# status.sh — ringkasan satu layar. Jalankan kapan saja tanpa mengganggu run.
#
# Dari laptop:
#   ssh -p PORT root@HOST 'bash /workspace/fidelity/status.sh'
set -uo pipefail
ROOT=/workspace/fidelity
PY=$ROOT/rtenv/bin/python
OUT=${OUT:-/workspace/fidelity/running_experiment_7b/hasil_massal/massal_int8_A_cfg0-143.csv}
TARGET=${TARGET:-13824}
PER_CFG=${PER_CFG:-96}   # 6 fault model x 8 bit x 2 run
N_CFG=${N_CFG:-144}    # konfigurasi yang jadi jatah mesin ini
AMBANG_DIAM=${AMBANG_DIAM:-1800}   # detik tanpa baris baru sebelum dicurigai

"$PY" - "$OUT" "$TARGET" "$PER_CFG" "$N_CFG" "$AMBANG_DIAM" <<'PY'
import csv, sys, collections, os
from datetime import datetime, timezone
f = sys.argv[1]; T = int(sys.argv[2])
PER = int(sys.argv[3]); NCFG = int(sys.argv[4]); AMBANG = int(sys.argv[5])
if not os.path.exists(f):
    print("CSV belum ada"); raise SystemExit
r = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
n = len(r)
ts = [datetime.fromisoformat(x["ts"]) for x in r]
span = (ts[-1] - ts[0]).total_seconds() or 1
laju = span / n
diam = (datetime.now() - ts[-1]).total_seconds()
laju_akhir = (ts[-1] - ts[-1000]).total_seconds() / 1000 if n > 1000 else laju
sisa = (T - n) * (laju_akhir if n < T else 0)
print(f"BARIS   : {n:,}/{T:,}  ({100*n/T:.2f}%)")
print(f"LAJU    : {laju:.2f} dtk/baris" + (f"   (1000 terakhir: {(ts[-1]-ts[-1000]).total_seconds()/1000:.2f})" if n > 1000 else ""))
print(f"SISA    : {sisa/3600:.1f} jam   perkiraan selesai {(datetime.now(timezone.utc).timestamp()+sisa)and datetime.fromtimestamp(datetime.now().timestamp()+sisa).strftime('%d %b %H:%M')}")
print(f"TERAKHIR: baris terakhir ditulis {diam/60:.1f} menit lalu"
      + ("   <-- MACET?" if diam > AMBANG else ("   (muat model, wajar)" if diam > 300 else "")))
cfg = collections.Counter((int(x["decoder_idx"]), x["operasi"]) for x in r)
print(f"CONFIG  : {len(cfg)} tersentuh, {sum(1 for v in cfg.values() if v==PER)} penuh / {NCFG}")
print(f"DECODER : {sorted({int(x['decoder_idx']) for x in r})}")
print(f"MUTU    : golden kosong {sum(1 for x in r if not x['golden_teks'].strip())}   "
      f"teks beda {100*sum(1 for x in r if x['golden_teks']!=x['faulty_teks'])/n:.2f}%")
PY

echo -n "PROSES  : "
pgrep -f "bash pengawas_fp16.sh" >/dev/null && echo -n "pengawas HIDUP" || echo -n "pengawas MATI"
pgrep -f "mass_loglik" >/dev/null && echo " | worker HIDUP" || echo " | worker mati"
echo "GPU     : $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "DISK    : $(df -h / | awk 'NR==2{print $3" / "$2" ("$5")"}')"
echo "ERROR   : $(cat $ROOT/logs/fp16.log 2>/dev/null | awk '/\[err\]/{n++} END{print n+0}') baris [err] di log"
echo "PENJAGA : $(cat $ROOT/logs/penjaga_vram.log 2>/dev/null | awk '/pengawas hidup lagi/{n++} END{print n+0}')x menghidupkan ulang pengawas"
tail -2 $ROOT/logs/penjaga_vram.log 2>/dev/null | sed 's/^/          /'
