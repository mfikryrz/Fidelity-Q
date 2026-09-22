import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
def dmax(r):
    return max(abs(float(r["g_logit_"+L]) - float(r["f_logit_"+L])) for L in "ABCD")
per = defaultdict(list)
for r in rows:
    per[int(r["Bit_Position"])].append(r)
print("%3s %8s %18s" % ("bit", "SDC", "|dlogit| median"))
print("-" * 34)
for b in sorted(per):
    rs = per[b]
    e = sum(1 for r in rs if r["is_error"] == "True")
    ds = [dmax(r) for r in rs]
    print("%3d %4d/%-3d %18.4f%s" % (b, e, len(rs), st.median(ds), "   <<<" if e else ""))
n = len(rows); e = sum(1 for r in rows if r["is_error"] == "True")
b14 = per.get(14, [])
e14 = sum(1 for r in b14 if r["is_error"] == "True")
lain = sum(1 for r in rows if int(r["Bit_Position"]) != 14 and r["is_error"] == "True")
print("-" * 34)
print("total SDC        : %d/%d = %.2f%%" % (e, n, 100 * e / n))
print("SDC di bit 14    : %d/%d" % (e14, len(b14)))
print("SDC di bit lain  : %d/%d" % (lain, n - len(b14)))
print()
ok = e14 >= len(b14) * 0.5 and lain == 0
print("GERBANG 2:", ">>> LOLOS - pola fp16 terkonfirmasi di 7B" if ok
      else ">>> PERIKSA - pola tidak seperti harapan")
