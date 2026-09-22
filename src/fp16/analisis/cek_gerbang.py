import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
n = len(rows)
def dmax(r):
    return max(abs(float(r["g_logit_"+L]) - float(r["f_logit_"+L])) for L in "ABCD")
ds = [dmax(r) for r in rows]
ident = sum(1 for d in ds if d == 0.0)
print("baris                :", n)
print("logit PERSIS SAMA    : %d/%d" % (ident, n))
print("|delta logit| median : %.4f" % st.median(ds))
print("|delta logit| maks   : %.4f" % max(ds))
print()
print("GERBANG 1:", ">>> LOLOS - injeksi masuk" if ident < n * 0.5 else ">>> GAGAL - injeksi tidak masuk")
print()
per = defaultdict(lambda: [0, 0])
for r in rows:
    per[r["Fault_Model"]][1] += 1
    per[r["Fault_Model"]][0] += (r["is_error"] == "True")
print("%-18s %7s %16s" % ("fault model", "SDC", "|dlogit| median"))
print("-" * 44)
for k in sorted(per, key=lambda x: -per[x][0] / max(1, per[x][1])):
    sub = [dmax(r) for r in rows if r["Fault_Model"] == k]
    e, t = per[k]
    print("%-18s %3d/%-3d %16.4f" % (k, e, t, st.median(sub)))
print()
dt = [float(r["detik"]) for r in rows]
print("detik/baris (forward): median %.3f | total %.1f s" % (st.median(dt), sum(dt)))
print("soal unik            : %d dari %d baris" % (len({r["sample_id"] for r in rows}), n))
print("  (berpasangan bekerja kalau jauh di bawah %d)" % n)
