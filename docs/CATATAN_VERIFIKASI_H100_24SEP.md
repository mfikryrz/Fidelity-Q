# Catatan verifikasi instance nyata — 24 September 2026

Verifikasi `setup_wizard.sh` di instance Vast.ai sungguhan (H100 NVL 95.8GB),
bukan simulasi/dry-run. Ditulis supaya temuan tidak hilang.

## Hasil

| | Status |
|---|---|
| **FP16** | **Berhasil penuh.** `run.sh --preset smoke` menghasilkan 82 baris CSV, 0 kosong, 8/17 baris tersampel menunjukkan `g_logit != f_logit` (bukti injeksi masuk). |
| **INT8** | **Tidak selesai — dihentikan atas permintaan user** saat ekspor ONNX di 34/35 berkas (lambat: kalibrasi SmoothQuant + kuantisasi 7B di CPU perlu ~15 menit, lalu ekspor ~15-20 menit lagi). Belum sampai `run.sh --preset smoke`. |

## Temuan — mana yang UNIVERSAL (sudah diperbaiki) vs KHUSUS SITUASI INI

### 1. Bug universal, SUDAH DIPERBAIKI: `bootstrap_vast.sh` exit 1 palsu

Baris terakhir skrip (`[ -f "$CODE/ringkas_vast.py" ] && ...`, dipakai tahap
90) jadi exit code SELURUH skrip kalau berkas itu tidak ada — yang normal
untuk run `10 40`/`10 30` (tahap 90 tidak ikut jalan). Akibatnya
`bootstrap_vast.sh` dilaporkan GAGAL padahal 35/35 ONNX sudah benar & sehat.
Diperbaiki: `src/fp16/runner/bootstrap_vast.sh` sekarang eksplisit `exit 0`
di akhir kalau semua tahap sebelumnya sukses.

**Dampak ke `setup_wizard.sh`**: kalau kamu jalankan wizard dan dia melapor
GAGAL tepat setelah baris `40c · Verifikasi 35 file bisa di-parse` sukses,
itu palsu (sudah tidak akan terjadi lagi setelah patch ini, tapi kalau
kamu pakai salinan lama, cek manual: `ls onnx/onnx_fp16/*.onnx | wc -l`).

### 2. BUKAN bug universal — spesifik ke cara pemulihan di sesi ini

Instance H100 ini awalnya gagal di tahap 30 (`bootstrap_vast.sh`) karena
repo kode default (`mfikryrz/llama2-7b-fidelity-onnx-fp16`) **privat** —
`HF_TOKEN` asli dibutuhkan, yang sengaja tidak pernah diisi otomatis (lihat
batasan penanganan kredensial). Sebagai pemulihan, kode dasar
(`FIdelity-ONNX-master` toolkit: `graph.py`, `inject_ops.py`,
`memory_pool.py`, `onnx_bitflip.so`) disalin manual dari cadangan
eksperimen lama di komputer lokal
(`experiments/int8_instance_49970225_true_smoke_20260905/code/`).

**Konsekuensi**: berkas-berkas dasar itu **tidak ikut ter-`git clone` dari
GitHub sama sekali** — repo publik ini cuma berisi `src/*/repo_terpatch/`
(berkas yang KAMI timpa), bukan seluruh toolkit `FIdelity-ONNX-master`.
Siapa pun yang clone repo ini dan tidak punya akses ke repo HF privat
tersebut (atau cadangan serupa) akan mentok di tahap yang sama persis.
**Ini gap nyata di framework, belum diperbaiki** — lihat bagian "Belum
diperbaiki" di bawah.

Karena toolkit dasarnya dipinjam dari snapshot eksperimen LAIN (cabang
"true INT8 + TensorRT", beda dari pendekatan fake-INT8/SmoothQuant yang
dipakai framework ini), dua hal berikut kemungkinan **cuma berlaku untuk
salinan `memory_pool.py`/`onnx_bitflip.so` spesifik itu**, bukan jaminan
berlaku di toolkit yang benar/privat:

- **`onnx_bitflip.so` minta `onnxruntime` versi tertentu lewat symbol
  version yang KETAT** (bukan sekadar ABI kompatibel). Tiga salinan `.so`
  berbeda yang ditemukan di berbagai cadangan lokal semuanya minta
  `VERS_1.20.1` (versi yang **tidak pernah dirilis publik** di PyPI —
  loncat dari 1.20.0 ke 1.20.2) kecuali satu yang minta `VERS_1.22.0`.
  Solusi yang berhasil: upgrade `onnxruntime-gpu` ke **`1.22.0`** persis
  (bukan `1.20.2` yang didokumentasikan `req_rtenv.txt`), lalu perbaiki
  symlink `libonnxruntime.so.1` (pip tidak selalu membuatnya otomatis).
- **`memory_pool.py` (dari snapshot ini) mencoba `TensorrtExecutionProvider`
  duluan secara default**, gagal karena `libnvinfer.so.10` tidak
  terpasang, lalu diam-diam *fallback* ke **CPU murni** berkali-kali
  (bukan CUDA) — jauh lebih lambat. Ada *env var* resmi untuk mematikan
  ini: `FIDELITY_USE_TRT=0` (dibaca langsung oleh `memory_pool.py`,
  bukan tambalan kami). **Belum diverifikasi apakah toolkit privat yang
  benar juga punya masalah default TRT ini** — mungkin sudah beda kalau
  TensorRT memang terpasang di image aslinya.

### 3. `injection_llm/` dipakai bersama FP16 & INT8 — bahaya kalau satu server dipakai dua-duanya

`$CODE` (`running_experiment_7b/`) sama untuk kedua presisi. Config
`injection_llm/*.json` yang di-generate `setup.sh` untuk FP16 (target
`/self_attn/...`) **beda tensor** dari yang INT8 butuhkan (target
`/mlp/down_proj/...`) — tapi JUMLAH filenya sama (288), sehingga
pengecekan "sudah lengkap" di `setup.sh` **tidak bisa membedakan
keduanya** dan akan salah pakai config FP16 untuk INT8 kalau FP16
dijalankan lebih dulu di server yang sama.

**Belum diperbaiki di kode** — kalau mau jalankan FP16 lalu INT8 (atau
sebaliknya) di SATU server yang sama, hapus manual
`repo/FIdelity-ONNX-master/injection_llm/*.json` sebelum ganti presisi,
baru jalankan `setup.sh` lagi supaya di-generate ulang untuk presisi yang
benar.

## Instruksi yang TERBUKTI jalan di instance ini (bukan simulasi)

```bash
# 1. clone (repo publik)
git clone https://github.com/mfikryrz/Fidelity-Q.git /workspace/fidelity_repo
cd /workspace/fidelity_repo

# 2. toolkit dasar TIDAK ada di repo publik -- salin dari cadangan yang
#    punya FIdelity-ONNX-master lengkap (ganti path sesuai cadanganmu)
#    dari KOMPUTER LOKAL (bukan di server):
rsync -a --exclude='injection_llm/' --exclude='injection_llm_smoke/' --exclude='__pycache__/' \
  -e "ssh -p PORT" \
  /path/ke/cadangan/code/FIdelity-ONNX-master/ \
  root@HOST:/workspace/fidelity/running_experiment_7b/repo/FIdelity-ONNX-master/
rsync -a -e "ssh -p PORT" \
  /path/ke/cadangan/code/mmlu_loglik_bench.py \
  /path/ke/cadangan/code/01_phase1_export.py \
  root@HOST:/workspace/fidelity/running_experiment_7b/
rsync -a -e "ssh -p PORT" \
  /path/ke/hasil_pgteks/results/fp16/pelengkap/work/mmlu_pool.csv \
  root@HOST:/workspace/fidelity/running_experiment_7b/work/mmlu_pool.csv

# 3. di SERVER: bootstrap (skip tahap 40 kalau ONNX FP16 mau diekspor lokal,
#    HF_TOKEN boleh dummy krn stage 30 dilewati -- toolkit sudah ada manual)
cd /workspace/fidelity_repo/src/fp16
export HF_TOKEN=dummy_boleh_apa_saja
bash runner/bootstrap_vast.sh 10 40

# 4. KALAU muncul "Failed to load library .../onnx_bitflip.so" dgn pesan
#    versi (mis. "version VERS_1.22.0 not found"):
/workspace/fidelity/rtenv/bin/pip install -q onnxruntime-gpu==1.22.0   # cocokkan ke versi yg diminta .so kamu
cd /workspace/fidelity/rtenv/lib/python3.10/site-packages/onnxruntime/capi/
ln -sf $(ls libonnxruntime.so.1.*) libonnxruntime.so.1
ln -sf $(ls libonnxruntime.so.1.*) libonnxruntime.so

# 5. setup + jalankan, WAJIB set FIDELITY_USE_TRT=0 kalau TensorRT tidak terpasang
cd /workspace/fidelity_repo/src/fp16
ROOT=/workspace/fidelity ./setup.sh
ROOT=/workspace/fidelity FIDELITY_USE_TRT=0 ./run.sh --preset smoke

# --- untuk INT8 di server YANG SAMA, sebelum lanjut: ---
rm -f /workspace/fidelity/running_experiment_7b/repo/FIdelity-ONNX-master/injection_llm/*.json
cd /workspace/fidelity_repo/src/int8
ROOT=/workspace/fidelity ./setup.sh
"$ROOT/expenv/bin/python" ".../export_llama_int8.py" --repo ... --model_id ... --out .../onnx_int8
ROOT=/workspace/fidelity FIDELITY_USE_TRT=0 ./run.sh --preset smoke
```

## Belum diperbaiki (di luar cakupan verifikasi ini)

1. **Toolkit dasar `FIdelity-ONNX-master` tidak ikut ter-`git clone`** —
   siapa pun tanpa akses repo HF privat mentok di tahap 30. Perlu diputuskan:
   apakah toolkit itu boleh ikut di-commit ke repo publik ini (perlu cek
   lisensi/ukuran), atau didokumentasikan eksplisit sebagai prasyarat
   terpisah.
2. **`patch_int8_memory_pool.sh` tidak cocok** dengan `memory_pool.py` dari
   toolkit yang dipakai verifikasi ini (pola kode beda) — perlu dicek ulang
   terhadap toolkit yang SEBENARNYA privat, bukan yang dipinjam.
3. **`injection_llm/` konflik FP16 vs INT8** di satu `$CODE` (lihat bagian 3
   di atas) — belum ada penanganan otomatis di `setup.sh`.
