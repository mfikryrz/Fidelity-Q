# Panduan Cepat

## 1. Varian FP16 — bootstrap penuh

```bash
cd src/fp16
export HF_TOKEN=hf_xxxxxxxxxxxxx

bash runner/bootstrap_vast.sh 10 40

ROOT=/workspace/fidelity
mkdir -p "$ROOT/running_experiment_7b/work"
gunzip -c "$ROOT/_dl/code/mmlu_pool.csv.gz" > "$ROOT/running_experiment_7b/work/mmlu_pool.csv"

cp config/fp16.env.example config/fp16.env
./setup.sh

./run.sh --preset smoke
./run.sh --preset full
./run.sh --status
```

---

## 2. Varian INT8 — bootstrap penuh

```bash
cd src/int8
export HF_TOKEN=hf_xxxxxxxxxxxxx

bash ../fp16/runner/bootstrap_vast.sh 10 30

ROOT=/workspace/fidelity
CODE=$ROOT/running_experiment_7b
INT8D=$CODE/onnx/onnx_int8
mkdir -p "$INT8D" "$CODE/work"

gunzip -c "$ROOT/_dl/code/mmlu_pool.csv.gz" > "$CODE/work/mmlu_pool.csv"

command -v zstd >/dev/null || apt-get install -y zstd

"$ROOT/rtenv/bin/hf" download mfikryrz/llama2-7b-fidelity-hasil --repo-type dataset \
  --include 'int8/portable/instance_49721366_v1/onnx/**' --local-dir "$ROOT/_dl_int8"
for z in "$ROOT"/_dl_int8/int8/portable/instance_49721366_v1/onnx/*.onnx.zst; do
  zstd -q -d -f "$z" -o "$INT8D/$(basename "${z%.zst}")"
done

"$ROOT/rtenv/bin/hf" download NousResearch/Llama-2-7b-hf tokenizer.model --local-dir "$ROOT/_dl_tok"
cp "$ROOT/_dl_tok/tokenizer.model" "$INT8D/tokenizer.model"

cp config/int8.env.example config/int8.env
./setup.sh

./run.sh --preset smoke
./run.sh --preset full
./run.sh --status
```

---

## 3. Dukungan model lain (model lokal non-HF, model selain Llama)

### (a) Model lokal, arsitektur sama Llama-2-7B (MHA)

FP16:

```bash
cd src/fp16
export HF_TOKEN=hf_xxx        # tetap wajib diisi utk preflight, walau bobotnya lokal
bash runner/bootstrap_vast.sh 10 30      # JANGAN lanjut ke 40 -- itu ONNX bobot HF asli, bukan bobot lokalmu

cp config/fp16.env.example config/fp16.env
# isi MODEL_ID=/path/ke/model/lokal di config/fp16.env (wajib: config.json + *.safetensors langsung di situ)
./setup.sh

ROOT=/workspace/fidelity
CODE=$ROOT/running_experiment_7b
REPO=$CODE/repo/FIdelity-ONNX-master
MODEL_ID=/path/ke/model/lokal   # sama seperti di config/fp16.env

mkdir -p "$CODE/work"
gunzip -c "$ROOT/_dl/code/mmlu_pool.csv.gz" > "$CODE/work/mmlu_pool.csv"

"$ROOT/expenv/bin/python" "$REPO/export_llama.py" \
  --repo "$REPO" --model_id "$MODEL_ID" \
  --out "$CODE/onnx/_onnx_raw_fp32" --fp16 "$CODE/onnx/onnx_fp16"

./setup.sh
./run.sh --preset smoke
./run.sh --preset full
./run.sh --status
```

INT8 (act_scales/llama-2-7b.pt yang dipakai tetap valid -- arsitekturnya sama):

```bash
cd src/int8
export HF_TOKEN=hf_xxx
bash ../fp16/runner/bootstrap_vast.sh 10 30

cp config/int8.env.example config/int8.env
# isi MODEL_ID=/path/ke/model/lokal di config/int8.env
./setup.sh

ROOT=/workspace/fidelity
CODE=$ROOT/running_experiment_7b
REPO=$CODE/repo/FIdelity-ONNX-master
MODEL_ID=/path/ke/model/lokal   # sama seperti di config/int8.env

mkdir -p "$CODE/onnx/onnx_int8" "$CODE/work"
gunzip -c "$ROOT/_dl/code/mmlu_pool.csv.gz" > "$CODE/work/mmlu_pool.csv"

"$ROOT/expenv/bin/python" "$REPO/export_llama_int8.py" \
  --repo "$REPO" --model_id "$MODEL_ID" --out "$CODE/onnx/onnx_int8"

"$ROOT/rtenv/bin/hf" download NousResearch/Llama-2-7b-hf tokenizer.model --local-dir "$ROOT/_dl_tok"
cp "$ROOT/_dl_tok/tokenizer.model" "$CODE/onnx/onnx_int8/tokenizer.model"

./setup.sh
./run.sh --preset smoke
./run.sh --preset full
./run.sh --status
```

### (b) Model lain arsitekturnya (GQA — Llama-3, Mistral, Gemma-2, dll.)

```bash
cd src/fp16   # FP16 dulu -- wajib walau targetmu INT8
export HF_TOKEN=hf_xxx        # lewati kalau MODEL_ID path lokal
bash runner/bootstrap_vast.sh 10 30      # JANGAN lanjut ke 40 -- itu ONNX Llama-2-7B, bukan modelmu

cp config/fp16.env.example config/fp16.env
# isi MODEL_ID=... di config/fp16.env, lalu:
./setup.sh

ROOT=/workspace/fidelity
CODE=$ROOT/running_experiment_7b
REPO=$CODE/repo/FIdelity-ONNX-master
MODEL_ID=...   # sama seperti di config/fp16.env

mkdir -p "$CODE/work"
gunzip -c "$ROOT/_dl/code/mmlu_pool.csv.gz" > "$CODE/work/mmlu_pool.csv"

"$ROOT/expenv/bin/python" "$REPO/export_llama.py" \
  --repo "$REPO" --model_id "$MODEL_ID" \
  --out "$CODE/onnx/_onnx_raw_fp32" --fp16 "$CODE/onnx/onnx_fp16"

./setup.sh   # generate injection_llm/*.json awal

"$ROOT/rtenv/bin/python" pipeline/discover_operasi.py \
  --injection-dir "$REPO/injection_llm" \
  --onnx "$CODE/onnx/onnx_fp16/decoder-merge-0.onnx" \
  --model-config "$REPO/configs/my_model.json"

./setup.sh   # jalan lagi -- pakai nama operasi hasil discover_operasi.py
./run.sh --preset smoke
./run.sh --preset full
```

```bash
# INT8 untuk model lain: BELUM BISA -- export_llama_int8.py hardcode
# act_scales/llama-2-7b.pt, belum ada skrip kalibrasi SmoothQuant untuk model lain.
```

---

## 4. Kalau gagal — temuan dari verifikasi instance H100 nyata (24 Sep 2026)

### Tahap 30 (`bootstrap_vast.sh`) gagal, "kode tidak ditemukan"

Repo kode default (`mfikryrz/llama2-7b-fidelity-onnx-fp16`) **privat** —
butuh `HF_TOKEN` asli dengan akses. Kalau tidak punya akses, salin toolkit
dasar manual dari cadangan lain yang punya `FIdelity-ONNX-master` lengkap,
dari **komputer lokal** (bukan di server):

```bash
rsync -a --exclude='injection_llm/' --exclude='injection_llm_smoke/' --exclude='__pycache__/' \
  -e "ssh -p PORT" \
  /path/ke/cadangan/code/FIdelity-ONNX-master/ \
  root@HOST:/workspace/fidelity/running_experiment_7b/repo/FIdelity-ONNX-master/
rsync -a -e "ssh -p PORT" \
  /path/ke/cadangan/code/mmlu_loglik_bench.py /path/ke/cadangan/code/01_phase1_export.py \
  root@HOST:/workspace/fidelity/running_experiment_7b/
rsync -a -e "ssh -p PORT" \
  /path/ke/hasil_pgteks/results/fp16/pelengkap/work/mmlu_pool.csv \
  root@HOST:/workspace/fidelity/running_experiment_7b/work/mmlu_pool.csv
```

Lalu `export HF_TOKEN=apa_saja` (dummy, cuma perlu tidak kosong) dan
`bootstrap_vast.sh 10 30` (skip 40 -- toolkit sudah ada manual).

### "Failed to load library .../onnx_bitflip.so ... version 'VERS_x.y.z' not found"

`onnx_bitflip.so` yang kamu punya minta versi `onnxruntime` tertentu lewat
symbol version yang KETAT (beda dari `req_rtenv.txt`==1.20.2 kalau
toolkit-nya dipinjam dari cadangan lain). Cocokkan persis ke versi yang
diminta pesan errornya:

```bash
/workspace/fidelity/rtenv/bin/pip install -q onnxruntime-gpu==X.Y.Z   # ganti sesuai pesan error
cd /workspace/fidelity/rtenv/lib/python3.10/site-packages/onnxruntime/capi/
ln -sf $(ls libonnxruntime.so.1.*) libonnxruntime.so.1
ln -sf $(ls libonnxruntime.so.1.*) libonnxruntime.so
```

### Baris `[err] ... Failed to load library libonnxruntime_providers_tensorrt.so`, lalu "Falling back to ['CPUExecutionProvider']" berkali-kali

`memory_pool.py` mencoba TensorRT duluan secara default, gagal (TensorRT
tidak terpasang), lalu diam-diam jatuh ke CPU murni (sangat lambat, 7B di
CPU). Matikan lewat env var resmi:

```bash
ROOT=/workspace/fidelity FIDELITY_USE_TRT=0 ./run.sh --preset smoke
```

### `bootstrap_vast.sh` melapor GAGAL padahal 35/35 ONNX sudah sehat

Bug di baris terakhir skrip (exit code skrip = exit code pengecekan
`ringkas_vast.py`, yang memang tidak ada untuk run `10 40`/`10 30`) —
**sudah diperbaiki** di `src/fp16/runner/bootstrap_vast.sh`. Kalau masih
kejadian, pastikan pakai salinan terbaru dari repo ini.

### Ganti presisi (FP16↔INT8) di server yang SAMA

`injection_llm/*.json` dipakai bersama tapi target tensornya beda per
presisi (jumlah filenya kebetulan sama, 288, jadi `setup.sh` tidak
mendeteksi salah). Hapus dulu sebelum ganti presisi:

```bash
rm -f /workspace/fidelity/running_experiment_7b/repo/FIdelity-ONNX-master/injection_llm/*.json
```

lalu jalankan `./setup.sh` lagi di folder presisi yang baru.
