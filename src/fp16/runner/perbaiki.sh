#!/usr/bin/env bash
# Melengkapi 496 baris yang gagal alokasi memori ONNX Runtime.
# Satu PROSES per konfigurasi -> VRAM direset penuh tiap kali.
set -uo pipefail
cd /workspace/fidelity/running_experiment_7b
ROOT=/workspace/fidelity
export FIDELITY_REPO=$PWD/repo/FIdelity-ONNX-master
export FIDELITY_WORK=$PWD/work FIDELITY_ONNX_ROOT=$PWD/onnx
export HF_HOME=$PWD/hf_home TMPDIR=$PWD/tmp
export MODEL_ID=NousResearch/Llama-2-7b-hf
export FIDELITY_GPU_LAYERS=32
export ORT_GPU_MEM_LIMIT_GB=20 FIDELITY_POOL_GB=20   # lebih ketat dari 30
export ORT_THREADS=30 OMP_NUM_THREADS=30
SP=$ROOT/rtenv/lib/python3.10/site-packages
LD=$SP/onnxruntime/capi; for d in $SP/nvidia/*/lib; do [ -d "$d" ] && LD="$LD:$d"; done
export LD_LIBRARY_PATH="$LD:$FIDELITY_REPO/llama"
OUT=$PWD/hasil_massal/massal_fp16.csv
echo "melengkapi 15 konfigurasi, total 496 baris hilang"
echo "--- [1/15] decoder 26 / self_attn_k_proj_MatMul  (150 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 26 --operasi self_attn_k_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [2/15] decoder 26 / mlp_gate_proj_MatMul  (58 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 26 --operasi mlp_gate_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [3/15] decoder 15 / mlp_down_proj_MatMul  (56 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 15 --operasi mlp_down_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [4/15] decoder 11 / mlp_up_proj_MatMul  (54 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 11 --operasi mlp_up_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [5/15] decoder 26 / mlp_up_proj_MatMul  (50 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 26 --operasi mlp_up_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [6/15] decoder 5 / mlp_gate_proj_MatMul  (22 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 5 --operasi mlp_gate_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [7/15] decoder 5 / mlp_up_proj_MatMul  (20 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 5 --operasi mlp_up_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [8/15] decoder 11 / self_attn_MatMul_1  (18 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 11 --operasi self_attn_MatMul_1 \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [9/15] decoder 11 / mlp_gate_proj_MatMul  (16 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 11 --operasi mlp_gate_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [10/15] decoder 14 / self_attn_q_proj_MatMul  (16 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 14 --operasi self_attn_q_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [11/15] decoder 11 / self_attn_MatMul  (12 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 11 --operasi self_attn_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [12/15] decoder 1 / self_attn_o_proj_MatMul  (8 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 1 --operasi self_attn_o_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [13/15] decoder 4 / self_attn_q_proj_MatMul  (6 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 4 --operasi self_attn_q_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [14/15] decoder 12 / self_attn_MatMul  (6 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 12 --operasi self_attn_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "--- [15/15] decoder 14 / self_attn_v_proj_MatMul  (4 baris) ---"
N0=$(( $(wc -l < "$OUT") - 1 ))
$ROOT/rtenv/bin/python mass_loglik_inject.py --decoders 14 --operasi self_attn_v_proj_MatMul \
    --runs 25 --soal 1 --shots 5 --posisi-token akhir \
    --out "$OUT" --resume 2>&1 | grep -aE "SELESAI|\[err\]" | tail -2
echo "    $N0 -> $(( $(wc -l < "$OUT") - 1 )) baris"
echo "=== SELESAI melengkapi ==="
echo "total baris: $(( $(wc -l < "$OUT") - 1 )) / 43200"
