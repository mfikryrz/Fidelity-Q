#!/usr/bin/env python3
"""SmoothQuant fake-INT8 ONNX export for FIdelity (NousResearch Llama-2-7b)."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model_id", required=True)
ap.add_argument("--out", required=True)  # onnx_int8 dir
ap.add_argument("--prompt", default="bonjour")
ap.add_argument("--max_new_tokens", type=int, default=2)
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
ap.add_argument("--alpha", type=float, default=0.85)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.repo, "configs"), exist_ok=True)
os.makedirs(os.path.join(args.repo, "act_scales"), exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Ensure SmoothQuant scales are where export-onnx historically expected them.
scales_src = os.path.join(args.repo, "int8", "llama-2-7b.pt")
scales_dst = os.path.join(args.repo, "act_scales", "llama-2-7b.pt")
assert os.path.isfile(scales_src), scales_src
if not os.path.isfile(scales_dst):
    shutil.copy(scales_src, scales_dst)

# 1) Patch transformers modeling_llama with INT8 exporter hooks; redirect output path.
import transformers

src = os.path.join(args.repo, "int8", "modeling_llama.py")
dst = os.path.join(os.path.dirname(transformers.__file__), "models", "llama", "modeling_llama.py")
assert os.path.isfile(src), src
txt = open(src).read()
txt = txt.replace("/workspace/llama.onnx/7B", args.out)
txt = txt.replace(
    "def verify_onnx(filepath, inkv, outv):\n",
    "def verify_onnx(filepath, inkv, outv):\n    return\n",
)
txt = txt.replace(", external_data=False", "").replace(", dynamo=False", "")
open(dst, "w").write(txt)
for m in [m for m in list(sys.modules) if "transformers.models.llama" in m]:
    del sys.modules[m]
importlib.invalidate_caches()
print("modeling_llama.py patched:", dst)

# Local SmoothQuant helpers
sys.path.insert(0, os.path.join(args.repo, "int8"))
from smoothquant.fake_quant import quantize_llama_like  # noqa: E402
from smoothquant.smooth import smooth_lm  # noqa: E402

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class CustomLlamaModel(LlamaForCausalLM, GenerationMixin):
    pass


cfg = AutoConfig.from_pretrained(args.model_id, token=HF_TOKEN)
nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
# DIUBAH 21 Sep 2026: GQA sekarang didukung (lihat repeat_kv() + LlamaAttention
# di repo_terpatch/modeling_llama.py) -- assert lama menolak GQA sama sekali.
assert cfg.num_attention_heads % nkv == 0, (
    "num_attention_heads harus kelipatan bulat dari num_key_value_heads -- bukan GQA yang valid. "
    f"num_attention_heads={cfg.num_attention_heads} num_key_value_heads={nkv}"
)
print(
    "{} OK: {} layers, {} query heads, {} kv heads, hidden={}".format(
        "MHA" if nkv == cfg.num_attention_heads else "GQA",
        cfg.num_hidden_layers, cfg.num_attention_heads, nkv, cfg.hidden_size
    )
)

device = args.device
if device == "cuda" and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable — falling back to CPU")
    device = "cpu"

print(f"Loading {args.model_id} (fp32, {device})...")
model = CustomLlamaModel.from_pretrained(
    args.model_id, torch_dtype=torch.float32, token=HF_TOKEN, trust_remote_code=True
)
act_scales = torch.load(scales_dst, map_location="cpu")
print(f"SmoothQuant alpha={args.alpha}...")
smooth_lm(model, act_scales, args.alpha)
print("Fake-INT8 quantize_llama_like...")
model = quantize_llama_like(model)
model.eval()
model.to(device)

tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, token=HF_TOKEN, trust_remote_code=True)
ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
print("Triggering ONNX export via generate()...")
with torch.no_grad():
    model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
del model
if device == "cuda":
    torch.cuda.empty_cache()
print("Export pass done.")

def _free_hf_weight_cache(model_id: str) -> None:
    hub = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.path.join(hub, "hub")
    slug = "models--" + model_id.replace("/", "--")
    repo_dir = os.path.join(hub, slug)
    if not os.path.isdir(repo_dir):
        return
    freed = 0
    for root, _dirs, files in os.walk(repo_dir):
        for fn in files:
            if fn.endswith((".safetensors", ".bin")) and "index" not in fn:
                p = os.path.join(root, fn)
                try:
                    if os.path.islink(p):
                        target = os.path.realpath(p)
                        os.unlink(p)
                        if os.path.isfile(target):
                            freed += os.path.getsize(target)
                            os.remove(target)
                    elif os.path.isfile(p):
                        freed += os.path.getsize(p)
                        os.remove(p)
                except OSError:
                    pass
    if freed:
        print("HF cache weights freed: {:.2f} GB".format(freed / 1e9))


_free_hf_weight_cache(args.model_id)

# tokenizer.model
#
# DIPATCH 21 Sep 2026 untuk dukungan MODEL LOKAL: kalau args.model_id adalah
# path lokal (bukan "org/repo" HF), hf_hub_download() di bawah akan gagal --
# ia cuma menerima ID repo HF. Model lokal sudah punya tokenizer.model di
# direktorinya sendiri (itulah kenapa model itu dianggap "lokal" oleh
# _download_7b.py, lihat is_local_model_dir() di sana), jadi cukup disalin
# langsung, tidak perlu menyentuh HF Hub sama sekali.
if not os.path.exists(os.path.join(args.out, "tokenizer.model")):
    local_tok = (
        os.path.join(args.model_id, "tokenizer.model")
        if os.path.isdir(args.model_id) else None
    )
    if local_tok and os.path.isfile(local_tok):
        shutil.copy(local_tok, os.path.join(args.out, "tokenizer.model"))
    else:
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(args.model_id, "tokenizer.model", token=HF_TOKEN)
        shutil.copy(f, os.path.join(args.out, "tokenizer.model"))

need = ["embed.onnx", "norm.onnx", "head.onnx"] + [
    f"decoder-merge-{i}.onnx" for i in range(cfg.num_hidden_layers)
]


def ok(d: str, f: str) -> bool:
    p = os.path.join(d, f)
    return os.path.exists(p) and os.path.getsize(p) > 0


miss = [f for f in need if not ok(args.out, f)]
print("MISSING int8:", miss[:6] if miss else "none")
assert not miss, f"Export INT8 tidak lengkap: {miss[:6]}"

head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
eos = cfg.eos_token_id
eos = int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos)
spec = {
    "decoder_count": cfg.num_hidden_layers,
    "eos_token_id": eos,
    "hidden_dim": cfg.hidden_size,
    "n_heads": cfg.num_attention_heads,
    "head_dim": head_dim,
    "decoder_template": "decoder-merge-{}.onnx",
    "tokenizer_file": "tokenizer.model",
    "embed_file": "embed.onnx",
    "norm_file": "norm.onnx",
    "head_file": "head.onnx",
    "precision": "int8",
    "input_names": {
        "hidden": "hidden_in",
        "attn_mask": "attn_mask",
        "position_ids": "position_ids",
        "past_key": "past_key_in",
        "past_value": "past_value_in",
    },
    "output_names": {"hidden": "hidden_out", "past_key": "past_key", "past_value": "past_value"},
    "embed_input": "input",
    "embed_output": "embed",
    "norm_input": "input",
    "norm_output": "output",
    "head_input": "input",
    "head_output": "output",
}
json.dump(spec, open(os.path.join(args.repo, "configs", "my_model.json"), "w"), indent=2)
print(f"OK: {len(need)} ONNX int8 + tokenizer + configs/my_model.json di {args.out}")
