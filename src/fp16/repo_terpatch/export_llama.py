
import os, sys, shutil, json, argparse, importlib
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model_id", required=True)
ap.add_argument("--out", required=True)        # folder ONNX fp32
ap.add_argument("--fp16", required=True)       # folder ONNX fp16 (hasil akhir)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--max_new_tokens", type=int, default=2)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.fp16, exist_ok=True)
os.makedirs(os.path.join(args.repo, "configs"), exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# 1) timpa modeling_llama.py transformers dengan exporter repo (int8/modeling_llama.py)
import transformers
src = os.path.join(args.repo, "int8", "modeling_llama.py")
dst = os.path.join(os.path.dirname(transformers.__file__), "models", "llama", "modeling_llama.py")
assert os.path.isfile(src), src
txt = open(src).read()
txt = txt.replace("/workspace/llama.onnx/7B", args.out)                  # arahkan output export
txt = txt.replace("def verify_onnx(filepath, inkv, outv):\n",
                  "def verify_onnx(filepath, inkv, outv):\n    return\n")  # netralkan verifikasi
# torch.onnx.export kwargs baru (external_data, dynamo) hanya ada di torch >= ~2.6;
# buang agar export jalan di torch versi apapun (memakai exporter TorchScript legacy)
txt = txt.replace(", external_data=False", "").replace(", dynamo=False", "")
open(dst, "w").write(txt)
for m in [m for m in list(sys.modules) if "transformers.models.llama" in m]:
    del sys.modules[m]
importlib.invalidate_caches()
print("modeling_llama.py dipatch:", dst)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

cfg = AutoConfig.from_pretrained(args.model_id, token=HF_TOKEN)
nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
# DIUBAH 21 Sep 2026: GQA sekarang didukung (lihat repeat_kv() + LlamaAttention
# di repo_terpatch/modeling_llama.py) -- assert lama menolak GQA sama sekali.
# Syarat yang tersisa cuma pembagian genap, yang memang wajib untuk GQA valid
# (num_key_value_groups = num_attention_heads // num_key_value_heads harus bulat).
assert cfg.num_attention_heads % nkv == 0, (
    "num_attention_heads ({}) harus kelipatan bulat dari num_key_value_heads ({}) "
    "-- bukan konfigurasi GQA yang valid.".format(cfg.num_attention_heads, nkv))
print("{} OK: {} layers, {} query heads, {} kv heads, hidden={}".format(
    "MHA" if nkv == cfg.num_attention_heads else "GQA",
    cfg.num_hidden_layers, cfg.num_attention_heads, nkv, cfg.hidden_size))

def _free_hf_weight_cache(model_id: str) -> None:
    """Hapus shard safetensors/bin dari cache HF setelah load ke RAM (hemat ~13GB disk)."""
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
                    freed += os.path.getsize(p)
                    os.remove(p)
                except OSError:
                    pass
    if freed:
        print("HF cache weights freed: {:.2f} GB".format(freed / 1e9))


print("Loading {} (fp32, CPU)...".format(args.model_id))
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, token=HF_TOKEN).eval()
_free_hf_weight_cache(args.model_id)
tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, token=HF_TOKEN)
ids = tok(args.prompt, return_tensors="pt").input_ids
print("Memicu export lewat generate (prefill->embed/norm/head, decode->decoder-merge)...")
with torch.no_grad():
    model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
del model
print("Export pass selesai.")

# 2) tokenizer.model
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

# 3) cek kelengkapan ONNX fp32
need = ["embed.onnx", "norm.onnx", "head.onnx"] + \
       ["decoder-merge-{}.onnx".format(i) for i in range(cfg.num_hidden_layers)]
def ok(d, f):
    p = os.path.join(d, f); return os.path.exists(p) and os.path.getsize(p) > 0
miss = [f for f in need if not ok(args.out, f)]
print("MISSING fp32:", miss[:6] if miss else "none")
assert not miss, "Export fp32 tidak lengkap: {}".format(miss[:6])

# 4) konversi ke FP16 (IO ikut fp16; FIdelity meng-feed fp16 saat config['fp16'])
import onnx
from onnxconverter_common import float16
# Bebaskan model torch (7B fp32 = ~27GB) SEBELUM konversi fp16.
# Tanpa ini, konversi berjalan sementara model masih residen -> RAM 31GB
# habis, swap 19GB terpakai penuh, dan mesin mati mendadak (kejadian 23 Agu 08:04).
import gc
try:
    del model
except NameError:
    pass
gc.collect()
print('model torch dibebaskan sebelum konversi fp16', flush=True)
for fn in sorted(os.listdir(args.out)):
    if not fn.endswith(".onnx"):
        continue
    src_fp32 = os.path.join(args.out, fn)
    dst_fp16 = os.path.join(args.fp16, fn)
    m = onnx.load(src_fp32)
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, dst_fp16)
    del m, m16
    os.remove(src_fp32)
shutil.copy(os.path.join(args.out, "tokenizer.model"), os.path.join(args.fp16, "tokenizer.model"))
miss16 = [f for f in need if not ok(args.fp16, f)]
assert not miss16, "Konversi fp16 tidak lengkap: {}".format(miss16[:6])

# 5) tulis config FIdelity (configs/my_model.json)
head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
eos = cfg.eos_token_id
eos = int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos)
spec = {
  "decoder_count": cfg.num_hidden_layers, "eos_token_id": eos,
  "hidden_dim": cfg.hidden_size, "n_heads": cfg.num_attention_heads, "head_dim": head_dim,
  "decoder_template": "decoder-merge-{}.onnx", "tokenizer_file": "tokenizer.model",
  "embed_file": "embed.onnx", "norm_file": "norm.onnx", "head_file": "head.onnx",
  "input_names": {"hidden": "hidden_in", "attn_mask": "attn_mask", "position_ids": "position_ids",
                  "past_key": "past_key_in", "past_value": "past_value_in"},
  "output_names": {"hidden": "hidden_out", "past_key": "past_key", "past_value": "past_value"},
  "embed_input": "input", "embed_output": "embed",
  "norm_input": "input", "norm_output": "output",
  "head_input": "input", "head_output": "output",
}
json.dump(spec, open(os.path.join(args.repo, "configs", "my_model.json"), "w"), indent=2)
print("OK: {} ONNX fp16 + tokenizer + configs/my_model.json siap di {}".format(len(need), args.fp16))
