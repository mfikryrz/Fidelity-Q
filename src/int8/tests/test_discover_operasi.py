"""Test discover_operasi.py + _config.operasi_suffixes()/fidelity_utils fallback.

Dibangun dengan graf ONNX SINTETIS kecil (bukan model 7B sungguhan) supaya test
ini bisa jalan di mana saja tanpa data eksperimen — hanya butuh package `onnx`
(CPU, ~10MB), bukan onnxruntime/GPU.

Dua hal yang dibuktikan:
1. REGRESI: untuk graf berstruktur Llama-2-7B, urutan yang ditemukan otomatis
   identik dengan OPERASI_SUFFIXES lama yang di-hardcode — supaya run yang
   sudah ada (FP16/INT8 Llama-2-7B) tidak berubah perilakunya.
2. PERBAIKAN BUG: untuk graf dengan operasi yang TIDAK ada di daftar lama
   (mensimulasikan arsitektur lain), operasi itu tetap DITEMUKAN, bukan
   diam-diam dibuang seperti perilaku lama.

Jalankan:
    cd src/int8/tests
    python -m pytest test_discover_operasi.py -v
    # atau tanpa pytest:
    python test_discover_operasi.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repo_terpatch"))

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from discover_operasi import discover  # noqa: E402

LLAMA_LIKE_OPS = [
    ("self_attn_q_proj_MatMul", "/layer/self_attn/q_proj/MatMul"),
    ("self_attn_k_proj_MatMul", "/layer/self_attn/k_proj/MatMul"),
    ("self_attn_v_proj_MatMul", "/layer/self_attn/v_proj/MatMul"),
    ("self_attn_MatMul", "/layer/self_attn/MatMul"),
    ("self_attn_MatMul_1", "/layer/self_attn/MatMul_1"),
    ("self_attn_o_proj_MatMul", "/layer/self_attn/o_proj/MatMul"),
    ("mlp_gate_proj_MatMul", "/layer/mlp/gate_proj/MatMul"),
    ("mlp_up_proj_MatMul", "/layer/mlp/up_proj/MatMul"),
    ("mlp_down_proj_MatMul", "/layer/mlp/down_proj/MatMul"),
]

OPERASI_SUFFIXES_LAMA = [op for op, _ in LLAMA_LIKE_OPS]


def _build_fake_decoder_onnx(path: str, ops: list[tuple[str, str]]) -> None:
    """Graf linear kecil: input -> MatMul -> MatMul -> ... -> output, dalam
    urutan `ops`. Nama node = target_layer di kolom kedua tuple."""
    x = "hidden_in"
    nodes = []
    value_info = [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 4])]
    w_inits = []
    for i, (_, node_name) in enumerate(ops):
        w_name = f"w{i}"
        y_name = f"h{i}"
        w_inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, [4, 4], [0.0] * 16))
        nodes.append(helper.make_node("MatMul", [x, w_name], [y_name], name=node_name))
        x = y_name
    value_info.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 4]))
    graph = helper.make_graph(nodes, "fake_decoder", [value_info[0]], [value_info[-1]], w_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.save(model, path)


def _write_injection_configs(inj_dir: str, decoder_idx: int, ops: list[tuple[str, str]]) -> None:
    os.makedirs(inj_dir, exist_ok=True)
    for suffix, target_layer in ops:
        cfg = {"decoder_idx": decoder_idx, "operation": suffix, "target_layer": target_layer}
        with open(os.path.join(inj_dir, f"decoder-merge-{decoder_idx}__{suffix}.json"), "w") as f:
            json.dump(cfg, f)


def test_regresi_llama_2_7b_urutan_sama_persis():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "decoder-merge-0.onnx")
        inj_dir = os.path.join(tmp, "injection_llm")
        _build_fake_decoder_onnx(onnx_path, LLAMA_LIKE_OPS)
        _write_injection_configs(inj_dir, 0, LLAMA_LIKE_OPS)

        found = discover(inj_dir, onnx_path, decoder_idx=0)
        assert found == OPERASI_SUFFIXES_LAMA, (
            f"Urutan berubah dari yang di-hardcode!\n  lama: {OPERASI_SUFFIXES_LAMA}\n  baru: {found}"
        )


def test_arsitektur_lain_tidak_diam_diam_dibuang():
    """Mensimulasikan model dengan operasi yang TIDAK ada di OPERASI_SUFFIXES
    lama (mis. QKV gabungan seperti sebagian model non-Llama, atau MLP dengan
    nama beda) -- harus tetap MUNCUL di hasil, bukan hilang tanpa peringatan.
    """
    ops_arsitektur_lain = [
        ("self_attn_qkv_proj_MatMul", "/layer/self_attn/qkv_proj/MatMul"),  # QKV fused, bukan 3 terpisah
        ("self_attn_MatMul", "/layer/self_attn/MatMul"),
        ("self_attn_MatMul_1", "/layer/self_attn/MatMul_1"),
        ("self_attn_o_proj_MatMul", "/layer/self_attn/o_proj/MatMul"),
        ("mlp_fused_geglu_MatMul", "/layer/mlp/fused_geglu/MatMul"),  # nama MLP beda dari Llama
        ("mlp_down_proj_MatMul", "/layer/mlp/down_proj/MatMul"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "decoder-merge-0.onnx")
        inj_dir = os.path.join(tmp, "injection_llm")
        _build_fake_decoder_onnx(onnx_path, ops_arsitektur_lain)
        _write_injection_configs(inj_dir, 0, ops_arsitektur_lain)

        found = discover(inj_dir, onnx_path, decoder_idx=0)
        expected = [op for op, _ in ops_arsitektur_lain]
        assert found == expected, f"Operasi arsitektur lain hilang/salah urutan: {found} != {expected}"

        # Bukti langsung anti-regresi bug lama: perilaku lama (filter terhadap
        # OPERASI_SUFFIXES tetap) akan membuang SEMUA operasi ini karena
        # namanya tidak ada di daftar Llama.
        lama_akan_tersisa = [op for op in found if op in OPERASI_SUFFIXES_LAMA]
        assert lama_akan_tersisa == ["self_attn_MatMul", "self_attn_MatMul_1", "self_attn_o_proj_MatMul", "mlp_down_proj_MatMul"], (
            "sanity check daftar lama berubah — perbarui test ini"
        )
        assert len(found) == 6, "keenam operasi arsitektur lain harus tetap ada, tidak dibuang diam-diam"


if __name__ == "__main__":
    test_regresi_llama_2_7b_urutan_sama_persis()
    print("[ok] regresi Llama-2-7B: urutan identik dengan OPERASI_SUFFIXES lama")
    test_arsitektur_lain_tidak_diam_diam_dibuang()
    print("[ok] arsitektur lain: operasi baru tidak lagi dibuang diam-diam")
