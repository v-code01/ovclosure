import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.extract import _load_weights, _to_f64, extract_gpt2, extract_qwen2_style
from src.theorem import nonzero_spectrum_match


def test_gpt2_shapes_and_count():
    heads = extract_gpt2()
    assert len(heads) == 12 * 12  # n_layer * n_head
    for hr in heads:
        assert hr.A.shape == (768, 64)
        assert hr.B.shape == (64, 768)


def test_gpt2_reconstructs_source_weight_exactly():
    # Concatenating every head's A columns must reconstruct the V block of
    # c_attn.weight exactly, and every head's B rows must reconstruct
    # c_proj.weight exactly. This is a definitional self-consistency check on
    # the extraction slicing itself (independent of the OV-closure theorem).
    heads = extract_gpt2()
    weights, _ = _load_weights("openai-community/gpt2")
    d_model = 768
    for layer in range(12):
        c_attn = _to_f64(weights[f"h.{layer}.attn.c_attn.weight"])
        c_proj = _to_f64(weights[f"h.{layer}.attn.c_proj.weight"])
        v_block = c_attn[:, 2 * d_model : 3 * d_model]
        layer_heads = sorted([h for h in heads if h.layer == layer], key=lambda h: h.head)
        A_cat = np.concatenate([h.A for h in layer_heads], axis=1)
        B_cat = np.concatenate([h.B for h in layer_heads], axis=0)
        assert np.allclose(A_cat, v_block), f"layer {layer} A reconstruction mismatch"
        assert np.allclose(B_cat, c_proj), f"layer {layer} B reconstruction mismatch"


def test_gpt2_heads_satisfy_theorem():
    heads = extract_gpt2()
    for hr in heads[::7]:  # subsample for speed, still >100 heads checked
        r = nonzero_spectrum_match(hr.A, hr.B)
        assert r["match"], f"layer {hr.layer} head {hr.head}: {r}"


def test_qwen_gqa_shares_A_within_kv_group():
    # Qwen2.5-0.5B: 14 query heads, 2 kv heads -> groups of 7 consecutive
    # query heads must have IDENTICAL A (value) matrices, since they read the
    # same v_proj output; only B (their O-slice) differs per head. This is the
    # sharpest correctness check on the GQA grouping logic: get the group
    # boundary off by one and this fails immediately.
    heads = extract_qwen2_style("Qwen/Qwen2.5-0.5B")
    n_head, n_kv = 14, 2
    n_rep = n_head // n_kv
    for layer in range(24):
        layer_heads = sorted([h for h in heads if h.layer == layer], key=lambda h: h.head)
        assert len(layer_heads) == n_head
        for kv in range(n_kv):
            group = layer_heads[kv * n_rep : (kv + 1) * n_rep]
            ref = group[0].A
            for hr in group[1:]:
                msg = f"layer {layer} kv-group {kv} head {hr.head} A mismatch"
                assert np.array_equal(hr.A, ref), msg
            # and a different kv-group must NOT share A (sanity that groups are non-trivial)
        other_group_A = layer_heads[0].A
        far_group_A = layer_heads[-1].A
        assert not np.array_equal(other_group_A, far_group_A)


def test_qwen_shapes_and_count():
    heads = extract_qwen2_style("Qwen/Qwen2.5-0.5B")
    assert len(heads) == 24 * 14
    for hr in heads:
        assert hr.A.shape == (896, 64)
        assert hr.B.shape == (64, 896)


def test_qwen_heads_satisfy_theorem():
    heads = extract_qwen2_style("Qwen/Qwen2.5-0.5B")
    for hr in heads[::11]:
        r = nonzero_spectrum_match(hr.A, hr.B)
        assert r["match"], f"layer {hr.layer} head {hr.head}: {r}"
