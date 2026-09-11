"""Per-head OV operator extraction from real downloaded model weights.

Two architecture families, both verified against their public HF config +
weight-shape conventions (no forward pass needed -- we only need V and O):

  - gpt2 (Conv1D layers): y = x @ W + b, weight stored (in_features, out_features).
    c_attn.weight is (d_model, 3*d_model), the V block is the last third.
    c_proj.weight is (d_model, d_model).

  - qwen2 / llama-style (nn.Linear layers, optionally GQA): y = x @ W.T + b,
    weight stored (out_features, in_features). v_proj.weight is
    (num_kv_heads*head_dim, d_model); o_proj.weight is (d_model,
    num_heads*head_dim). Each of the num_heads query heads is assigned to a
    kv head in contiguous blocks of size num_heads // num_kv_heads (the
    standard `repeat_kv` grouping used by HF's Qwen2/Llama attention).

For every head we return (A, B) with A: (d_model, head_dim) such that
v = x @ A, and B: (head_dim, d_model) such that the head's contribution to the
residual stream is v @ B. So T = A @ B is the OV operator in the convention
used by src/theorem.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import ml_dtypes  # noqa: F401 -- side effect: registers bfloat16 as a numpy dtype for safetensors
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open


@dataclass
class HeadRef:
    layer: int
    head: int
    A: np.ndarray  # (d_model, head_dim), float64
    B: np.ndarray  # (head_dim, d_model), float64


def _to_f64(t: np.ndarray) -> np.ndarray:
    # safetensors' numpy backend returns ml_dtypes.bfloat16 arrays for BF16
    # tensors; cast through float32 first since numpy has no native bf16 path
    # for some ops, then to float64 for the linear-algebra work.
    return np.asarray(t, dtype=np.float32).astype(np.float64)


def _load_config(repo_id: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, filename="config.json")
    return json.load(open(path))


def _load_weights(repo_id: str) -> tuple[dict, str]:
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    weights = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118 -- safetensors' keys() isn't a dict
            weights[k] = f.get_tensor(k)
    return weights, path


def extract_gpt2(repo_id: str = "openai-community/gpt2") -> list[HeadRef]:
    cfg = _load_config(repo_id)
    weights, _ = _load_weights(repo_id)
    d_model = cfg["n_embd"]
    n_head = cfg["n_head"]
    n_layer = cfg["n_layer"]
    d_head = d_model // n_head
    heads: list[HeadRef] = []
    for layer in range(n_layer):
        c_attn = _to_f64(weights[f"h.{layer}.attn.c_attn.weight"])  # (d_model, 3*d_model)
        c_proj = _to_f64(weights[f"h.{layer}.attn.c_proj.weight"])  # (d_model, d_model)
        v_block = c_attn[:, 2 * d_model : 3 * d_model]  # (d_model, d_model), x @ v_block
        for h in range(n_head):
            A = v_block[:, h * d_head : (h + 1) * d_head]
            B = c_proj[h * d_head : (h + 1) * d_head, :]
            heads.append(HeadRef(layer=layer, head=h, A=A, B=B))
    return heads


def extract_qwen2_style(repo_id: str) -> list[HeadRef]:
    cfg = _load_config(repo_id)
    weights, _ = _load_weights(repo_id)
    d_model = cfg["hidden_size"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)
    n_layer = cfg["num_hidden_layers"]
    d_head = cfg.get("head_dim", d_model // n_head)
    if n_head % n_kv != 0:
        raise ValueError(
            f"{repo_id}: num_attention_heads {n_head} not divisible by "
            f"num_key_value_heads {n_kv}"
        )
    n_rep = n_head // n_kv
    heads: list[HeadRef] = []
    for layer in range(n_layer):
        v_key = f"model.layers.{layer}.self_attn.v_proj.weight"  # (n_kv*d_head, d_model)
        o_key = f"model.layers.{layer}.self_attn.o_proj.weight"  # (d_model, n_head*d_head)
        v_w = _to_f64(weights[v_key])
        o_w = _to_f64(weights[o_key])
        v_wT = v_w.T  # (d_model, n_kv*d_head): x @ v_wT gives all kv-head values
        o_wT = o_w.T  # (n_head*d_head, d_model): concat(heads) @ o_wT gives contribution
        for h in range(n_head):
            kv = h // n_rep
            A = v_wT[:, kv * d_head : (kv + 1) * d_head]
            B = o_wT[h * d_head : (h + 1) * d_head, :]
            heads.append(HeadRef(layer=layer, head=h, A=A, B=B))
    return heads


ARCHITECTURES = {
    "gpt2": ("openai-community/gpt2", extract_gpt2),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", extract_qwen2_style),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", extract_qwen2_style),
    "smollm2-135m-instruct": ("HuggingFaceTB/SmolLM2-135M-Instruct", extract_qwen2_style),
    "smollm2-360m-instruct": ("HuggingFaceTB/SmolLM2-360M-Instruct", extract_qwen2_style),
}


def extract(model_key: str) -> list[HeadRef]:
    repo_id, fn = ARCHITECTURES[model_key]
    if fn is extract_gpt2:
        return fn(repo_id)
    return fn(repo_id)
