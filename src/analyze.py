"""Orchestrates the full analysis: extract heads, verify the AB/BA theorem on
every real head, compute the idempotence residual, and score each head against
a matched-shape random-matrix null. Produces a flat list of per-head records
that scripts/reproduce.sh dumps to results/summary.json.
"""
from __future__ import annotations

import numpy as np

from src.extract import ARCHITECTURES, extract
from src.null_baseline import null_residual_samples, percentile_of
from src.theorem import idempotence_residual, nonzero_spectrum_match, reduced_matrix


def analyze_model(
    model_key: str, null_samples_per_shape: int = 300, null_seed: int = 0
) -> list[dict]:
    heads = extract(model_key)
    n_layers = max(h.layer for h in heads) + 1
    d_head = heads[0].A.shape[1]
    null_cache: dict[int, np.ndarray] = {}
    records = []
    for hr in heads:
        check = nonzero_spectrum_match(hr.A, hr.B)
        if not check["match"]:
            raise RuntimeError(
                f"{model_key} layer {hr.layer} head {hr.head}: AB/BA spectral "
                f"reduction did not hold numerically ({check}) -- indicates a "
                f"bug in the eig/reduction code or a pathological-scale weight, "
                f"not something to silently skip"
            )
        M = reduced_matrix(hr.A, hr.B)
        rho, alpha = idempotence_residual(M)
        d = M.shape[0]
        if d not in null_cache:
            null_cache[d] = null_residual_samples(d, null_samples_per_shape, seed=null_seed)
        pct = percentile_of(rho, null_cache[d]) if np.isfinite(rho) else 0.0
        records.append({
            "model": model_key,
            "layer": hr.layer,
            "head": hr.head,
            "n_layers": n_layers,
            "depth_frac": hr.layer / max(1, n_layers - 1),
            "d_head": d_head,
            "alpha": alpha,
            "rho": rho,
            "null_percentile": pct,
        })
    return records


def analyze_all(model_keys: list[str] | None = None) -> list[dict]:
    model_keys = model_keys or list(ARCHITECTURES.keys())
    out = []
    for mk in model_keys:
        out.extend(analyze_model(mk))
    return out
