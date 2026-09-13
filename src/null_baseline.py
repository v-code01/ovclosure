"""Random-matrix null distribution for the idempotence residual.

For a d x d matrix with i.i.d. standard-normal entries, computes the residual
distribution of src.theorem.idempotence_residual via Monte Carlo, so a real
head's residual can be reported as a bootstrap z-score / percentile against a
matched-shape null rather than just an unadorned number.
"""
from __future__ import annotations

import numpy as np

from src.theorem import idempotence_residual


def null_residual_samples(d: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        M = rng.standard_normal((d, d))
        rho, _ = idempotence_residual(M)
        out[i] = rho
    return out


def null_summary(d: int, n_samples: int = 400, seed: int = 0) -> dict:
    samples = null_residual_samples(d, n_samples, seed)
    return {
        "d": d,
        "n_samples": n_samples,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "p01": float(np.percentile(samples, 1)),
        "p50": float(np.percentile(samples, 50)),
        "p99": float(np.percentile(samples, 99)),
    }


def percentile_of(value: float, null_samples: np.ndarray) -> float:
    """Fraction of null samples >= value that a real head's residual falls below (0..100)."""
    return float(100.0 * np.mean(null_samples >= value))
