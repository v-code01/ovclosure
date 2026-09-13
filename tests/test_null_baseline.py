import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.null_baseline import null_residual_samples, null_summary, percentile_of
from src.theorem import idempotence_residual


def test_null_samples_reproducible_with_seed():
    a = null_residual_samples(16, 50, seed=42)
    b = null_residual_samples(16, 50, seed=42)
    assert np.array_equal(a, b)


def test_null_samples_differ_across_seeds():
    a = null_residual_samples(16, 50, seed=1)
    b = null_residual_samples(16, 50, seed=2)
    assert not np.array_equal(a, b)


def test_null_summary_shape():
    s = null_summary(d=8, n_samples=200, seed=0)
    assert s["p01"] < s["p50"] < s["p99"]
    assert s["n_samples"] == 200
    assert s["d"] == 8


def test_percentile_of_monotonic():
    samples = null_residual_samples(16, 500, seed=7)
    lo = percentile_of(np.percentile(samples, 10), samples)
    hi = percentile_of(np.percentile(samples, 90), samples)
    # a value at a higher percentile of the null has fewer samples >= it
    assert hi < lo


def test_exact_projector_scores_far_below_null():
    # An exact scaled projector (rho == 0) must land far outside a random-matrix
    # null distribution -- this is the actual discriminative use of the null.
    rng = np.random.default_rng(11)
    d, r = 64, 8
    U = rng.standard_normal((d, r))
    Vt = rng.standard_normal((r, d))
    Vt = np.linalg.solve(Vt @ U, Vt)
    P = U @ Vt
    rho, _ = idempotence_residual(3.0 * P)
    null = null_residual_samples(d, 300, seed=3)
    pct = percentile_of(rho, null)
    assert pct > 99.0, f"expected exact projector residual to beat >99% of null, got {pct}"
