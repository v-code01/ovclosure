import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bench.bench_reduction import bench_shape


def test_bench_shape_smoke():
    r = bench_shape(d_model=128, d_head=16, n_repeats=5)
    assert r["naive_median_ms"] > 0
    assert r["reduced_median_ms"] > 0
    assert r["theoretical_ratio"] == (128 / 16) ** 3


def test_reduced_eig_matches_naive_nonzero_spectrum():
    # bench_shape only measures time; separately confirm on the SAME random
    # A,B it times that the two computations it's racing actually agree
    # (otherwise a "speedup" over a numerically-different computation would
    # be meaningless).
    rng = np.random.default_rng(0)
    d_model, d_head = 128, 16
    A = rng.standard_normal((d_model, d_head))
    B = rng.standard_normal((d_head, d_model))
    eig_T = np.linalg.eigvals(A @ B)
    eig_M = np.linalg.eigvals(B @ A)
    nz_T = eig_T[np.abs(eig_T) > 1e-6 * np.max(np.abs(eig_T))]
    nz_T = np.sort_complex(nz_T)
    eig_M = np.sort_complex(eig_M)
    assert len(nz_T) <= len(eig_M)
    assert np.max(np.abs(nz_T - eig_M[: len(nz_T)])) < 1e-6 * np.max(np.abs(eig_M))
