"""Measured M4 wall-clock law: naive (d_model^3) vs reduced (d_head^3)
eigendecomposition cost for the OV-closure diagnostic, across the actual
(d_model, d_head) shapes of the four analyzed models. This is the systems half
of the finding: the AB/BA spectral-equivalence theorem (src/theorem.py) says
the reduced computation is EXACT, not approximate, so any measured speedup is
pure engineering payoff with zero accuracy cost -- worth confirming that's
true on real M4 hardware rather than assumed from FLOP-counting alone, since
LAPACK's dgeev has real overhead (Hessenberg reduction, QR iteration) whose
constant factors don't have to track the naive O(n^3) FLOP count exactly.
"""
from __future__ import annotations

import time

import numpy as np

# (label, d_model, d_head) taken directly from the four models in results/summary.json
SHAPES = [
    ("gpt2", 768, 64),
    ("qwen2.5-0.5b", 896, 64),
    ("smollm2-135m-instruct", 576, 64),
    ("qwen2.5-1.5b-instruct", 1536, 128),
]


def _time_call(fn, n_repeats: int) -> np.ndarray:
    times = np.empty(n_repeats, dtype=np.float64)
    for i in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t0
    return times


def bench_shape(d_model: int, d_head: int, n_repeats: int = 100, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_model, d_head))
    B = rng.standard_normal((d_head, d_model))

    def naive():
        T = A @ B
        return np.linalg.eigvals(T)

    def reduced():
        M = B @ A
        return np.linalg.eigvals(M)

    # warm up BLAS / page faults before timing
    naive()
    reduced()

    t_naive = _time_call(naive, n_repeats)
    t_reduced = _time_call(reduced, n_repeats)
    # Report both median (typical case under whatever contention was present)
    # and min (the standard robust estimator for wall-clock microbenchmarks:
    # the fastest observed run is the closest proxy to an uncontended runtime,
    # since scheduler/contention noise only ever ADDS time, never subtracts
    # it). This machine was not exclusively quiesced for this run -- another
    # concurrent process was doing CPU-bound work -- so min is reported
    # alongside median rather than presenting median alone as ground truth.
    return {
        "d_model": d_model,
        "d_head": d_head,
        "theoretical_ratio": (d_model / d_head) ** 3,
        "naive_median_ms": float(np.median(t_naive) * 1e3),
        "naive_min_ms": float(np.min(t_naive) * 1e3),
        "naive_std_ms": float(np.std(t_naive, ddof=1) * 1e3),
        "reduced_median_ms": float(np.median(t_reduced) * 1e3),
        "reduced_min_ms": float(np.min(t_reduced) * 1e3),
        "reduced_std_ms": float(np.std(t_reduced, ddof=1) * 1e3),
        "measured_speedup_median": float(np.median(t_naive) / np.median(t_reduced)),
        "measured_speedup_min": float(np.min(t_naive) / np.min(t_reduced)),
    }


def run_all(n_repeats: int = 100) -> list[dict]:
    return [
        dict(label=label, **bench_shape(d_model, d_head, n_repeats))
        for label, d_model, d_head in SHAPES
    ]


if __name__ == "__main__":
    import json
    results = run_all()
    for r in results:
        print(
            f"{r['label']:24s} d_model={r['d_model']:5d} d_head={r['d_head']:4d}  "
            f"naive median={r['naive_median_ms']:8.3f}ms min={r['naive_min_ms']:8.3f}ms  "
            f"reduced median={r['reduced_median_ms']:7.4f}ms min={r['reduced_min_ms']:7.4f}ms  "
            f"speedup(median)={r['measured_speedup_median']:7.1f}x "
            f"speedup(min)={r['measured_speedup_min']:7.1f}x  "
            f"theoretical=({r['d_model']}/{r['d_head']})^3={r['theoretical_ratio']:7.1f}x"
        )
    with open("results/bench.json", "w") as f:
        json.dump(results, f, indent=1)
