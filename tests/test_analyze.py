import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import src.analyze as analyze_mod
from src.analyze import analyze_model
from src.extract import HeadRef


def test_analyze_gpt2_record_shape():
    records = analyze_model("gpt2", null_samples_per_shape=50)
    assert len(records) == 12 * 12
    for r in records:
        assert 0.0 <= r["null_percentile"] <= 100.0
        assert r["rho"] >= 0.0
        assert r["d_head"] == 64
        assert 0.0 <= r["depth_frac"] <= 1.0


def test_analyze_raises_if_reduction_theorem_ever_fails(monkeypatch):
    # analyze_model must treat a nonzero_spectrum_match failure as a hard
    # error, not a silent skip -- this guards against a future regression in
    # the eig/reduction code shipping wrong numbers instead of crashing
    # loudly. We force the failure directly (see src/theorem.py: the AB/BA
    # theorem holds for ANY compatible A, B regardless of provenance, so a
    # mismatched-head pair does NOT actually fail it and can't be used to
    # trigger this path -- that was the wrong test until caught here).
    rng = np.random.default_rng(0)
    heads = [
        HeadRef(layer=0, head=0, A=rng.standard_normal((32, 8)), B=rng.standard_normal((8, 32)))
    ]

    def fake_extract(model_key):
        return heads

    def fake_match(A, B):
        return {
            "match": False,
            "eig_T_nonzero_count": 0,
            "eig_M_count": 0,
            "max_pairwise_dist": 1.0,
        }

    monkeypatch.setattr(analyze_mod, "extract", fake_extract)
    monkeypatch.setattr(analyze_mod, "nonzero_spectrum_match", fake_match)
    with pytest.raises(RuntimeError):
        analyze_model("gpt2", null_samples_per_shape=10)
