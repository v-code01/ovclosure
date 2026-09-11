import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.theorem import (
    best_alpha,
    full_operator,
    idempotence_residual,
    nonzero_spectrum_match,
    reduced_matrix,
)


def test_spectrum_match_random_generic():
    rng = np.random.default_rng(0)
    for (n, p) in [(20, 5), (100, 8), (7, 7), (50, 1), (13, 12)]:
        A = rng.standard_normal((n, p))
        B = rng.standard_normal((p, n))
        r = nonzero_spectrum_match(A, B)
        assert r["match"], f"shape {(n,p)} failed: {r}"


def test_spectrum_match_rank_deficient_A():
    # A itself rank-deficient (fewer independent columns than p): still must hold,
    # since the theorem only assumes A is n x p and B is p x n, no rank hypothesis.
    rng = np.random.default_rng(1)
    n, p = 30, 6
    base = rng.standard_normal((n, 3))
    coeffs = rng.standard_normal((3, p))
    A = base @ coeffs  # rank <= 3 < p = 6
    B = rng.standard_normal((p, n))
    r = nonzero_spectrum_match(A, B)
    assert r["match"], r


def test_full_operator_shape_and_rank():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 6))
    B = rng.standard_normal((6, 40))
    T = full_operator(A, B)
    assert T.shape == (40, 40)
    assert np.linalg.matrix_rank(T) <= 6


def test_shape_mismatch_raises():
    A = np.zeros((10, 4))
    B = np.zeros((5, 10))
    with pytest.raises(ValueError):
        reduced_matrix(A, B)


def test_exact_scaled_projector_zero_residual():
    # Construct M = alpha * P where P is an oblique (non-orthogonal) projector,
    # i.e. P@P == P exactly. Then M@M == alpha*M exactly (up to float roundoff),
    # so the residual must be ~0 regardless of rank split or non-orthogonality.
    rng = np.random.default_rng(3)
    for d, r in [(10, 3), (16, 1), (16, 15), (8, 4)]:
        U = rng.standard_normal((d, r))
        Vt = rng.standard_normal((r, d))
        # Make U@Vt idempotent by normalizing so Vt @ U == I_r.
        VtU = Vt @ U
        Vt = np.linalg.solve(VtU, Vt)
        P = U @ Vt
        assert np.allclose(P @ P, P, atol=1e-9), "constructed P is not idempotent"
        alpha_true = rng.uniform(0.1, 5.0)
        M = alpha_true * P
        rho, alpha_fit = idempotence_residual(M)
        assert rho < 1e-8, f"expected ~0 residual for exact scaled projector, got {rho}"
        assert abs(alpha_fit - alpha_true) < 1e-6, f"fitted alpha {alpha_fit} != true {alpha_true}"


def test_generic_random_matrix_has_large_residual():
    # A generic random matrix should NOT be close to satisfying M@M = alpha*M;
    # this is the negative control that shows the residual metric actually
    # discriminates (a metric that reads ~0 on everything is useless).
    rng = np.random.default_rng(4)
    resids = []
    for _ in range(30):
        M = rng.standard_normal((32, 32))
        rho, _ = idempotence_residual(M)
        resids.append(rho)
    resids = np.array(resids)
    msg = f"expected generic random matrices to have large residual, got mean {resids.mean()}"
    assert resids.mean() > 0.3, msg


def test_scale_invariance_of_residual():
    # rho(M) must equal rho(c*M) for any nonzero c -- this pins the fix for a
    # real bug caught during development: an earlier formula normalized only
    # by ||M||_F, which has units of alpha and scales linearly with c instead
    # of staying fixed.
    rng = np.random.default_rng(6)
    for _ in range(10):
        M = rng.standard_normal((15, 15))
        c = rng.uniform(0.01, 50.0) * rng.choice([-1.0, 1.0])
        rho_M, _ = idempotence_residual(M)
        rho_cM, _ = idempotence_residual(c * M)
        assert abs(rho_M - rho_cM) < 1e-8 * max(1.0, rho_M), (rho_M, rho_cM, c)


def test_best_alpha_matches_closed_form_brute_force():
    rng = np.random.default_rng(5)
    M = rng.standard_normal((12, 12))
    alpha = best_alpha(M)
    # brute-force check alpha is (near) optimal by comparing to a fine grid
    grid = np.linspace(alpha - 2.0, alpha + 2.0, 4001)
    losses = [np.linalg.norm(M @ M - a * M, ord="fro") for a in grid]
    best_grid_alpha = grid[int(np.argmin(losses))]
    assert abs(best_grid_alpha - alpha) < 2e-3
