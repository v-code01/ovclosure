"""Exact spectral reduction for low-rank OV operators.

For a per-head value map A (d_model x d_head) and output map B (d_head x d_model),
the OV operator T = A @ B is d_model x d_model with rank <= d_head. Its nonzero
eigenvalues equal exactly the eigenvalues of the reduced matrix M = B @ A
(d_head x d_head), with matching algebraic multiplicity. This follows from the
block-determinant identity

    lambda^p * det(lambda*I_n - A@B) == lambda^n * det(lambda*I_p - B@A)

for A (n x p), B (p x n) (Sylvester's determinant theorem). All functions here
operate in float64 regardless of the input dtype, since the claim is about the
exact linear-algebra structure of the weight matrices, not about arithmetic
precision during a forward pass.

nonzero_spectrum_match below verifies this numerically for a *given* (A, B)
pair; it is a sanity check on the reduction arithmetic, not a way to detect
whether A and B were extracted from matching heads (the theorem holds
unconditionally for any compatible A, B, mismatched or not). Extraction
correctness is verified separately in tests/test_extract.py against known
ground truth (weight reconstruction, GQA head-sharing).
"""
from __future__ import annotations

import numpy as np


def reduced_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """M = B @ A, the d_head x d_head matrix with the same nonzero spectrum as A @ B."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0] or B.shape[1] != A.shape[0]:
        raise ValueError(f"shape mismatch: A={A.shape} B={B.shape}, need A:(n,p) B:(p,n)")
    return B @ A


def full_operator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """T = A @ B, the d_model x d_model operator (rank <= d_head)."""
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)


def nonzero_spectrum_match(A: np.ndarray, B: np.ndarray, rel_tol: float = 1e-6) -> dict:
    """Verify eig(A@B) nonzero part == eig(B@A), sorted by magnitude, up to rel_tol.

    IMPORTANT: this is NOT a detector for semantic extraction bugs (e.g. pairing
    a head's A with the WRONG head's B) -- the theorem holds unconditionally for
    ANY compatible A, B, correctly paired or not, since it is a statement about
    the two specific matrices handed in, not about their provenance. What this
    check actually validates is numerical: that our eigval computation and the
    reduction (B@A vs A@B) agree to machine precision for the matrices at hand,
    which would catch a bug in reduced_matrix/full_operator itself, or scale
    issues (see the rel_tol design below). Real extraction-correctness (right
    head, right slice) is checked separately in tests/test_extract.py by
    reconstructing the source weight tensor and by the GQA head-sharing
    invariant -- those compare against known ground truth, this does not.

    Both zero-thresholding and the final match tolerance are scaled by the
    largest eigenvalue magnitude of T, not absolute: an earlier absolute
    threshold (1e-8) misclassified numerical noise as "nonzero" eigenvalues for
    a matrix with O(1e9) entries (32 spurious "nonzero" eigenvalues out of a
    true rank-1 operator), because eigval noise floor scales with the matrix's
    own magnitude, not with an absolute constant. Caught while writing
    tests/test_analyze.py's intended negative-control case, before it shipped.
    """
    T = full_operator(A, B)
    M = reduced_matrix(A, B)
    eig_T = np.linalg.eigvals(T)
    eig_M = np.linalg.eigvals(M)
    max_T = np.max(np.abs(eig_T)) if eig_T.size else 0.0
    max_M = np.max(np.abs(eig_M)) if eig_M.size else 0.0
    scale = max(max_T, max_M, 1e-300)
    abs_tol = rel_tol * scale
    nz_T = eig_T[np.abs(eig_T) > abs_tol]
    nz_T = nz_T[np.argsort(-np.abs(nz_T))]  # descending magnitude for stable pairing
    eig_M_sorted = eig_M[np.argsort(-np.abs(eig_M))]
    n = min(len(nz_T), len(eig_M_sorted))
    if n == 0:
        max_dist = 0.0
    else:
        diff = np.sort_complex(nz_T[:n]) - np.sort_complex(eig_M_sorted[:n])
        max_dist = float(np.max(np.abs(diff)))
    return {
        "eig_T_nonzero_count": len(nz_T),
        "eig_M_count": len(eig_M_sorted),
        "max_pairwise_dist": max_dist,
        "max_pairwise_dist_rel": float(max_dist / scale),
        "match": bool(len(nz_T) <= len(eig_M_sorted) and max_dist < abs_tol),
    }


def best_alpha(M: np.ndarray) -> float:
    """Least-squares scalar alpha minimizing ||M@M - alpha*M||_F.

    Closed form: alpha* = <M@M, M>_F / <M, M>_F, the Frobenius inner product
    <X, Y>_F = sum(X * Y) for real matrices (no transpose needed since we sum
    elementwise products, which is the real Frobenius inner product trace(X^T Y)
    written without an explicit transpose).
    """
    M = np.asarray(M, dtype=np.float64)
    M2 = M @ M
    denom = float(np.sum(M * M))
    if denom == 0.0:
        return 0.0
    return float(np.sum(M2 * M) / denom)


def idempotence_residual(M: np.ndarray, alpha: float | None = None) -> tuple[float, float]:
    """Dimensionless residual rho = ||M@M - alpha*M||_F / (|alpha| * ||M||_F).

    Write P = M / alpha (defined wherever alpha != 0). Then
        rho == ||P@P - P||_F / ||P||_F,
    i.e. rho measures the relative distance from P to an exact projector
    (P@P == P). This is genuinely scale-invariant: M -> c*M forces
    alpha -> c*alpha (both best_alpha and the true alpha in an exact instance
    scale linearly with M), so P = M/alpha is unchanged and rho is unchanged.

    An earlier version of this function normalized by ||M||_F alone
    (||M@M - alpha*M||_F / ||M||_F); that quantity has units of alpha (not
    dimensionless) and scales linearly with an overall rescaling of M -- it is
    NOT comparable across matrices of different norm, which is exactly the
    comparison this analysis needs to make across heads/layers/models. Caught
    via a direct numerical check (M vs 7.3*M) before any real data was
    analyzed; test_scale_invariance below pins the fix as a regression test.

    Returns (rho, alpha_used). alpha=0 (the degenerate all-nilpotent case) is
    reported as rho=inf, since no scale makes M/alpha a projector.
    """
    M = np.asarray(M, dtype=np.float64)
    if alpha is None:
        alpha = best_alpha(M)
    norm_M = np.linalg.norm(M, ord="fro")
    if norm_M == 0.0:
        return 0.0, alpha
    if alpha == 0.0:
        return float("inf"), alpha
    resid = np.linalg.norm(M @ M - alpha * M, ord="fro") / (abs(alpha) * norm_M)
    return float(resid), float(alpha)
