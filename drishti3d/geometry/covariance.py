"""BA covariance propagation: turning the normal equations into per-point uncertainty.

This is half of the project's actual novelty (the other half,
``observability.py``, turns this covariance into a corrective flight
plan). Existing photogrammetry pipelines report reprojection RMSE as their
one accuracy number; that hides the single most important fact about a
single-pass drone reconstruction, which is that uncertainty is
*anisotropic* -- lateral position is well constrained by parallax, but
depth along the viewing direction is not, especially in the near-collinear
strip geometry this whole project targets (see ``geometry.bundle``'s
module docstring). A single RMSE number cannot express "this point is
solid to 2 cm sideways but could be off by half a metre in depth"; a 3x3
covariance matrix can.

Why Schur complement, not a direct inverse
--------------------------------------------
The bundle adjustment normal-equation matrix ``H = J^T J`` (``J`` being
``BAResult.jacobian``, see ``geometry.bundle``) has the classic BA block
structure::

    H = [[A, B],
         [B^T, D]]

where ``A`` (camera-camera) and ``D`` (point-point, the "structure" block)
are each themselves **block-diagonal** -- no single reprojection residual
depends on two different cameras or two different points, so cross terms
between distinct cameras, or between distinct points, are exactly zero.
Only ``B`` (camera<->point) couples them. Forming ``H^{-1}`` directly
(dense) is ``O((6C + 3N)^3)`` for ``C`` cameras and ``N`` points -- for
even a few hundred points this is already impractical, and Ba: for real
reconstructions with tens of thousands of points it is impossible.

Instead we use the identity (Schur complement of ``A`` in ``H``, applied
via ``D``'s block-diagonal cheap invertibility rather than the usual
"eliminate structure to solve for motion" direction):

    S   = A - B @ D^{-1} @ B^T                      (camera Schur complement)
    Cov_points = D^{-1} + D^{-1} @ B^T @ S^{-1} @ B @ D^{-1}      (Woodbury identity)

``D^{-1}`` is cheap because ``D`` is block-diagonal (invert each point's
own 3x3 block independently). ``S`` has the size of the *camera* parameter
block only (typically far smaller than the point block), so ``S^{-1}`` is
cheap too. Critically, the per-point 3x3 diagonal block of ``Cov_points``
can be extracted **one point at a time** -- ``D_i^{-1} + D_i^{-1} B_i^T
S^{-1} B_i D_i^{-1}`` for point ``i``'s own 3x3 ``D_i`` block and
``B_i = B[:, point i's columns]`` -- without ever forming the dense
``(3N, 3N)`` matrix. This is what "do not invert the full matrix" means in
practice here.

Gauge / rank-deficiency handling (read before trusting a covariance number)
-----------------------------------------------------------------------------
``bundle_adjust`` removes the reconstruction's 7-DOF similarity gauge
freedom by holding some cameras' poses fixed (``BAConfig.fixed_gauge`` /
``BAProblem.fixed_camera_indices`` -- see ``geometry.bundle``). Given that,
``S`` (and ``H`` as a whole, restricted to the free parameters) *should*
be full rank for a well-posed problem. It is not, in general, "safely
invertible" in the near-collinear single-pass regime this project targets:
a flight strip can leave a direction in parameter space (e.g. roll about
the flight axis, or point depth along the mean viewing ray) with a very
small but *nonzero* eigenvalue in ``S`` -- that is not a gauge artefact,
it is a real, physical statement that the data barely constrains that
direction, and the correct covariance there is *large*, not
undefined/infinite and not zero.

We therefore invert ``S`` (and each small ``D_i`` block) with
``numpy.linalg.pinv`` at its *default*, near-machine-epsilon ``rcond``,
not a loose one. This distinguishes two cases that a naive
``numpy.linalg.inv`` cannot:

- A direction with a *tiny but nonzero* singular value (the collinear
  degeneracy) still gets inverted -- ``1/singular_value`` is large but
  finite, which is exactly the "huge, anisotropic uncertainty" signal
  this module exists to produce. ``numpy.linalg.inv`` would technically
  also do this, but blows up/raises or returns garbage the moment the
  matrix is numerically singular to working precision, which *does*
  happen in practice from floating-point round-off even when gauge is
  correctly fixed.
- A direction with an *exactly* (to machine precision) zero singular
  value -- which would only happen if gauge fixing were incomplete, e.g.
  a caller passed ``fixed_gauge=False`` with too few external priors --
  is mapped to zero contribution rather than a divide-by-zero/``inf``.
  This is the standard convention for reporting a covariance in a fixed
  gauge: directions with literally no information get no manufactured
  uncertainty (there is nothing principled to report), which is a
  deliberately conservative choice; it does mean a caller who disables
  gauge fixing and skips priors can get an artificially small covariance
  back, which is why ``bundle_adjust`` fixes gauge by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from drishti3d.geometry.bundle import BAResult
from drishti3d.types import Confidence

_PINV_RCOND = 1e-12


def _to_csr(mat) -> sparse.csr_matrix:
    if sparse.issparse(mat):
        return mat.tocsr()
    return sparse.csr_matrix(mat)


def _normal_equations(ba_result: BAResult) -> sparse.csr_matrix:
    if ba_result.jacobian is None:
        raise ValueError(
            "BAResult.jacobian is None (nothing was optimized -- see "
            "bundle_adjust's 'nothing to optimize' short-circuit); no "
            "covariance can be computed."
        )
    J = _to_csr(ba_result.jacobian)
    return (J.T @ J).tocsr()


def _schur_pieces(ba_result: BAResult) -> tuple[np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    """Return dense ``A`` (camera-camera), sparse ``B`` (camera-point), sparse ``D`` (point-point)."""
    layout = ba_result.param_layout
    H = _normal_equations(ba_result)
    pb = layout.points_base_col

    A = H[:pb, :pb].toarray() if pb > 0 else np.zeros((0, 0))
    B = H[:pb, pb:]
    D = H[pb:, pb:]
    return A, B, D


def _invert_point_blocks(D: sparse.csr_matrix, n_points: int) -> list[np.ndarray]:
    """Invert each point's 3x3 diagonal block of ``D`` independently (block-diagonal -- cheap)."""
    D = D.tocsr()
    d_inv = []
    for i in range(n_points):
        block = D[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].toarray()
        d_inv.append(np.linalg.pinv(block, rcond=_PINV_RCOND))
    return d_inv


def _camera_schur_complement(ba_result: BAResult) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], sparse.csr_matrix]:
    """Shared work between ``point_covariances`` and ``camera_covariances``.

    Returns ``(A, S_inv, d_inv_blocks, B)``.
    """
    A, B, D = _schur_pieces(ba_result)
    layout = ba_result.param_layout
    d_inv_blocks = _invert_point_blocks(D, layout.n_points)

    if A.shape[0] == 0:
        S_inv = np.zeros((0, 0))
        return A, S_inv, d_inv_blocks, B.tocsc()

    # B @ D^{-1} @ B^T, assembled point-block by point-block (D^{-1} is
    # block-diagonal, so this never touches a dense (3N,3N) matrix).
    B_csc = B.tocsc()
    BDinvBt = np.zeros_like(A)
    for i, d_inv in enumerate(d_inv_blocks):
        Bi = B_csc[:, 3 * i : 3 * i + 3].toarray()  # (n_cam_params, 3)
        if not Bi.any():
            continue
        BDinvBt += Bi @ d_inv @ Bi.T

    S = A - BDinvBt
    S_inv = np.linalg.pinv(S, rcond=_PINV_RCOND)
    return A, S_inv, d_inv_blocks, B_csc


def point_covariances(ba_result: BAResult, method: str = "schur") -> np.ndarray:
    """Per-point 3x3 covariance matrices, ``(N, 3, 3)``, via Schur complement (see module docstring).

    ``method`` is currently always ``"schur"`` (the parameter exists for
    interface stability / documentation; no other method is implemented,
    since a dense full-matrix inverse is exactly what this module exists
    to avoid).
    """
    if method != "schur":
        raise ValueError(f"unsupported covariance method {method!r}; only 'schur' is implemented")

    layout = ba_result.param_layout
    _, S_inv, d_inv_blocks, B = _camera_schur_complement(ba_result)

    covariances = np.zeros((layout.n_points, 3, 3), dtype=np.float64)
    for i in range(layout.n_points):
        d_inv = d_inv_blocks[i]
        if S_inv.size == 0:
            covariances[i] = d_inv
            continue
        Bi = B[:, 3 * i : 3 * i + 3].toarray()
        covariances[i] = d_inv + d_inv @ Bi.T @ S_inv @ Bi @ d_inv
    return covariances


def camera_covariances(ba_result: BAResult) -> np.ndarray:
    """Per-camera covariance of the full free camera parameter block, ``(n_cameras, k, k)``.

    ``k`` is ``param_layout.camera_param_size`` (6: rotvec + translation;
    10 if intrinsics were refined). Fixed/gauge-anchored cameras (see
    ``BAProblem.fixed_camera_indices``) get an all-zero block -- a fixed
    parameter has, by construction, zero variance in this estimation.

    The translation (position) sub-block is rows/columns ``3:6`` of each
    camera's block -- e.g. ``camera_covariances(result)[i, 3:6, 3:6]`` is
    the position-only covariance most report-card consumers actually want.
    """
    layout = ba_result.param_layout
    k = layout.camera_param_size
    _A, S_inv, _, _ = _camera_schur_complement(ba_result)

    out = np.zeros((layout.n_cameras, k, k), dtype=np.float64)
    for cam_idx, col in layout.camera_col.items():
        out[cam_idx] = S_inv[col : col + k, col : col + k]
    return out


# ---------------------------------------------------------------------------
# Ellipsoid / confidence-tier / anisotropy summaries
# ---------------------------------------------------------------------------


def covariance_to_ellipsoid(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a 3x3 covariance into ``(semi_axes, rotation)``.

    ``semi_axes`` (3,): 1-sigma semi-axis lengths, descending order.
    ``rotation`` (3,3): columns are the corresponding (unit) eigenvectors
    -- ``rotation[:, 0]`` is the direction of greatest uncertainty (the
    "worst-constrained direction"; see ``observability.py``).

    Works on a single ``(3,3)`` matrix or a batch ``(N,3,3)`` (returns
    ``(N,3)`` / ``(N,3,3)`` respectively).
    """
    cov = np.asarray(cov, dtype=np.float64)
    single = cov.ndim == 2
    if single:
        cov = cov[None, ...]

    # Symmetrize defensively -- Schur-complement round-off can leave a
    # covariance matrix very slightly asymmetric.
    cov_sym = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    # eigh returns ascending order; we want descending (largest first).
    order = np.argsort(-eigvals, axis=-1)
    eigvals_sorted = np.take_along_axis(eigvals, order, axis=-1)
    eigvecs_sorted = np.take_along_axis(eigvecs, order[:, None, :], axis=-1)

    semi_axes = np.sqrt(np.clip(eigvals_sorted, 0.0, None))

    if single:
        return semi_axes[0], eigvecs_sorted[0]
    return semi_axes, eigvecs_sorted


def anisotropy_ratio(cov: np.ndarray) -> np.ndarray | float:
    """Largest / smallest 1-sigma semi-axis -- the single number characterising single-pass degeneracy.

    ~1 for a well-conditioned, near-isotropic multi-view configuration;
    orders of magnitude larger for a collinear single-pass strip, where
    depth along the mean viewing ray is far less constrained than lateral
    position. Accepts a single ``(3,3)`` covariance or a batch ``(N,3,3)``.
    """
    semi_axes, _ = covariance_to_ellipsoid(cov)
    semi_axes = np.atleast_2d(semi_axes) if semi_axes.ndim == 1 and np.asarray(cov).ndim > 2 else semi_axes
    single = np.asarray(cov).ndim == 2
    smallest = semi_axes[..., -1]
    largest = semi_axes[..., 0]
    # Avoid divide-by-zero for a (numerically) exactly-zero smallest axis;
    # such a point has no meaningfully-defined ratio, so report +inf
    # rather than a fabricated finite number.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(smallest > 0, largest / np.maximum(smallest, 1e-300), np.inf)
    if single:
        return float(ratio)
    return ratio


@dataclass
class ConfidenceThresholds:
    """Metre thresholds mapping a point's largest 1-sigma semi-axis to a ``Confidence`` tier."""

    measured_max_m: float = 0.05
    low_confidence_max_m: float = 0.5


def confidence_from_covariance(
    cov: np.ndarray, thresholds: ConfidenceThresholds | None = None
) -> np.ndarray | Confidence:
    """Map per-point covariance to a ``types.Confidence`` tier via the largest semi-axis.

    This is what actually populates the trust layer with real
    uncertainty-propagated math (largest 1-sigma semi-axis of the BA
    covariance ellipsoid) rather than a heuristic proxy like raw backbone
    confidence or observation count alone. Accepts a single ``(3,3))``
    covariance or a batch ``(N,3,3)``.
    """
    thresholds = thresholds or ConfidenceThresholds()
    semi_axes, _ = covariance_to_ellipsoid(cov)
    single = np.asarray(cov).ndim == 2
    largest = semi_axes[..., 0] if not single else semi_axes[0]

    def _tier(value: float) -> Confidence:
        if value <= thresholds.measured_max_m:
            return Confidence.MEASURED
        if value <= thresholds.low_confidence_max_m:
            return Confidence.LOW_CONFIDENCE
        return Confidence.INFERRED

    if single:
        return _tier(float(largest))
    return np.array([_tier(float(v)) for v in largest], dtype=np.int64)
