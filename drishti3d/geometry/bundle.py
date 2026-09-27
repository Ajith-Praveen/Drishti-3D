"""Sparse bundle adjustment: the accuracy stage of DRISHTI-3D.

Feed-forward backbones (MapAnything et al., see ``geometry.backbone``) and
GPS/telemetry-conditioned submap merging (``geometry.submap``) give a
*robust* initial reconstruction, but not an *accurate* one -- their errors
come from network biases, per-window independent scale/orientation fits,
and GPS noise, none of which get corrected just by averaging more windows
together. Bundle adjustment (BA) is the refinement step that actually
buys accuracy: it jointly re-optimizes every camera pose, every 3D point,
and (optionally) intrinsics against the full set of 2D observations that
produced them, minimizing reprojection error directly.

Why this module exists instead of GTSAM/Ceres/g2o
---------------------------------------------------
Those libraries are the "normal" way to do this, but they are heavy,
platform-fussy C++ dependencies with painful offline/air-gapped install
stories -- unacceptable for an installer that has to work on-site with no
network access. Everything here is built on ``numpy``/``scipy.sparse``
instead: an explicit analytic sparsity pattern (see ``_build_sparsity``)
feeding ``scipy.optimize.least_squares(method="trf", jac_sparsity=...)``,
which internally uses a sparse-aware finite-difference Jacobian (via graph
colouring over the sparsity pattern) and an iterative sparse linear solver
(LSMR) rather than ever forming/factoring a dense Hessian. This is exactly
the class of problem ``trf`` + ``jac_sparsity`` is designed for, and it
gives us the sparse Jacobian at the solution (``BAResult.jacobian``) for
free -- which ``covariance.py`` needs for the Schur-complement uncertainty
propagation that is this project's actual novelty.

The single-flight-strip problem this module has to survive
-------------------------------------------------------------
A single drone pass is a near-collinear camera trajectory (see
``geometry.submap`` and ``geometry.windows`` module docstrings for the
Sim(3)-alignment version of this same story). For bundle adjustment
specifically, near-collinearity means:

- Camera *centres* are well constrained by reprojection + GPS: they
  clearly have to move along the strip to explain the parallax seen in
  the images, and GPS pins down roughly where.
- Camera *roll/pitch* (tilt around/along the flight axis) are barely
  constrained by reprojection alone in this geometry: a small rotation of
  the whole rig about the flight-line axis moves image points by an
  amount that is easily absorbed by a compensating, equally small change
  in the 3D points' positions, so the reprojection cost barely notices
  it. Left unconstrained, the optimizer is free to drift into a tilted
  solution that fits the pixels just as well as the untilted (correct)
  one -- camera centres will look fine, but the whole point cloud comes
  out visibly rotated.
- Depth along the viewing direction is far less constrained than lateral
  position, for the same reason a narrow-baseline stereo pair gives poor
  depth: triangulation angle is small end-to-end along a single strip.

This module's answer to the *first* of those is the **gravity/Z-up
prior** (see ``_gravity_up_reference`` and the ``CameraPrior.gimbal_*``
fields): it directly penalizes camera tilt against IMU/gimbal attitude,
which is measured independently of the reprojection geometry and so is
not fooled by the strip's degeneracy. This is not a nice-to-have; it is
what prevents the tilted-model failure mode described above, and
``tests/test_bundle.py::test_gravity_prior_fixes_tilt_in_collinear_flight``
exists specifically to prove it. The *second* problem (depth
uncertainty) is not something BA can fix by construction -- it is a
property of the observation geometry -- but it is exactly what
``covariance.py``/``observability.py`` exist to quantify and then correct
for by planning a genuinely new viewing direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera-frame gravity reference (see module docstring: the gravity prior)
# ---------------------------------------------------------------------------
#
# Convention: world is ENU, Z-up (drishti3d.types module docstring). Camera
# frame is OpenCV (X right, Y down, Z forward/into the scene).
#
# We parametrize a camera's orientation, for the *purpose of the gravity
# prior only*, by (yaw, pitch, roll) via
#
#   R(yaw, pitch, roll) = Rz(yaw) @ R_CAM0 @ Rx(pitch) @ Rz(roll)
#
# where R_CAM0 is the fixed reference orientation at yaw=pitch=roll=0
# (camera level, boresight pointing along world +Y / "North"; see
# ``_R_CAM0`` below), pitch tilts the boresight down about the camera's own
# right axis (DJI convention: pitch=0 horizontal, pitch=-90 nadir/straight
# down), and roll rotates about the boresight. Yaw is a rotation about the
# *world* vertical axis, applied last (outermost).
#
# The key property this buys us: because Rz(yaw) rotates about the world Z
# axis, it leaves the world "up" vector [0,0,1] invariant. So
#
#   up_in_camera_frame = R(yaw, pitch, roll)^T @ [0, 0, 1]
#
# does not depend on yaw at all -- only on pitch and roll. That is exactly
# the physical content of a gravity/level prior: gimbal pitch and roll are
# measured relative to the horizon (independent of compass heading), and a
# prior built this way constrains tilt without fighting yaw, which
# reprojection is perfectly capable of constraining on its own.
_R_CAM0 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rotx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gimbal_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build a world-from-camera rotation matrix from (yaw, pitch, roll) degrees.

    Yaw is internal ENU yaw: counterclockwise from North about world +Z.
    Negate a clockwise compass heading before calling this function.

    See the module-level convention note above ``_R_CAM0``. Used to
    synthesize ground truth / expected orientations from telemetry-style
    Euler angles; ``_gravity_up_reference`` is the yaw-invariant quantity
    actually used as the BA prior.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _rotz(yaw) @ _R_CAM0 @ _rotx(pitch) @ _rotz(roll)


def _gravity_up_reference(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """World "up" [0,0,1] expressed in the camera frame, from gimbal pitch/roll.

    Yaw-invariant by construction (see the convention note above
    ``_R_CAM0``): this is the quantity the gravity prior residual compares
    against ``R_est.T @ [0, 0, 1]``, so an error in yaw/heading never gets
    penalized by this prior, only genuine tilt.
    """
    pitch, roll = np.radians([pitch_deg, roll_deg])
    r_local = _R_CAM0 @ _rotx(pitch) @ _rotz(roll)
    return r_local.T @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Problem / prior data structures
# ---------------------------------------------------------------------------


@dataclass
class CameraPrior:
    """Optional external priors for one camera, addressed by index into ``BAProblem.cameras``.

    gps_position / gps_sigma_m:
        Camera-centre position prior in world (ENU) metres and its 1-sigma
        uncertainty. Use the GPS receiver's own reported horizontal/vertical
        accuracy when available (see ``types.GeoPoint.accuracy_h`` /
        ``accuracy_v``); fall back to ``BAConfig.gps_sigma_m_default``
        (several metres -- standalone-GPS-realistic) otherwise. This is
        *not* an absolute-accuracy statement about the final map (see
        ``georef.py`` for that split) -- it is just how hard the optimizer
        should trust this one camera's GPS fix relative to reprojection.
    gimbal_pitch_deg / gimbal_roll_deg / tilt_sigma_deg:
        Gravity/level prior source (IMU or gimbal attitude report -- both
        are, physically, a measurement of the local vertical, which is
        what this prior actually constrains; see the module docstring's
        "gravity prior" discussion). ``tilt_sigma_deg`` is how much we
        trust that attitude reading; 1-3 degrees is typical for a
        stabilized camera gimbal.
    """

    camera_idx: int
    gps_position: np.ndarray | None = None
    gps_sigma_m: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_roll_deg: float | None = None
    tilt_sigma_deg: float | None = None


@dataclass
class GCP:
    """A ground control point: a 3D point index tied to an independently surveyed position.

    ``sigma_m`` should reflect the survey method's accuracy (centimetres
    for RTK/total-station GCPs), not GPS-on-the-drone accuracy -- GCPs are
    the mechanism by which absolute accuracy can actually be tightened
    below standalone-GPS-bias levels (see ``georef.py``).
    """

    point_idx: int
    xyz: np.ndarray
    sigma_m: float = 0.02


@dataclass
class BAProblem:
    """A bundle adjustment problem instance: cameras, points, and 2D observations.

    cameras / intrinsics:
        One ``Pose`` (world-from-camera) and one ``CameraIntrinsics`` per
        camera, same order/length.
    points:
        ``(N, 3)`` world-frame (ENU) 3D point positions.
    obs_camera_idx / obs_point_idx / obs_uv:
        Parallel ``(M,)`` / ``(M,)`` / ``(M, 2)`` arrays: observation ``k``
        is "point ``obs_point_idx[k]`` was seen in camera
        ``obs_camera_idx[k]`` at pixel ``obs_uv[k]``".
    camera_priors:
        Optional ``CameraPrior`` entries (GPS position / gravity), one per
        constrained camera (cameras with no entry get no prior factor).
    gcps:
        Optional ``GCP`` entries.
    fixed_camera_indices:
        Camera indices excluded from optimization -- their ``cameras[i]``
        pose is treated as a constant. See ``BAConfig.fixed_gauge`` for why
        this matters and the default policy for choosing these. Note this
        fixes that camera's *entire* parameter block, not just pose: if
        ``BAConfig.refine_intrinsics`` is also set, a fixed camera's
        intrinsics are held constant too (gauge-fixing is about removing
        pose/scale ambiguity, but there is no separate mechanism to fix
        pose while still refining that same camera's intrinsics -- seed
        ``intrinsics[i]`` correctly for any camera you plan to fix).
    """

    cameras: list[Pose]
    intrinsics: list[CameraIntrinsics]
    points: np.ndarray
    obs_camera_idx: np.ndarray
    obs_point_idx: np.ndarray
    obs_uv: np.ndarray
    camera_priors: list[CameraPrior] = field(default_factory=list)
    gcps: list[GCP] = field(default_factory=list)
    fixed_camera_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.obs_camera_idx = np.asarray(self.obs_camera_idx, dtype=np.int64).reshape(-1)
        self.obs_point_idx = np.asarray(self.obs_point_idx, dtype=np.int64).reshape(-1)
        self.obs_uv = np.asarray(self.obs_uv, dtype=np.float64).reshape(-1, 2)
        n_obs = self.obs_camera_idx.shape[0]
        if self.obs_point_idx.shape[0] != n_obs or self.obs_uv.shape[0] != n_obs:
            raise ValueError("obs_camera_idx / obs_point_idx / obs_uv must have matching length")
        if len(self.cameras) != len(self.intrinsics):
            raise ValueError("cameras and intrinsics must have matching length")


@dataclass
class BAConfig:
    """Bundle adjustment options.

    fix_intrinsics / refine_intrinsics:
        Complementary flags on the same decision: whether ``fx, fy, cx,
        cy`` become free optimization variables, refined *independently
        per camera* (each camera in ``BAProblem.cameras`` gets its own
        4 intrinsics parameters, matching ``types.Keyframe.intrinsics``
        being per-keyframe in this codebase -- e.g. from a per-frame
        EXIF/estimator pass in ``ingest.intrinsics`` -- rather than
        assuming one fixed shared lens for the whole flight).
        ``refine_intrinsics`` is authoritative when the two disagree (e.g.
        both left at their defaults is ``fix_intrinsics=True,
        refine_intrinsics=False``, meaning fixed; explicitly passing
        ``refine_intrinsics=True`` refines regardless of
        ``fix_intrinsics``).

        Caution: refining a camera's intrinsics *and* its pose
        simultaneously from that camera's own observations alone is a
        classic ambiguous case -- a camera's focal length and its depth
        along the viewing axis trade off against each other (moving a
        camera back while proportionally increasing its focal length
        reproduces the same image), and reprojection error alone cannot
        tell them apart. An external position constraint (a GPS prior --
        see ``CameraPrior.gps_position``/``gps_sigma_m`` -- or a fixed
        pose) on that same camera removes the ambiguity by pinning depth
        independently of reprojection. Without one, refined intrinsics can
        converge to a self-consistent but physically wrong value even as
        reprojection RMSE goes to (numerically) zero -- see
        ``tests/test_bundle.py::test_refine_intrinsics_recovers_focal_length``
        for a worked example and its GPS-prior fix.
    robust_loss:
        ``"huber"``, ``"cauchy"``, or ``None`` (plain least squares),
        passed straight through to ``scipy.optimize.least_squares``. All
        residuals (reprojection and priors) are whitened by their assumed
        1-sigma noise before the optimizer sees them (see module docstring
        of ``_residuals``), so a residual of magnitude ~1 is "as expected"
        regardless of which factor it came from -- ``f_scale`` is
        therefore expressed in those whitened sigma units, and the default
        of 1.345 is the textbook Huber tuning constant for ~95% efficiency
        under Gaussian noise.
    max_iterations:
        Upper bound on optimizer iterations. TRF does not expose a
        separate outer-iteration count, so this is turned into scipy's
        own ``max_nfev`` (residual-*evaluation* budget, not outer
        iterations) via ``max_nfev`` below when that field is left at its
        default -- see that field's docstring for why this is no longer
        ``max_iterations * n_params`` (the old formula, and the actual
        root cause of bundle adjustment costing 5x the neural backbone on
        real footage -- see the module docstring's profiling note).
    max_nfev:
        Explicit override for scipy's ``max_nfev`` (total residual-
        function evaluations, not outer iterations). ``None`` (default)
        derives ``max_iterations * 30`` -- a constant per-"iteration"
        allowance, *not* scaled by the parameter count.

        Why not scale by parameter count (the old, buggy default)
        ------------------------------------------------------------
        The old default was ``max_iterations * n_params``, i.e. up to
        tens of thousands of evaluations even for a modest problem (720
        params -> 72,000). That reasoning silently assumed a from-scratch
        finite-difference Jacobian costs one evaluation per parameter,
        but ``jac_sparsity`` (see module docstring) makes scipy compute
        the FD Jacobian via graph colouring, whose per-iteration
        evaluation cost is bounded by the sparsity pattern's *local*
        density (roughly ``camera_param_size + 3`` for this block
        structure) -- a small constant independent of the total parameter
        count, not something that grows with problem size. Multiplying by
        ``n_params`` anyway did not buy extra convergence robustness; it
        only raised the ceiling an already-slow tail could grind up to.

        On real (noisy, never-exactly-zero-residual) data this ceiling
        gets hit almost every time: profiling a real 8-keyframe run
        (see the module docstring) showed cost dropping from 780 to ~48
        (whitened units) within the first ~100 evaluations, then crawling
        from 48.0 to 47.3 -- a 1.5% further improvement, invisible in the
        final reprojection RMSE -- over the *next 41,000* evaluations
        before finally tripping ``ftol``. That slow tail is an inherent
        property of a numerically-differentiated Jacobian chasing a
        noise-limited residual floor (real pixel/matching noise means the
        true minimum is never exactly reached), not a sign that more
        iterations would have bought more accuracy; a much smaller,
        constant budget reaches the same practical accuracy in a small
        fraction of the time (see ``tests/test_bundle.py`` and the
        module's profiling notes for before/after numbers).
    fixed_gauge:
        Whether to remove the 7-DOF similarity gauge freedom (3 rotation +
        3 translation + 1 scale) of the reconstruction by holding some
        cameras' poses fixed, when the caller hasn't already specified
        ``BAProblem.fixed_camera_indices`` explicitly (e.g. via
        ``windowed_bundle_adjust``'s inter-window anchors). See
        ``_default_gauge_fix`` for the exact policy and why it removes all
        7 DOF, not just 6. When GPS position priors (plus gravity priors
        for a straight strip) already define position, scale and
        orientation, no camera is fixed: the priors ARE the datum, and
        freezing two seed poses would only import their errors.
    refine_shared_focal / refine_distortion:
        Shared lens self-calibration: one physical lens for every camera
        of a video. ``refine_shared_focal`` adds ONE multiplicative factor
        on every camera's ``fx``/``fy``; ``refine_distortion`` adds ONE
        OpenCV radial pair ``(k1, k2)`` that replaces every camera's
        ``dist_coeffs[0:2]`` (tangential ``p1, p2`` and ``k3`` stay as
        given). Both apply to fixed cameras too -- the lens does not change
        because a pose is held for gauge.

        Why distortion matters here: an unmodelled radial lens error is the
        textbook cause of "doming" in parallel-axis (nadir) blocks. The
        optimizer cannot move pixels, so it bends the camera strip into a
        bowl instead. flight01 (k1 ~ -0.2, ~70 px at the frame edges)
        converged to 0.4 px reprojection with cameras tilted up to 33 deg,
        heights bowed +/-20 m and the ground 30 m too close; with the
        shared pair refined the same tracks gave 0.2 px, cameras 1.8 m from
        GPS and flat heights.

        Focal length and ground depth trade off exactly over flat terrain
        (``(f, k1, k2, depth) ~ (c f, c^2 k1, c^4 k2, c depth)``), so only
        refine the shared focal when it is a guess; a measured focal is
        the better depth anchor.
    """

    fix_intrinsics: bool = True
    refine_intrinsics: bool = False
    refine_shared_focal: bool = False
    refine_distortion: bool = False
    robust_loss: str | None = "huber"
    f_scale: float = 1.345
    max_iterations: int = 100
    max_nfev: int | None = None
    pixel_sigma_px: float = 1.0
    # 1-sigma of a GPS position prior when the log gives none. 2.5 m (a
    # consumer receiver's horizontal accuracy) rather than 5: with the exact
    # Schur solver, flight01's camera track came out closer to COLMAP's
    # 720-camera solution at 2.5 m than at 5 m (p90 4.7 vs 6.2 m, vertical
    # 1.1 vs 1.6 m); weaker priors let the single-pass block bend (8x: p90 15 m).
    gps_sigma_m_default: float = 2.5
    # A GPS fix's vertical sigma, as a multiple of its horizontal one.
    # Consumer GPS altitude drifts by metres between passes; at 1 (the old
    # isotropic prior) a block whose strips are tied by image matches is
    # tilted to fit those altitudes, moving the ground horizontally.
    gps_vertical_sigma_factor: float = 1.0
    tilt_sigma_deg_default: float = 2.0
    # Residual evaluations for a LINEAR-loss warm start before a robust
    # solve (0 disables). A Huber loss scaled to ~1 px caps every residual's
    # gradient while the start is many pixels off, so trust-region steps
    # stay tiny and scipy declares convergence with the cameras untouched:
    # flight01's pose prior stopped at 18.6 px after 75 evaluations having
    # moved no camera at all. 100 linear evaluations reached 0.79 px and
    # corrected the gimbal tilt by ~6 deg; the robust pass then refines.
    warm_start_linear_nfev: int = 100
    # Loss of that warm start (``"linear"`` or a robust loss at
    # ``warm_start_f_scale`` px). Plain least squares follows a few
    # inconsistent cameras anywhere: flight01's 20 s clip at the correct
    # clock (three banking end-of-leg frames at 2.6 px) warm-started into
    # the bowl solution, which the tight robust pass could not leave.
    warm_start_loss: str = "linear"
    warm_start_f_scale: float = 1.0
    fixed_gauge: bool = True
    # Relative cost-decrease tolerance. 1e-10 is never met on real, noisy
    # tracks: the robust pass always ran to max_nfev (3000 evaluations on
    # flight01's 12-camera clip, 26 s) while 1e-6 stops after 17 with
    # cameras within 1.4 cm / 0.008 deg of that answer (8 s).
    # Log-space 1-sigma of a prior holding the shared focal factor at 1.
    # Over flat ground seen straight down, focal length and ground depth
    # trade off exactly and GPS pins only the cameras, not the ground:
    # DJI_1001 (no altitude log) ran to 2.2x the guess. The prior only
    # decides that unobservable direction; real evidence (relief, oblique
    # views) still moves it. None disables.
    # 0.01, not 0.3: the Schur solver resolves this direction exactly. On
    # DJI_1001 (nadir, no ground-height reference) the images pulled the
    # focal 1066 -> 1720 px at 0.3 and still to 1640 px at 0.05, sinking the
    # DSM from the real terrain (147-177 m, Austin) to ~0-35 m. A focal that
    # only a guess supports stays put; lens distortion is still solved.
    shared_focal_prior_sigma: float | None = 0.01
    ftol: float = 1e-6
    xtol: float = 1e-8
    gtol: float = 1e-10
    verbose: int = 0
    # Options for the inner LSMR solve of each trust-region step (scipy's
    # ``tr_options``). None keeps scipy's defaults: tolerances 1e-6 and up to
    # min(m, n) inner iterations per step. Capping them was tried on
    # flight01 and rejected: 4x faster, but steps then miss the directions
    # only the GPS priors pin, and cameras ended 4.5 m from GPS instead of 2.6.
    lsmr_options: dict | None = None
    # "schur": Levenberg-Marquardt on the reduced camera system -- every
    # point's 3x3 block is eliminated (Schur complement) and the ~1k camera
    # and lens unknowns are solved exactly each step, the standard bundle
    # adjustment solver. "scipy": least_squares(trf, lsmr).
    solver: str = "schur"
    # Schur LM stopping rules: relative cost decrease per accepted step. The
    # linear warm start only has to reach the basin (flight01: 33.5 M ->
    # 39.7 k in 10 steps, then 0.4% per step); the robust pass passes
    # scipy's final cost by step 5 and then creeps (~1e-4 per step).
    schur_warm_ftol: float = 1e-2
    schur_ftol: float = 1e-4


# Constant per-``max_iterations`` unit residual-evaluation allowance behind
# ``BAConfig.max_nfev``'s default -- see that field's docstring for why this
# replaces the old ``max_iterations * n_params`` formula. 30 comfortably
# covers this problem's sparse-coloured FD Jacobian (bounded by local
# sparsity density, typically well under 20 colours for this block
# structure) plus a handful of trust-region backtracking evaluations per
# real outer iteration, without leaving room for the unbounded-tail grind
# a parameter-count-scaled cap allowed on real, noisy data.
_MAX_NFEV_PER_ITERATION = 30


def _default_max_nfev(max_iterations: int) -> int:
    """Default residual-evaluation budget for ``least_squares``' ``max_nfev``.

    See ``BAConfig.max_nfev``'s docstring for the full rationale: this is
    deliberately a constant multiple of ``max_iterations``, not scaled by
    the parameter count.
    """
    return max(1, max_iterations) * _MAX_NFEV_PER_ITERATION


@dataclass
class ParamLayout:
    """Maps ``BAProblem`` cameras/points onto columns of the optimization vector ``x``.

    Consumed by ``covariance.py`` to slice the normal-equation matrix
    ``J^T J`` back into per-camera / per-point blocks for the Schur
    complement -- it needs to know exactly which columns are which without
    re-deriving this bookkeeping itself. The shared lens block (if any)
    sits between the camera blocks and the points, so everything before
    ``points_base_col`` is the reduced camera system.
    """

    n_cameras: int
    n_points: int
    fixed_camera_indices: set[int]
    camera_param_size: int  # 6, or 10 if intrinsics are refined
    refine_intrinsics: bool
    camera_col: dict[int, int]  # free camera_idx -> starting column
    points_base_col: int  # points_base_col + 3*point_idx -> point's starting column
    n_params: int
    lens_col: int | None = None  # shared lens block: [focal factor][k1, k2]
    lens_size: int = 0
    refine_shared_focal: bool = False
    refine_distortion: bool = False

    def point_cols(self, point_idx: int) -> slice:
        c = self.points_base_col + 3 * point_idx
        return slice(c, c + 3)

    def camera_cols(self, camera_idx: int) -> slice | None:
        c = self.camera_col.get(camera_idx)
        if c is None:
            return None
        return slice(c, c + self.camera_param_size)


@dataclass
class BAResult:
    """Output of ``bundle_adjust``.

    jacobian:
        The final sparse Jacobian of the *whitened* residual vector
        (``scipy.sparse`` matrix, shape ``(n_residuals, n_params)``) at the
        converged solution. Whitened means each residual was already
        divided by its assumed 1-sigma noise before scipy ever saw it
        (see ``_residuals``), so ``jacobian.T @ jacobian`` is directly the
        Gauss-Newton approximation to the Fisher information matrix in
        physical units -- exactly what ``covariance.py`` needs, with no
        extra sigma bookkeeping required on its end.
    residuals_px:
        Final per-observation reprojection residual, ``(M, 2)`` pixels
        (*not* whitened -- this is for human/report consumption).
    """

    poses: list[Pose]
    points: np.ndarray
    intrinsics: list[CameraIntrinsics]
    residuals_px: np.ndarray
    rmse_before_px: float
    rmse_after_px: float
    n_iterations: int
    converged: bool
    jacobian: object  # scipy.sparse matrix
    param_layout: ParamLayout
    message: str = ""
    #: Shared lens solution when one was refined: ``focal_factor``, ``k1``, ``k2``.
    lens: dict | None = None


# ---------------------------------------------------------------------------
# Parameter (de)serialization
# ---------------------------------------------------------------------------


def _priors_fix_datum(problem: BAProblem, config: BAConfig | None = None) -> bool:
    """True when the camera priors alone pin position, scale and orientation.

    GPS positions on three or more cameras spread well beyond their noise
    fix translation, scale and heading. A straight strip leaves rotation
    about the flight line to the gravity priors, so collinear GPS needs at
    least one tilt prior as well.
    """
    fixed = problem.fixed_camera_indices
    gps = [p for p in problem.camera_priors if p.gps_position is not None and p.camera_idx not in fixed]
    if len(gps) < 3:
        return False
    default_sigma = config.gps_sigma_m_default if config is not None else 5.0
    pos = np.array([p.gps_position for p in gps], dtype=np.float64).reshape(-1, 3)
    sigma = float(np.median([p.gps_sigma_m if p.gps_sigma_m is not None else default_sigma for p in gps]))
    spread = np.linalg.svd(pos - pos.mean(axis=0), compute_uv=False) / np.sqrt(len(pos))
    if not np.isfinite(spread).all() or spread[0] < 4.0 * sigma:
        return False
    has_tilt = any(p.gimbal_pitch_deg is not None and p.camera_idx not in fixed for p in problem.camera_priors)
    return has_tilt or spread[1] >= 4.0 * sigma


def _default_gauge_fix(problem: BAProblem, config: BAConfig | None = None) -> set[int]:
    """Default gauge-fixing policy: none when priors define the datum, else the first two cameras.

    With GPS on most cameras (and gravity for a straight strip) the priors
    already remove all 7 similarity DOF, so nothing is fixed. Fixing seed
    poses anyway is actively harmful: their priors are dropped and their
    seed rotation becomes a hard constraint. flight01's clip starts in a
    turn where cameras 0 and 1 had ~29 deg yaw errors; freezing them
    rotated the whole solution ~30 deg and moved the far end 169 m.

    Without such priors, hold the first (up to) two cameras fully fixed.
    Bundle adjustment here always starts from an already-reasonable
    initial guess (feed-forward backbone + GPS/gravity-conditioned submap
    merge), so "adjustment" -- small local refinement -- is the right
    mental model, and anchoring gauge to two of the *given* initial poses
    is safe. Fixing camera 0's full pose removes the 3 rotation + 3
    translation DOF of "where is the world frame" (6 DOF); on its own that
    still leaves global *scale* free (the 7th DOF: a single similarity
    transform can rescale everything around the fixed camera 0 and still
    fit reprojections identically, since reprojection is scale
    ambiguous -- this is the classic monocular SfM scale ambiguity).
    Additionally fixing camera 1's position pins the camera 0 <-> camera 1
    distance, which removes scale too. With fewer than 2 cameras there is
    only one to fix (translation/rotation gauge still matters, though a
    single-camera problem has no meaningful scale to fix in the first
    place).
    """
    n = len(problem.cameras)
    if n <= 1:
        return set(range(n))
    if _priors_fix_datum(problem, config):
        return set()
    return {0, 1}


def _build_layout(problem: BAProblem, config: BAConfig) -> ParamLayout:
    refine_intrinsics = bool(config.refine_intrinsics)  # authoritative per docstring
    if refine_intrinsics and config.refine_shared_focal:
        raise ValueError("refine_intrinsics (per-camera) and refine_shared_focal are mutually exclusive")
    camera_param_size = 10 if refine_intrinsics else 6

    camera_col: dict[int, int] = {}
    col = 0
    for i in range(len(problem.cameras)):
        if i in problem.fixed_camera_indices:
            continue
        camera_col[i] = col
        col += camera_param_size

    lens_size = int(bool(config.refine_shared_focal)) + 2 * int(bool(config.refine_distortion))
    lens_col = col if lens_size else None
    col += lens_size

    points_base_col = col
    n_points = problem.points.shape[0]
    col += 3 * n_points

    return ParamLayout(
        n_cameras=len(problem.cameras),
        n_points=n_points,
        fixed_camera_indices=set(problem.fixed_camera_indices),
        camera_param_size=camera_param_size,
        refine_intrinsics=refine_intrinsics,
        camera_col=camera_col,
        points_base_col=points_base_col,
        n_params=col,
        lens_col=lens_col,
        lens_size=lens_size,
        refine_shared_focal=bool(config.refine_shared_focal),
        refine_distortion=bool(config.refine_distortion),
    )


def _dist5(k: CameraIntrinsics) -> np.ndarray:
    """OpenCV ``(k1, k2, p1, p2, k3)``; missing coefficients are zero."""
    out = np.zeros(5, dtype=np.float64)
    if k.dist_coeffs is not None:
        d = np.asarray(k.dist_coeffs, dtype=np.float64).reshape(-1)[:5]
        out[: d.size] = d
    return out


def _lens_x0(problem: BAProblem, layout: ParamLayout) -> np.ndarray:
    values: list[float] = []
    if layout.refine_shared_focal:
        values.append(1.0)
    if layout.refine_distortion:
        d = _dist5(problem.intrinsics[0]) if problem.intrinsics else np.zeros(5)
        values += [float(d[0]), float(d[1])]
    return np.asarray(values, dtype=np.float64)


def _apply_lens(K_all: np.ndarray, D_all: np.ndarray, lens: np.ndarray, layout: ParamLayout):
    """Per-camera ``(fx, fy, cx, cy)`` / distortion arrays with the shared lens block applied."""
    if not layout.lens_size:
        return K_all, D_all
    k = 0
    if layout.refine_shared_focal:
        K_all = K_all.copy()
        K_all[:, 0:2] *= lens[0]
        k = 1
    if layout.refine_distortion:
        D_all = D_all.copy()
        D_all[:, 0] = lens[k]
        D_all[:, 1] = lens[k + 1]
    return K_all, D_all


def _project(R: np.ndarray, t: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray):
    """Pixel ``(u, v)`` of world points ``P`` through world-from-camera ``(R, t)``, OpenCV lens model."""
    Xc = np.einsum("mji,mj->mi", R, P - t)
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (D[:, 0] + r2 * (D[:, 1] + r2 * D[:, 4]))
    xd = x * radial + 2.0 * D[:, 2] * x * y + D[:, 3] * (r2 + 2.0 * x * x)
    yd = y * radial + D[:, 2] * (r2 + 2.0 * y * y) + 2.0 * D[:, 3] * x * y
    return K[:, 0] * xd + K[:, 2], K[:, 1] * yd + K[:, 3]


def _focal_prior_residual(x: np.ndarray, layout: ParamLayout, sizes: dict) -> np.ndarray:
    """Whitened log of the shared focal factor (see ``BAConfig.shared_focal_prior_sigma``)."""
    if not sizes.get("n_focal"):
        return np.zeros(0)
    return np.array([np.log(max(float(x[layout.lens_col]), 1e-9)) / sizes["focal_sigma"]])


def _pack_x(problem: BAProblem, layout: ParamLayout) -> np.ndarray:
    x = np.zeros(layout.n_params, dtype=np.float64)
    for i, pose in enumerate(problem.cameras):
        c = layout.camera_col.get(i)
        if c is None:
            continue
        rotvec = Rotation.from_matrix(pose.R).as_rotvec()
        x[c : c + 3] = rotvec
        x[c + 3 : c + 6] = pose.t
        if layout.refine_intrinsics:
            k = problem.intrinsics[i]
            x[c + 6 : c + 10] = [k.fx, k.fy, k.cx, k.cy]
    if layout.lens_size:
        x[layout.lens_col : layout.lens_col + layout.lens_size] = _lens_x0(problem, layout)
    x[layout.points_base_col : layout.points_base_col + 3 * layout.n_points] = problem.points.reshape(-1)
    return x


def _unpack_x(
    x: np.ndarray, problem: BAProblem, layout: ParamLayout
) -> tuple[list[Pose], np.ndarray, list[CameraIntrinsics]]:
    poses: list[Pose] = []
    intrinsics: list[CameraIntrinsics] = []
    for i, orig_pose in enumerate(problem.cameras):
        c = layout.camera_col.get(i)
        if c is None:
            poses.append(orig_pose)
            intrinsics.append(problem.intrinsics[i])
            continue
        rotvec = x[c : c + 3]
        t = x[c + 3 : c + 6]
        R = Rotation.from_rotvec(rotvec).as_matrix()
        poses.append(Pose(R=R, t=t.copy()))
        if layout.refine_intrinsics:
            fx, fy, cx, cy = x[c + 6 : c + 10]
            orig_k = problem.intrinsics[i]
            intrinsics.append(
                CameraIntrinsics(
                    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                    width=orig_k.width, height=orig_k.height, dist_coeffs=orig_k.dist_coeffs,
                )
            )
        else:
            intrinsics.append(problem.intrinsics[i])
    if layout.lens_size:
        lens = x[layout.lens_col : layout.lens_col + layout.lens_size]
        K_all = np.array([[k.fx, k.fy, k.cx, k.cy] for k in intrinsics], dtype=np.float64)
        D_all = np.stack([_dist5(k) for k in intrinsics])
        K_all, D_all = _apply_lens(K_all, D_all, lens, layout)
        intrinsics = [
            CameraIntrinsics(
                fx=float(kv[0]), fy=float(kv[1]), cx=float(kv[2]), cy=float(kv[3]),
                width=k.width, height=k.height,
                dist_coeffs=dv.copy() if (layout.refine_distortion or k.dist_coeffs is not None) else None,
            )
            for k, kv, dv in zip(intrinsics, K_all, D_all, strict=True)
        ]
    points = x[layout.points_base_col : layout.points_base_col + 3 * layout.n_points].reshape(-1, 3).copy()
    return poses, points, intrinsics


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------


def _residual_layout_sizes(problem: BAProblem, layout: ParamLayout | None = None, config: BAConfig | None = None) -> dict:
    """How many residual rows each factor type contributes, and in what order."""
    n_reproj = 2 * problem.obs_camera_idx.shape[0]
    gps_priors = [p for p in problem.camera_priors if p.gps_position is not None and p.camera_idx not in problem.fixed_camera_indices]
    tilt_priors = [
        p
        for p in problem.camera_priors
        if p.gimbal_pitch_deg is not None and p.camera_idx not in problem.fixed_camera_indices
    ]
    n_gps = 3 * len(gps_priors)
    n_tilt = 3 * len(tilt_priors)
    n_gcp = 3 * len(problem.gcps)
    focal_sigma = config.shared_focal_prior_sigma if config is not None else None
    n_focal = int(bool(layout is not None and layout.refine_shared_focal and focal_sigma))
    return {
        "n_reproj": n_reproj,
        "gps_priors": gps_priors,
        "n_gps": n_gps,
        "tilt_priors": tilt_priors,
        "n_tilt": n_tilt,
        "n_gcp": n_gcp,
        "n_focal": n_focal,
        "focal_sigma": float(focal_sigma) if n_focal else None,
        "total": n_reproj + n_gps + n_tilt + n_gcp + n_focal,
    }


def _fast_constants(problem: BAProblem, layout: ParamLayout, config: BAConfig) -> dict:
    """Everything ``_residuals`` needs that does not depend on ``x``, as arrays. Built once per solve."""
    n = len(problem.cameras)
    free = np.array([i for i in range(n) if layout.camera_col.get(i) is not None], dtype=np.int64)
    cols = np.array([layout.camera_col[i] for i in free], dtype=np.int64)
    R0 = np.stack([np.asarray(p.R, dtype=np.float64) for p in problem.cameras])
    t0 = np.stack([np.asarray(p.t, dtype=np.float64).reshape(3) for p in problem.cameras])
    K0 = np.array([[k.fx, k.fy, k.cx, k.cy] for k in problem.intrinsics], dtype=np.float64)
    sizes = _residual_layout_sizes(problem)
    gps = sizes["gps_priors"]
    tilt = sizes["tilt_priors"]
    return {
        "free": free,
        "cols": cols,
        "R0": R0,
        "t0": t0,
        "K0": K0,
        "D0": np.stack([_dist5(k) for k in problem.intrinsics]) if n else np.zeros((0, 5)),
        "gps_idx": np.array([p.camera_idx for p in gps], dtype=np.int64),
        "gps_pos": np.array([p.gps_position for p in gps], dtype=np.float64).reshape(-1, 3),
        "gps_sigma": np.array(
            [p.gps_sigma_m if p.gps_sigma_m is not None else config.gps_sigma_m_default for p in gps], dtype=np.float64
        ).reshape(-1, 1) * np.array([1.0, 1.0, float(config.gps_vertical_sigma_factor)]),
        "tilt_idx": np.array([p.camera_idx for p in tilt], dtype=np.int64),
        "tilt_up": np.array(
            [
                _gravity_up_reference(p.gimbal_pitch_deg, p.gimbal_roll_deg if p.gimbal_roll_deg is not None else 0.0)
                for p in tilt
            ],
            dtype=np.float64,
        ).reshape(-1, 3),
        "tilt_sigma": np.radians(
            np.array(
                [p.tilt_sigma_deg if p.tilt_sigma_deg is not None else config.tilt_sigma_deg_default for p in tilt],
                dtype=np.float64,
            )
        ),
        "gcp_idx": np.array([g.point_idx for g in problem.gcps], dtype=np.int64),
        "gcp_xyz": np.array([g.xyz for g in problem.gcps], dtype=np.float64).reshape(-1, 3),
        "gcp_sigma": np.array([g.sigma_m for g in problem.gcps], dtype=np.float64),
    }


def _residuals(
    x: np.ndarray,
    problem: BAProblem,
    layout: ParamLayout,
    config: BAConfig,
    sizes: dict,
) -> np.ndarray:
    """All residual rows, fully vectorised. Numerically identical to ``_residuals_reference``.

    The finite-difference Jacobian calls this thousands of times per solve.
    The reference version rebuilt a ``Pose``/``CameraIntrinsics`` per camera
    and looped over every prior in Python on each call, which on a 150-camera
    flight made bundle adjustment take ten minutes; here the per-camera
    work is one batched rotation conversion and the priors are array ops.
    """
    const = sizes.get("_fast")
    if const is None:
        const = sizes["_fast"] = _fast_constants(problem, layout, config)
    free, cols = const["free"], const["cols"]
    R_all = const["R0"].copy()
    t_all = const["t0"].copy()
    K_all = const["K0"].copy()
    if free.size:
        blocks = x[cols[:, None] + np.arange(layout.camera_param_size)[None, :]]
        R_all[free] = Rotation.from_rotvec(blocks[:, 0:3]).as_matrix()
        t_all[free] = blocks[:, 3:6]
        if layout.refine_intrinsics:
            K_all[free] = blocks[:, 6:10]
    D_all = const["D0"]
    if layout.lens_size:
        K_all, D_all = _apply_lens(K_all, D_all, x[layout.lens_col : layout.lens_col + layout.lens_size], layout)
    points = x[layout.points_base_col : layout.points_base_col + 3 * layout.n_points].reshape(-1, 3)

    cam_idx = problem.obs_camera_idx
    pt_idx = problem.obs_point_idx
    u, v = _project(R_all[cam_idx], t_all[cam_idx], K_all[cam_idx], D_all[cam_idx], points[pt_idx])
    sigma = config.pixel_sigma_px
    res_reproj = np.empty(sizes["n_reproj"], dtype=np.float64)
    res_reproj[0::2] = (u - problem.obs_uv[:, 0]) / sigma
    res_reproj[1::2] = (v - problem.obs_uv[:, 1]) / sigma

    res_gps = ((t_all[const["gps_idx"]] - const["gps_pos"]) / const["gps_sigma"]).reshape(-1)
    # R^T @ [0, 0, 1] is R's third row.
    res_tilt = ((R_all[const["tilt_idx"], 2, :] - const["tilt_up"]) / const["tilt_sigma"][:, None]).reshape(-1)
    res_gcp = ((points[const["gcp_idx"]] - const["gcp_xyz"]) / const["gcp_sigma"][:, None]).reshape(-1)
    res_focal = _focal_prior_residual(x, layout, sizes)
    return np.concatenate([res_reproj, res_gps, res_tilt, res_gcp, res_focal])


_FD_REL_STEP = np.sqrt(np.finfo(np.float64).eps)  # scipy's own "2-point" relative step


def _jacobian(
    x: np.ndarray,
    problem: BAProblem,
    layout: ParamLayout,
    config: BAConfig,
    sizes: dict,
):
    """Sparse Jacobian of ``_residuals``, one projection pass per parameter slot.

    Forward differences with scipy's own step rule, but organised by the
    problem's structure instead of by generic column colouring: every
    reprojection row depends on exactly one camera and one point, so
    perturbing parameter slot ``d`` of EVERY camera at once and projecting
    all observations once yields that slot's column for every camera. That
    is ``camera_param_size + 3`` projection passes per Jacobian (13 with
    intrinsics) against the ~37 full residual evaluations scipy's greedy
    colouring needed, and no per-observation Python loop to build a
    sparsity pattern. Prior rows are exact: GPS and GCP residuals are
    linear, tilt uses the same per-slot rotation perturbation. Each shared
    lens parameter is one more projection pass and one dense column.
    """
    from scipy.sparse import csr_matrix

    const = sizes.get("_fast")
    if const is None:
        const = sizes["_fast"] = _fast_constants(problem, layout, config)
    free, cols = const["free"], const["cols"]
    psize = layout.camera_param_size
    R_all = const["R0"].copy()
    t_all = const["t0"].copy()
    K_all = const["K0"].copy()
    blocks = x[cols[:, None] + np.arange(psize)[None, :]] if free.size else np.zeros((0, psize))
    if free.size:
        R_all[free] = Rotation.from_rotvec(blocks[:, 0:3]).as_matrix()
        t_all[free] = blocks[:, 3:6]
        if layout.refine_intrinsics:
            K_all[free] = blocks[:, 6:10]
    K_base, D_base = K_all, const["D0"]
    lens = x[layout.lens_col : layout.lens_col + layout.lens_size] if layout.lens_size else np.zeros(0)
    K_all, D_all = _apply_lens(K_base, D_base, lens, layout)
    points = x[layout.points_base_col : layout.points_base_col + 3 * layout.n_points].reshape(-1, 3)
    sigma = config.pixel_sigma_px

    cam_idx = problem.obs_camera_idx
    pt_idx = problem.obs_point_idx
    m = cam_idx.shape[0]
    col_of_cam = np.full(len(problem.cameras), -1, dtype=np.int64)
    col_of_cam[free] = cols
    obs_col = col_of_cam[cam_idx]
    obs_free = obs_col >= 0
    rows_u = 2 * np.arange(m)
    D_o = D_all[cam_idx]

    def project(R, t, K, P, D=None):
        return _project(R, t, K, D_o if D is None else D, P)

    R_o, t_o, K_o, P_o = R_all[cam_idx], t_all[cam_idx], K_all[cam_idx], points[pt_idx]
    u0, v0 = project(R_o, t_o, K_o, P_o)

    r_parts, c_parts, v_parts = [], [], []

    def emit(rows, cols_, vals):
        r_parts.append(rows)
        c_parts.append(cols_)
        v_parts.append(vals)

    n_rep = sizes["n_reproj"]
    gps_row0 = n_rep
    tilt_row0 = gps_row0 + sizes["n_gps"]
    gcp_row0 = tilt_row0 + sizes["n_tilt"]
    gps_cols = col_of_cam[const["gps_idx"]]
    tilt_cols = col_of_cam[const["tilt_idx"]]

    # --- camera slots -----------------------------------------------------
    for d in range(psize if free.size else 0):
        h_cam = np.zeros(len(problem.cameras))
        h_cam[free] = _FD_REL_STEP * np.maximum(1.0, np.abs(blocks[:, d]))
        R_p, t_p, K_p = R_all, t_all, K_all
        if d < 3:
            rv = blocks[:, 0:3].copy()
            rv[:, d] += h_cam[free]
            R_p = R_all.copy()
            R_p[free] = Rotation.from_rotvec(rv).as_matrix()
        elif d < 6:
            t_p = t_all.copy()
            t_p[free, d - 3] += h_cam[free]
        else:
            K_p = K_all.copy()
            K_p[free, d - 6] += h_cam[free]
        sel = obs_free
        u1, v1 = project(R_p[cam_idx[sel]], t_p[cam_idx[sel]], K_p[cam_idx[sel]], P_o[sel], D_o[sel])
        h = h_cam[cam_idx[sel]]
        c = obs_col[sel] + d
        emit(rows_u[sel], c, (u1 - u0[sel]) / h / sigma)
        emit(rows_u[sel] + 1, c, (v1 - v0[sel]) / h / sigma)
        if 3 <= d < 6 and gps_cols.size:
            ok = gps_cols >= 0
            k = np.nonzero(ok)[0]
            emit(gps_row0 + 3 * k + (d - 3), gps_cols[ok] + d, 1.0 / const["gps_sigma"][ok, d - 3])
        if d < 3 and tilt_cols.size:
            ok = tilt_cols >= 0
            k = np.nonzero(ok)[0]
            idx = const["tilt_idx"][ok]
            dup = (R_p[idx, 2, :] - R_all[idx, 2, :]) / h_cam[idx][:, None] / const["tilt_sigma"][ok][:, None]
            for a in range(3):
                emit(tilt_row0 + 3 * k + a, tilt_cols[ok] + d, dup[:, a])

    # --- shared lens slots ------------------------------------------------
    for j in range(layout.lens_size):
        lp = lens.copy()
        h = _FD_REL_STEP * max(1.0, abs(float(lp[j])))
        lp[j] += h
        K_l, D_l = _apply_lens(K_base, D_base, lp, layout)
        u1, v1 = project(R_o, t_o, K_l[cam_idx], P_o, D_l[cam_idx])
        c = np.full(m, layout.lens_col + j, dtype=np.int64)
        emit(rows_u, c, (u1 - u0) / h / sigma)
        emit(rows_u + 1, c, (v1 - v0) / h / sigma)

    # --- point slots ------------------------------------------------------
    for k3 in range(3):
        h_pt = _FD_REL_STEP * np.maximum(1.0, np.abs(points[:, k3]))
        P_p = P_o.copy()
        P_p[:, k3] += h_pt[pt_idx]
        u1, v1 = project(R_o, t_o, K_o, P_p)
        c = layout.points_base_col + 3 * pt_idx + k3
        h = h_pt[pt_idx]
        emit(rows_u, c, (u1 - u0) / h / sigma)
        emit(rows_u + 1, c, (v1 - v0) / h / sigma)
    if const["gcp_idx"].size:
        g = np.arange(const["gcp_idx"].size)
        for k3 in range(3):
            emit(gcp_row0 + 3 * g + k3, layout.points_base_col + 3 * const["gcp_idx"] + k3, 1.0 / const["gcp_sigma"])

    if sizes.get("n_focal"):
        factor = float(x[layout.lens_col])
        emit(np.array([sizes["total"] - 1]), np.array([layout.lens_col]), np.array([1.0 / (factor * sizes["focal_sigma"])]))

    rows = np.concatenate(r_parts)
    cols_all = np.concatenate(c_parts)
    vals = np.concatenate(v_parts)
    return csr_matrix((vals, (rows, cols_all)), shape=(sizes["total"], layout.n_params))


def _residuals_reference(
    x: np.ndarray,
    problem: BAProblem,
    layout: ParamLayout,
    config: BAConfig,
    sizes: dict,
) -> np.ndarray:
    poses, points, intrinsics = _unpack_x(x, problem, layout)

    R_all = np.stack([p.R for p in poses], axis=0)
    t_all = np.stack([p.t for p in poses], axis=0)
    K_all = np.array([[k.fx, k.fy, k.cx, k.cy] for k in intrinsics], dtype=np.float64)
    D_all = np.stack([_dist5(k) for k in intrinsics])

    cam_idx = problem.obs_camera_idx
    pt_idx = problem.obs_point_idx

    # R^T @ (Pw - t) per observation, then the OpenCV lens model.
    u_pred, v_pred = _project(R_all[cam_idx], t_all[cam_idx], K_all[cam_idx], D_all[cam_idx], points[pt_idx])

    sigma = config.pixel_sigma_px
    res_reproj = np.empty(sizes["n_reproj"], dtype=np.float64)
    res_reproj[0::2] = (u_pred - problem.obs_uv[:, 0]) / sigma
    res_reproj[1::2] = (v_pred - problem.obs_uv[:, 1]) / sigma

    res_gps = np.empty(sizes["n_gps"], dtype=np.float64)
    for k, prior in enumerate(sizes["gps_priors"]):
        s = prior.gps_sigma_m if prior.gps_sigma_m is not None else config.gps_sigma_m_default
        s = s * np.array([1.0, 1.0, float(config.gps_vertical_sigma_factor)])
        res_gps[3 * k : 3 * k + 3] = (t_all[prior.camera_idx] - prior.gps_position) / s

    res_tilt = np.empty(sizes["n_tilt"], dtype=np.float64)
    for k, prior in enumerate(sizes["tilt_priors"]):
        s = np.radians(prior.tilt_sigma_deg if prior.tilt_sigma_deg is not None else config.tilt_sigma_deg_default)
        up_est = R_all[prior.camera_idx].T @ np.array([0.0, 0.0, 1.0])
        roll = prior.gimbal_roll_deg if prior.gimbal_roll_deg is not None else 0.0
        up_prior = _gravity_up_reference(prior.gimbal_pitch_deg, roll)
        res_tilt[3 * k : 3 * k + 3] = (up_est - up_prior) / s

    res_gcp = np.empty(sizes["n_gcp"], dtype=np.float64)
    for k, gcp in enumerate(problem.gcps):
        res_gcp[3 * k : 3 * k + 3] = (points[gcp.point_idx] - gcp.xyz) / gcp.sigma_m

    return np.concatenate([res_reproj, res_gps, res_tilt, res_gcp, _focal_prior_residual(x, layout, sizes)])


def _build_sparsity(problem: BAProblem, layout: ParamLayout, sizes: dict) -> lil_matrix:
    """Explicit sparsity pattern of d(residual)/d(x). See module docstring for why this matters.

    Reprojection residual ``k`` only depends on camera ``obs_camera_idx[k]``'s
    params and point ``obs_point_idx[k]``'s params (never any other camera
    or point) -- this is the classic BA sparsity structure (block-diagonal
    camera block, block-diagonal point block, coupled only through a sparse
    camera<->point cross block) that makes a dense Jacobian wasteful for
    anything beyond toy problems and that ``covariance.py``'s Schur
    complement exploits directly.
    """
    n_rows = sizes["total"]
    n_cols = layout.n_params
    sp = lil_matrix((n_rows, n_cols), dtype=np.int8)

    cam_idx = problem.obs_camera_idx
    pt_idx = problem.obs_point_idx
    for k in range(cam_idx.shape[0]):
        row0 = 2 * k
        cam_cols = layout.camera_cols(int(cam_idx[k]))
        pt_cols = layout.point_cols(int(pt_idx[k]))
        if cam_cols is not None:
            sp[row0 : row0 + 2, cam_cols] = 1
        sp[row0 : row0 + 2, pt_cols] = 1
        if layout.lens_size:
            sp[row0 : row0 + 2, layout.lens_col : layout.lens_col + layout.lens_size] = 1

    row = sizes["n_reproj"]
    for prior in sizes["gps_priors"]:
        cam_cols = layout.camera_cols(prior.camera_idx)
        if cam_cols is not None:
            sp[row : row + 3, cam_cols] = 1
        row += 3

    for prior in sizes["tilt_priors"]:
        cam_cols = layout.camera_cols(prior.camera_idx)
        if cam_cols is not None:
            sp[row : row + 3, cam_cols] = 1
        row += 3

    for gcp in problem.gcps:
        sp[row : row + 3, layout.point_cols(gcp.point_idx)] = 1
        row += 3

    if sizes.get("n_focal"):
        sp[row, layout.lens_col] = 1

    return sp


def _reprojection_rmse_px(residual_vec: np.ndarray, n_reproj: int, sigma: float) -> float:
    if n_reproj == 0:
        return 0.0
    reproj = residual_vec[:n_reproj] * sigma  # un-whiten back to pixels
    return float(np.sqrt(np.mean(reproj**2)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _reprojection_loss(name: str, n_reproj: int):
    """Robustify image outliers while retaining quadratic physical priors.

    scipy applies a string loss to EVERY residual, which otherwise makes a
    large GPS/IMU violation cheaper just when that constraint is needed most.
    """
    def loss(z):
        rho = np.vstack([z.copy(), np.ones_like(z), np.zeros_like(z)])
        x = z[:n_reproj]
        if name == "huber":
            out = x > 1
            roots = np.sqrt(x[out])
            rows = np.flatnonzero(out)
            rho[0, rows] = 2 * roots - 1
            rho[1, rows] = 1 / roots
            rho[2, rows] = -0.5 / (x[out] * roots)
        elif name == "soft_l1":
            t = 1 + x
            rho[:, :n_reproj] = np.array([2 * (np.sqrt(t) - 1), t ** -0.5, -0.5 * t ** -1.5])
        elif name == "cauchy":
            t = 1 + x
            rho[:, :n_reproj] = np.array([np.log1p(x), 1 / t, -1 / t**2])
        elif name == "arctan":
            t = 1 + x**2
            rho[:, :n_reproj] = np.array([np.arctan(x), 1 / t, -2 * x / t**2])
        elif name != "linear":
            raise ValueError(f"Unsupported robust loss: {name}")
        return rho
    return loss


def _loss_function(name: str, n_reproj: int, f_scale: float):
    """scipy's callable-loss convention: residuals -> rho (3, m), already scaled by ``f_scale``."""
    if name == "linear":
        return None
    base = _reprojection_loss(name, n_reproj)

    def fn(f: np.ndarray) -> np.ndarray:
        rho = base((f / f_scale) ** 2)
        rho[0] *= f_scale**2
        rho[2] /= f_scale**2
        return rho

    return fn


def _point_blocks(Hpp, n_points: int) -> np.ndarray:
    """The 3x3 diagonal blocks of the (block-diagonal) point normal matrix, as (N, 3, 3)."""
    coo = Hpp.tocoo()
    same = (coo.row // 3) == (coo.col // 3)
    B = np.zeros((n_points, 3, 3))
    np.add.at(B, (coo.row[same] // 3, coo.row[same] % 3, coo.col[same] % 3), coo.data[same])
    return B


def _least_squares_schur(x0, problem, layout, config, sizes, loss_fn, max_iterations: int, ftol: float, xtol: float):
    """Levenberg-Marquardt with the points eliminated by Schur complement.

    Same objective as ``least_squares`` with the same robust loss (scipy's
    IRLS scaling of rows, including the second-order term). Every step
    solves the reduced camera system exactly (dense Cholesky: ~1k unknowns)
    and back-substitutes the points, so directions that only the priors
    pin are resolved as well as the well-observed ones. Marquardt damping
    (lambda * diag) handles the mixed units of pixels, metres and radians.
    """
    from types import SimpleNamespace

    import scipy.linalg as sla
    from scipy.sparse import bsr_matrix, diags

    P = int(layout.points_base_col)
    n_pts = int(layout.n_points) if (layout.n_params - P) == 3 * int(layout.n_points) else (layout.n_params - P) // 3
    eps = np.finfo(float).eps

    def evaluate(x):
        f = _residuals(x, problem, layout, config, sizes)
        if loss_fn is None:
            return f, 0.5 * float(f @ f), None
        rho = loss_fn(f)
        return f, 0.5 * float(rho[0].sum()), rho

    x = np.asarray(x0, dtype=np.float64).copy()
    f, cost, rho = evaluate(x)
    lam, nu = 1e-4, 2.0
    converged, message, it = False, "maximum iterations reached", 0
    for it in range(1, max_iterations + 1):
        J = _jacobian(x, problem, layout, config, sizes).tocsr()
        fw = f
        if rho is not None:
            # Plain IRLS: weight sqrt(rho') on rows and residuals -- the exact
            # robust gradient J^T (rho' f) with a positive Gauss-Newton
            # Hessian. scipy's second-order correction (rho' + 2 rho'' f^2)
            # is exactly zero on Huber's linear branch, so every outlier row
            # left the system while still pulling on the gradient, and LM
            # stalled at twice scipy's cost on flight01.
            js = np.sqrt(np.maximum(rho[1], eps))
            fw = f * js
            J = diags(js) @ J
        Jc, Jp = J[:, :P], J[:, P:]
        Hcc = (Jc.T @ Jc).toarray()
        Hcp = (Jc.T @ Jp).tocsr()
        B = _point_blocks((Jp.T @ Jp), n_pts)
        gc, gp = Jc.T @ fw, Jp.T @ fw
        # Marquardt scaling (lambda * diag). A parameter no residual touches
        # (a dropped camera's yaw) has a zero diagonal: only those get a
        # tiny stand-in, so the system stays positive definite without
        # damping the well-measured ones. (A floor on EVERY parameter,
        # scaled to the largest diagonal, over-damped radians against
        # focal pixels and made convergence linear.)
        dcc = np.diag(Hcc).copy()
        dpp = np.einsum("nii->ni", B).copy()
        sc = max(float(dcc.max()) if dcc.size else 0.0, 1e-300)
        sp = max(float(dpp.max()) if dpp.size else 0.0, 1e-300)
        dead_c, dead_p = dcc <= 1e-14 * sc, dpp <= 1e-14 * sp
        dcc[dead_c] = 1e-9 * sc
        dpp[dead_p] = 1e-9 * sp
        improved, tries = False, 0
        while lam < 1e16:
            tries += 1
            Bd = B.copy()
            Bd[:, [0, 1, 2], [0, 1, 2]] += lam * dpp + np.where(dead_p, 1e-9 * sp, 0.0)
            Binv = np.linalg.inv(Bd)
            Binv_sp = bsr_matrix((Binv, np.arange(n_pts), np.arange(n_pts + 1)), shape=(3 * n_pts, 3 * n_pts))
            W = (Hcp @ Binv_sp).tocsr()
            S = Hcc - (W @ Hcp.T).toarray()
            S[np.diag_indices_from(S)] += lam * dcc + np.where(dead_c, 1e-9 * sc, 0.0)
            try:
                dc = sla.cho_solve(sla.cho_factor(S, check_finite=False), -(gc - W @ gp), check_finite=False)
            except np.linalg.LinAlgError:
                logger.debug("schur LM: Cholesky failed at lambda %.3g", lam)
                lam *= nu
                nu *= 2.0
                continue
            dp = Binv_sp @ (-gp - Hcp.T @ dc)
            dx = np.concatenate([dc, dp])
            # Gain ratio against the (weighted) linear model: Nielsen's update.
            lin = fw + J @ dx
            predicted = cost - 0.5 * float(lin @ lin) if rho is None else 0.5 * float(fw @ fw) - 0.5 * float(lin @ lin)
            x_new = x + dx
            f_new, cost_new, rho_new = evaluate(x_new)
            actual = cost - cost_new
            if np.isfinite(cost_new) and actual > 0 and predicted > 0:
                ratio = actual / predicted
                rel = actual / max(cost, 1e-300)
                small_step = np.linalg.norm(dx) <= xtol * (xtol + np.linalg.norm(x))
                x, f, cost, rho = x_new, f_new, cost_new, rho_new
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                lam = max(lam, 1e-15)
                nu = 2.0
                improved = True
                if rel < ftol or small_step:
                    converged, message = True, "relative cost decrease below ftol" if rel < ftol else "step below xtol"
                break
            logger.debug("schur LM it %d: cost %.6g -> %.6g rejected at lambda %.3g", it, cost, cost_new, lam)
            lam *= nu
            nu *= 2.0
        logger.debug("schur LM it %d: cost %.6g, lambda %.3g, %d tries, improved=%s", it, cost, lam, tries, improved)
        if not improved:
            converged, message = True, "no damped step decreases the cost"
        if converged:
            break
    jac = _jacobian(x, problem, layout, config, sizes).tocsr()
    if rho is not None:
        jac = diags(np.sqrt(np.maximum(rho[1], eps))) @ jac
    return SimpleNamespace(x=x, fun=f, jac=jac, nfev=it, success=converged, message=message, cost=cost)


def bundle_adjust(problem: BAProblem, config: BAConfig | None = None) -> BAResult:
    """Refine cameras/points/(optionally) intrinsics against reprojection + priors.

    See the module docstring for why priors (GPS position, gravity/tilt)
    and gauge fixing are not optional extras but load-bearing parts of
    making this well-posed for a single, near-collinear flight strip.
    """
    config = config or BAConfig()
    import time as _time

    t_start = _time.monotonic()

    if config.fixed_gauge and not problem.fixed_camera_indices:
        problem.fixed_camera_indices = _default_gauge_fix(problem, config)

    layout = _build_layout(problem, config)
    sizes = _residual_layout_sizes(problem, layout, config)
    x0 = _pack_x(problem, layout)

    sigma = config.pixel_sigma_px
    r0 = _residuals(x0, problem, layout, config, sizes)
    rmse_before = _reprojection_rmse_px(r0, sizes["n_reproj"], sigma)

    if layout.n_params == 0:
        # Nothing to optimize (e.g. every camera fixed and no points) --
        # degenerate but not an error; just report the fixed configuration.
        poses, points, intrinsics = _unpack_x(x0, problem, layout)
        return BAResult(
            poses=poses, points=points, intrinsics=intrinsics,
            residuals_px=np.zeros((0, 2)), rmse_before_px=rmse_before, rmse_after_px=rmse_before,
            n_iterations=0, converged=True, jacobian=None, param_layout=layout,
            message="nothing to optimize",
        )

    loss = config.robust_loss if config.robust_loss is not None else "linear"
    max_nfev = config.max_nfev if config.max_nfev is not None else _default_max_nfev(config.max_iterations)

    if config.solver == "schur":
        if loss != "linear" and config.warm_start_linear_nfev > 0:
            warm_loss = config.warm_start_loss or "linear"
            warm = _least_squares_schur(
                x0, problem, layout, config, sizes,
                _loss_function(warm_loss, sizes["n_reproj"], config.warm_start_f_scale),
                max_iterations=int(config.warm_start_linear_nfev), ftol=config.schur_warm_ftol, xtol=config.xtol,
            )
            x0 = warm.x
        result = _least_squares_schur(
            x0, problem, layout, config, sizes, _loss_function(loss, sizes["n_reproj"], config.f_scale),
            max_iterations=max(1, int(config.max_iterations)), ftol=config.schur_ftol, xtol=config.xtol,
        )
    elif loss != "linear" and config.warm_start_linear_nfev > 0:
        warm_loss = config.warm_start_loss or "linear"
        warm = least_squares(
            _residuals,
            x0,
            jac=_jacobian,
            method="trf",
            tr_solver="lsmr",
            tr_options=dict(config.lsmr_options or {}),
            loss="linear" if warm_loss == "linear" else _reprojection_loss(warm_loss, sizes["n_reproj"]),
            f_scale=config.warm_start_f_scale,
            x_scale="jac",
            max_nfev=int(config.warm_start_linear_nfev),
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            args=(problem, layout, config, sizes),
        )
        x0 = warm.x

    if config.solver != "schur":
        result = least_squares(
            _residuals,
            x0,
            # Structured finite-difference Jacobian (see _jacobian): 13 cheap
            # projection passes instead of scipy's colour-grouped full residual
            # evaluations, which were ~85% of solve time at 150 cameras.
            jac=_jacobian,
            method="trf",
            # Explicit rather than relying on scipy's own sparse-Jacobian
            # default: LSMR is the iterative sparse linear solver this
            # problem's block structure needs (see module docstring) --
            # never form/factor the dense normal-equations matrix.
            tr_solver="lsmr",
            tr_options=dict(config.lsmr_options or {}),
            loss=_reprojection_loss(loss, sizes["n_reproj"]),
            f_scale=config.f_scale,
            x_scale="jac",  # badly-scaled params otherwise (pixels/metres/radians/focal-px mixed)
            max_nfev=max_nfev,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            verbose=config.verbose,
            args=(problem, layout, config, sizes),
        )

    poses, points, intrinsics = _unpack_x(result.x, problem, layout)
    rmse_after = _reprojection_rmse_px(result.fun, sizes["n_reproj"], sigma)
    import logging as _logging

    _lf = _loss_function(loss, sizes["n_reproj"], config.f_scale)
    final_cost = 0.5 * float(result.fun @ result.fun) if _lf is None else 0.5 * float(_lf(result.fun)[0].sum())
    _logging.getLogger(__name__).info(
        "bundle adjustment solve (%s): %d cameras, %d points, %d observations, %d parameters; "
        "%d iterations/evaluations, %.1f s, rmse %.3f -> %.3f px, cost %.2f",
        config.solver, len(problem.cameras), int(problem.points.shape[0]), int(sizes["n_reproj"] // 2),
        int(layout.n_params), int(result.nfev), _time.monotonic() - t_start, rmse_before, rmse_after, final_cost,
    )

    n_reproj = sizes["n_reproj"]
    residuals_px = np.stack(
        [result.fun[0:n_reproj:2] * sigma, result.fun[1:n_reproj:2] * sigma], axis=1
    ) if n_reproj else np.zeros((0, 2))

    lens = None
    if layout.lens_size:
        values = [float(v) for v in result.x[layout.lens_col : layout.lens_col + layout.lens_size]]
        lens = {"focal_factor": values.pop(0) if layout.refine_shared_focal else 1.0}
        if layout.refine_distortion:
            lens.update(k1=values[0], k2=values[1])

    return BAResult(
        poses=poses,
        points=points,
        intrinsics=intrinsics,
        residuals_px=residuals_px,
        rmse_before_px=rmse_before,
        rmse_after_px=rmse_after,
        n_iterations=int(result.nfev),
        converged=bool(result.success),
        jacobian=result.jac,
        param_layout=layout,
        message=str(result.message),
        lens=lens,
    )


def windowed_bundle_adjust(
    problems: list[BAProblem],
    config: BAConfig | None = None,
    shared_camera_pairs: list[list[tuple[int, int]]] | None = None,
    n_anchor: int = 3,
) -> list[BAResult]:
    """Run ``bundle_adjust`` window-by-window, anchoring each to the previous window's refined poses.

    Why pose-anchoring (not just re-running Sim(3) alignment after the fact)
    -----------------------------------------------------------------------
    Each window is its own independent optimization instance; nothing
    stops window ``i``'s BA from quietly re-scaling or re-orienting itself
    relative to window ``i - 1`` even if both individually fit their own
    observations well (bundle adjustment has the same 7-DOF gauge freedom
    per-window that a lone submap does -- see ``geometry.submap``'s module
    docstring). If we only fixed gauge *within* each window independently
    (e.g. via ``BAConfig.fixed_gauge``'s default of anchoring cameras 0/1
    of *that* window), consecutive windows' gauges would be chosen
    independently and could disagree, reintroducing exactly the
    scale/orientation drift that ``geometry.submap.merge_submaps`` already
    has to fight at the merge step.

    The fix: for every window after the first, take the last ``n_anchor``
    cameras it shares with the previous window (chronologically -- the
    tail of the overlap, closest to the new, not-yet-seen frames) and hold
    *their poses fixed at the previous window's already-refined values*
    for the duration of this window's optimization. This makes each
    window's gauge literally identical to its predecessor's (same
    reference cameras, same values), so scale and orientation cannot drift
    at all between adjacent windows -- only the genuinely new cameras and
    points in the non-overlapping part of the window are free to move.

    Parameters
    ----------
    problems:
        One ``BAProblem`` per window, in order. Each problem's camera list
        should include, at the appropriate local indices, the cameras
        shared with the neighbouring window(s) -- exactly as
        ``geometry.windows.Window.shared_with_previous`` /
        ``types.Submap.keyframe_indices`` already track for the
        backbone/merge stage.
    shared_camera_pairs:
        ``shared_camera_pairs[j]`` (for junction ``j`` between
        ``problems[j]`` and ``problems[j + 1]``) is a list of
        ``(local_idx_in_problems[j], local_idx_in_problems[j + 1])`` pairs
        for the same physical camera, ordered chronologically. If omitted,
        no inter-window anchoring is done (each window gauge-fixes itself
        independently via ``BAConfig.fixed_gauge``) -- fine for a single
        window, but drift-prone for a real multi-window flight.
    n_anchor:
        How many of the chronologically-last shared cameras to anchor per
        junction. Must be >= 2 for the anchored cameras alone to fix full
        gauge (translation + rotation + scale) the way
        ``_default_gauge_fix`` does for cameras 0/1 of the first window;
        3 is used as the default for a bit of extra robustness against any
        one anchor camera being poorly constrained.
    """
    config = config or BAConfig()
    if not problems:
        return []

    results: list[BAResult] = []
    first_config = BAConfig(**{**config.__dict__, "fixed_gauge": True})
    results.append(bundle_adjust(problems[0], first_config))

    for j in range(1, len(problems)):
        problem = problems[j]
        prev_result = results[j - 1]

        if shared_camera_pairs is not None and j - 1 < len(shared_camera_pairs):
            pairs = shared_camera_pairs[j - 1][-n_anchor:]
            anchors = set()
            for prev_local, curr_local in pairs:
                problem.cameras[curr_local] = prev_result.poses[prev_local]
                anchors.add(curr_local)
            problem.fixed_camera_indices = anchors
            window_config = BAConfig(**{**config.__dict__, "fixed_gauge": False})
        else:
            window_config = BAConfig(**{**config.__dict__, "fixed_gauge": True})

        results.append(bundle_adjust(problem, window_config))

    return results
