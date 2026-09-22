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
        7 DOF, not just 6.
    """

    fix_intrinsics: bool = True
    refine_intrinsics: bool = False
    robust_loss: str | None = "huber"
    f_scale: float = 1.345
    max_iterations: int = 100
    max_nfev: int | None = None
    pixel_sigma_px: float = 1.0
    gps_sigma_m_default: float = 5.0
    tilt_sigma_deg_default: float = 2.0
    fixed_gauge: bool = True
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    verbose: int = 0


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
    re-deriving this bookkeeping itself.
    """

    n_cameras: int
    n_points: int
    fixed_camera_indices: set[int]
    camera_param_size: int  # 6, or 10 if intrinsics are refined
    refine_intrinsics: bool
    camera_col: dict[int, int]  # free camera_idx -> starting column
    points_base_col: int  # points_base_col + 3*point_idx -> point's starting column
    n_params: int

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


# ---------------------------------------------------------------------------
# Parameter (de)serialization
# ---------------------------------------------------------------------------


def _default_gauge_fix(problem: BAProblem) -> set[int]:
    """Default gauge-fixing policy: hold the first (up to) two cameras fully fixed.

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
    return {0, 1}


def _build_layout(problem: BAProblem, config: BAConfig) -> ParamLayout:
    refine_intrinsics = config.refine_intrinsics or (not config.fix_intrinsics and config.refine_intrinsics is not False)
    refine_intrinsics = bool(config.refine_intrinsics)  # authoritative per docstring
    camera_param_size = 10 if refine_intrinsics else 6

    camera_col: dict[int, int] = {}
    col = 0
    for i in range(len(problem.cameras)):
        if i in problem.fixed_camera_indices:
            continue
        camera_col[i] = col
        col += camera_param_size

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
    )


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
    points = x[layout.points_base_col : layout.points_base_col + 3 * layout.n_points].reshape(-1, 3).copy()
    return poses, points, intrinsics


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------


def _residual_layout_sizes(problem: BAProblem) -> dict:
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
    return {
        "n_reproj": n_reproj,
        "gps_priors": gps_priors,
        "n_gps": n_gps,
        "tilt_priors": tilt_priors,
        "n_tilt": n_tilt,
        "n_gcp": n_gcp,
        "total": n_reproj + n_gps + n_tilt + n_gcp,
    }


def _residuals(
    x: np.ndarray,
    problem: BAProblem,
    layout: ParamLayout,
    config: BAConfig,
    sizes: dict,
) -> np.ndarray:
    poses, points, intrinsics = _unpack_x(x, problem, layout)

    R_all = np.stack([p.R for p in poses], axis=0)
    t_all = np.stack([p.t for p in poses], axis=0)
    fx_all = np.array([k.fx for k in intrinsics])
    fy_all = np.array([k.fy for k in intrinsics])
    cx_all = np.array([k.cx for k in intrinsics])
    cy_all = np.array([k.cy for k in intrinsics])

    cam_idx = problem.obs_camera_idx
    pt_idx = problem.obs_point_idx

    R_obs = R_all[cam_idx]  # (M,3,3)
    t_obs = t_all[cam_idx]  # (M,3)
    Pw = points[pt_idx]  # (M,3)
    Xc = np.einsum("mji,mj->mi", R_obs, Pw - t_obs)  # R^T @ (Pw - t), per-obs

    z = Xc[:, 2]
    u_pred = fx_all[cam_idx] * Xc[:, 0] / z + cx_all[cam_idx]
    v_pred = fy_all[cam_idx] * Xc[:, 1] / z + cy_all[cam_idx]

    sigma = config.pixel_sigma_px
    res_reproj = np.empty(sizes["n_reproj"], dtype=np.float64)
    res_reproj[0::2] = (u_pred - problem.obs_uv[:, 0]) / sigma
    res_reproj[1::2] = (v_pred - problem.obs_uv[:, 1]) / sigma

    res_gps = np.empty(sizes["n_gps"], dtype=np.float64)
    for k, prior in enumerate(sizes["gps_priors"]):
        s = prior.gps_sigma_m if prior.gps_sigma_m is not None else config.gps_sigma_m_default
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

    return np.concatenate([res_reproj, res_gps, res_tilt, res_gcp])


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

    return sp


def _reprojection_rmse_px(residual_vec: np.ndarray, n_reproj: int, sigma: float) -> float:
    if n_reproj == 0:
        return 0.0
    reproj = residual_vec[:n_reproj] * sigma  # un-whiten back to pixels
    return float(np.sqrt(np.mean(reproj**2)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bundle_adjust(problem: BAProblem, config: BAConfig | None = None) -> BAResult:
    """Refine cameras/points/(optionally) intrinsics against reprojection + priors.

    See the module docstring for why priors (GPS position, gravity/tilt)
    and gauge fixing are not optional extras but load-bearing parts of
    making this well-posed for a single, near-collinear flight strip.
    """
    config = config or BAConfig()

    if config.fixed_gauge and not problem.fixed_camera_indices:
        problem.fixed_camera_indices = _default_gauge_fix(problem)

    layout = _build_layout(problem, config)
    sizes = _residual_layout_sizes(problem)
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

    sparsity = _build_sparsity(problem, layout, sizes)

    loss = config.robust_loss if config.robust_loss is not None else "linear"
    max_nfev = config.max_nfev if config.max_nfev is not None else _default_max_nfev(config.max_iterations)

    result = least_squares(
        _residuals,
        x0,
        jac="2-point",
        jac_sparsity=sparsity,
        method="trf",
        # Explicit rather than relying on scipy's own sparse-Jacobian
        # default: LSMR is the iterative sparse linear solver this
        # problem's block structure needs (see module docstring) --
        # never form/factor the dense normal-equations matrix.
        tr_solver="lsmr",
        loss=loss,
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

    n_reproj = sizes["n_reproj"]
    residuals_px = np.stack(
        [result.fun[0:n_reproj:2] * sigma, result.fun[1:n_reproj:2] * sigma], axis=1
    ) if n_reproj else np.zeros((0, 2))

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
