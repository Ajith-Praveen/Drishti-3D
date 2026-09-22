"""Tests for drishti3d.fusion: cleanup filters, confidence-weighted TSDF, and mesh ops.

Everything here must pass with no open3d, no rasterio, no torch -- every
test exercises the pure numpy/scipy fallback paths, using synthetic data
(no video/model weights needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.config import FusionConfig
from drishti3d.fusion.filters import (
    confidence_filter,
    crop_to_bounds,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)
from drishti3d.fusion.mesh import (
    compute_mesh_stats,
    decimate_mesh,
    estimate_normals,
    poisson_reconstruct,
)
from drishti3d.fusion.tsdf import TSDFVolume, fuse_submaps
from drishti3d.geometry.windows import Window
from drishti3d.types import CameraIntrinsics, Confidence, PointCloud, Pose, Submap

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_cloud_with_outliers(rng: np.random.Generator, n_inliers: int = 300, n_outliers: int = 8) -> PointCloud:
    inliers = rng.normal(scale=1.0, size=(n_inliers, 3))
    outliers = rng.normal(scale=1.0, size=(n_outliers, 3)) + rng.choice([-1.0, 1.0], size=(n_outliers, 3)) * 40.0
    xyz = np.concatenate([inliers, outliers], axis=0)
    rgb = rng.integers(0, 255, size=(xyz.shape[0], 3)).astype(np.uint8)
    confidence = rng.integers(0, 3, size=(xyz.shape[0],)).astype(np.uint8)
    covariance = np.tile(np.eye(3) * 0.01, (xyz.shape[0], 1, 1))
    return PointCloud(xyz=xyz, rgb=rgb, covariance=covariance, confidence=confidence)


# Camera orientations for six axis-aligned views of a box centred at the
# origin, each looking straight at the origin from one face's side.
# Columns of R are [right_world, down_world, forward_world]; forward is
# always the world direction each camera actually looks along, verified
# (see tests/test_fusion.py history) to satisfy right x down == forward
# so each R is a proper (det +1) rotation.
_BOX_VIEW_ROTATIONS = {
    "+x": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),  # camera at +X, looks -X
    "-x": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),  # camera at -X, looks +X
    "+y": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),  # camera at +Y, looks -Y
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),  # camera at -Y, looks +Y
    "+z": np.diag([1.0, -1.0, -1.0]),  # camera at +Z, looks -Z (nadir)
    "-z": np.eye(3),  # camera at -Z, looks +Z
}


def _render_box_depth(
    box_min: np.ndarray, box_max: np.ndarray, intr: CameraIntrinsics, pose: Pose, h: int, w: int
) -> np.ndarray:
    """Ray/AABB-intersect depth render of an axis-aligned box, for one camera."""
    K_inv = np.linalg.inv(intr.K())
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1)
    dirs_cam = pix @ K_inv.T
    dirs_world = dirs_cam @ pose.R.T
    origin = pose.t

    eps = 1e-12
    d = np.where(np.abs(dirs_world) < eps, eps, dirs_world)
    t1 = (box_min - origin) / d
    t2 = (box_max - origin) / d
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)
    t_enter = np.max(t_near, axis=-1)
    t_exit = np.min(t_far, axis=-1)
    hit = (t_exit >= t_enter) & (t_exit >= 1e-6)
    depth = np.where(hit, t_enter, 0.0)
    return depth.astype(np.float64)


def _six_view_box_submap_free_integration(voxel_size: float, sdf_trunc: float, box_half: float = 1.0):
    """Integrate six axis-aligned views of a box into a fresh TSDFVolume. Returns the volume."""
    box_min = np.full(3, -box_half)
    box_max = np.full(3, box_half)
    distance = 5.0
    h, w = 48, 64
    intr = CameraIntrinsics.from_hfov(60.0, w, h)

    volume = TSDFVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc, use_open3d=False)
    axis_offsets = {"+x": (1, distance), "-x": (1, -distance), "+y": (0, -distance), "-y": (0, distance), "+z": (2, -distance), "-z": (2, distance)}
    # axis_offsets values are unused directly; camera positions are set explicitly below for clarity.
    positions = {
        "+x": np.array([distance, 0.0, 0.0]),
        "-x": np.array([-distance, 0.0, 0.0]),
        "+y": np.array([0.0, distance, 0.0]),
        "-y": np.array([0.0, -distance, 0.0]),
        "+z": np.array([0.0, 0.0, distance]),
        "-z": np.array([0.0, 0.0, -distance]),
    }
    del axis_offsets

    for name, R in _BOX_VIEW_ROTATIONS.items():
        pose = Pose(R=R, t=positions[name])
        depth = _render_box_depth(box_min, box_max, intr, pose, h, w)
        color = np.full((h, w, 3), 180, dtype=np.uint8)
        confidence_map = np.full((h, w), 0.9)
        volume.integrate(depth, color, intr, pose, confidence_map=confidence_map)

    return volume, box_min, box_max


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_fn",
    [
        lambda pc: statistical_outlier_removal(pc, k=10, std_ratio=2.0),
        lambda pc: radius_outlier_removal(pc, radius=1.0, min_neighbors=3),
        lambda pc: confidence_filter(pc, min_confidence=1),
        lambda pc: crop_to_bounds(pc, (None, None, None, None, None, None)),
    ],
)
def test_filters_carry_rgb_covariance_confidence(filter_fn):
    rng = np.random.default_rng(0)
    pc = _make_cloud_with_outliers(rng)
    out = filter_fn(pc)

    assert out.rgb is not None and out.rgb.shape[0] == out.xyz.shape[0]
    assert out.covariance is not None and out.covariance.shape[0] == out.xyz.shape[0]
    assert out.confidence is not None and out.confidence.shape[0] == out.xyz.shape[0]
    assert out.covariance.shape[1:] == (3, 3)


def test_statistical_outlier_removal_drops_outliers():
    rng = np.random.default_rng(1)
    pc = _make_cloud_with_outliers(rng, n_inliers=300, n_outliers=8)
    out = statistical_outlier_removal(pc, k=15, std_ratio=2.0)

    assert out.xyz.shape[0] < pc.xyz.shape[0]
    # Every surviving point should be an inlier (within a generous radius
    # of the origin, since outliers were placed ~40 units away).
    assert np.all(np.linalg.norm(out.xyz, axis=1) < 10.0)


def test_radius_outlier_removal_drops_isolated_points():
    rng = np.random.default_rng(2)
    pc = _make_cloud_with_outliers(rng, n_inliers=300, n_outliers=8)
    out = radius_outlier_removal(pc, radius=0.5, min_neighbors=3)

    assert out.xyz.shape[0] < pc.xyz.shape[0]
    assert np.all(np.linalg.norm(out.xyz, axis=1) < 10.0)


def test_voxel_downsample_takes_minimum_confidence():
    # Two points in the same voxel: one MEASURED, one INFERRED. The
    # merged voxel must report INFERRED (the minimum), not an average.
    xyz = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    confidence = np.array([Confidence.MEASURED, Confidence.INFERRED], dtype=np.uint8)
    rgb = np.array([[200, 200, 200], [100, 100, 100]], dtype=np.uint8)
    pc = PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)

    out = voxel_downsample(pc, voxel_size=1.0)

    assert out.xyz.shape[0] == 1
    assert out.confidence[0] == Confidence.INFERRED
    # Position/colour should still be a plain average.
    np.testing.assert_allclose(out.xyz[0], xyz.mean(axis=0))
    np.testing.assert_allclose(out.rgb[0], rgb.mean(axis=0), atol=1.0)


def test_voxel_downsample_carries_covariance():
    rng = np.random.default_rng(3)
    pc = _make_cloud_with_outliers(rng, n_inliers=100, n_outliers=0)
    out = voxel_downsample(pc, voxel_size=0.5)

    assert out.xyz.shape[0] < pc.xyz.shape[0]
    assert out.covariance is not None
    assert out.covariance.shape == (out.xyz.shape[0], 3, 3)
    assert out.confidence is not None and out.confidence.shape[0] == out.xyz.shape[0]


def test_confidence_filter_requires_confidence():
    pc = PointCloud(xyz=np.zeros((5, 3)))
    with pytest.raises(ValueError):
        confidence_filter(pc, min_confidence=1)


def test_crop_to_bounds():
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]])
    pc = PointCloud(xyz=xyz)
    out = crop_to_bounds(pc, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    assert out.xyz.shape[0] == 1
    np.testing.assert_allclose(out.xyz[0], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# TSDF
# ---------------------------------------------------------------------------


def test_tsdf_integrate_and_extract_box_mesh_bbox():
    voxel_size = 0.1
    sdf_trunc = 0.3
    volume, box_min, box_max = _six_view_box_submap_free_integration(voxel_size, sdf_trunc)

    vertices, faces, colors, confidence = volume.extract_triangle_mesh()

    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert colors.shape == (vertices.shape[0], 3)
    assert confidence.shape == (vertices.shape[0],)

    tolerance = 2.0 * voxel_size
    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    np.testing.assert_allclose(mesh_min, box_min, atol=tolerance)
    np.testing.assert_allclose(mesh_max, box_max, atol=tolerance)


def test_tsdf_confidence_has_multiple_tiers():
    volume, _box_min, _box_max = _six_view_box_submap_free_integration(voxel_size=0.15, sdf_trunc=0.45)
    _vertices, _faces, _colors, confidence = volume.extract_triangle_mesh()

    assert confidence.shape[0] > 0
    assert len(np.unique(confidence)) > 1


def test_tsdf_extract_point_cloud():
    volume, box_min, box_max = _six_view_box_submap_free_integration(voxel_size=0.2, sdf_trunc=0.6)
    pc = volume.extract_point_cloud()

    assert pc.xyz.shape[0] > 0
    assert pc.confidence is not None
    assert pc.rgb is not None
    # Every extracted point should be within (a small margin of) the box's
    # truncation-widened bounding volume.
    margin = 0.6 + 0.2
    assert np.all(pc.xyz >= box_min - margin)
    assert np.all(pc.xyz <= box_max + margin)


def test_tsdf_confidence_map_scales_weight():
    """A low-confidence observation should move a voxel's value less than a high-confidence one."""
    voxel_size = 0.1
    sdf_trunc = 0.3
    h, w = 32, 32
    intr = CameraIntrinsics.from_hfov(60.0, w, h)
    pose = Pose(R=np.eye(3), t=np.array([0.0, 0.0, -3.0]))

    depth_true = np.full((h, w), 2.0)
    depth_noisy = np.full((h, w), 2.2)  # a "wrong" observation, still inside the truncation band
    color = np.full((h, w, 3), 128, dtype=np.uint8)

    vol_high_trust = TSDFVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc, use_open3d=False)
    vol_high_trust.integrate(depth_true, color, intr, pose, confidence_map=np.ones((h, w)))
    vol_high_trust.integrate(depth_noisy, color, intr, pose, confidence_map=np.full((h, w), 0.01))

    vol_equal_trust = TSDFVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc, use_open3d=False)
    vol_equal_trust.integrate(depth_true, color, intr, pose, confidence_map=np.ones((h, w)))
    vol_equal_trust.integrate(depth_noisy, color, intr, pose, confidence_map=np.ones((h, w)))

    # Sample the voxel value near the true surface (z=2 along the camera
    # ray from t=(0,0,-3), i.e. world z ~ -1) in both volumes.
    idx = np.round((np.array([0.0, 0.0, -1.0]) - vol_high_trust._origin) / voxel_size - 0.5).astype(int)
    val_high_trust = vol_high_trust._values[idx[0], idx[1], idx[2]]
    val_equal_trust = vol_equal_trust._values[idx[0], idx[1], idx[2]]

    # The low-weight noisy observation should have pulled the equal-trust
    # volume's value further from zero (the true surface) than the
    # confidence-weighted one.
    assert abs(val_equal_trust) > abs(val_high_trust)


def test_fuse_submaps_end_to_end():
    rng = np.random.default_rng(4)

    def make_submap(idx: int, keyframe_indices: list[int], offset: float, n: int = 250) -> Submap:
        xyz = rng.normal(size=(n, 3)) * 1.0 + np.array([offset, 0.0, 0.0])
        confidence = rng.integers(1, 3, size=(n,)).astype(np.uint8)
        rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
        pc = PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)
        poses = [Pose(R=np.eye(3), t=np.array([offset + i * 0.5, 0.1 * i, 5.0])) for i in range(len(keyframe_indices))]
        window = Window(index=idx, start=keyframe_indices[0], end=keyframe_indices[-1] + 1)
        return Submap(
            window=window,
            poses=poses,
            points=pc,
            confidence=confidence.astype(np.float64) / 2.0,
            keyframe_indices=keyframe_indices,
            local_origin=poses[0],
        )

    s0 = make_submap(0, [0, 1, 2, 3, 4], offset=0.0)
    s1 = make_submap(1, [2, 3, 4, 5, 6], offset=0.05)
    s1.window.shared_with_previous = [2, 3, 4]

    cfg = FusionConfig(voxel_size=0.2, outlier_std_ratio=2.5, min_confidence=1)
    vertices, faces, colors, confidence, raw_point_cloud = fuse_submaps([s0, s1], cfg)

    assert vertices.shape[0] > 0
    assert faces.shape[1] == 3
    assert colors.shape == (vertices.shape[0], 3)
    assert confidence.shape == (vertices.shape[0],)
    # Fix 2: the raw, cleaned, pre-TSDF point cloud is always returned
    # alongside the mesh, independent of TSDF meshing outcome.
    assert raw_point_cloud.xyz.shape[0] > 0


def _two_submaps(rng):
    """Shared fixture for the point_filter tests: two overlapping submaps."""

    def make(idx: int, kfs: list[int], offset: float, n: int = 250) -> Submap:
        xyz = rng.normal(size=(n, 3)) * 1.0 + np.array([offset, 0.0, 0.0])
        conf = rng.integers(1, 3, size=(n,)).astype(np.uint8)
        rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
        poses = [Pose(R=np.eye(3), t=np.array([offset + i * 0.5, 0.1 * i, 5.0])) for i in range(len(kfs))]
        return Submap(
            window=Window(index=idx, start=kfs[0], end=kfs[-1] + 1),
            poses=poses,
            points=PointCloud(xyz=xyz, rgb=rgb, confidence=conf),
            confidence=conf.astype(np.float64) / 2.0,
            keyframe_indices=kfs,
            local_origin=poses[0],
        )

    s0 = make(0, [0, 1, 2, 3, 4], offset=0.0)
    s1 = make(1, [2, 3, 4, 5, 6], offset=0.05)
    s1.window.shared_with_previous = [2, 3, 4]
    return [s0, s1]


def test_point_filter_runs_before_meshing_and_removes_points():
    """The filter must shape the mesh, not merely annotate it afterwards.

    This is the regression test for the ordering defect: photometric
    verification used to run only after fusion, so points the source frames
    disagreed about were meshed first and labelled second. Asserting that
    the rejected points are absent from the *returned cloud* is what pins
    the filter to the pre-mesh position -- a post-hoc re-grading would leave
    the point count unchanged and still pass a weaker assertion.
    """
    rng = np.random.default_rng(4)
    cfg = FusionConfig(voxel_size=0.2, outlier_std_ratio=2.5, min_confidence=1)

    seen: dict = {}

    def reject_high_z(cloud):
        # Deterministic stand-in for "the views disagree here": drop the
        # upper half in Z. Any predicate works; what matters is that the
        # points it rejects cannot appear downstream.
        seen["n_in"] = cloud.xyz.shape[0]
        return cloud.xyz[:, 2] <= float(np.median(cloud.xyz[:, 2]))

    stats_filtered: dict = {}
    _, _, _, _, filtered = fuse_submaps(
        _two_submaps(rng), cfg, stats=stats_filtered, point_filter=reject_high_z
    )

    stats_plain: dict = {}
    _, _, _, _, plain = fuse_submaps(_two_submaps(np.random.default_rng(4)), cfg, stats=stats_plain)

    assert seen["n_in"] > 0, "filter was never called"
    assert stats_filtered["point_filter_rejected"] > 0
    assert stats_filtered["after_point_filter"] < stats_plain["after_outlier_removal"]
    assert filtered.xyz.shape[0] < plain.xyz.shape[0]
    # The rejected half is genuinely gone, not just down-weighted.
    assert filtered.xyz[:, 2].max() <= plain.xyz[:, 2].max()


def test_point_filter_rejecting_everything_reports_the_step():
    """Emptying the cloud must name the step, not return a silent empty mesh.

    Every other cleaning step in fuse_submaps records ``empty_at_step`` when
    it consumes the last point; a filter that could delete everything
    without saying so would make a fully-rejected run indistinguishable from
    a backbone that produced nothing.
    """
    cfg = FusionConfig(voxel_size=0.2, outlier_std_ratio=2.5, min_confidence=1)
    stats: dict = {}
    vertices, _, _, _, cloud = fuse_submaps(
        _two_submaps(np.random.default_rng(4)),
        cfg,
        stats=stats,
        point_filter=lambda c: np.zeros(c.xyz.shape[0], dtype=bool),
    )
    assert vertices.shape[0] == 0
    assert cloud.xyz.shape[0] == 0
    assert "point_filter" in stats["empty_at_step"]


def test_point_filter_wrong_mask_shape_raises():
    """A mask of the wrong length is a caller bug, and silently broadcasting
    it would mis-delete points rather than fail."""
    cfg = FusionConfig(voxel_size=0.2, outlier_std_ratio=2.5, min_confidence=1)
    with pytest.raises(ValueError, match="expected"):
        fuse_submaps(
            _two_submaps(np.random.default_rng(4)),
            cfg,
            point_filter=lambda c: np.ones(3, dtype=bool),
        )


def test_fuse_submaps_empty_input():
    cfg = FusionConfig()
    vertices, faces, colors, confidence, raw_point_cloud = fuse_submaps([], cfg)
    assert vertices.shape == (0, 3)
    assert faces.shape[0] == 0
    assert colors.shape[0] == 0
    assert confidence.shape[0] == 0
    assert raw_point_cloud.xyz.shape[0] == 0


def test_fuse_submaps_produces_all_three_confidence_tiers():
    """The headline "trust layer" must actually reach MEASURED, not just LOW/INFERRED.

    Before the views-bookkeeping fix in ``TSDFVolume.integrate_point_cloud``
    (see that method's docstring), ``fuse_submaps`` made exactly one
    ``integrate_point_cloud`` call over the whole merged cloud, so every
    voxel's accumulated view count capped at 1 -- below
    ``FusionConfig.measured_min_views`` -- and MEASURED was mathematically
    unreachable regardless of how confident or well-observed the input was.
    This test builds a scene with genuinely different observation levels
    (mirroring a real flight: some ground seen redundantly from many
    overlapping high-confidence windows, some far-off geometry only barely
    glimpsed once at low confidence) and asserts all three tiers show up --
    it fails against the pre-fix behaviour (MEASURED never appears) even
    though the scene clearly contains well-observed geometry.
    """
    rng = np.random.default_rng(7)

    def make_submap(
        idx: int, keyframe_indices: list[int], center: np.ndarray, n: int, raw_confidence: float, spread: float
    ) -> Submap:
        xyz = rng.normal(scale=spread, size=(n, 3)) + center
        confidence = np.full(n, raw_confidence, dtype=np.float64)
        rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
        pc = PointCloud(xyz=xyz, rgb=rgb)
        poses = [Pose(R=np.eye(3), t=center + np.array([i * 0.5, 0.1 * i, 5.0])) for i in range(len(keyframe_indices))]
        window = Window(index=idx, start=keyframe_indices[0], end=keyframe_indices[-1] + 1)
        return Submap(
            window=window,
            poses=poses,
            points=pc,
            confidence=confidence,
            keyframe_indices=keyframe_indices,
            local_origin=poses[0],
        )

    submaps: list[Submap] = []

    # Well-observed geometry: 6 overlapping submaps (30-keyframe flight,
    # each window overlapping the previous by 3 keyframes -- exactly the
    # >=3-shared-camera minimum merge_submaps' Umeyama fit needs) all
    # independently reconstructing the *same* small patch of ground at
    # high (box-hit-like) raw backbone confidence. Real multi-view
    # redundancy, not a single lucky high-confidence guess.
    kf = 0
    for i in range(6):
        kf_indices = list(range(kf, kf + 5))
        sm = make_submap(i, kf_indices, center=np.zeros(3), n=150, raw_confidence=0.9, spread=0.4)
        if i > 0:
            sm.window.shared_with_previous = kf_indices[:3]
        submaps.append(sm)
        kf += 2

    # Thinly-observed geometry: one submap, far away, low (ground-hit-like
    # but weaker) raw confidence, few points -- seen, but not corroborated.
    kf_indices = list(range(kf, kf + 5))
    thin = make_submap(len(submaps), kf_indices, center=np.array([15.0, 0.0, 0.0]), n=15, raw_confidence=0.5, spread=0.2)
    thin.window.shared_with_previous = kf_indices[:3]
    submaps.append(thin)

    cfg = FusionConfig(voxel_size=0.3, outlier_std_ratio=3.0, min_confidence=0)
    stats: dict = {}
    vertices, faces, colors, confidence, _raw_point_cloud = fuse_submaps(submaps, cfg, stats=stats)

    assert vertices.shape[0] > 0
    assert faces.shape[1] == 3
    assert colors.shape == (vertices.shape[0], 3)

    tiers_present = set(np.unique(confidence).tolist())
    assert tiers_present == {Confidence.INFERRED, Confidence.LOW_CONFIDENCE, Confidence.MEASURED}, (
        f"expected all three confidence tiers, got {tiers_present}"
    )
    # The well-observed cluster's redundant high-confidence coverage must
    # actually win some MEASURED vertices -- not just exist as a tier value
    # with zero members.
    assert int(np.sum(confidence == Confidence.MEASURED)) > 0

    # No covariance was attached to any submap's points, so this must have
    # gone through the backbone-confidence + view-count fallback, not the
    # (currently unreachable without covariance) BA-covariance path.
    assert stats.get("confidence_source") == "backbone_confidence_and_view_count"


# ---------------------------------------------------------------------------
# Fix 1: GSD-derived voxel sizing (replaces the scene-extent heuristic that
# erased buildings on real footage -- see fusion.tsdf's module docstring)
# ---------------------------------------------------------------------------


def test_derive_voxel_size_uses_gsd_not_scene_extent():
    """A huge scene extent must not, by itself, produce a huge voxel.

    Before this fix, ``_auto_voxel_size`` picked ``max_extent / 200``: a
    259 m-wide real scene gave voxel=1.30 m, well over an entire house's
    width. Here the point cloud spans 250 m, but the camera-to-scene depth
    (and therefore GSD) implies a small, sub-metre voxel -- the derived
    size must track GSD, not the bounding box.
    """
    from drishti3d.fusion.tsdf import _auto_voxel_size, _derive_voxel_size

    rng = np.random.default_rng(11)
    n = 500
    xyz = np.zeros((n, 3))
    xyz[:, 0] = rng.uniform(0, 250, n)  # a 250 m-wide scene, like the real bug report
    xyz[:, 1] = rng.uniform(0, 250, n)
    xyz[:, 2] = rng.uniform(0, 5, n)

    # Poses scattered near the scene at ~120 m altitude AGL (nadir survey).
    poses = [Pose(R=np.eye(3), t=np.array([x, y, 120.0])) for x, y in rng.uniform(0, 250, size=(20, 2))]
    intr = CameraIntrinsics(fx=2289.0, fy=2289.0, cx=1920.0, cy=1080.0, width=3840, height=2160)
    keyframe_intrinsics = {i: intr for i in range(len(poses))}

    stats: dict = {}
    voxel = _derive_voxel_size(xyz, poses, keyframe_intrinsics, gsd_multiplier=3.0, voxel_count_budget=10_000_000_000, stats=stats)

    old_bbox_voxel = _auto_voxel_size(xyz)
    assert old_bbox_voxel == pytest.approx(250.0 / 200.0, rel=0.05)  # the old, now-superseded behaviour

    assert stats["voxel_size_source"] == "gsd"
    assert stats["voxel_size_gsd_m"] is not None
    # GSD = depth/fx ~= 120/2289 ~= 0.0524 m/px; voxel = 3x that ~= 0.157m --
    # an order of magnitude finer than the old bounding-box heuristic.
    assert voxel < old_bbox_voxel / 3.0
    assert voxel == pytest.approx(stats["voxel_size_gsd_m"] * 3.0, rel=1e-6)


def test_derive_voxel_size_falls_back_without_poses_or_intrinsics():
    from drishti3d.fusion.tsdf import _auto_voxel_size, _derive_voxel_size

    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    stats: dict = {}
    voxel = _derive_voxel_size(xyz, poses=[], keyframe_intrinsics=None, gsd_multiplier=3.0, voxel_count_budget=1_000_000, stats=stats)

    assert stats["voxel_size_source"] == "bbox_fallback (no GSD available)"
    assert voxel == pytest.approx(_auto_voxel_size(xyz))


def test_derive_voxel_size_budget_clamp_coarsens_and_warns(caplog):
    """A tiny voxel-count budget must force coarsening, loudly logged, with both sizes reported."""
    from drishti3d.fusion.tsdf import _derive_voxel_size

    rng = np.random.default_rng(12)
    n = 500
    xyz = np.zeros((n, 3))
    xyz[:, 0] = rng.uniform(0, 250, n)
    xyz[:, 1] = rng.uniform(0, 250, n)
    xyz[:, 2] = rng.uniform(0, 5, n)

    poses = [Pose(R=np.eye(3), t=np.array([x, y, 120.0])) for x, y in rng.uniform(0, 250, size=(20, 2))]
    intr = CameraIntrinsics(fx=2289.0, fy=2289.0, cx=1920.0, cy=1080.0, width=3840, height=2160)
    keyframe_intrinsics = {i: intr for i in range(len(poses))}

    stats: dict = {}
    with caplog.at_level("WARNING", logger="drishti3d.fusion.tsdf"):
        voxel = _derive_voxel_size(
            xyz, poses, keyframe_intrinsics, gsd_multiplier=3.0, voxel_count_budget=1000, stats=stats
        )

    assert stats["voxel_size_budget_exceeded"] is True
    assert stats["voxel_size_ideal_m"] is not None
    assert voxel > stats["voxel_size_ideal_m"]  # coarsened relative to the ideal GSD-derived size
    assert any("VOXEL BUDGET WARNING" in rec.message and "sacrificed" in rec.message for rec in caplog.records)


def test_fuse_submaps_tiles_when_footprint_is_sparse_in_its_bounding_box():
    """Block-sparse tiled fusion (mesh voxel-lattice fix): a curved/partial-coverage
    flight's occupied footprint is typically a narrow strip through a much
    larger bounding box (measured on real footage: ~15% occupied at a 0.5m
    planview grid) -- a single dense TSDF grid over the whole bbox wastes
    almost all of ``voxel_count_budget`` on empty space, forcing the voxel
    size to coarsen far below the GSD-ideal size even though the ideal size
    itself would need very few voxels if only the *occupied* ground counted.
    This builds exactly that scenario (two small flat patches, 300 m apart,
    everything else empty) and checks that ``fuse_submaps`` recovers a
    voxel size close to the ideal GSD-derived one instead of the coarse
    single-grid fallback, by tiling.
    """
    rng = np.random.default_rng(17)

    def cluster_points(center: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        xyz = center + np.stack(
            [rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), rng.normal(scale=0.01, size=n)], axis=1
        )
        rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
        return xyz, rgb

    cluster_a = np.array([0.0, 0.0, 0.0])
    cluster_b = np.array([300.0, 0.0, 0.0])
    xyz_a, rgb_a = cluster_points(cluster_a)
    xyz_b, rgb_b = cluster_points(cluster_b)
    xyz = np.concatenate([xyz_a, xyz_b], axis=0)
    rgb = np.concatenate([rgb_a, rgb_b], axis=0)
    confidence = np.full(xyz.shape[0], 2, dtype=np.uint8)
    pc = PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)

    # One window/submap covering both clusters -- keyframe poses near each
    # cluster (for GSD's own nearest-camera depth estimate), no cross
    # -submap merge/alignment machinery needed since there's only one
    # submap (merge_submaps' n=1 case is an identity transform).
    keyframe_indices = [0, 1, 2, 3, 4, 5]
    poses = [Pose(R=np.eye(3), t=cluster_a + np.array([i * 0.2, 0.0, 120.0])) for i in range(3)] + [
        Pose(R=np.eye(3), t=cluster_b + np.array([i * 0.2, 0.0, 120.0])) for i in range(3)
    ]
    window = Window(index=0, start=0, end=6)
    s0 = Submap(
        window=window,
        poses=poses,
        points=pc,
        confidence=confidence.astype(np.float64),
        keyframe_indices=keyframe_indices,
        local_origin=poses[0],
    )

    intr = CameraIntrinsics(fx=2289.0, fy=2289.0, cx=1920.0, cy=1080.0, width=3840, height=2160)
    keyframe_intrinsics = {i: intr for i in range(6)}

    # A budget too small even for the two clusters' own OCCUPIED voxels
    # at GSD-ideal resolution, so a single grid must coarsen while two
    # tiles can each stay fine.
    #
    # This used to be 20_000, which only failed because the budget was
    # computed over the whole bounding box -- including the 300 m of
    # empty space between the clusters. That overcount is fixed (see
    # _voxel_count in fusion.tsdf), so triggering tiling now takes a
    # budget that the real occupied set genuinely exceeds.
    cfg = FusionConfig(
        voxel_count_budget=2_000, outlier_std_ratio=4.0, min_confidence=0, pre_tsdf_downsample_voxel_fraction=0.0
    )
    stats: dict = {}
    vertices, faces, colors, confidence, raw_point_cloud = fuse_submaps(
        [s0], cfg, stats=stats, keyframe_intrinsics=keyframe_intrinsics
    )

    # A sparse footprint must NOT be charged for the empty space between
    # its clusters. The budget counts the voxels the block-sparse TSDF
    # actually allocates -- the surface and its truncation band -- so 300 m
    # of nothing between two 2 m clusters costs nothing, and the voxel
    # stays near the GSD ideal on a single grid.
    #
    # This previously asserted tiling, which existed to work around the
    # opposite behaviour: the budget used to count the whole bounding box,
    # so the empty gap consumed it and forced a coarse voxel that only
    # per-tile fusion could recover from. With the count fixed, tiling
    # correctly declines (it only engages when it beats a single grid by
    # >10%), and the single grid is already fine.
    assert stats["voxel_size"] == pytest.approx(stats["voxel_size_ideal_m"], rel=0.35), stats
    assert stats["voxel_size"] < 1.0

    assert vertices.shape[0] > 0
    assert faces.shape[1] == 3
    assert colors.shape == (vertices.shape[0], 3)
    assert confidence.shape == (vertices.shape[0],)
    assert raw_point_cloud.xyz.shape[0] > 0

    # Both clusters must actually be represented in the output mesh, not
    # just whichever tile happened to be processed first.
    near_a = np.any(np.linalg.norm(vertices - cluster_a, axis=1) < 5.0)
    near_b = np.any(np.linalg.norm(vertices - cluster_b, axis=1) < 5.0)
    assert near_a and near_b


def test_fuse_submaps_passes_keyframe_intrinsics_through_to_gsd_sizing():
    """End to end: fuse_submaps with no voxel_size override must use GSD sizing when intrinsics are given."""
    rng = np.random.default_rng(13)

    def make_submap(idx: int, keyframe_indices: list[int], n: int = 200) -> Submap:
        xyz = rng.normal(size=(n, 3)) * 0.5
        confidence = rng.integers(1, 3, size=(n,)).astype(np.uint8)
        rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
        pc = PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)
        poses = [Pose(R=np.eye(3), t=np.array([i * 0.5, 0.1 * i, 5.0])) for i in range(len(keyframe_indices))]
        window = Window(index=idx, start=keyframe_indices[0], end=keyframe_indices[-1] + 1)
        return Submap(
            window=window,
            poses=poses,
            points=pc,
            confidence=confidence.astype(np.float64) / 2.0,
            keyframe_indices=keyframe_indices,
            local_origin=poses[0],
        )

    s0 = make_submap(0, [0, 1, 2, 3, 4])
    intr = CameraIntrinsics.from_hfov(84.0, 3840, 2160)
    keyframe_intrinsics = {i: intr for i in range(5)}

    cfg = FusionConfig(outlier_std_ratio=3.0, min_confidence=0)  # voxel_size left None -> GSD auto-derive
    stats: dict = {}
    fuse_submaps([s0], cfg, stats=stats, keyframe_intrinsics=keyframe_intrinsics)

    assert stats["voxel_size_source"] == "gsd"
    assert stats["voxel_size_gsd_m"] is not None
    assert "grid_dims" in stats or "empty_at_step" in stats


# ---------------------------------------------------------------------------
# Fix 3: pre-TSDF downsampling is decoupled from TSDF voxel_size
# ---------------------------------------------------------------------------


def test_pre_tsdf_downsample_skipped_below_point_budget():
    """Below the point budget, the cleaned cloud must reach TSDF integration undownsampled."""
    rng = np.random.default_rng(14)
    n = 500
    xyz = rng.normal(size=(n, 3)) * 0.3
    confidence = np.full(n, 2, dtype=np.uint8)
    pc = PointCloud(xyz=xyz, rgb=rng.integers(0, 255, size=(n, 3)).astype(np.uint8), confidence=confidence)
    poses = [Pose(R=np.eye(3), t=np.array([i * 0.5, 0.0, 5.0])) for i in range(5)]
    window = Window(index=0, start=0, end=5)
    submap = Submap(
        window=window, poses=poses, points=pc, confidence=confidence.astype(np.float64), keyframe_indices=[0, 1, 2, 3, 4], local_origin=poses[0]
    )

    # A very fine explicit voxel_size: the pre-TSDF downsample grid
    # (voxel_size * 0.25 = 0.0125m) is far finer than this sparse cloud's
    # actual point spacing, so it should be a no-op (every point already
    # lands in its own cell) -- no pre-TSDF downsampling should show up as
    # having actually fired.
    cfg = FusionConfig(voxel_size=0.05, outlier_std_ratio=5.0, min_confidence=0)
    stats: dict = {}
    fuse_submaps([submap], cfg, stats=stats)

    assert "pre_tsdf_downsample_voxel_size" not in stats
    assert stats["after_voxel_downsample"] == stats["pre_tsdf_points"]


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def _sphere_cloud(n: int = 1500, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    xyz = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )
    confidence = np.full(n, Confidence.MEASURED, dtype=np.uint8)
    rgb = np.full((n, 3), 180, dtype=np.uint8)
    return PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)


def test_estimate_normals_orientation_outward_without_poses():
    pc = _sphere_cloud()
    normals = estimate_normals(pc, k=20)

    assert normals.shape == pc.xyz.shape
    dots = np.einsum("ij,ij->i", normals, pc.xyz)
    assert (dots > 0).mean() > 0.9


def test_estimate_normals_orientation_toward_camera():
    pc = _sphere_cloud()
    # A single camera far along +X. Orientation is resolved per-point
    # toward the *nearest* camera (see estimate_normals' docstring on why
    # that's a documented approximation, not a full visibility solve) --
    # with only one camera in the whole scene, every point (including the
    # occluded far hemisphere, which no single real camera could ever
    # have actually seen) gets oriented toward it, so only the
    # near-camera hemisphere is a meaningful check here.
    camera_positions = np.array([[10.0, 0.0, 0.0]])
    normals = estimate_normals(pc, k=20, camera_positions=camera_positions)
    dots = np.einsum("ij,ij->i", normals, pc.xyz)

    near_hemisphere = pc.xyz[:, 0] > 0.5
    assert near_hemisphere.sum() > 20
    assert (dots[near_hemisphere] > 0).mean() > 0.9


def test_poisson_reconstruct_flags_low_density_as_inferred():
    pc = _sphere_cloud(n=1200)
    vertices, faces, colors, confidence = poisson_reconstruct(pc, depth=6)

    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert colors.shape == (vertices.shape[0], 3)
    assert confidence.shape[0] == vertices.shape[0]
    tiers = set(np.unique(confidence).tolist())
    assert Confidence.INFERRED in tiers or Confidence.LOW_CONFIDENCE in tiers


def test_decimate_mesh_reduces_triangle_count():
    pc = _sphere_cloud(n=2000)
    vertices, faces, _colors, _confidence = poisson_reconstruct(pc, depth=6)
    assert faces.shape[0] > 200

    target = faces.shape[0] // 4
    new_vertices, new_faces = decimate_mesh(vertices, faces, target_triangles=target)

    assert new_faces.shape[0] < faces.shape[0]
    assert new_vertices.shape[1] == 3
    assert new_faces.max() < new_vertices.shape[0]


def test_compute_mesh_stats():
    # A single unit-right-triangle mesh with a known area.
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    confidence = np.array([Confidence.MEASURED, Confidence.LOW_CONFIDENCE, Confidence.INFERRED], dtype=np.uint8)

    stats = compute_mesh_stats(vertices, faces, confidence)

    assert stats["triangle_count"] == 1
    assert stats["vertex_count"] == 3
    np.testing.assert_allclose(stats["surface_area_m2"], 0.5)
    assert stats["watertight"] is False  # a single triangle can't be watertight
    breakdown = stats["confidence_breakdown_pct"]
    np.testing.assert_allclose(breakdown["measured"], 100.0 / 3.0)
    np.testing.assert_allclose(breakdown["low_confidence"], 100.0 / 3.0)
    np.testing.assert_allclose(breakdown["inferred"], 100.0 / 3.0)


def test_compute_mesh_stats_without_confidence():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    stats = compute_mesh_stats(vertices, faces)
    assert stats["confidence_breakdown_pct"] is None


# ---------------------------------------------------------------------------
# Mesh cleanup: Poisson spikes and speck islands
# ---------------------------------------------------------------------------


def _grid_mesh_with_artifacts():
    """Flat 8x8 grid + one 50 m spike triangle + a detached 2-face speck."""
    g = 8
    axis = np.linspace(0, 7, g)
    gx, gy = np.meshgrid(axis, axis)
    vertices = np.c_[gx.ravel(), gy.ravel(), np.zeros(g * g)]

    faces = []
    for r in range(g - 1):
        for c in range(g - 1):
            a = r * g + c
            faces += [[a, a + 1, a + g + 1], [a, a + g + 1, a + g]]
    faces = np.array(faces)

    # A spike: one vertex flung 50 m up, tethered by a single sliver.
    vertices = np.vstack([vertices, [[3.5, 3.5, 50.0]]])
    spike_vertex = len(vertices) - 1
    faces = np.vstack([faces, [[0, 1, spike_vertex]]])

    # A speck island, far from everything.
    base = len(vertices)
    vertices = np.vstack(
        [vertices, [[100, 100, 0], [100.1, 100, 0], [100, 100.1, 0], [100.1, 100.1, 0]]]
    )
    faces = np.vstack([faces, [[base, base + 1, base + 2], [base + 1, base + 3, base + 2]]])
    return vertices, faces, spike_vertex


def test_clean_mesh_removes_spikes_and_specks_but_keeps_the_surface():
    from drishti3d.fusion.mesh import clean_mesh

    vertices, faces, _spike = _grid_mesh_with_artifacts()
    v_out, f_out, stats = clean_mesh(vertices, faces)

    assert stats["removed_spike_faces"] == 1
    assert stats["removed_component_faces"] == 2
    # The real surface -- 98 grid faces -- survives.
    assert len(f_out) == 98
    # The 50 m spike vertex is gone.
    assert v_out[:, 2].max() < 1.0
    # So is the distant speck.
    assert v_out[:, 0].max() < 50.0


def test_clean_mesh_only_ever_removes_geometry():
    """Never adds, smooths or moves a vertex -- measurements must not shift."""
    from drishti3d.fusion.mesh import clean_mesh

    vertices, faces, _ = _grid_mesh_with_artifacts()
    v_out, f_out, _stats = clean_mesh(vertices, faces)

    assert len(v_out) <= len(vertices)
    assert len(f_out) <= len(faces)
    # Every surviving vertex is bit-identical to one of the inputs.
    surviving = {tuple(v) for v in v_out}
    original = {tuple(v) for v in vertices}
    assert surviving <= original


def test_clean_mesh_returns_a_vertex_mapping_for_parallel_channels():
    """Colour/confidence/semantics must follow the compaction."""
    from drishti3d.fusion.mesh import clean_mesh

    vertices, faces, _ = _grid_mesh_with_artifacts()
    colors = np.arange(len(vertices) * 3, dtype=np.uint8).reshape(-1, 3)

    v_out, _f_out, stats = clean_mesh(vertices, faces)
    kept = stats["kept_vertices"]

    assert len(kept) == len(v_out)
    # Gathering through `kept` keeps colour attached to the right point.
    np.testing.assert_array_equal(colors[kept][0], colors[kept[0]])


def test_clean_mesh_keeps_large_disconnected_regions():
    """A single pass legitimately yields several big unconnected surfaces."""
    from drishti3d.fusion.mesh import clean_mesh

    def slab(offset):
        g = 10
        axis = np.linspace(0, 9, g)
        gx, gy = np.meshgrid(axis, axis)
        v = np.c_[gx.ravel() + offset, gy.ravel(), np.zeros(g * g)]
        f = []
        for r in range(g - 1):
            for c in range(g - 1):
                a = r * g + c
                f += [[a, a + 1, a + g + 1], [a, a + g + 1, a + g]]
        return v, np.array(f)

    v1, f1 = slab(0.0)
    v2, f2 = slab(500.0)  # far away, genuinely disconnected, genuinely large
    vertices = np.vstack([v1, v2])
    faces = np.vstack([f1, f2 + len(v1)])

    _v, f_out, stats = clean_mesh(vertices, faces)

    # Both slabs survive: neither is a speck.
    assert stats["components_kept"] == 2
    assert len(f_out) == len(faces)


def test_clean_mesh_component_cap_protects_real_structure():
    """A bare ratio would delete whole buildings on a multi-million-face mesh.

    1% of 6.5M faces is 65,000 -- an entire building. The cap keeps the
    rule aimed at the tens-to-hundreds-of-faces specks Poisson leaves.
    """
    from drishti3d.fusion.mesh import clean_mesh

    vertices, faces, _ = _grid_mesh_with_artifacts()
    _v, _f, stats = clean_mesh(vertices, faces, min_component_ratio=0.5, component_face_cap=10)

    # Without the cap, a 50% ratio would demand ~49 faces and delete
    # everything small; the cap holds the threshold at 10.
    assert stats["component_face_threshold"] == 10


def test_clean_mesh_handles_an_empty_mesh():
    from drishti3d.fusion.mesh import clean_mesh

    v, f, stats = clean_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert len(v) == 0 and len(f) == 0
    assert stats["faces_out"] == 0


# ---------------------------------------------------------------------------
# Photometric verification: grade geometry against the source frames
# ---------------------------------------------------------------------------


def _world_textured_views(alt: float = 25.0, offsets=(-4.0, -1.5, 1.5, 4.0)):
    """Nadir views rendered from a WORLD-space texture.

    The fixture has to be world-space, not pixel-space: if the image
    content is a function of pixel coordinates, a geometrically CORRECT
    point still samples different colours from different views, and the
    test measures nothing. Rendering by inverse-projecting each pixel onto
    z=0 means the same physical spot has the same colour in every view --
    which is exactly the property photometric consistency checks.
    """
    from drishti3d.types import CameraIntrinsics, Pose

    intr = CameraIntrinsics.from_hfov(70.0, 320, 240)
    rot = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])

    def texture(x, y):
        r = ((np.sin(x * 1.7) + np.cos(y * 2.1)) * 0.5 + 0.5) * 255
        g = (np.sin(x * 0.9 + y * 1.3) * 0.5 + 0.5) * 255
        b = (np.cos(x * 2.3 - y * 0.7) * 0.5 + 0.5) * 255
        return np.stack([r, g, b], -1).astype(np.uint8)

    views = []
    for tx in offsets:
        pose = Pose(R=rot.copy(), t=np.array([tx, 0.0, alt]))
        v, u = np.mgrid[0:240, 0:320]
        d_cam = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u, float)], -1)
        d_world = d_cam @ rot.T
        t = (0.0 - pose.t[2]) / d_world[..., 2]
        x = pose.t[0] + t * d_world[..., 0]
        y = pose.t[1] + t * d_world[..., 1]
        views.append((pose, intr, texture(x, y)))
    return views


def test_photometric_separates_correct_geometry_from_floating_points():
    """The core claim: views agree about real surfaces and disagree about wrong ones."""
    from drishti3d.fusion.photometric import photometric_consistency

    views = _world_textured_views()
    rng = np.random.default_rng(0)
    on_surface = np.c_[rng.uniform(-6, 6, (400, 2)), np.zeros(400)]
    floating = np.c_[rng.uniform(-6, 6, (100, 2)), np.full(100, 5.0)]
    xyz = np.vstack([on_surface, floating])

    result = photometric_consistency(xyz, views, min_contrast=1.0)

    assert result.stats["views_used"] == 4
    # Correct geometry is overwhelmingly verified; floating geometry never is.
    assert result.verified[:400].mean() > 0.8
    assert result.verified[400:].mean() < 0.05
    # And the error signal itself separates cleanly.
    assert np.median(result.error[:400]) < 10.0
    assert np.median(result.error[400:]) > 30.0


def test_verify_confidence_promotes_verified_and_demotes_inconsistent():
    from drishti3d.fusion.photometric import photometric_consistency, verify_confidence
    from drishti3d.types import Confidence

    views = _world_textured_views()
    rng = np.random.default_rng(1)
    on_surface = np.c_[rng.uniform(-6, 6, (300, 2)), np.zeros(300)]
    floating = np.c_[rng.uniform(-6, 6, (100, 2)), np.full(100, 5.0)]
    xyz = np.vstack([on_surface, floating])

    result = photometric_consistency(xyz, views, min_contrast=1.0)

    # Start with the backbone confidently wrong about everything.
    confidence = np.full(len(xyz), int(Confidence.MEASURED), dtype=np.uint8)
    graded, stats = verify_confidence(confidence, result)

    # Floating points were MEASURED and get demoted on photometric evidence.
    assert stats["photometric_demoted"] > 50
    assert (graded[300:] == int(Confidence.MEASURED)).mean() < 0.1
    # Real surface keeps its MEASURED tier.
    assert (graded[:300] == int(Confidence.MEASURED)).mean() > 0.8


def test_unverifiable_points_are_not_graded_either_way():
    """A textureless surface agrees across views regardless of geometry.

    Agreement there is uninformative, so those points must be reported
    UNVERIFIABLE rather than promoted -- promoting them would manufacture
    confidence from the absence of evidence.
    """
    from drishti3d.fusion.photometric import photometric_consistency, verify_confidence
    from drishti3d.types import CameraIntrinsics, Confidence, Pose

    intr = CameraIntrinsics.from_hfov(70.0, 320, 240)
    rot = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    # Uniform grey everywhere: zero texture contrast.
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    views = [(Pose(R=rot.copy(), t=np.array([tx, 0.0, 25.0])), intr, flat) for tx in (-3.0, 0.0, 3.0)]

    xyz = np.c_[np.random.default_rng(0).uniform(-5, 5, (200, 2)), np.zeros(200)]
    result = photometric_consistency(xyz, views)

    assert result.unverifiable.all()
    assert not result.verified.any()

    confidence = np.full(len(xyz), int(Confidence.INFERRED), dtype=np.uint8)
    graded, stats = verify_confidence(confidence, result)
    assert stats["photometric_promoted"] == 0
    # Nothing invented, nothing destroyed.
    np.testing.assert_array_equal(graded, confidence)


def test_photometric_handles_no_views_and_no_points():
    from drishti3d.fusion.photometric import photometric_consistency

    empty = photometric_consistency(np.zeros((0, 3)), [])
    assert empty.stats["points"] == 0

    no_views = photometric_consistency(np.zeros((10, 3)), [])
    assert no_views.unverifiable.all()
    assert not no_views.verified.any()


def test_photometric_never_moves_a_vertex():
    """Grades and deletes; never reshapes. A nudged surface is unmeasured."""
    from drishti3d.fusion.photometric import photometric_consistency

    views = _world_textured_views()
    xyz = np.c_[np.random.default_rng(0).uniform(-5, 5, (100, 2)), np.zeros(100)]
    before = xyz.copy()

    photometric_consistency(xyz, views, min_contrast=1.0)

    np.testing.assert_array_equal(xyz, before)


def test_local_texture_measures_detail_not_saturation():
    """The contrast floor must react to texture, not to colour.

    The regression this pins: contrast used to be the spread across R/G/B
    of the mean colour, which scores a vividly coloured flat surface high
    and a sharply textured grey surface zero -- backwards for the thing it
    gates. On desaturated aerial imagery that marked 96.5% of points
    UNVERIFIABLE and disabled the photometric check entirely.
    """
    from drishti3d.fusion.photometric import _local_texture

    # Saturated but perfectly flat: no detail to match on.
    flat_colour = np.zeros((64, 64, 3), dtype=np.uint8)
    flat_colour[..., 2] = 220
    # Grey but sharply textured: plenty to match on.
    checks = np.indices((64, 64)).sum(axis=0) % 2
    grey_textured = np.repeat((checks * 200).astype(np.uint8)[..., None], 3, axis=2)

    assert _local_texture(flat_colour).mean() < 1.0
    assert _local_texture(grey_textured).mean() > 50.0


def test_umeyama_fixed_rotation_honours_locked_scale():
    """A locked scale must survive the fit exactly, not be nudged toward it."""
    from drishti3d.geometry.submap import umeyama_fixed_rotation

    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    # Ground truth applies a 1.7x scale; locking to 1.0 must NOT recover it.
    dst = 1.7 * src + np.array([5.0, -2.0, 1.0])

    free = umeyama_fixed_rotation(src, dst, np.eye(3))
    locked = umeyama_fixed_rotation(src, dst, np.eye(3), scale=1.0)

    assert free.scale == pytest.approx(1.7, rel=1e-6)
    assert locked.scale == 1.0
    # Translation still absorbs what it can, so the fit stays centred.
    assert locked.apply(src).mean(axis=0) == pytest.approx(dst.mean(axis=0), abs=1e-9)


def test_locked_scale_keeps_submap_grounds_co_located():
    """Two submaps of the same flat ground must land at the same elevation.

    This is the 68-component / 54 m split in miniature. Each submap sees the
    same ground from a short, nearly-straight camera run -- the geometry
    that makes a free scale ill-conditioned. With scale free, the two
    submaps' grounds are allowed to drift apart; with scale locked at 1.0
    (correct for a metric backbone) they cannot.
    """
    from drishti3d.geometry.submap import merge_submaps

    ground_z = -120.0
    rng = np.random.default_rng(7)

    def make(idx, kfs, x0):
        # Cameras in a short straight run at z=0, ground 120 m below.
        poses = [Pose(R=np.eye(3), t=np.array([x0 + 8.0 * j, 0.0, 0.0])) for j in range(len(kfs))]
        pts = np.column_stack([
            rng.uniform(x0 - 10, x0 + 40, 400),
            rng.uniform(-15, 15, 400),
            np.full(400, ground_z) + rng.normal(0, 0.05, 400),
        ])
        conf = np.full(400, 2, dtype=np.uint8)
        return Submap(
            window=Window(index=idx, start=kfs[0], end=kfs[-1] + 1),
            poses=poses,
            points=PointCloud(xyz=pts, confidence=conf),
            confidence=conf.astype(np.float64) / 2.0,
            keyframe_indices=kfs,
            local_origin=poses[0],
        )

    s0 = make(0, [0, 1, 2, 3], x0=0.0)
    s1 = make(1, [2, 3, 4, 5], x0=16.0)
    s1.window.shared_with_previous = [2, 3]
    gps = {k: np.array([8.0 * k, 0.0, 0.0]) for k in range(6)}

    merged, _ = merge_submaps(
        [s0, s1], camera_gps_enu=gps, strategy="gps_anchored", lock_scale=1.0
    )
    z = merged.xyz[:, 2]
    # One ground plane, not two: the spread must stay at the noise level,
    # nowhere near the tens of metres a free scale can introduce.
    assert np.ptp(z) < 1.0, f"ground split by {np.ptp(z):.1f} m with scale locked"
    assert abs(np.median(z) - ground_z) < 1.0


def _altitude_submap(recon_alt_m, n_kf=4, seed=0):
    """A window flying at a known height over flat ground `recon_alt_m` below."""
    rng = np.random.default_rng(seed)
    kfs = list(range(n_kf))
    poses = [Pose(R=np.eye(3), t=np.array([20.0 * j, 0.0, 0.0])) for j in range(n_kf)]
    pts = np.column_stack([
        rng.uniform(-30, 20 * n_kf + 30, 3000),
        rng.uniform(-40, 40, 3000),
        np.full(3000, -recon_alt_m) + rng.normal(0, 0.3, 3000),
    ])
    conf = np.full(3000, 2, dtype=np.uint8)
    return Submap(
        window=Window(index=0, start=0, end=n_kf),
        poses=poses,
        points=PointCloud(xyz=pts, confidence=conf),
        confidence=conf.astype(np.float64),
        keyframe_indices=kfs,
        local_origin=poses[0],
    )


@pytest.mark.parametrize("recon_alt_m", [18.0, 70.0, 120.0])
def test_altitude_anchor_recovers_true_flying_height(recon_alt_m):
    """Both measured backbone failures must land back at the true altitude.

    18 m and 70 m are not invented: they are the reconstructed
    camera-to-ground distances measured on real 77-keyframe footage flown at
    a GPS-confirmed 120 m AGL, where MapAnything reported is_metric=True.
    120 m is the already-correct case, which must be left alone.
    """
    from drishti3d.geometry.submap import altitude_anchor_scale

    true_alt = 120.0
    sm = _altitude_submap(recon_alt_m)
    scale, diag = altitude_anchor_scale(sm, {k: true_alt for k in sm.keyframe_indices})

    assert scale is not None, diag
    assert recon_alt_m * scale == pytest.approx(true_alt, rel=0.05)


def test_altitude_anchor_refuses_absurd_corrections():
    """A backbone off by 60x has failed, not mis-scaled.

    Rescaling it would turn unusable output into something that looks
    plausible, which is worse than reporting the failure.
    """
    from drishti3d.geometry.submap import altitude_anchor_scale

    sm = _altitude_submap(2.0)
    scale, diag = altitude_anchor_scale(sm, {k: 120.0 for k in sm.keyframe_indices})

    assert scale is None
    assert "outside" in diag["failure"]


def test_altitude_anchor_reports_missing_telemetry_rather_than_guessing():
    from drishti3d.geometry.submap import altitude_anchor_scale

    sm = _altitude_submap(120.0)
    scale, diag = altitude_anchor_scale(sm, {})
    assert scale is None
    assert "telemetry altitude" in diag["failure"]


def test_altitude_anchoring_co_locates_windows_with_different_depth_errors():
    """The 50 m ground split, reproduced and fixed.

    Two windows over the same flat ground, one reconstructed at 0.15x true
    depth and one at 0.59x -- the exact per-window inconsistency measured on
    real footage. Without anchoring their grounds sit tens of metres apart;
    with it they must agree.
    """
    from drishti3d.geometry.submap import altitude_anchor_scale

    true_alt = 120.0
    bad = _altitude_submap(18.0, seed=1)
    ok = _altitude_submap(70.0, seed=2)

    s_bad, _ = altitude_anchor_scale(bad, {k: true_alt for k in bad.keyframe_indices})
    s_ok, _ = altitude_anchor_scale(ok, {k: true_alt for k in ok.keyframe_indices})

    ground_bad = -18.0 * s_bad
    ground_ok = -70.0 * s_ok
    # Before anchoring these differ by 52 m; after, they must not.
    assert abs(-18.0 - -70.0) > 50.0
    assert abs(ground_bad - ground_ok) < 5.0


def test_point_budget_voxel_sizes_by_surface_not_volume():
    """A surface cloud with a tall bounding box must not be collapsed to a few thousand points.

    Reproduces the measured failure: two planar slabs far apart in Z
    inflate the bounding-box volume, and a volumetric estimate then picks a
    metres-wide voxel that leaves ~20k points from millions. The budget is
    a *ceiling*; the returned voxel must land near it from below.
    """
    from drishti3d.fusion.filters import voxel_downsample
    from drishti3d.fusion.tsdf import _voxel_size_for_point_budget

    rng = np.random.default_rng(0)
    n = 400_000
    xy = rng.uniform(0, 300, (n, 2))
    z = np.where(rng.uniform(size=n) < 0.5, 0.0, 60.0) + rng.normal(0, 0.05, n)
    xyz = np.column_stack([xy, z])

    budget = 50_000
    voxel = _voxel_size_for_point_budget(xyz, budget)
    kept = voxel_downsample(PointCloud(xyz=xyz), voxel_size=voxel).xyz.shape[0]

    assert kept <= budget
    # Near the ceiling, not orders of magnitude under it.
    assert kept >= 0.5 * budget, f"voxel {voxel:.3f} m kept only {kept} of a {budget} budget"
    # For 2 slabs of 300x300 m at ~25k points each, that is a ~1.9 m voxel;
    # the volumetric estimate would have said ~(300*300*60/50000)**(1/3) = 10 m.
    assert voxel < 4.0


def test_voxel_sized_to_face_budget_not_overbuilt_then_decimated():
    """Building 5x the faces and decimating them away is wasted work.

    Measured: a 0.66 m voxel gave 10,003,138 faces, decimated to
    1,999,999 -- 345 s of TSDF integration plus 97 s of decimation spent
    on geometry discarded immediately. The voxel must be chosen so the
    mesh lands near the budget in the first place.
    """
    from drishti3d.fusion.tsdf import _FACES_PER_AREA_CONSTANT

    area = 250_000.0  # m2 of occupied ground, the measured site
    target = 2_000_000

    voxel = (_FACES_PER_AREA_CONSTANT * area / target) ** 0.5
    predicted = _FACES_PER_AREA_CONSTANT * area / voxel**2
    assert predicted == pytest.approx(target, rel=1e-6)

    # The measured run's own numbers must come back out of the model:
    # 0.66 m over that area predicted ~10M faces, which is what it built.
    measured = _FACES_PER_AREA_CONSTANT * area / 0.66**2
    assert 8e6 < measured < 12e6

    # And the chosen voxel is coarser than the one that overbuilt.
    assert voxel > 0.66
