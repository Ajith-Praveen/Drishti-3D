"""Height-field multi-view stereo (geometry.heightfield) on a rendered nadir scene with known heights."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drishti3d.geometry.heightfield import (
    HeightfieldConfig,
    HeightfieldSurface,
    reconstruct_heightfield,
)
from drishti3d.types import CameraIntrinsics, Confidence, Pose

_W, _H, _F = 320, 240, 300.0
_ALT = 80.0
_PLATFORM = (-20.0, 20.0, -15.0, 15.0, 8.0)  # x0, x1, y0, y1, height
_EXTENT = 130.0  # texture covers [-130, 130] m
_TEX_RES = 0.25  # m per texel

_rng = np.random.default_rng(11)
_TEXTURE = cv2.GaussianBlur((_rng.random((int(2 * _EXTENT / _TEX_RES),) * 2) * 255).astype(np.float32), (0, 0), 1.5)
# Nadir camera: image x -> world +X, image y -> world -Y (image up = north), optical axis -> world -Z.
_R_NADIR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _render(centre: np.ndarray) -> np.ndarray:
    """Ray-cast the textured ground + platform-top scene from a nadir camera at ``centre``."""
    u, v = np.meshgrid(np.arange(_W) + 0.5, np.arange(_H) + 0.5)
    d = np.stack([(u - _W / 2) / _F, (v - _H / 2) / _F, np.ones_like(u)], -1) @ _R_NADIR.T
    x0, x1, y0, y1, h = _PLATFORM
    t_top = (h - centre[2]) / d[..., 2]
    p_top = centre + t_top[..., None] * d
    on_top = (p_top[..., 0] >= x0) & (p_top[..., 0] <= x1) & (p_top[..., 1] >= y0) & (p_top[..., 1] <= y1)
    t_ground = (0.0 - centre[2]) / d[..., 2]
    p = np.where(on_top[..., None], p_top, centre + t_ground[..., None] * d)
    col = ((p[..., 0] + _EXTENT) / _TEX_RES).astype(np.float32)
    row = ((_EXTENT - p[..., 1]) / _TEX_RES).astype(np.float32)
    grey = cv2.remap(_TEXTURE, col, row, cv2.INTER_LINEAR)
    return cv2.cvtColor(np.clip(grey, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _views():
    centres = [np.array([x, y, _ALT]) for x in np.arange(-40.0, 41.0, 16.0) for y in (-14.0, 14.0)]
    images = [_render(c) for c in centres]
    intr = [CameraIntrinsics(_F, _F, _W / 2, _H / 2, _W, _H) for _ in centres]
    poses = [Pose(R=_R_NADIR.copy(), t=c) for c in centres]
    return images, intr, poses


@pytest.fixture(scope="module")
def reconstructed() -> tuple[HeightfieldSurface, dict]:
    images, intr, poses = _views()
    cfg = HeightfieldConfig(coarse_below_m=15.0, coarse_above_m=20.0, tile_m=60.0, fine_range_m=2.0)
    return reconstruct_heightfield(images, intr, poses, agl_m=_ALT, config=cfg)


def test_recovers_ground_and_platform_heights(reconstructed) -> None:
    surf, diag = reconstructed
    X, Y = surf.cell_centres()
    x0, x1, y0, y1, h = _PLATFORM
    top = (X > x0 + 4) & (X < x1 - 4) & (Y > y0 + 4) & (Y < y1 - 4) & surf.valid
    ground = ((np.abs(X) > 28) | (np.abs(Y) > 22)) & (np.abs(X) < 55) & (np.abs(Y) < 35) & surf.valid
    assert top.sum() > 500 and ground.sum() > 2000, (top.sum(), ground.sum())
    assert np.median(np.abs(surf.z[top] - h)) < 0.3
    assert np.median(np.abs(surf.z[ground] - 0.0)) < 0.2
    assert diag["method"] == "heightfield_mvs" and diag["cells_reconstructed"] > 0


def test_mesh_closes_the_platform_edge_with_an_inferred_wall(reconstructed) -> None:
    """The 8 m step is a wall now, built from duplicated vertices tiered INFERRED, not an open gap."""
    surf, _ = reconstructed
    pc, faces = surf.mesh()
    n_cells = int(surf.valid.sum())
    assert faces.shape[0] > 0 and pc.xyz.shape[0] > n_cells
    z = pc.xyz[faces][..., 2]
    span = z.max(axis=1) - z.min(axis=1)
    wall = span > surf.max_step_m + 1e-6
    assert wall.any(), "the platform edge must be closed"
    assert (faces[wall] >= n_cells).all()  # walls only use the duplicated vertices ...
    assert (pc.confidence[n_cells:] == int(Confidence.INFERRED)).all()  # ... which are INFERRED
    assert np.isnan(pc.uncertainty_m[n_cells:]).all()
    assert np.allclose(pc.xyz[n_cells:], pc.xyz[surf._wall_src])
    # Flat faces stay counter-clockwise seen from above: normals point up.
    a, b, c = (pc.xyz[faces[~wall, k]] for k in range(3))
    assert (np.cross(b - a, c - a)[:, 2] > 0).mean() > 0.99
    assert surf.mesh_vertex_uv().shape == (pc.xyz.shape[0], 2)


def test_textured_ground_is_measured(reconstructed) -> None:
    surf, _ = reconstructed
    pc = surf.point_cloud()
    assert pc.rgb is not None and pc.rgb.shape == (pc.xyz.shape[0], 3)
    stereo = ~surf.inferred[surf.valid]  # the visible-scene fill is INFERRED by construction
    assert np.mean(pc.confidence[stereo] == int(Confidence.MEASURED)) > 0.5


def test_needs_at_least_three_views() -> None:
    images, intr, poses = _views()
    with pytest.raises(ValueError):
        reconstruct_heightfield(images[:2], intr[:2], poses[:2], agl_m=_ALT)


def test_geometry_branch_hands_fusion_a_meshed_surface(tmp_path) -> None:
    """GeometryStage's height-field branch -> placement check -> FusionStage's height-map read-out."""
    from drishti3d.config import Config
    from drishti3d.pipeline.stages import FusionStage, GeometryStage, PipelineState
    from drishti3d.types import FrameMetrics, Keyframe

    images, intr, poses = _views()
    cfg = Config()
    cfg.export.output_dir = str(tmp_path / "output")
    state = PipelineState(video_path=tmp_path / "v.mp4", telemetry_path=None, config=cfg, backbone_name="mapanything")
    state.keyframes = [
        Keyframe(frame_index=i, timestamp=float(i), metrics=FrameMetrics(i, float(i), 100.0, 1.0, 120.0, 0.0),
                 intrinsics=intr[i], pose=poses[i])
        for i in range(len(images))
    ]
    state.keyframe_cache = type(
        "Cache", (), {"get": lambda self, i: images[i], "scale_for": lambda self, i: 1.0, "__len__": lambda self: len(images)}
    )()
    # Sparse "bundle-adjusted" points on the true surface: the prior and the placement reference.
    xs, ys = np.meshgrid(np.arange(-60.0, 61.0, 3.0), np.arange(-40.0, 41.0, 3.0))
    x0, x1, y0, y1, h = _PLATFORM
    zs = np.where((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1), h, 0.0)
    state.ba_points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], -1)
    state.geometry_world_frame = True

    artifacts, _message = GeometryStage()._run_heightfield(
        state, state.keyframes, poses, None, None, flight_profile=None, camera_gps_enu={}
    )
    assert artifacts["backbone"] == "heightfield_mvs"
    assert artifacts["heightfield"]["prior"] == "bundle_adjusted_points"
    assert state.geometry_heightmap and len(state.submaps) == 1
    assert state.placement_report is not None and state.placement_report.verdict == "PASS", state.placement_report.summary()
    assert "bundle-adjusted ground" in state.placement_report.summary()

    fusion_artifacts, _ = FusionStage()._from_heightmap(state, state.incremental_fusion)
    assert state.mesh_faces is not None and state.mesh_faces.shape[0] > 1000
    tiers = fusion_artifacts["tier_fractions"]
    assert tiers["measured"] > 0.5 * (1.0 - tiers["inferred"])  # majority of what stereo measured


def test_true_ortho_texture_and_planar_uvs(reconstructed) -> None:
    surf, diag = reconstructed
    ny, nx = surf.z.shape
    assert surf.texture_rgb is not None
    assert surf.texture_rgb.shape[0] >= ny and surf.texture_rgb.shape[1] >= nx
    uv = surf.vertex_uv()
    pc = surf.point_cloud()
    assert uv.shape == (pc.xyz.shape[0], 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    # Planar: u grows east, v grows north.
    assert np.corrcoef(uv[:, 0], pc.xyz[:, 0])[0, 1] > 0.999
    assert np.corrcoef(uv[:, 1], pc.xyz[:, 1])[0, 1] > 0.999
    assert diag["texture_px"] == [surf.texture_rgb.shape[0], surf.texture_rgb.shape[1]]


def test_planar_texture_survives_fusion_and_exports(tmp_path, reconstructed) -> None:
    import trimesh

    from drishti3d.config import Config
    from drishti3d.pipeline.stages import ExportStage, FusionStage, PipelineState

    surf, _ = reconstructed
    cfg = Config()
    cfg.fusion.photometric_reject_before_mesh = False
    state = PipelineState(video_path=tmp_path / "v.mp4", telemetry_path=None, config=cfg, backbone_name="mapanything")
    FusionStage()._from_heightmap(state, surf)
    assert state.mesh_uv is not None and len(state.mesh_uv) == len(state.point_cloud.xyz)
    written: dict = {}
    stats = ExportStage()._write_planar_texture(state, tmp_path, written)
    assert stats is not None and stats["method"] == "heightfield_true_ortho"
    for name in ("model_textured.obj", "model_textured.mtl", "model_textured.png", "model_textured.glb"):
        assert (tmp_path / name).is_file(), name
    scene = trimesh.load(tmp_path / "model_textured.glb")
    geom = next(iter(scene.geometry.values())) if hasattr(scene, "geometry") else scene
    assert geom.visual.kind == "texture"
    assert geom.visual.material.baseColorTexture is not None


def test_small_enclosed_holes_are_filled_and_tiered_inferred() -> None:
    from drishti3d.geometry.heightfield import _fill_small_holes

    z = np.zeros((20, 20), dtype=np.float32)
    z[5:7, 5:7] = np.nan  # 4-cell pinhole inside the surface: filled
    z[10:17, 10:17] = np.nan  # 49 cells: a real gap, left open
    z[0:3, 0:3] = np.nan  # touches the border: outside the survey, left open
    out, filled = _fill_small_holes(z, max_cells=25)
    assert filled[5:7, 5:7].all() and np.allclose(out[5:7, 5:7], 0.0)
    assert np.isnan(out[10:17, 10:17]).all() and not filled[10:17, 10:17].any()
    assert np.isnan(out[0:3, 0:3]).all()

    surf = HeightfieldSurface(
        z=out, score=np.ones_like(out), views=np.full_like(out, 6.0), rgb=np.zeros(out.shape + (3,), np.uint8),
        xmin=0.0, ymax=10.0, cell_m=0.5, inferred=filled,
    )
    tiers = surf.confidence()
    assert (tiers[filled] == int(Confidence.INFERRED)).all()
    assert (tiers[np.isfinite(out) & ~filled] == int(Confidence.MEASURED)).all()


def test_heightfield_accepts_a_partial_camera_set(tmp_path) -> None:
    """Cameras the solve dropped (None) contribute nothing; the rest still measure the surface."""
    from drishti3d.config import Config
    from drishti3d.pipeline.stages import (
        GeometryStage,
        PipelineState,
        _heightfield_ready,
    )
    from drishti3d.types import FrameMetrics, Keyframe

    images, intr, poses = _views()
    partial = list(poses)
    partial[0] = None
    partial[-1] = None
    cfg = Config()
    cfg.export.output_dir = str(tmp_path / "output")
    state = PipelineState(video_path=tmp_path / "v.mp4", telemetry_path=None, config=cfg, backbone_name="mapanything")
    state.keyframes = [
        Keyframe(frame_index=i, timestamp=float(i), metrics=FrameMetrics(i, float(i), 100.0, 1.0, 120.0, 0.0),
                 intrinsics=intr[i], pose=partial[i])
        for i in range(len(images))
    ]
    state.keyframe_cache = type(
        "Cache", (), {"get": lambda self, i: images[i], "scale_for": lambda self, i: 1.0, "__len__": lambda self: len(images)}
    )()
    xs, ys = np.meshgrid(np.arange(-60.0, 61.0, 3.0), np.arange(-40.0, 41.0, 3.0))
    x0, x1, y0, y1, h = _PLATFORM
    zs = np.where((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1), h, 0.0)
    state.ba_points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], -1)
    state.pose_prior_refined = True

    assert _heightfield_ready(state, state.keyframes, partial)
    assert not _heightfield_ready(state, state.keyframes, [None] * len(partial))
    artifacts, _ = GeometryStage()._run_heightfield(
        state, state.keyframes, partial, None, None, flight_profile=None, camera_gps_enu={}
    )
    assert artifacts["views_used"] == len(images) - 2
    assert state.placement_report.verdict == "PASS", state.placement_report.summary()
    assert "bundle-adjusted ground" in state.placement_report.summary()

    # The pose prior's convention: unrefined cameras keep a telemetry pose and are listed instead.
    state.unrefined_keyframes = [1, 2]
    artifacts, _ = GeometryStage()._run_heightfield(
        state, state.keyframes, poses, None, None, flight_profile=None, camera_gps_enu={}
    )
    assert artifacts["views_used"] == len(images) - 2
    state.unrefined_keyframes = list(range(len(images) - 2))
    assert not _heightfield_ready(state, state.keyframes, poses)


def test_fusion_does_not_photometrically_filter_a_heightfield_surface(tmp_path, reconstructed) -> None:
    """The surface is photo-consistent by construction; re-filtering punched holes in it on flight01."""
    from drishti3d.config import Config
    from drishti3d.pipeline.stages import FusionStage, PipelineState

    surf, _ = reconstructed
    cfg = Config()
    cfg.fusion.photometric_reject_before_mesh = True
    state = PipelineState(video_path=tmp_path / "v.mp4", telemetry_path=None, config=cfg, backbone_name="mapanything")
    calls = []
    stage = FusionStage()
    stage._photometric_point_filter = lambda st: calls.append(1) or (lambda pc: np.zeros(len(pc.xyz), bool))
    artifacts, _ = stage._from_heightmap(state, surf)
    assert not calls
    assert artifacts["photometric_points_rejected"] == 0
    assert len(state.point_cloud.xyz) == len(surf.mesh()[0].xyz)  # every cell, plus wall vertices


def test_outlier_prior_points_do_not_create_a_false_surface() -> None:
    """Low-parallax sparse points near the cameras must not seed a surface up there."""
    images, intr, poses = _views()
    xs, ys = np.meshgrid(np.arange(-60.0, 61.0, 3.0), np.arange(-40.0, 41.0, 3.0))
    ground = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], -1)
    rng = np.random.default_rng(1)
    junk = np.c_[rng.uniform(-30, 30, 400), rng.uniform(-20, 20, 400), rng.uniform(40.0, 70.0, 400)]
    surf, _diag = reconstruct_heightfield(images, intr, poses, prior_points=np.vstack([ground, junk]), agl_m=_ALT)
    z = surf.z[surf.valid]
    assert z.size > 0
    assert np.nanmax(z) < _ALT * 0.5, np.nanmax(z)
    assert (surf.rgb[surf.valid].max(axis=1) > 0).all()  # every kept cell was seen by some view


def test_every_measured_cell_carries_a_height_uncertainty(reconstructed) -> None:
    """Metres, per cell, from the sweep itself; tight on sharply textured ground."""
    surf, diag = reconstructed
    X, Y = surf.cell_centres()
    measured = surf.valid & ~surf.inferred
    assert surf.sigma_m is not None and np.isfinite(surf.sigma_m[measured]).all()
    ground = ((np.abs(X) > 28) | (np.abs(Y) > 22)) & (np.abs(X) < 55) & (np.abs(Y) < 35) & measured
    assert np.median(surf.sigma_m[ground]) < 0.5, np.median(surf.sigma_m[ground])
    pc = surf.point_cloud()
    assert pc.uncertainty_m is not None and len(pc.uncertainty_m) == len(pc.xyz)
    assert np.isnan(pc.uncertainty_m[surf.inferred[surf.valid]]).all()  # filled cells are not measurements
    assert diag["height_uncertainty_m"]["median"] < 0.5


def test_blurred_texture_pins_heights_less_tightly(monkeypatch) -> None:
    """The uncertainty must track how well the imagery constrains the height, not be a constant.

    Same sensor noise on both; blurring the ground texture removes the
    detail stereo locks onto, so the measured cells must report more.
    """
    import sys

    cfg = HeightfieldConfig(coarse_below_m=15.0, coarse_above_m=20.0, tile_m=60.0, fine_range_m=2.0)

    def run(texture):
        monkeypatch.setattr(sys.modules[__name__], "_TEXTURE", texture)
        images, intr, poses = _views()
        rng = np.random.default_rng(1)
        noisy = [np.clip(im + rng.normal(0.0, 8.0, im.shape), 0, 255).astype(np.uint8) for im in images]
        surf, _ = reconstruct_heightfield(noisy, intr, poses, agl_m=_ALT, config=cfg)
        return surf.sigma_m[surf.valid & ~surf.inferred]

    sharp = run(_TEXTURE)
    soft = run(cv2.GaussianBlur(_TEXTURE, (0, 0), 6.0))
    assert np.median(soft) > 1.5 * np.median(sharp), (np.median(sharp), np.median(soft))


def test_live_partials_stream_while_the_fine_level_sweeps() -> None:
    images, intr, poses = _views()
    cfg = HeightfieldConfig(coarse_below_m=15.0, coarse_above_m=20.0, tile_m=30.0, fine_range_m=2.0)
    got = []
    reconstruct_heightfield(images, intr, poses, agl_m=_ALT, config=cfg, partial=got.append, partial_every_s=0.0)
    assert len(got) >= 2 and len(got[-1].xyz) > len(got[0].xyz)
    last = got[-1]
    assert last.confidence is not None and last.uncertainty_m is not None
    assert len(last.uncertainty_m) == len(last.xyz) and last.rgb is None


def test_visible_scene_is_completed_and_the_fill_is_labelled_inferred(reconstructed) -> None:
    """Every cell a camera saw is in the model; what stereo did not measure is INFERRED, not a measurement."""
    surf, diag = reconstructed
    assert diag["visible_cells"] > 0
    assert diag["visible_measured_fraction"] + diag["visible_inferred_fraction"] > 0.99
    assert diag["visible_filled_cells"] > 0
    pc = surf.point_cloud()
    inferred = surf.inferred[surf.valid]
    assert (pc.confidence[inferred] == int(Confidence.INFERRED)).all()
    assert np.isnan(pc.uncertainty_m[inferred]).all()
    # The fill follows the ground: filled cells well away from the platform sit near z = 0.
    X, Y = surf.cell_centres()
    x0, x1, y0, y1, _h = _PLATFORM
    far = surf.inferred & ((X < x0 - 10) | (X > x1 + 10) | (Y < y0 - 10) | (Y > y1 + 10))
    assert far.any() and np.median(np.abs(surf.z[far])) < 1.0, np.median(np.abs(surf.z[far]))
    # Nothing is invented where no camera looked: the grid corners stay empty.
    assert not surf.valid[0, 0] and not surf.valid[-1, -1]


def test_fill_follows_the_ground_beside_a_roof_and_the_roof_inside_it() -> None:
    from drishti3d.geometry.heightfield import _fill_visible

    z = np.zeros((120, 120), dtype=np.float32)
    z[30:90, 30:90] = 8.0  # a 30 m roof at 0.5 m cells
    z[45:75, 45:75] = np.nan  # untextured middle of the roof (900 cells)
    z[30:90, 95:115] = np.nan  # bare field beside it, bordered by roof on one side only
    guide = np.zeros((120, 120))
    guide[30:90, 30:90] = 8.0  # the coarse level sees the roof, not its untextured middle
    out, filled, roof_like = _fill_visible(z, np.ones(z.shape, bool), 0.5, 2.5, guide=guide)
    assert filled.sum() == int(np.isnan(z).sum()) and roof_like == 1
    assert np.allclose(out[45:75, 45:75], 8.0, atol=0.5)  # the roof hole is roof
    assert np.abs(out[30:90, 97:115]).max() < 0.5  # the field is ground: no ramp from the roof


def test_small_hole_fill_never_divides_filter_residue() -> None:
    """Cells with no known neighbour must not be 'filled' from uniform_filter's numerical residue."""
    from drishti3d.geometry.heightfield import _fill_small_holes

    rng = np.random.default_rng(0)
    z = (-280.0 + rng.normal(0.0, 0.5, (300, 300))).astype(np.float32)
    z[rng.random(z.shape) < 0.35] = np.nan  # ragged gaps of every size
    out, filled = _fill_small_holes(z, 400)
    got = out[filled]
    assert got.size and np.isfinite(got).all()
    assert got.min() >= np.nanmin(z) - 1e-3 and got.max() <= np.nanmax(z) + 1e-3  # a mean of neighbours stays in range
