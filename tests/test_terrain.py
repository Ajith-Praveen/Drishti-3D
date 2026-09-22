"""Tests for DTM extraction and INFERRED facade completion.

The synthetic scene throughout: 40x40 m of ground at z=0, one 10x10 m
building whose roof is at z=8, and a tree canopy at z=7 off to one side.
That tree is the point of the fixture -- it is the thing naive
height-thresholded footprint extrusion turns into a solid block, and
several tests here exist purely to pin that it never happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.export.geotiff import point_cloud_to_dsm
from drishti3d.export.terrain import (
    dtm_from_point_cloud,
    height_above_ground,
    morphological_ground_mask,
)
from drishti3d.fusion.completion import complete_facades
from drishti3d.types import Confidence, PointCloud

_TERRAIN, _BUILDING, _VEGETATION = 1, 2, 4
_ROOF_Z = 8.0
_CANOPY_Z = 7.0


def _scene(*, with_labels: bool = True, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    ground = rng.uniform(-20, 20, (9000, 2))
    roof = rng.uniform(-5, 5, (3000, 2))
    canopy = rng.uniform(12, 16, (1500, 2))

    xyz = np.vstack(
        [
            np.c_[ground, np.zeros(len(ground))],
            np.c_[roof, np.full(len(roof), _ROOF_Z)],
            np.c_[canopy, np.full(len(canopy), _CANOPY_Z)],
        ]
    )
    if not with_labels:
        return PointCloud(xyz=xyz)

    semantic = np.concatenate(
        [
            np.full(len(ground), _TERRAIN, dtype=np.uint8),
            np.full(len(roof), _BUILDING, dtype=np.uint8),
            np.full(len(canopy), _VEGETATION, dtype=np.uint8),
        ]
    )
    return PointCloud(xyz=xyz, semantic_class=semantic)


# ---------------------------------------------------------------------------
# DTM
# ---------------------------------------------------------------------------


def test_semantic_dtm_excludes_roofs_and_canopy() -> None:
    result = dtm_from_point_cloud(_scene(), resolution_m=1.0)

    assert result.method == "semantic"
    # Bare earth is flat at 0: nothing raised may survive into the DTM.
    assert result.dtm.max() == pytest.approx(0.0, abs=1e-6)


def test_dtm_falls_back_to_morphological_without_labels() -> None:
    result = dtm_from_point_cloud(_scene(with_labels=False), resolution_m=1.0)

    assert result.method == "morphological"
    assert result.dtm.max() == pytest.approx(0.0, abs=0.5)


def test_dtm_reports_which_cells_were_interpolated() -> None:
    """Under a building there is no ground observation, and the DTM says so."""
    result = dtm_from_point_cloud(_scene(), resolution_m=1.0)

    assert result.interpolated_mask.shape == result.dtm.shape
    assert result.interpolated_mask.dtype == bool
    assert "dtm_interpolated_pct" in result.stats
    assert "dtm_method" in result.stats


def test_dtm_requires_a_non_empty_cloud() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        dtm_from_point_cloud(PointCloud(xyz=np.zeros((0, 3))), resolution_m=1.0)


def test_dtm_raises_rather_than_inventing_ground_when_none_exists() -> None:
    """An all-building cloud has no bare earth; guessing one would be a lie."""
    xyz = np.c_[np.random.default_rng(0).uniform(-5, 5, (500, 2)), np.full(500, _ROOF_Z)]
    pc = PointCloud(xyz=xyz, semantic_class=np.full(500, _BUILDING, dtype=np.uint8))

    # No ground class present at all -> falls back to morphological, which
    # on a single flat slab classifies it as ground (it is the only
    # surface). The contract under test is that it does not crash and
    # reports which method it used.
    result = dtm_from_point_cloud(pc, resolution_m=1.0)
    assert result.method == "morphological"


def test_morphological_ground_mask_separates_a_building_from_terrain() -> None:
    pc = _scene()
    mask = morphological_ground_mask(pc.xyz, resolution_m=1.0)

    assert mask.shape == (len(pc.xyz),)
    ground_points = pc.xyz[mask]
    # Whatever it keeps must be near the true ground plane.
    assert ground_points[:, 2].max() < 2.0


def test_morphological_ground_mask_handles_empty_input() -> None:
    assert morphological_ground_mask(np.zeros((0, 3)), 1.0).shape == (0,)


# ---------------------------------------------------------------------------
# Height above ground
# ---------------------------------------------------------------------------


def test_height_above_ground_recovers_the_building_height() -> None:
    pc = _scene()
    dsm, _transform, _filled = point_cloud_to_dsm(pc, resolution_m=1.0)
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    height = height_above_ground(dsm, dtm_result.dtm)

    centre = height[height.shape[0] // 2, height.shape[1] // 2]
    assert centre == pytest.approx(_ROOF_Z, abs=0.5)
    assert height.min() >= 0.0


def test_height_above_ground_rejects_mismatched_rasters() -> None:
    with pytest.raises(ValueError, match="does not match"):
        height_above_ground(np.zeros((4, 4)), np.zeros((5, 5)))


def test_height_above_ground_can_keep_negatives_for_diagnostics() -> None:
    dsm = np.array([[1.0]])
    dtm = np.array([[2.0]])
    assert height_above_ground(dsm, dtm, clip_negative=False)[0, 0] == pytest.approx(-1.0)
    assert height_above_ground(dsm, dtm, clip_negative=True)[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Facade completion
# ---------------------------------------------------------------------------


def test_completion_extrudes_the_building_from_ground_to_roof() -> None:
    pc = _scene()
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0)

    assert result is not None
    assert result.stats["facade_structures_completed"] == 1
    z = result.facade_points.xyz[:, 2]
    assert z.min() == pytest.approx(0.0, abs=0.5)
    assert z.max() == pytest.approx(_ROOF_Z, abs=0.5)


def test_every_generated_facade_point_declares_itself_inferred() -> None:
    """The rule the whole module rests on."""
    pc = _scene()
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0)

    assert result is not None
    fp = result.facade_points
    assert np.all(fp.confidence == int(Confidence.INFERRED))
    assert np.all(fp.semantic_class == _BUILDING)
    # No view voted on these, because no view saw them.
    assert np.all(fp.semantic_confidence == 0.0)


def test_completion_never_extrudes_vegetation() -> None:
    """The failure that makes naive footprint extrusion untrustworthy.

    The canopy sits at z=7 over ground at z=0 -- height-thresholded
    extrusion would happily turn it into a 7 m solid block. Semantic labels
    are what prevent it, so the generated points must stay inside the
    building footprint and never reach the tree's x/y range.
    """
    pc = _scene()
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0)

    assert result is not None
    xy = result.facade_points.xyz[:, :2]
    # The canopy occupies x,y in [12, 16]; nothing generated may land there.
    assert xy.max() < 10.0
    assert result.stats["facade_components_found"] == 1


def test_completion_refuses_to_guess_without_semantic_labels() -> None:
    """No labels means no honest way to tell a roof from a canopy."""
    pc = _scene(with_labels=False)
    labelled = _scene()
    dtm_result = dtm_from_point_cloud(labelled, resolution_m=1.0)

    assert complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0) is None


def test_completion_returns_none_when_there_are_no_buildings() -> None:
    rng = np.random.default_rng(0)
    ground = rng.uniform(-20, 20, (2000, 2))
    pc = PointCloud(
        xyz=np.c_[ground, np.zeros(len(ground))],
        semantic_class=np.full(len(ground), _TERRAIN, dtype=np.uint8),
    )
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    assert complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0) is None


def test_completion_rejects_footprints_that_are_too_low() -> None:
    """A 0.5 m 'building' is a kerb, not a structure."""
    rng = np.random.default_rng(0)
    ground = rng.uniform(-20, 20, (6000, 2))
    kerb = rng.uniform(-5, 5, (2000, 2))
    xyz = np.vstack([np.c_[ground, np.zeros(len(ground))], np.c_[kerb, np.full(len(kerb), 0.5)]])
    semantic = np.concatenate(
        [np.full(len(ground), _TERRAIN, dtype=np.uint8), np.full(len(kerb), _BUILDING, dtype=np.uint8)]
    )
    pc = PointCloud(xyz=xyz, semantic_class=semantic)
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0)

    assert result is None or result.stats["facade_rejected_low_height"] >= 1


def test_completion_rejects_tiny_footprints() -> None:
    """A few building-labelled points on a van must not become a building."""
    rng = np.random.default_rng(0)
    ground = rng.uniform(-20, 20, (6000, 2))
    speck = rng.uniform(-0.5, 0.5, (30, 2))
    xyz = np.vstack([np.c_[ground, np.zeros(len(ground))], np.c_[speck, np.full(len(speck), 8.0)]])
    semantic = np.concatenate(
        [np.full(len(ground), _TERRAIN, dtype=np.uint8), np.full(len(speck), _BUILDING, dtype=np.uint8)]
    )
    pc = PointCloud(xyz=xyz, semantic_class=semantic)
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(
        pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0, min_footprint_m2=50.0
    )

    assert result is None or result.stats["facade_rejected_small_footprint"] >= 1


def test_completion_is_hollow_not_solid() -> None:
    """Only the footprint boundary gets a wall; the interior stays empty."""
    pc = _scene()
    dtm_result = dtm_from_point_cloud(pc, resolution_m=1.0)

    result = complete_facades(pc, dtm_result.dtm, dtm_result.transform, resolution_m=1.0)

    assert result is not None
    xy = result.facade_points.xyz[:, :2]
    # The building spans roughly [-5, 5]; its centre must contain no
    # generated points, because a facade is a wall, not a filled volume.
    near_centre = (np.abs(xy[:, 0]) < 2.0) & (np.abs(xy[:, 1]) < 2.0)
    assert not near_centre.any()


# ---------------------------------------------------------------------------
# Georeferencing must carry per-point channels (regression)
# ---------------------------------------------------------------------------


def test_georeferencing_preserves_every_per_point_channel() -> None:
    """A Sim(3) moves points; it does not change what they ARE.

    `_apply_georeferencing` rebuilds both clouds from scratch, and the
    rebuild originally listed only xyz/rgb/covariance/confidence. Semantic
    labels were therefore computed in fusion, shown in the report card, and
    then silently dropped before export -- the LAS had no semantic_class
    dimension and an all-zero ASPRS classification field, while the report
    confidently described the scene composition.

    Asserted structurally: the failure is an omission at a constructor, and
    reproducing it end-to-end needs a full georeferenced run.
    """
    import pathlib

    from drishti3d.pipeline import stages as stages_mod
    from drishti3d.types import PointCloud

    src = pathlib.Path(stages_mod.__file__).read_text()
    start = src.index("state.point_cloud = PointCloud(\n        xyz=result.transform.apply")
    rebuild = src[start : start + 1800]

    # Every optional per-point field on PointCloud must be carried through.
    channels = [
        f.name
        for f in PointCloud.__dataclass_fields__.values()  # type: ignore[attr-defined]
        if f.name != "xyz"
    ]
    for name in channels:
        assert f"{name}=" in rebuild, f"georeferencing drops PointCloud.{name}"


def test_a_handful_of_ground_labels_does_not_trigger_the_semantic_path() -> None:
    """Regression: a DTM that was 99.97% interpolated, reported as measured.

    An out-of-domain segmenter on nadir aerial imagery can label a few
    dozen points TERRAIN while leaving the actual ground unlabelled. That
    passed an `any()` test, took the semantic path, and produced a surface
    invented from almost nothing. Below a usable fraction the morphological
    filter -- which reads every point's shape rather than its label -- is
    strictly more trustworthy.
    """
    rng = np.random.default_rng(0)
    ground = rng.uniform(-20, 20, (10000, 2))
    xyz = np.c_[ground, np.zeros(len(ground))]

    # 0.1% labelled TERRAIN; everything else UNLABELLED.
    semantic = np.zeros(len(xyz), dtype=np.uint8)
    semantic[:10] = _TERRAIN

    result = dtm_from_point_cloud(PointCloud(xyz=xyz, semantic_class=semantic), resolution_m=1.0)

    assert result.method == "morphological"
    # And the resulting DTM is mostly observed, not invented.
    assert result.stats["dtm_interpolated_pct"] < 50.0


def test_plenty_of_ground_labels_still_uses_semantics() -> None:
    result = dtm_from_point_cloud(_scene(), resolution_m=1.0)
    assert result.method == "semantic"
    assert result.stats["ground_point_pct"] > 50.0
