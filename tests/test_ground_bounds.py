"""Tests for rejecting geometry that cannot physically be terrain.

The artifact these are written against is a real one: a 174 m tall,
142 m long column of dense points standing in the middle of a model of
flat farmland, which the existing sliver cleanup removed exactly 0 faces
of across two runs because its triangles are ordinary -- it is good
geometry in an impossible place.

The tests therefore check two opposite things. A column like that must
go. A tall building, a valley, and an otherwise-healthy reconstruction
must not.
"""

from __future__ import annotations

import numpy as np

from drishti3d.fusion.ground_bounds import GroundBand, filter_to_ground_band

GROUND = -119.8


def _terrain(n=20000, relief=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-300, 300, (n, 2)), GROUND + rng.normal(0, relief, n)]


def test_the_column_is_removed_and_the_terrain_is_not():
    """The measured failure, reproduced: a 174 m column over flat ground."""
    ground = _terrain()
    rng = np.random.default_rng(1)
    column = np.c_[
        rng.uniform(-70, 70, (900, 1)),
        rng.uniform(-24, 24, (900, 1)),
        GROUND + rng.uniform(0, 174, 900),
    ]
    xyz = np.vstack([ground, column])

    keep, diag = filter_to_ground_band(xyz, GROUND)

    assert diag["applied"]
    # Every terrain point survives.
    assert keep[: len(ground)].all()
    # The top of the column is gone; its base is legitimately inside the
    # band and is not what made the model unusable.
    assert not keep[len(ground) :][column[:, 2] > GROUND + 100].any()
    assert diag["worst_above_m"] > 150


def test_a_tall_building_survives():
    """Deleting real structure would be worse than the artifact."""
    ground = _terrain()
    rng = np.random.default_rng(2)
    tower = np.c_[
        rng.uniform(-15, 15, (3000, 1)),
        rng.uniform(-15, 15, (3000, 1)),
        GROUND + rng.uniform(0, 60, 3000),
    ]
    keep, diag = filter_to_ground_band(np.vstack([ground, tower]), GROUND)
    assert diag["applied"]
    assert keep.all(), "a 60 m building is real structure"


def test_a_deep_valley_survives():
    ground = _terrain()
    rng = np.random.default_rng(3)
    valley = np.c_[rng.uniform(-50, 50, (2000, 2)), GROUND - rng.uniform(0, 50, 2000)]
    keep, _ = filter_to_ground_band(np.vstack([ground, valley]), GROUND)
    assert keep.all()


def test_a_healthy_reconstruction_loses_nothing():
    keep, diag = filter_to_ground_band(_terrain(), GROUND)
    assert keep.all()
    assert diag["points_rejected"] == 0


def test_it_declines_rather_than_delete_most_of_the_model():
    """A broken run must stay visibly broken.

    If the depth scale is wrong, most points land outside any plausible
    band. Trimming there would return a small, clean-looking cloud and
    hide the real failure, so the filter refuses and says why.
    """
    rng = np.random.default_rng(4)
    scattered = np.c_[rng.uniform(-300, 300, (20000, 2)), GROUND + rng.uniform(-400, 400, 20000)]
    keep, diag = filter_to_ground_band(scattered, GROUND)

    assert diag["applied"] is False
    assert "reject" in diag["reason"]
    assert keep.all(), "declining means changing nothing"


def test_no_ground_reference_means_no_filtering():
    xyz = _terrain()
    keep, diag = filter_to_ground_band(xyz, None)
    assert diag["applied"] is False
    assert keep.all()


def test_empty_cloud_is_survivable():
    keep, diag = filter_to_ground_band(np.zeros((0, 3)), GROUND)
    assert keep.shape == (0,)
    assert diag["applied"] is False


def test_band_reports_its_own_bounds():
    band = GroundBand(GROUND, above_m=80.0, below_m=60.0)
    assert band.low == GROUND - 60.0
    assert band.high == GROUND + 80.0
    d = band.as_dict()
    assert d["ground_z_m"] == round(GROUND, 3)


def test_the_sliver_cleanup_genuinely_cannot_catch_this():
    """Why a separate filter exists at all.

    The column's faces have ordinary edge lengths, so an edge-length
    test -- which is what fusion.mesh's spike cleanup applies -- cannot
    distinguish them from the terrain's.
    """
    rng = np.random.default_rng(5)
    # A patch of terrain and a patch of column, sampled at the same
    # density. The column is ~4% of the cloud, as the real one was --
    # a fixture where it dominates would (correctly) trip the
    # decline-rather-than-delete guard instead of testing this.
    terrain = np.c_[rng.uniform(0, 10, (12000, 2)), GROUND + rng.normal(0, 0.05, 12000)]
    column = np.c_[rng.uniform(0, 10, (500, 1)), rng.uniform(0, 0.5, (500, 1)), GROUND + 150 + rng.uniform(0, 10, 500)]

    def nn_spacing(p):
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))

    # Comparable local spacing -> comparable triangle edges -> invisible
    # to a median-edge test, while the ground band separates them outright.
    assert 0.3 < nn_spacing(column[:500]) / nn_spacing(terrain[:500]) < 3.0
    keep, _ = filter_to_ground_band(np.vstack([terrain, column]), GROUND)
    assert keep[: len(terrain)].all()
    assert not keep[len(terrain) :].any()
