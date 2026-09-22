"""Tests for drishti3d.ingest.photometric: CLAHE and exposure gain fitting.

The gain solver is the load-bearing piece here: it is what removes visible
brightness seams from the texture atlas, and it is easy to write a version
that looks right on one pair of views and is scale-degenerate across many.
These tests pin both the pairwise ratio it recovers and the behaviour that
keeps it well-conditioned (the prior, and leaving unobserved views alone).
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.ingest.photometric import (
    apply_gains,
    normalize_illumination,
    solve_exposure_gains,
)

# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------


def test_normalize_illumination_preserves_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    out = normalize_illumination(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_normalize_illumination_raises_on_non_bgr_input() -> None:
    with pytest.raises(ValueError, match="expected a BGR"):
        normalize_illumination(np.zeros((10, 10), dtype=np.uint8))


def test_normalize_illumination_recovers_contrast_in_a_dark_region() -> None:
    """A crushed shadow must come back with more usable gradient."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # A low-contrast dark patch: values 10..20, the kind of shadow detail
    # global auto-exposure throws away.
    rng = np.random.default_rng(1)
    image[:, :] = rng.integers(10, 21, size=(64, 64, 1), dtype=np.uint8)

    out = normalize_illumination(image, clip_limit=3.0)

    assert out.std() > image.std()


def test_normalize_illumination_does_not_shift_a_grey_image_off_grey() -> None:
    """Equalising L only (not RGB separately) must keep neutral tones neutral."""
    rng = np.random.default_rng(2)
    grey = rng.integers(40, 200, size=(64, 64, 1), dtype=np.uint8)
    image = np.repeat(grey, 3, axis=2)

    out = normalize_illumination(image)

    channel_spread = out.astype(np.int16).max(axis=2) - out.astype(np.int16).min(axis=2)
    assert channel_spread.mean() < 3.0


def test_normalize_illumination_passes_empty_input_through() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert normalize_illumination(empty).size == 0


# ---------------------------------------------------------------------------
# Exposure gains
# ---------------------------------------------------------------------------


def test_gains_recover_a_known_brightness_ratio() -> None:
    """View 0 reads 2x view 1, so its gain must be half view 1's."""
    observations = [(0, 1, 200.0, 100.0)] * 40

    gains = solve_exposure_gains(observations, 2)

    assert gains[0] / gains[1] == pytest.approx(0.5, abs=0.02)


def test_gains_equalise_corrected_luminance() -> None:
    """After applying gains the two views very nearly agree.

    Not exactly: the toward-1.0 prior is a real term in the objective, so
    the fit settles slightly short of perfect pairwise agreement. That
    trade is the point -- a solver that equalised exactly would be the
    scale-degenerate one this prior exists to prevent (see
    ``test_prior_keeps_the_solution_from_drifting_in_scale``). The residual
    disagreement shrinks as observations outweigh the prior.
    """
    observations = [(0, 1, 180.0, 90.0)] * 30

    gains = solve_exposure_gains(observations, 2)

    corrected_a = gains[0] * 180.0
    corrected_b = gains[1] * 90.0
    assert corrected_a == pytest.approx(corrected_b, rel=0.02)

    # And the gap really does close with more evidence.
    many = solve_exposure_gains([(0, 1, 180.0, 90.0)] * 300, 2)
    tighter = abs(many[0] * 180.0 - many[1] * 90.0)
    assert tighter < abs(corrected_a - corrected_b)


def test_unobserved_views_keep_a_gain_of_exactly_one() -> None:
    """Nothing justifies correcting a view that shares no surface."""
    observations = [(0, 1, 200.0, 100.0)] * 10

    gains = solve_exposure_gains(observations, 3)

    assert gains[2] == 1.0


def test_no_observations_is_an_identity_correction() -> None:
    assert solve_exposure_gains([], 4).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert solve_exposure_gains([], 0).size == 0


def test_already_consistent_views_are_left_alone() -> None:
    observations = [(0, 1, 120.0, 120.0), (1, 2, 120.0, 120.0)]

    gains = solve_exposure_gains(observations, 3)

    assert np.allclose(gains, 1.0, atol=1e-6)


def test_prior_keeps_the_solution_from_drifting_in_scale() -> None:
    """The pairwise constraints alone are scale-degenerate; the prior fixes that.

    A chain of equal-luminance views has infinitely many solutions that
    satisfy every pairwise constraint (any constant multiple). The prior is
    what selects the one near 1.0 instead of letting the whole model drift
    darker or brighter run to run.
    """
    observations = [(i, i + 1, 100.0, 100.0) for i in range(5)]

    gains = solve_exposure_gains(observations, 6)

    assert np.allclose(gains, 1.0, atol=1e-6)
    assert gains.mean() == pytest.approx(1.0, abs=1e-6)


def test_extreme_gains_are_clamped_not_applied() -> None:
    """A 10x disagreement is a broken frame, not a mis-exposed one."""
    observations = [(0, 1, 250.0, 5.0)] * 20

    gains = solve_exposure_gains(observations, 2)

    assert gains.min() >= 0.4
    assert gains.max() <= 2.5


def test_near_black_samples_are_ignored_as_noise() -> None:
    """Log-ratio of a near-zero luminance is noise, not exposure information."""
    observations = [(0, 1, 0.0005, 100.0)] * 10

    gains = solve_exposure_gains(observations, 2)

    assert np.allclose(gains, 1.0)


def test_apply_gains_scales_per_source_view_and_saturates() -> None:
    samples = np.array([[100.0, 100, 100], [200.0, 200, 200]])
    gains = np.array([2.0, 1.0])
    view_indices = np.array([0, 0])

    out = apply_gains(samples, gains, view_indices)

    assert out[0].tolist() == [200.0, 200.0, 200.0]
    # Clipped, not rescaled: protecting one blown highlight by darkening
    # the whole frame would be worse than saturating it.
    assert out[1].tolist() == [255.0, 255.0, 255.0]


def test_apply_gains_uses_the_right_gain_per_sample() -> None:
    samples = np.array([[100.0, 100, 100], [100.0, 100, 100]])
    gains = np.array([0.5, 2.0])
    view_indices = np.array([0, 1])

    out = apply_gains(samples, gains, view_indices)

    assert out[0, 0] == pytest.approx(50.0)
    assert out[1, 0] == pytest.approx(200.0)
