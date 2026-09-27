"""Tests for drishti3d.semantics: taxonomy, projection, and multi-view voting.

Must pass with no torch and no transformers -- every test here either uses
``NullSegmenter`` or feeds ``label_points`` hand-built masks directly, so
the vote logic is exercised without any model at all. That is deliberate:
the voting rules (and their demotion-to-UNLABELLED behaviour) are what this
project's semantic claims rest on, and they must be verifiable on a laptop
with no GPU.

Camera convention under test throughout: ``Pose.R`` is world-from-camera
with ``t`` the camera centre in world coordinates, and the camera frame is
OpenCV (+X right, +Y down in image, +Z forward out of the lens). A nadir
camera therefore has ``R = diag(1, -1, -1)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.semantics.classes import (
    ADE20K_TO_CANONICAL,
    ASPRS_FROM_CANONICAL,
    DYNAMIC_CLASSES,
    EXCLUDED_CLASSES,
    SemanticClass,
    class_histogram,
    map_label_array,
)
from drishti3d.semantics.labelling import label_points, project_points, visible_mask
from drishti3d.semantics.segmenter import NullSegmenter, create_segmenter
from drishti3d.types import CameraIntrinsics, Pose

_NADIR_R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])


def _intr(width: int = 640, height: int = 480) -> CameraIntrinsics:
    return CameraIntrinsics.from_hfov(60.0, width, height)


def _nadir_pose(x: float = 0.0, y: float = 0.0, alt: float = 20.0) -> Pose:
    return Pose(R=_NADIR_R.copy(), t=np.array([x, y, alt]))


def _view(label: int, *, x: float = 0.0, conf: float = 0.9, intr: CameraIntrinsics | None = None):
    intr = intr or _intr()
    labels = np.full((intr.height, intr.width), label, dtype=np.uint8)
    confidence = np.full((intr.height, intr.width), conf, dtype=np.float32)
    return _nadir_pose(x=x), intr, labels, confidence


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_semantic_class_values_are_a_stable_file_format_contract() -> None:
    """These integers are written into exported LAS/PLY and must never move."""
    assert SemanticClass.UNLABELLED == 0
    assert SemanticClass.TERRAIN == 1
    assert SemanticClass.BUILDING == 2
    assert SemanticClass.ROAD == 3
    assert SemanticClass.VEGETATION == 4
    assert SemanticClass.INFRASTRUCTURE == 5
    assert SemanticClass.WATER == 6
    assert SemanticClass.VEHICLE == 7
    assert SemanticClass.PERSON == 8
    assert SemanticClass.SKY == 9
    assert SemanticClass.OBSTACLE == 10


def test_excluded_classes_are_dynamic_plus_sky() -> None:
    assert DYNAMIC_CLASSES == {SemanticClass.VEHICLE, SemanticClass.PERSON}
    assert EXCLUDED_CLASSES == DYNAMIC_CLASSES | {SemanticClass.SKY}


def test_ade20k_lookup_maps_the_classes_the_deliverables_require() -> None:
    # Spot-check one ADE20K index per required reconstruction target.
    assert ADE20K_TO_CANONICAL[1] == SemanticClass.BUILDING  # building
    assert ADE20K_TO_CANONICAL[6] == SemanticClass.ROAD  # road
    assert ADE20K_TO_CANONICAL[4] == SemanticClass.VEGETATION  # tree
    assert ADE20K_TO_CANONICAL[13] == SemanticClass.TERRAIN  # earth
    assert ADE20K_TO_CANONICAL[20] == SemanticClass.VEHICLE  # car
    assert ADE20K_TO_CANONICAL[12] == SemanticClass.PERSON  # person
    assert ADE20K_TO_CANONICAL[2] == SemanticClass.SKY  # sky
    assert ADE20K_TO_CANONICAL[21] == SemanticClass.WATER  # water


def test_ade20k_indoor_classes_are_unlabelled_not_guessed() -> None:
    """An indoor class in an aerial mask is model confusion, not a surface."""
    for indoor in (7, 15, 19, 23, 65):  # bed, table, chair, sofa, toilet
        assert ADE20K_TO_CANONICAL[indoor] == SemanticClass.UNLABELLED


def test_map_label_array_is_out_of_range_safe() -> None:
    labels = np.array([[1, 6], [999, -1]], dtype=np.int64)
    out = map_label_array(labels, ADE20K_TO_CANONICAL)
    assert out[0, 0] == SemanticClass.BUILDING
    assert out[0, 1] == SemanticClass.ROAD
    # Out-of-range indices must degrade to UNLABELLED rather than raising.
    assert out[1, 0] == SemanticClass.UNLABELLED
    assert out[1, 1] == SemanticClass.UNLABELLED


def test_asprs_mapping_never_invents_a_plausible_near_miss() -> None:
    """Classes with no honest ASPRS equivalent become 1/unclassified."""
    assert ASPRS_FROM_CANONICAL[SemanticClass.TERRAIN] == 2  # ground
    assert ASPRS_FROM_CANONICAL[SemanticClass.BUILDING] == 6  # building
    assert ASPRS_FROM_CANONICAL[SemanticClass.ROAD] == 11  # road surface
    assert ASPRS_FROM_CANONICAL[SemanticClass.VEGETATION] == 5  # high vegetation
    assert ASPRS_FROM_CANONICAL[SemanticClass.WATER] == 9  # water
    # A vehicle must NOT be written as a building just because 6 exists.
    assert ASPRS_FROM_CANONICAL[SemanticClass.VEHICLE] == 1
    assert ASPRS_FROM_CANONICAL[SemanticClass.PERSON] == 1
    assert ASPRS_FROM_CANONICAL[SemanticClass.INFRASTRUCTURE] == 1


def test_class_histogram_distinguishes_never_ran_from_found_nothing() -> None:
    assert class_histogram(None) == {}
    assert class_histogram(np.array([], dtype=np.uint8)) == {}

    hist = class_histogram(np.array([1, 1, 1, 2], dtype=np.uint8))
    assert hist == {"terrain": 75.0, "building": 25.0}
    # Absent classes are omitted, not reported as 0.0 -- a 0.0 row reads as
    # "we looked and found none", which is a different claim.
    assert "vehicle" not in hist


# ---------------------------------------------------------------------------
# NullSegmenter
# ---------------------------------------------------------------------------


def test_null_segmenter_is_always_available_and_invents_nothing() -> None:
    seg = NullSegmenter()
    assert seg.is_available()
    seg.load("cpu")
    result = seg.predict([np.zeros((30, 40, 3), dtype=np.uint8)])[0]
    assert result.labels.shape == (30, 40)
    assert result.labels.dtype == np.uint8
    assert np.all(result.labels == SemanticClass.UNLABELLED)
    assert np.all(result.confidence == 0.0)
    seg.unload()


def test_create_segmenter_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown segmenter"):
        create_segmenter("definitely-not-a-segmenter")


def test_dynamic_mask_selects_only_excluded_classes() -> None:
    seg = NullSegmenter()
    result = seg.predict([np.zeros((4, 4, 3), dtype=np.uint8)])[0]
    result.labels[0, 0] = SemanticClass.VEHICLE
    result.labels[1, 1] = SemanticClass.PERSON
    result.labels[2, 2] = SemanticClass.SKY
    result.labels[3, 3] = SemanticClass.BUILDING

    mask = result.dynamic_mask(EXCLUDED_CLASSES)
    assert mask[0, 0] and mask[1, 1] and mask[2, 2]
    assert not mask[3, 3]  # a building is structure, not content


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_project_points_matches_the_documented_camera_convention() -> None:
    intr = _intr()
    pose = _nadir_pose(alt=20.0)

    uv, depth, in_view = project_points(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [0.0, 5, 0]]), pose, intr
    )

    # Directly below the camera -> principal point, at the flight altitude.
    assert uv[0] == pytest.approx([intr.cx, intr.cy])
    assert depth[0] == pytest.approx(20.0)
    # +X (East) moves right in the image; +Y (North) moves *up* (v decreases).
    assert uv[1, 0] > intr.cx and uv[1, 1] == pytest.approx(intr.cy)
    assert uv[2, 1] < intr.cy and uv[2, 0] == pytest.approx(intr.cx)
    assert in_view.tolist() == [True, True, True]


def test_project_points_rejects_points_behind_the_camera() -> None:
    intr = _intr()
    pose = _nadir_pose(alt=20.0)
    # Above a down-looking camera == behind it.
    _, depth, in_view = project_points(np.array([[0.0, 0.0, 40.0]]), pose, intr)
    assert depth[0] < 0
    assert not in_view[0]


def test_visible_mask_occludes_points_behind_a_nearer_surface() -> None:
    intr = _intr()
    pose = _nadir_pose(alt=20.0)
    # Two points on the same ray: a rooftop at z=10 and ground at z=0.
    pts = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    uv, depth, in_view = project_points(pts, pose, intr)

    vis = visible_mask(uv, depth, in_view, width=intr.width, height=intr.height, tolerance_m=0.5)
    assert vis[0]  # the roof is seen
    assert not vis[1]  # the ground beneath it is not


def test_visible_mask_tolerance_prevents_a_surface_occluding_itself() -> None:
    intr = _intr()
    pose = _nadir_pose(alt=20.0)
    # Same surface, 10 cm of depth noise -- must all stay visible.
    pts = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, -0.1]])
    uv, depth, in_view = project_points(pts, pose, intr)
    vis = visible_mask(uv, depth, in_view, width=intr.width, height=intr.height, tolerance_m=0.5)
    assert vis.all()


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def test_unanimous_views_produce_a_confident_label() -> None:
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    views = [_view(SemanticClass.BUILDING, x=dx) for dx in (-1.0, 0.0, 1.0)]

    result = label_points(pts, views, min_views=2, min_vote_ratio=0.5)

    assert np.all(result.semantic_class == SemanticClass.BUILDING)
    assert np.allclose(result.vote_ratio, 1.0)
    assert np.all(result.view_count == 3)
    assert result.stats["labelled_points"] == 2


def test_a_disputed_point_is_demoted_rather_than_guessed() -> None:
    """A 50/50 split under a 0.6 threshold must not become a coin flip."""
    pts = np.array([[0.0, 0, 0]])
    views = [_view(SemanticClass.BUILDING), _view(SemanticClass.ROAD)]

    result = label_points(pts, views, min_views=2, min_vote_ratio=0.6)

    assert result.semantic_class[0] == SemanticClass.UNLABELLED
    assert result.stats["disputed_points"] == 1
    assert result.stats["unseen_points"] == 0


def test_a_majority_survives_the_same_threshold() -> None:
    pts = np.array([[0.0, 0, 0]])
    views = [
        _view(SemanticClass.BUILDING, x=-1.0),
        _view(SemanticClass.BUILDING, x=1.0),
        _view(SemanticClass.ROAD),
    ]

    result = label_points(pts, views, min_views=2, min_vote_ratio=0.6)

    assert result.semantic_class[0] == SemanticClass.BUILDING
    assert result.vote_ratio[0] > 0.6


def test_unseen_and_disputed_are_reported_separately() -> None:
    """They are different defects -- flight coverage vs segmentation quality."""
    pts = np.array([[0.0, 0, 0], [5000.0, 5000.0, 0.0]])
    views = [_view(SemanticClass.BUILDING), _view(SemanticClass.ROAD)]

    result = label_points(pts, views, min_views=2, min_vote_ratio=0.6)

    assert result.stats["unseen_points"] == 1
    assert result.stats["disputed_points"] == 1
    assert result.view_count[1] == 0
    assert result.semantic_class[1] == SemanticClass.UNLABELLED


def test_min_views_rejects_single_view_points() -> None:
    pts = np.array([[0.0, 0, 0]])
    views = [_view(SemanticClass.BUILDING)]

    lenient = label_points(pts, views, min_views=1, min_vote_ratio=0.5)
    strict = label_points(pts, views, min_views=2, min_vote_ratio=0.5)

    assert lenient.semantic_class[0] == SemanticClass.BUILDING
    assert strict.semantic_class[0] == SemanticClass.UNLABELLED
    assert strict.stats["unseen_points"] == 1


def test_segmentation_confidence_breaks_a_two_way_tie() -> None:
    """Equal view counts, unequal confidence -> the confident view wins."""
    pts = np.array([[0.0, 0, 0]])
    views = [
        _view(SemanticClass.BUILDING, conf=0.95),
        _view(SemanticClass.ROAD, conf=0.20),
    ]

    result = label_points(pts, views, min_views=1, min_vote_ratio=0.5)

    assert result.semantic_class[0] == SemanticClass.BUILDING


def test_occlusion_stops_ground_points_stealing_a_roof_label() -> None:
    """The failure this module exists to prevent, end to end.

    The roof has to be a *surface*, not a single point: the z-buffer works
    by taking the nearest depth per image block, so a lone point occludes
    nothing (and correctly so -- one sample is not a roof). A dense slab
    spanning the ground point's image neighbourhood is the real geometry.
    """
    # Sample spacing must put several roof points inside every image block
    # the ground point could land in: at 10 m depth, 0.025 m of world
    # spacing is ~1.4 px, comfortably under the 4 px block size.
    step = np.linspace(-2.0, 2.0, 160)
    gx, gy = np.meshgrid(step, step)
    roof = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 10.0)], axis=1)
    ground_behind = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([roof, ground_behind])
    ground_idx = len(pts) - 1
    views = [_view(SemanticClass.BUILDING, x=dx) for dx in (-0.2, 0.0, 0.2)]

    with_occlusion = label_points(pts, views, min_views=1, use_occlusion=True)
    without = label_points(pts, views, min_views=1, use_occlusion=False)

    # Without the visibility test the hidden ground point happily collects
    # the building's label; with it, the point is never sampled at all.
    assert without.semantic_class[ground_idx] == SemanticClass.BUILDING
    assert with_occlusion.view_count[ground_idx] == 0
    assert with_occlusion.semantic_class[ground_idx] == SemanticClass.UNLABELLED
    # The roof itself is unaffected either way.
    assert with_occlusion.semantic_class[0] == SemanticClass.BUILDING


def test_mismatched_mask_and_intrinsics_is_skipped_not_misindexed() -> None:
    pts = np.array([[0.0, 0, 0]])
    pose, intr, labels, conf = _view(SemanticClass.BUILDING)
    bad = (pose, intr, labels[:10, :10], conf[:10, :10])

    result = label_points(pts, [bad], min_views=1)

    assert result.stats["views_voted"] == 0
    assert result.semantic_class[0] == SemanticClass.UNLABELLED


def test_no_views_leaves_everything_unlabelled() -> None:
    result = label_points(np.zeros((5, 3)), [], min_views=1)
    assert np.all(result.semantic_class == SemanticClass.UNLABELLED)
    assert result.stats["labelled_points"] == 0


# ---------------------------------------------------------------------------
# Checkpoint auth (regression: 401 on a PUBLIC model when HF_TOKEN is set)
# ---------------------------------------------------------------------------


def test_public_checkpoint_is_fetched_anonymously_by_default(monkeypatch) -> None:
    """A scoped HF token in the environment must not break a public model.

    A fine-grained token scoped to another namespace makes the Hub answer
    401 for a public repo instead of serving it anonymously, and
    transformers reports that as "not a valid model identifier" -- which
    reads like a typo, not an auth failure. So the loader passes
    `token=False` unless a checkpoint token was explicitly configured.
    """
    from drishti3d.semantics.segmenter import SegFormerSegmenter

    captured: dict = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, source, token=None, **kw):
            captured["processor_token"] = token
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source, torch_dtype=None, token=None, **kw):
            captured["model_token"] = token
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_tf = type(
        "M",
        (),
        {"SegformerForSemanticSegmentation": FakeModel, "SegformerImageProcessor": FakeProcessor},
    )
    fake_torch = type("T", (), {"float16": "f16", "float32": "f32"})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tf)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setenv("HF_TOKEN", "hf_a_scoped_token_for_some_other_namespace")

    SegFormerSegmenter().load("cpu")

    assert captured["processor_token"] is False
    assert captured["model_token"] is False


def test_explicit_token_is_used_for_a_private_checkpoint(monkeypatch) -> None:
    """A gated/private checkpoint still needs a real token, when configured."""
    from drishti3d.semantics.segmenter import SegFormerSegmenter

    captured: dict = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, source, token=None, **kw):
            captured["processor_token"] = token
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source, torch_dtype=None, token=None, **kw):
            captured["model_token"] = token
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_tf = type(
        "M",
        (),
        {"SegformerForSemanticSegmentation": FakeModel, "SegformerImageProcessor": FakeProcessor},
    )
    fake_torch = type("T", (), {"float16": "f16", "float32": "f32"})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tf)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    SegFormerSegmenter(hf_token="hf_explicit").load("cpu")

    assert captured["processor_token"] == "hf_explicit"
    assert captured["model_token"] == "hf_explicit"


def test_anonymous_hub_access_clears_every_token_source(monkeypatch) -> None:
    """token=False is not enough on huggingface_hub 1.x; remove the credential.

    An ambient token still reaches the request on the httpx-based client,
    and the Hub answers 401 for a public repo when that token is
    fine-grained and scoped elsewhere. So the credential is removed at
    source for the duration of the download.
    """
    import os

    from drishti3d.semantics.segmenter import _HF_TOKEN_ENV_VARS, _anonymous_hub_access

    for name in _HF_TOKEN_ENV_VARS:
        monkeypatch.setenv(name, "hf_scoped_elsewhere")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)

    with _anonymous_hub_access():
        for name in _HF_TOKEN_ENV_VARS:
            assert name not in os.environ, f"{name} still set inside the block"
        assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"

    # Restored on exit -- a private PROJECT repo downloaded elsewhere in the
    # same process must still authenticate.
    for name in _HF_TOKEN_ENV_VARS:
        assert os.environ[name] == "hf_scoped_elsewhere"
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in os.environ


def test_anonymous_hub_access_restores_after_an_exception(monkeypatch) -> None:
    import os

    import pytest as _pytest

    from drishti3d.semantics.segmenter import _anonymous_hub_access

    monkeypatch.setenv("HF_TOKEN", "hf_keepme")

    with _pytest.raises(RuntimeError), _anonymous_hub_access():
        raise RuntimeError("download blew up")

    assert os.environ["HF_TOKEN"] == "hf_keepme"


def test_anonymous_hub_access_is_a_noop_for_a_private_checkpoint(monkeypatch) -> None:
    """A genuinely gated checkpoint needs its token left alone."""
    import os

    from drishti3d.semantics.segmenter import _anonymous_hub_access

    monkeypatch.setenv("HF_TOKEN", "hf_needed")

    with _anonymous_hub_access(enabled=False):
        assert os.environ["HF_TOKEN"] == "hf_needed"


def test_default_checkpoint_is_one_that_actually_exists() -> None:
    """Guard against the bug that cost three Kaggle runs.

    `nvidia/segformer-b4-finetuned-ade-640-640` was asserted from memory and
    does not exist: NVIDIA publishes B4 for ADE20K at 512 only, and 640 for
    B5 alone. The Hub answers a nonexistent repo with 401 rather than 404
    (it will not confirm which private repos exist), and transformers
    reports that as "not a valid model identifier" -- so the typo presented
    as an authentication failure and sent two fixes down the wrong path.
    """
    from drishti3d.config import SemanticsConfig
    from drishti3d.semantics.segmenter import (
        _DEFAULT_CHECKPOINT,
        _KNOWN_ADE_CHECKPOINTS,
    )

    assert _DEFAULT_CHECKPOINT in _KNOWN_ADE_CHECKPOINTS
    assert SemanticsConfig().checkpoint in _KNOWN_ADE_CHECKPOINTS
    # The specific string that did not exist.
    assert "b4-finetuned-ade-640-640" not in _DEFAULT_CHECKPOINT


def test_a_mistyped_checkpoint_reports_the_real_cause(monkeypatch) -> None:
    """A 401 must not be reported as an auth problem without qualification."""
    from drishti3d.semantics.segmenter import SegFormerSegmenter

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, source, token=None, **kw):
            raise OSError(f"{source} is not a valid model identifier listed on 'https://huggingface.co/models'")

    fake_tf = type(
        "M",
        (),
        {"SegformerForSemanticSegmentation": object, "SegformerImageProcessor": FakeProcessor},
    )
    fake_torch = type("T", (), {"float16": "f16", "float32": "f32"})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tf)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    with pytest.raises(OSError) as excinfo:
        SegFormerSegmenter(checkpoint="nvidia/segformer-b4-finetuned-ade-640-640").load("cpu")

    message = str(excinfo.value)
    assert "NONEXISTENT repo" in message
    # And it must list what does exist, so the fix is obvious from the error.
    assert "nvidia/segformer-b4-finetuned-ade-512-512" in message


# ---------------------------------------------------------------------------
# Memory (regression: 4K softmax OOM'd a 16 GB T4)
# ---------------------------------------------------------------------------


def test_is_cuda_accepts_indexed_devices() -> None:
    """`device == "cuda"` is False for "cuda:1" -- and that silently cost bf16.

    GeometryStage loads one backbone per GPU and passes "cuda:0"/"cuda:1".
    Two equality tests in the MapAnything adapter answered False there,
    dropping the model to float32 and disabling autocast on every device,
    which roughly doubled its memory and turned a comfortable fit into an
    illegal memory access.
    """
    from drishti3d.geometry.mapanything import _is_cuda

    assert _is_cuda("cuda")
    assert _is_cuda("cuda:0")
    assert _is_cuda("cuda:1")
    assert not _is_cuda("cpu")
    assert not _is_cuda("mps")
    assert not _is_cuda(None)
    assert not _is_cuda("")


def test_class_axis_is_reduced_before_upsampling() -> None:
    """The ordering that decides whether a 4K frame needs 10 GB or 10 MB.

    Upsampling 150-channel logits to the frame size before argmax builds a
    (150, 2160, 3840) float32 tensor -- 5.0 GB, with softmax needing a
    second one -- which OOM'd a 16 GB T4 on a single frame. Reducing the
    class axis first keeps the 150-channel tensor at the network's own 1/4
    resolution and upsamples only two single-channel maps.

    Asserted against the source rather than by running the model: the bug
    is an ORDERING, it only manifests at 4K on real CUDA, and there is no
    GPU here to reproduce it on.
    """
    import pathlib
    import sys

    src = pathlib.Path(sys.modules["drishti3d.semantics.segmenter"].__file__).read_text()

    assert "torch.softmax(logits[i].float(), dim=0)" in src
    assert "conf_small, native_small = probs.max(dim=0)" in src
    # The ordering that caused the OOM must be gone.
    assert "probs = torch.softmax(up, dim=0)" not in src


def test_labels_are_upsampled_nearest_not_bilinear() -> None:
    """Averaging class INDICES invents classes that were never predicted.

    The mean of road(3) and vegetation(5) is infrastructure(4) -- a class
    the network did not choose. Confidence is continuous and interpolates
    bilinearly; labels must not.
    """
    import pathlib
    import sys

    src = pathlib.Path(sys.modules["drishti3d.semantics.segmenter"].__file__).read_text()
    # The label upsample must request nearest.
    assert 'native_small[None, None].float(), size=(h, w), mode="nearest"' in src
    # The confidence upsample may be bilinear.
    assert 'conf_small[None, None], size=(h, w), mode="bilinear"' in src


@pytest.mark.parametrize("pitch_down,checkpoint,skipped", [
    (True, "nvidia/segformer-b4-finetuned-ade-512-512", True),     # nadir + ground-level model: pointless
    (False, "nvidia/segformer-b4-finetuned-ade-512-512", False),   # forward footage: in domain
    (True, "chribark/segformer-b3-finetuned-UAVid", False),        # an aerial checkpoint runs on nadir
])
def test_ground_level_segmenter_is_skipped_on_downward_flights(tmp_path, monkeypatch, pitch_down, checkpoint, skipped):
    from drishti3d.config import Config
    from drishti3d.pipeline import stages
    from drishti3d.pipeline.stages import PipelineState, SemanticsStage, StageUnavailable
    from drishti3d.types import FrameMetrics, Keyframe, Pose

    cfg = Config()
    cfg.semantics.checkpoint = checkpoint
    state = PipelineState(tmp_path / "v.mp4", None, cfg, "null")
    R = np.diag([1.0, -1.0, -1.0]) if pitch_down else np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    state.keyframes = [Keyframe(i, float(i), FrameMetrics(i, float(i), 100.0, 1.0, 120.0, 0.0),
                                pose=Pose(R=R, t=np.array([float(i), 0.0, 50.0]))) for i in range(4)]
    state.video = object()

    class Reached(Exception):
        pass

    def fake_create(*a, **kw):
        raise Reached

    monkeypatch.setattr("drishti3d.semantics.segmenter.create_segmenter", fake_create)
    expected = StageUnavailable if skipped else Reached
    with pytest.raises(expected):
        SemanticsStage().run(state, None, None)
    assert stages._ground_level_checkpoint(checkpoint) == ("ade" in checkpoint)
