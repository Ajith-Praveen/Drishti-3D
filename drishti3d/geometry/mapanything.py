"""Adapter around MapAnything (Meta/CMU, ``facebookresearch/map-anything``).

Why MapAnything, and why conditioning matters
----------------------------------------------
Unlike VGGT or Pi3, MapAnything regresses *metric* geometry (real-world
scale, not scale-up-to-an-unknown-constant) and accepts camera intrinsics
and poses as optional per-view conditioning inputs. We have both: rough
intrinsics from EXIF/a camera database (``ingest.intrinsics``) and rough
poses from GPS/IMU telemetry (``ingest.telemetry`` via triage's
``Keyframe.telemetry``). On the UAVFF3D benchmark, feeding that
conditioning in cut ray error from 6.55 to 0.56 -- so this adapter always
passes both through when available; it does not silently fall back to
image-only inference by choice, only because a caller didn't have priors
to give it.

This module is torch/mapanything-free at import time
------------------------------------------------------
``torch`` and ``mapanything`` are imported lazily, strictly inside methods,
never at module scope. ``drishti3d.geometry.mapanything`` must remain
importable (and ``MapAnythingBackbone.is_available()`` must return
``False`` cleanly, not raise) on a machine with neither installed -- e.g.
this project's own macOS dev machine, which has no CUDA and where MapAnything
is not part of the base dependency set (see ``pyproject.toml``'s ``ml``
extra).

Checkpoint choice and licensing
--------------------------------
MapAnything ships two Hugging Face checkpoints with identical APIs:
``facebook/map-anything`` (CC-BY-NC 4.0 -- research-only, but scores higher
on published benchmarks) and ``facebook/map-anything-apache``
(Apache-2.0 -- commercial-friendly). ``DEFAULT_CHECKPOINT`` here is the
Apache-2.0 one, because this is meant to ship as a product, not stay a
research artifact; the checkpoint id is a constructor argument precisely so
a caller who has explicitly accepted the CC-BY-NC terms (e.g. for an
internal/research deployment) can opt into the stronger
``NONCOMMERCIAL_CHECKPOINT`` instead.

Offline weights
----------------
The end product is an air-gapped desktop app: a ``from_pretrained()`` call
that silently reaches out to the Hugging Face Hub at first run is a bug,
not a convenience, on a machine that may have no network access at deploy
time. Set ``DRISHTI3D_MAPANYTHING_WEIGHTS`` (``LOCAL_WEIGHTS_ENV_VAR``) to a
local directory containing the predownloaded checkpoint (or pass
``local_weights_dir`` to the constructor) and this adapter loads from
there; if neither is given it logs a warning and falls through to
``from_pretrained(checkpoint)`` (network-dependent), and if a local
directory *is* given but doesn't exist, it fails immediately with an
actionable message rather than a confusing downstream error.

Coordinate conventions -- read this before touching the conversion helpers
----------------------------------------------------------------------------
``drishti3d.types`` fixes world = ENU, Z-up, metres, and camera = OpenCV
(X-right, Y-down, Z-forward); ``Pose.matrix()`` is exactly the 4x4
cam2world transform in that convention. MapAnything's ``camera_poses``
input/output is documented as "OpenCV cam2world" and its "world" frame is
*not* a fixed absolute convention -- it is simply whatever frame the input
conditioning poses were expressed in (when no pose conditioning is given,
the model picks an arbitrary one, typically anchored at the first camera).
Since we always condition on our own ENU-frame poses, MapAnything's "world"
coincides with our ENU frame for both the input and the output, and *no
axis remapping is needed* -- ``pose_to_opencv_c2w``/``opencv_c2w_to_pose``
are (deliberately) thin wrappers. Getting this wrong (e.g. flipping an axis
"to be safe") is exactly the classic bug that would silently mirror or
rotate the reconstructed model; don't add a flip here without a very good,
tested reason. Verified directly against the real installed package: a
synthetic 3-view sequence conditioned with camera_poses translating along
world +Z by 0, 2, 4 m came back with predicted camera_poses translating
along +Z by ~0.02, 1.73, 3.38 m -- monotonically increasing in the same
axis and direction as the input, i.e. no sign flip / axis swap. (Magnitude
tracks but does not exactly reproduce the input because pose conditioning
is a soft prior the model can refine, not a hard constraint; direction is
what matters here.)

Real required per-view schema (confirmed against the installed package,
not guessed -- see ``mapanything.utils.inference.validate_input_views_for_inference``
and ``MapAnything.infer``'s docstring)
----------------------------------------------------------------------------
Required:
  - ``img``: ``(1, 3, H, W)`` float32 tensor, normalized per the encoder's
    ``data_norm_type`` (NOT raw 0-255 RGB -- see ``_normalize_image_batch``).
  - ``data_norm_type``: ``[str]`` -- a *list* containing the norm type name
    (the code indexes it as ``[0]``), and it must equal
    ``self._model.encoder.data_norm_type`` (``"dinov2"`` for the Apache
    checkpoint, but read from the model rather than hard-coded since a
    different checkpoint's encoder could use a different value).
Optional conditioning (all with a leading batch dim of 1, matching ``img``):
  - ``intrinsics``: ``(1, 3, 3)`` float32 -- conflicts with ``ray_directions``.
  - ``camera_poses``: ``(1, 4, 4)`` float32 OpenCV cam2world, or a
    ``(quats (1,4), trans (1,3))`` tuple. If any view has this, view 0 must
    too.
  - ``depth_z``: ``(1, H, W)`` float32 -- requires ``intrinsics`` too.
  - ``is_metric_scale``: ``(1,)`` bool -- defaults to ``True`` when omitted,
    which matches our assumption (real GPS poses + real intrinsics), so
    this adapter does not pass it explicitly.
Output (confirmed): a ``list`` of one dict per view (NOT a dict of stacked
tensors -- see ``_normalize_predictions``), each tensor still carrying that
same leading batch dim of 1: ``pts3d``/``depth_z``/``mask`` are
``(1, H, W, {3,1,1})``, ``conf`` is ``(1, H, W)``, ``camera_poses`` is
``(1, 4, 4)``, ``intrinsics`` is ``(1, 3, 3)``. ``conf`` is *not* a
probability in [0, 1] -- see ``_predictions_to_result``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drishti3d.geometry.backbone import Backbone, BackboneResult
from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)


def _autocast_dtype(device: str | None) -> str | None:
    """Half-precision dtype to run this model under, or ``None`` for fp32.

    bf16 on both CUDA and MPS; fp32 on CPU (where autocast buys nothing
    and torch's CPU bf16 path is slower than fp32 for this model).

    MPS was fp32 until it was measured. The original caution was right --
    "a smoke test on synthetic noise is not enough evidence to trust for a
    metric reconstruction pipeline" -- so the evidence was collected
    instead (``scripts/benchmark_mps.py``, 8 real keyframes at 518 px,
    every one of 1,160,320 valid pixels compared against the fp32 run):

        bf16 vs fp32 depth   median 0.0035%  p99 0.027%  max 0.38%
        bf16 vs fp32 points  median 1.03 mm  p99 2.8 mm  max 35 mm
        wall clock           14.72 s -> 5.30 s  (2.78x)

    A 1 mm median displacement against a 1 m accuracy specification is
    three orders of magnitude inside tolerance, and MapAnything is trained
    in bf16 to begin with. fp16 measured identically; bf16 is preferred
    for its wider exponent range, which matters for the unbounded
    confidence head (``1 + exp(x)``).
    """
    if _is_cuda(device):
        return "bf16"
    if device and str(device).startswith("mps"):
        return "bf16"
    return None


def _is_cuda(device: str | None) -> bool:
    """Whether ``device`` names a CUDA device, indexed or not.

    Exists because ``device == "cuda"`` is the wrong test once devices are
    addressed per-index. ``GeometryStage`` loads one backbone per GPU and
    passes ``"cuda:0"``/``"cuda:1"``, for which equality answers False --
    silently selecting float32 and disabling autocast on every device,
    which roughly doubles this model's memory footprint.
    """
    return bool(device) and str(device).startswith("cuda")

# Apache-2.0, commercial-friendly -- the safe default for a shipped product.
DEFAULT_CHECKPOINT = "facebook/map-anything-apache"
# CC-BY-NC 4.0 -- research use only. Scores higher on published benchmarks
# per MapAnything's own reporting, but callers must opt in explicitly.
NONCOMMERCIAL_CHECKPOINT = "facebook/map-anything"

# Env var pointing at a local directory of predownloaded weights, for
# offline/air-gapped deployment. See module docstring.
LOCAL_WEIGHTS_ENV_VAR = "DRISHTI3D_MAPANYTHING_WEIGHTS"


class MapAnythingBackbone(Backbone):
    """``Backbone`` adapter for MapAnything.

    Ships complete and correct even though it cannot be exercised on this
    project's macOS/no-CUDA dev machine: every conversion helper below
    (resize+intrinsics scaling, pose conversion, prediction unpacking) is
    plain numpy/opencv and is unit-tested directly in ``tests
    /test_geometry.py`` without needing torch or mapanything installed at
    all. Only ``load``/``predict`` themselves need the real packages.
    """

    name = "mapanything"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        local_weights_dir: str | Path | None = None,
        max_image_size: int = 518,
        mask_edges: bool = True,
        multiview_confidence: bool = False,
        confidence_percentile: float | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_image_size = max_image_size
        # See GeometryConfig.mapanything_mask_edges for why the pipeline
        # turns this off: the edge mask is a smoothness prior, and on
        # noisy, mis-scaled depth it discards most of every frame.
        self.mask_edges = mask_edges
        # Replace the network's self-reported confidence with MEASURED
        # cross-view depth agreement (mapanything.utils.multiview_confidence:
        # each pixel's depth is projected into every other view of the
        # window and scored on whether they concur). This is the same class
        # of evidence as fusion.photometric, but computed where every view
        # of the window is in memory and pixel-aligned -- so it does not
        # suffer the "too few views saw this point" starvation that leaves
        # 96% of points UNVERIFIABLE downstream.
        self.multiview_confidence = multiview_confidence
        # When set, zero out the lowest-confidence percentile at the source
        # rather than carrying the points downstream to be filtered later.
        self.confidence_percentile = confidence_percentile
        self._local_weights_dir = Path(local_weights_dir) if local_weights_dir else self._env_weights_dir()
        self._model: Any = None
        self._device: str | None = None
        self._dtype: Any = None

    @staticmethod
    def _env_weights_dir() -> Path | None:
        raw = os.environ.get(LOCAL_WEIGHTS_ENV_VAR)
        return Path(raw) if raw else None

    def is_available(self) -> bool:
        try:
            import mapanything  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, device: str = "cpu", dtype: Any = None) -> None:
        if not self.is_available():
            raise RuntimeError(
                "MapAnythingBackbone.load() called but the 'mapanything' package "
                "and/or torch is not installed. Install the 'ml' extra "
                "(`uv sync --extra ml`) and a checkout of "
                "https://github.com/facebookresearch/map-anything "
                "(`pip install -e .` inside it), then re-run."
            )

        import torch
        from mapanything.models import MapAnything

        self._device = device
        # bf16 autocast on CUDA and MPS, fp32 on CPU -- see
        # _autocast_dtype for the per-pixel measurement (1.03 mm median
        # displacement, 2.78x faster) that moved MPS off fp32.
        # `startswith`, not `==`: the device may be an INDEXED string
        # ("cuda:0"/"cuda:1") when one backbone is loaded per GPU. An
        # equality test silently answers False there, which quietly drops
        # this model to float32 and roughly doubles its memory -- enough to
        # turn a comfortable fit into an OOM or an illegal memory access.
        self._dtype = dtype if dtype is not None else (torch.bfloat16 if _is_cuda(device) else torch.float32)

        weights_source = self.checkpoint
        if self._local_weights_dir is not None:
            if not self._local_weights_dir.exists():
                raise RuntimeError(
                    f"{LOCAL_WEIGHTS_ENV_VAR} is set to '{self._local_weights_dir}', but that "
                    "path does not exist. This is meant to be an air-gapped deployment: "
                    "MapAnything weights must be predownloaded there ahead of time (e.g. "
                    f"`huggingface-cli download {self.checkpoint} --local-dir "
                    f"{self._local_weights_dir}` on a machine with network access), not "
                    "fetched from the Hugging Face Hub at runtime."
                )
            weights_source = str(self._local_weights_dir)
        else:
            logger.warning(
                "%s is not set and no local_weights_dir was given; "
                "MapAnything.from_pretrained(%r) may attempt a Hugging Face Hub "
                "download over the network. Set %s to a local weights directory "
                "for offline/air-gapped runs.",
                LOCAL_WEIGHTS_ENV_VAR,
                self.checkpoint,
                LOCAL_WEIGHTS_ENV_VAR,
            )

        self._model = MapAnything.from_pretrained(weights_source).to(device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    #: Advertised so the pipeline can ask for a depth-prior refinement pass
    #: without special-casing this class by name.
    supports_depth_prior = True

    def predict(
        self,
        images: list[np.ndarray],
        intrinsics: list[CameraIntrinsics] | None = None,
        poses: list[Pose] | None = None,
        depth_priors: list[np.ndarray] | None = None,
    ) -> BackboneResult:
        """See ``Backbone.predict``.

        ``depth_priors`` (optional, this adapter only): one ``(H, W)``
        camera-frame z-depth per view at the backbone's own grid resolution
        -- i.e. the ``depth`` of a previous ``BackboneResult`` from the same
        images, typically after ``geometry.depth_anchor`` has rescaled it.
        Passed to MapAnything as ``depth_z`` so the network regresses on top
        of a correctly-scaled field (its depth-completion mode) rather than
        guessing scale from a range prior that does not cover survey
        altitude. Requires ``intrinsics``; a prior without intrinsics is a
        contract violation in MapAnything's own validator and is rejected
        here for the same reason.
        """
        if self._model is None:
            raise RuntimeError("MapAnythingBackbone.predict() called before load()")
        if depth_priors is not None and intrinsics is None:
            raise ValueError("depth_priors require intrinsics (MapAnything's depth_z contract)")
        if depth_priors is not None and len(depth_priors) != len(images):
            raise ValueError("depth_priors list must be the same length as images")

        import torch
        from uniception.models.encoders.image_normalizations import (
            IMAGE_NORMALIZATION_DICT,
        )

        n = len(images)
        if intrinsics is not None and len(intrinsics) != n:
            raise ValueError("intrinsics list must be the same length as images")
        if poses is not None and len(poses) != n:
            raise ValueError("poses list must be the same length as images")

        # Required per `validate_input_views_for_inference`: every view's
        # `data_norm_type` must equal the *loaded* encoder's, not a
        # hard-coded string -- a different checkpoint's encoder could use a
        # different normalization (e.g. "dinov2" vs "identity").
        data_norm_type = self._model.encoder.data_norm_type
        try:
            image_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]
        except KeyError as exc:
            raise RuntimeError(
                f"MapAnything encoder reports data_norm_type={data_norm_type!r}, which is "
                f"not in uniception's IMAGE_NORMALIZATION_DICT ({sorted(IMAGE_NORMALIZATION_DICT)}). "
                "Cannot normalize input images correctly without this."
            ) from exc
        mean = image_norm.mean.view(3, 1, 1).to(self._device, dtype=torch.float32)
        std = image_norm.std.view(3, 1, 1).to(self._device, dtype=torch.float32)

        # MapAnything's patch embedding conv hard-asserts the input spatial
        # dims are an exact multiple of the encoder's patch size (confirmed
        # via a real `AssertionError: Input shape must be divisible by
        # patch size: 14` when they aren't -- `resize_preserving_aspect`'s
        # max-side scaling has no reason to land on a multiple of 14, e.g.
        # `--sizes 256,384` in the benchmark script both fail this). Read
        # the patch size from the model rather than hard-coding 14, same
        # reasoning as `data_norm_type`.
        patch_size = getattr(self._model.encoder, "patch_size", 14)

        views: list[dict[str, Any]] = []
        used_intrinsics: list[CameraIntrinsics] = []
        # Fix 3: the exact RGB pixels the encoder actually saw for each
        # view, pixel-aligned with pts3d/depth/conf (same H, W -- both are
        # produced from this same cropped_rgb). Captured here (not
        # recomputed downstream) because this is the one place that knows
        # the resize scale + patch-alignment crop offset actually applied;
        # a caller trying to re-derive matching colour from the original
        # image would have to duplicate resize_preserving_aspect +
        # crop_to_patch_multiple exactly, including any off-by-one in the
        # crop centering, to land on the identical pixel grid.
        view_images_rgb: list[np.ndarray] = []

        for i, img in enumerate(images):
            resized_bgr, scale = resize_preserving_aspect(img, self.max_image_size)
            # drishti3d.types.Frame.image (and therefore every image this
            # pipeline hands around) is BGR uint8 per drishti3d.ingest.video;
            # MapAnything, like essentially every ML vision model, expects
            # RGB. Converting here (not upstream) keeps that BGR convention
            # local to this file's boundary with the outside model.
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            cropped_rgb, crop_top, crop_left = crop_to_patch_multiple(resized_rgb, patch_size)
            view_images_rgb.append(cropped_rgb)

            # `img` must be a (1, 3, H, W) float32 tensor normalized per
            # `data_norm_type` -- NOT raw (H, W, 3) uint8 RGB. Getting this
            # wrong doesn't raise a validation error (unlike the missing
            # `data_norm_type` key); it silently feeds the encoder garbage.
            img_chw = torch.from_numpy(np.ascontiguousarray(cropped_rgb))
            img_chw = img_chw.to(self._device).permute(2, 0, 1).float() / 255.0
            img_chw = (img_chw - mean) / std

            view: dict[str, Any] = {
                "img": img_chw.unsqueeze(0),
                "data_norm_type": [data_norm_type],
            }

            if intrinsics is not None:
                intr = scale_intrinsics(intrinsics[i], scale)
                # The patch-alignment crop above is centered, so it shifts
                # the principal point by exactly the crop offset (focal
                # length is unaffected -- cropping isn't rescaling).
                intr = CameraIntrinsics(
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx - crop_left,
                    cy=intr.cy - crop_top,
                    width=cropped_rgb.shape[1],
                    height=cropped_rgb.shape[0],
                    dist_coeffs=intr.dist_coeffs,
                )
                used_intrinsics.append(intr)
                view["intrinsics"] = (
                    torch.from_numpy(intr.K()).to(self._device, dtype=torch.float32).unsqueeze(0)
                )

            if poses is not None and poses[i] is not None:
                c2w = pose_to_opencv_c2w(poses[i])
                view["camera_poses"] = (
                    torch.from_numpy(c2w).to(self._device, dtype=torch.float32).unsqueeze(0)
                )

            if depth_priors is not None and depth_priors[i] is not None:
                prior = np.asarray(depth_priors[i], dtype=np.float32)
                if prior.shape != cropped_rgb.shape[:2]:
                    raise ValueError(
                        f"depth prior for view {i} has shape {prior.shape}, expected the backbone grid "
                        f"{cropped_rgb.shape[:2]} (pass a previous BackboneResult.depth from the same images)"
                    )
                # Masked pixels carry depth 0 in a BackboneResult. Zero is
                # not "unknown" to the depth encoder, it is "at the camera",
                # so holes are filled with the view's median valid depth --
                # a flat, honest guess the network is free to overrule.
                valid = prior > 1e-6
                if not valid.any():
                    prior_filled = None
                else:
                    prior_filled = np.where(valid, prior, float(np.median(prior[valid])))
                if prior_filled is not None:
                    view["depth_z"] = (
                        torch.from_numpy(prior_filled).to(self._device, dtype=torch.float32)[None, ..., None]
                    )
                    view["is_metric_scale"] = torch.ones(1, dtype=torch.bool, device=self._device)

            views.append(view)

        # bf16 autocast on CUDA only; fp32 (no autocast -- `infer()` forces
        # `amp_dtype=torch.float32` internally whenever `use_amp=False`)
        # everywhere else. See `load()`'s docstring comment for why MPS
        # doesn't get autocast by default despite not raising in testing.
        # Same indexed-device trap as in `load()`: "cuda:1" == "cuda" is
        # False, which would disable autocast on every GPU but the first.
        # See _autocast_dtype: bf16 on CUDA and MPS, fp32 on CPU, with the
        # per-pixel equivalence measurement that justifies MPS.
        amp_dtype = _autocast_dtype(self._device)
        use_amp = amp_dtype is not None
        amp_dtype = amp_dtype or "bf16"  # ignored when use_amp is False
        try:
            with torch.inference_mode():
                raw_predictions = self._model.infer(
                    views,
                    memory_efficient_inference=True,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    mask_edges=self.mask_edges,
                    use_multiview_confidence=self.multiview_confidence,
                    apply_confidence_mask=self.confidence_percentile is not None,
                    confidence_percentile=(
                        self.confidence_percentile if self.confidence_percentile is not None else 10.0
                    ),
                )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"MapAnything.infer() hit an operation not implemented on device "
                f"{self._device!r}: {exc}. This is typically a torch op missing an MPS "
                "kernel; either run on CUDA/CPU, or set PYTORCH_ENABLE_MPS_FALLBACK=1 to "
                "let torch fall back to CPU for that op (silently slower, so only use "
                "this as a stopgap)."
            ) from exc

        per_view = _normalize_predictions(raw_predictions, n)
        fallback_intrinsics = used_intrinsics if used_intrinsics else None
        result = _predictions_to_result(
            per_view,
            self.checkpoint,
            fallback_intrinsics,
            confidence_is_unit_range=self.multiview_confidence,
        )
        # Fix 3: all views were resized+cropped from the same max_image_size/
        # patch_size, so (barring an input with a wildly different aspect
        # ratio than the rest of the batch -- already an implicit
        # requirement for np.stack over pts3d/depth/conf above) every entry
        # in view_images_rgb shares one (H, W), same as pts3d itself.
        result.images = np.stack(view_images_rgb, axis=0)
        return result


# ---------------------------------------------------------------------------
# Confirmed against the real installed package (`MapAnything.infer`'s type
# hint and a live run): the return value is always a `list`, one dict per
# view, in input order -- never a dict of tensors stacked along a leading
# view dimension. This function no longer guesses between the two; it just
# validates the confirmed shape so a future version returning something
# unexpected fails loudly here instead of downstream with a confusing KeyError.
# ---------------------------------------------------------------------------


def _normalize_predictions(raw_predictions: Any, n_views: int) -> list[dict[str, Any]]:
    """Validate MapAnything's `infer` output is the expected list-of-per-view-dicts shape."""
    per_view = list(raw_predictions)
    if len(per_view) != n_views:
        raise RuntimeError(
            f"MapAnything returned {len(per_view)} view predictions for {n_views} input views"
        )
    return per_view


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _predictions_to_result(
    per_view: list[dict[str, Any]],
    checkpoint: str,
    fallback_intrinsics: list[CameraIntrinsics] | None,
    confidence_is_unit_range: bool = False,
) -> BackboneResult:
    """Convert MapAnything's per-view prediction dicts into a ``BackboneResult``.

    See the module docstring's "Coordinate conventions" section: no axis
    remapping happens here, by design, because MapAnything's world frame
    (when conditioned on our poses, which we always do when we have them)
    already *is* our ENU world frame.

    Every tensor in ``pred`` carries a leading batch dim of 1 (this adapter
    always calls ``infer`` with exactly one image per view -- see the
    module docstring's schema section) -- index ``[0]`` unwraps it.

    ``conf`` is NOT a [0, 1] probability: MapAnything's confidence head is
    ``ConfidenceAdaptor(confidence_type="exp", vmin=1, vmax=inf)`` (verified
    by inspecting the loaded model's ``dense_adaptor.confidence_adaptor``),
    i.e. ``conf_raw = 1 + exp(x)`` in ``[1, inf)``. ``1 - 1/conf_raw`` is
    exactly ``sigmoid(x)``, the monotonic inverse of that encoding back into
    ``[0, 1)`` -- matching ``BackboneResult.confidence``'s documented
    contract without an arbitrary min/max normalization that would make the
    scale depend on what happened to be in the batch. Pixels the library's
    own ``mask`` (non-ambiguous + edge + confidence mask, when enabled) says
    are invalid are zeroed here too: ``apply_mask=True`` (this adapter's
    default via ``infer``'s own default) already zeroes ``pts3d``/``depth_z``
    for those pixels, but MapAnything does *not* zero ``conf`` itself, so
    without this a masked-out (0, 0, 0) point could carry a high raw
    confidence and get treated as real by downstream tiering.
    """
    poses: list[Pose] = []
    points_list: list[np.ndarray] = []
    depth_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    out_intrinsics: list[CameraIntrinsics] = []
    masked_fraction: list[float] = []

    for i, pred in enumerate(per_view):
        c2w = _to_numpy(pred["camera_poses"])[0].astype(np.float64)
        poses.append(Pose(R=c2w[:3, :3].copy(), t=c2w[:3, 3].copy()))

        pts = _to_numpy(pred["pts3d"])[0].astype(np.float64)
        points_list.append(pts)

        depth = _to_numpy(pred["depth_z"])[0].astype(np.float64)
        depth_list.append(depth[..., 0] if depth.ndim == 3 and depth.shape[-1] == 1 else depth)

        conf_raw = _to_numpy(pred["conf"])[0].astype(np.float64)
        if confidence_is_unit_range:
            # Multi-view depth-consistency confidence (see
            # mapanything.utils.multiview_confidence) is documented as
            # "values in [0, 1]" -- the fraction of other views whose
            # reprojected depth agrees. It is ALREADY this contract's
            # scale, so it is taken as-is.
            #
            # Applying the learned head's decoding to it is catastrophic
            # and silent: clip(conf, 1.0, None) flattens every value in
            # [0, 1] to exactly 1.0, and 1 - 1/1 is 0.0, so every pixel
            # reports zero confidence. Measured: the confidence filter
            # then dropped all 4,099,909 points and fusion produced no
            # vertices at all.
            confidence = np.clip(conf_raw, 0.0, 1.0)
        else:
            # Learned confidence head: ConfidenceAdaptor(type="exp",
            # vmin=1) emits 1 + exp(x) in [1, inf). 1 - 1/conf is exactly
            # sigmoid(x), the monotonic inverse back into [0, 1).
            confidence = 1.0 - 1.0 / np.clip(conf_raw, 1.0, None)
        if "mask" in pred:
            mask = _to_numpy(pred["mask"])[0]
            mask = mask[..., 0] if mask.ndim == 3 and mask.shape[-1] == 1 else mask
            confidence = np.where(mask.astype(bool), confidence, 0.0)
            # Recorded so a run can say how much of each frame the backbone
            # threw away before anything downstream saw it. This number is
            # what turned a 216 m theoretical footprint into a measured
            # 13-25 m one, and it was invisible until it was logged.
            masked_fraction.append(float(1.0 - mask.astype(bool).mean()))
        conf_list.append(confidence.astype(np.float32))

        if "intrinsics" in pred:
            K = _to_numpy(pred["intrinsics"])[0].astype(np.float64)
            h, w = pts.shape[:2]
            out_intrinsics.append(
                CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=w, height=h)
            )
        elif fallback_intrinsics is not None:
            out_intrinsics.append(fallback_intrinsics[i])
        else:
            h, w = pts.shape[:2]
            out_intrinsics.append(CameraIntrinsics.from_hfov(60.0, w, h))

    return BackboneResult(
        poses=poses,
        points=np.stack(points_list, axis=0),
        depth=np.stack(depth_list, axis=0),
        confidence=np.stack(conf_list, axis=0),
        intrinsics=out_intrinsics,
        # MapAnything regresses metric geometry by construction when
        # conditioned with real (metric) intrinsics -- which this adapter
        # always supplies whenever the caller has them.
        is_metric=True,
        metadata={"checkpoint": checkpoint, "num_views": len(per_view), "masked_fraction": masked_fraction},
    )


# ---------------------------------------------------------------------------
# Resize + intrinsics scaling (pure numpy/opencv -- unit-tested directly)
# ---------------------------------------------------------------------------


def resize_preserving_aspect(img: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Resize ``img`` so its longer side is ``max_size``, preserving aspect ratio.

    Returns ``(resized_img, scale)`` where ``scale = new_side / old_side``
    is the single scalar factor BOTH axes were rescaled by (aspect ratio is
    preserved, so there is only one factor). Callers must rescale
    intrinsics by this exact ``scale`` (see ``scale_intrinsics``) -- not by
    a factor independently recomputed from the rounded integer output
    size, which silently drifts the principal point by a pixel or more on
    some resolutions. That drift is exactly the "classic bug source" this
    function exists to avoid.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid image shape {img.shape}")

    scale = max_size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def scale_intrinsics(intrinsics: CameraIntrinsics, scale: float) -> CameraIntrinsics:
    """Scale ``intrinsics`` for an image resized by the uniform ``scale`` factor from ``resize_preserving_aspect``."""
    return CameraIntrinsics(
        fx=intrinsics.fx * scale,
        fy=intrinsics.fy * scale,
        cx=intrinsics.cx * scale,
        cy=intrinsics.cy * scale,
        width=max(1, round(intrinsics.width * scale)),
        height=max(1, round(intrinsics.height * scale)),
        dist_coeffs=intrinsics.dist_coeffs,
    )


def crop_to_patch_multiple(img: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    """Center-crop ``img`` (``(H, W, C)``) so both spatial dims are exact multiples of ``patch_size``.

    MapAnything's patch embedding conv hard-asserts the input spatial shape
    is exactly divisible by the encoder's patch size (confirmed directly:
    ``AssertionError: Input shape must be divisible by patch size: 14`` when
    it isn't). ``resize_preserving_aspect``'s max-side scaling has no reason
    to land on a multiple of the patch size, so this crop is a required
    second step whenever building a MapAnything view, independent of the
    ``max_image_size`` chosen.

    Returns ``(cropped_img, crop_top, crop_left)``; the offsets are needed
    to shift a principal point (``cx``, ``cy``) that was computed for the
    pre-crop image -- cropping doesn't rescale, so focal length is
    unaffected, but the principal point moves by exactly the crop offset.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid image shape {img.shape}")
    if patch_size <= 0:
        raise ValueError(f"invalid patch_size {patch_size}")

    aligned_h = max(patch_size, (h // patch_size) * patch_size)
    aligned_w = max(patch_size, (w // patch_size) * patch_size)
    crop_top = (h - aligned_h) // 2
    crop_left = (w - aligned_w) // 2
    cropped = img[crop_top : crop_top + aligned_h, crop_left : crop_left + aligned_w]
    return cropped, crop_top, crop_left


# ---------------------------------------------------------------------------
# Pose conversion (pure numpy -- unit-tested directly)
# ---------------------------------------------------------------------------


def pose_to_opencv_c2w(pose: Pose) -> np.ndarray:
    """Convert a ``types.Pose`` to the 4x4 OpenCV cam2world matrix MapAnything's ``camera_poses`` input expects.

    No axis remapping: see the module docstring's "Coordinate conventions"
    section for why ``pose.matrix()`` (already a cam2world transform in
    OpenCV camera-axis convention) is directly usable as-is.
    """
    return pose.matrix().astype(np.float32)


def opencv_c2w_to_pose(matrix: np.ndarray) -> Pose:
    """Inverse of ``pose_to_opencv_c2w``: wrap a raw 4x4 cam2world matrix back into a ``types.Pose``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return Pose(R=matrix[:3, :3].copy(), t=matrix[:3, 3].copy())
