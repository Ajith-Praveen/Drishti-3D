"""Swappable per-frame semantic segmentation, mirroring ``geometry.backbone``.

This module is to segmentation what ``geometry.backbone`` is to depth: a
narrow ABC, a name registry, lazy imports, and a deterministic fallback
that lets the whole pipeline run with no model and no torch installed.
The shape is deliberately identical so there is one mental model for
"swappable learned component" in this codebase, not two.

What a segmenter is allowed to return
-------------------------------------
``predict`` returns a ``SegmentationResult`` whose ``labels`` are already
``SemanticClass`` values (see ``semantics.classes``), never the backend's
native label space. Translation happens inside the backend, next to the
checkpoint that defines the mapping -- so a checkpoint swap can never
silently change what a stored ``semantic_class`` byte means downstream.

``confidence`` is the per-pixel max softmax probability. It is a *relative*
quality signal, not a calibrated probability: modern segmentation heads are
systematically overconfident, and an out-of-domain nadir drone frame makes
that worse, not better. ``labelling.label_points`` therefore uses it only
to weight votes between views, never as an absolute accept/reject
threshold on a single view.

The honest note about domain
----------------------------
The default checkpoint is trained on ground-level photography (ADE20K).
Aerial nadir imagery is out of its training distribution. Everything in
this package is built around that fact rather than pretending otherwise:
multi-view voting, a stored per-point agreement ratio, an explicit
``UNLABELLED`` class, and a report card that shows how much of the cloud
the segmenter actually agreed with itself about. If an aerial-domain
checkpoint is available for a deployment, register it here and pass its
name via ``SemanticsConfig.model`` -- the rest of the pipeline needs no
change.
"""

from __future__ import annotations

import contextlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from drishti3d.semantics.classes import (
    ADE20K_TO_CANONICAL,
    SemanticClass,
    map_label_array,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NullSegmenter",
    "SegFormerSegmenter",
    "SegmentationResult",
    "Segmenter",
    "create_segmenter",
    "register_segmenter",
]


#: Every environment variable huggingface_hub will read a token from.
#: Cleared together, because clearing only one leaves the others to supply
#: the same rejected credential.
_HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)


@contextlib.contextmanager
def _anonymous_hub_access(*, enabled: bool = True):
    """Guarantee an unauthenticated Hub download for the duration of the block.

    Passing ``token=False`` to ``from_pretrained`` is *supposed* to be
    enough, and on older huggingface_hub it was. On the httpx-based 1.x
    client it is not: an ambient token still reaches the request, the Hub
    rejects a fine-grained token scoped to another namespace with 401 even
    for a public repo, and transformers reports that as "not a valid model
    identifier" -- which reads like a typo rather than an auth failure.

    So this removes the credential at its source instead of asking the
    library to ignore it: every token environment variable is unset, and
    ``HF_HUB_DISABLE_IMPLICIT_TOKEN`` is set to stop the cached
    ``huggingface-cli login`` token being used either. Both are restored on
    exit, so a private *project* repo downloaded elsewhere in the same
    process is unaffected.

    ``enabled=False`` makes this a no-op, for a genuinely private or gated
    checkpoint that needs its own token.
    """
    if not enabled:
        yield
        return

    saved = {name: os.environ.pop(name, None) for name in _HF_TOKEN_ENV_VARS}
    saved_implicit = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
        if saved_implicit is None:
            os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
        else:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = saved_implicit


@dataclass
class SegmentationResult:
    """One frame's segmentation, already in this project's taxonomy.

    ``labels`` and ``confidence`` are always the same ``(H, W)`` as the
    image that was passed in -- backends that run at a lower internal
    resolution are responsible for upsampling before returning, so callers
    never have to track two resolutions.
    """

    labels: np.ndarray  # (H, W) uint8, SemanticClass values
    confidence: np.ndarray  # (H, W) float32 in [0, 1], max softmax
    model_name: str = "unknown"

    def dynamic_mask(self, excluded: frozenset[int]) -> np.ndarray:
        """``(H, W)`` bool, True where the pixel must NOT become geometry."""
        return np.isin(self.labels, list(excluded))


class Segmenter(ABC):
    """Interface every segmentation backend implements."""

    name: str = "unnamed"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can run right now (deps + weights present).

        Must never raise. A missing optional dependency is reported here,
        not thrown, so ``SemanticsStage`` can check-then-skip cleanly and
        record itself as ``"skipped"`` instead of ``"failed"``.
        """

    @abstractmethod
    def load(self, device: str, dtype: Any = None) -> None:
        """Load weights onto ``device`` ("cuda", "mps", "cpu")."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and any device memory it holds."""

    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        """Segment a batch of BGR uint8 ``(H, W, 3)`` frames.

        BGR, not RGB: that is what ``ingest.video.VideoSource`` produces
        (see its ``_to_upright_bgr``), and converting once inside the
        backend is less error-prone than asking every caller to remember.
        """


_REGISTRY: dict[str, type[Segmenter]] = {}


def register_segmenter(name: str, cls: type[Segmenter]) -> None:
    _REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# NullSegmenter -- the "no model" path
# ---------------------------------------------------------------------------


class NullSegmenter(Segmenter):
    """Labels every pixel ``UNLABELLED``. Always available, never useful.

    This exists so the pipeline shape is identical with and without a
    segmentation model: ``SemanticsStage`` always runs, always produces
    masks of the right dtype and size, and the difference is only whether
    those masks carry information. Nothing downstream needs an
    ``if semantics is None`` branch.

    Crucially it does NOT invent labels. A run that used this segmenter
    produces a 100%-``UNLABELLED`` histogram, and the report card renders
    that as "not computed" -- which is the truth -- rather than a
    plausible-looking class breakdown that no model ever produced.
    """

    name = "null"

    def is_available(self) -> bool:
        return True

    def load(self, device: str, dtype: Any = None) -> None:
        return None

    def unload(self) -> None:
        return None

    def predict(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        out = []
        for img in images:
            h, w = img.shape[:2]
            out.append(
                SegmentationResult(
                    labels=np.full((h, w), SemanticClass.UNLABELLED, dtype=np.uint8),
                    confidence=np.zeros((h, w), dtype=np.float32),
                    model_name=self.name,
                )
            )
        return out


# ---------------------------------------------------------------------------
# SegFormerSegmenter -- the real path
# ---------------------------------------------------------------------------

#: Default checkpoint. SegFormer-B4 is the accuracy/VRAM knee for this use
#: case: ~64M params, and it fits alongside MapAnything on a 6 GB target
#: *because the two never run simultaneously* -- SemanticsStage completes
#: and unloads before GeometryStage loads. B5 is ~2 points better mIoU for
#: nearly double the memory, which is not worth it when the domain gap
#: (ground-level training, aerial input) dominates the error budget anyway.
#:
#: NOTE THE RESOLUTION: NVIDIA publishes B4 for ADE20K at **512x512** only.
#: There is no `...-b4-finetuned-ade-640-640`; 640 exists for B5 alone. The
#: Hub answers a nonexistent repo with **401**, not 404 (so it cannot leak
#: which private repos exist), and transformers reports that as "not a
#: valid model identifier" -- so a typo in this string presents as an
#: authentication failure. Verify any replacement against
#: `_KNOWN_ADE_CHECKPOINTS` below before changing it.
_DEFAULT_CHECKPOINT = "nvidia/segformer-b4-finetuned-ade-512-512"

#: Every ADE20K SegFormer checkpoint NVIDIA actually publishes, largest
#: last. Kept here so the docstring above is checkable rather than
#: assertable, and so a future edit has the real names to hand.
_KNOWN_ADE_CHECKPOINTS = (
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "nvidia/segformer-b5-finetuned-ade-640-640",
)


class SegFormerSegmenter(Segmenter):
    """SegFormer (NVIDIA/HuggingFace ``transformers``) over ADE20K's 150 classes.

    Why SegFormer and not a Cityscapes model: Cityscapes has a cleaner
    label set for roads and vehicles, but it is trained exclusively on
    forward-facing dashcam views, which is a *worse* domain mismatch for
    nadir aerial frames than ADE20K's mixed indoor/outdoor photography.
    ADE20K also carries ``building``/``tree``/``water``/``earth`` classes
    that Cityscapes lacks entirely, and those map directly onto required
    reconstruction targets.

    Why not SAM: SAM segments *instances* without naming them. This
    pipeline needs names -- "which points are vegetation" is the actual
    deliverable -- so a class-agnostic segmenter would still need a
    classification head bolted on top.
    """

    name = "segformer"

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        max_image_size: int = 1024,
        batch_size: int = 4,
        local_weights: str | None = None,
        hf_token: str | bool | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_image_size = int(max_image_size)
        self.batch_size = max(1, int(batch_size))
        self.local_weights = local_weights
        self.hf_token = hf_token
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._dtype: Any = None

    # -- availability ---------------------------------------------------
    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    # -- lifecycle ------------------------------------------------------
    def load(self, device: str, dtype: Any = None) -> None:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        source = self.local_weights or self.checkpoint
        token: str | bool = self.hf_token if self.hf_token else False

        # fp16 on CUDA only. MPS's fp16 softmax is numerically flaky in a
        # way that shows up as speckled labels along class boundaries, and
        # on CPU fp16 is emulated and slower than fp32 -- so both get
        # fp32 regardless of what the caller asked for.
        if dtype is None:
            dtype = torch.float16 if device.startswith("cuda") else torch.float32

        try:
            with _anonymous_hub_access(enabled=not self.hf_token):
                self._processor = SegformerImageProcessor.from_pretrained(source, token=token)
                self._model = SegformerForSemanticSegmentation.from_pretrained(
                    source, torch_dtype=dtype, token=token
                )
        except OSError as exc:
            # The Hub returns 401 for a repo that does not exist, not 404 --
            # it will not confirm or deny the existence of private repos. So
            # "unauthorized" here most often means a MISTYPED CHECKPOINT,
            # and the raw error sends you hunting for an auth bug instead.
            # Say so, and list what actually exists.
            if "401" in str(exc) or "not a valid model identifier" in str(exc):
                known = "\n  ".join(_KNOWN_ADE_CHECKPOINTS)
                raise OSError(
                    f"Could not load segmentation checkpoint {source!r}.\n"
                    "The Hub answers 401 for a NONEXISTENT repo as well as for a genuinely "
                    "unauthorized one, so the most likely cause is that this checkpoint id is "
                    "wrong, not that authentication failed.\n"
                    f"ADE20K SegFormer checkpoints that exist:\n  {known}\n"
                    "For a genuinely private or gated checkpoint, set SemanticsConfig.hf_token."
                ) from exc
            raise
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._dtype = dtype
        logger.info("segformer loaded: %s on %s (%s)", source, device, dtype)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- inference ------------------------------------------------------
    def predict(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        if self._model is None:
            raise RuntimeError("SegFormerSegmenter.predict() called before load()")

        import torch

        results: list[SegmentationResult] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            results.extend(self._predict_chunk(chunk, torch))
        return results

    def _predict_chunk(self, images: list[np.ndarray], torch: Any) -> list[SegmentationResult]:
        # The processor wants RGB; VideoSource gives BGR. Downscale first
        # so the expensive part scales with max_image_size, not with the
        # source 4K frame.
        rgb_small: list[np.ndarray] = []
        for img in images:
            small = _downscale(img, self.max_image_size)
            rgb_small.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        inputs = self._processor(images=rgb_small, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)

        with torch.no_grad():
            logits = self._model(pixel_values=pixel_values).logits  # (B, 150, h/4, w/4)

        out: list[SegmentationResult] = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]

            # Reduce the 150-class axis BEFORE upsampling, never after.
            #
            # The obvious ordering -- upsample logits to the frame size,
            # then softmax+argmax -- gives marginally crisper class
            # boundaries, and is what this did originally. It is also
            # catastrophic on 4K input: a (150, 2160, 3840) float32 tensor
            # is 5.0 GB, and softmax needs a second one, so a single frame
            # asked for ~10 GB and OOM'd a 16 GB T4.
            #
            # Reducing first keeps the 150-channel tensor at the network's
            # own 1/4 resolution (a few MB), then upsamples exactly two
            # single-channel maps. Confidence interpolates bilinearly
            # because it is continuous; labels must use nearest, since
            # averaging class *indices* would invent classes that were
            # never predicted -- the mean of "road"(3) and "vegetation"(5)
            # is "infrastructure"(4), which is nonsense.
            probs = torch.softmax(logits[i].float(), dim=0)
            conf_small, native_small = probs.max(dim=0)
            del probs

            conf = torch.nn.functional.interpolate(
                conf_small[None, None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
            native = torch.nn.functional.interpolate(
                native_small[None, None].float(), size=(h, w), mode="nearest"
            )[0, 0]

            out.append(
                SegmentationResult(
                    labels=map_label_array(native.to(torch.int64).cpu().numpy(), ADE20K_TO_CANONICAL),
                    confidence=conf.cpu().numpy().astype(np.float32),
                    model_name=self.checkpoint,
                )
            )
            del conf, native, conf_small, native_small

        del logits, pixel_values
        return out


def _downscale(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink-only resize, matching ``ingest.video.downscale_image``'s contract."""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


register_segmenter("null", NullSegmenter)
register_segmenter("segformer", SegFormerSegmenter)


def create_segmenter(name: str, **kwargs: Any) -> Segmenter:
    """Instantiate the registered segmenter ``name``.

    Follows ``geometry.backbone.create_backbone``'s contract exactly,
    including the "unknown kwargs are dropped rather than raising" rule --
    every registered segmenter must stay constructible with zero
    arguments so a caller that doesn't know which one is configured
    doesn't have to know which kwargs it takes.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown segmenter {name!r}; available: {sorted(_REGISTRY)}")

    cls = _REGISTRY[name]
    if not kwargs:
        return cls()
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
