"""Replace regressed depth with depth MEASURED by matching pixels between views.

The problem this exists to solve
---------------------------------
MapAnything predicts depth from a learned prior. Nothing in that
prediction is tied to the baseline the drone actually flew, so its error
is whatever the prior gives -- measured on real survey footage, **0.38 m
of local planarity noise at a 4.4 cm/px ground sampling**, roughly a 9x
noise floor. Every correction the pipeline applies downstream
(``geometry.depth_anchor``, ``fusion.reanchor``) is a *scale* correction
or a *rejection*: they fix where a surface sits, never how noisy it is.
A pixel wrong by 0.4 m in a way consistent with its neighbours survives
all of them, and that residual is what makes roofs read as bumps rather
than planes.

Matching is a different kind of measurement. Triangulated depth error
follows ``dZ = Z^2 / (fx * B) * disparity_error``; on this footage
(Z = 120 m, fx = 2697 px, B = 23 m) one pixel of disparity error is
0.23 m and sub-pixel matching reaches **~0.05 m** -- about 7x better than
the regression, and comfortably inside the 1 m specification.

What this module does
---------------------
Classic plane sweep, run per reference view over the neighbours already
present in the same window:

1. Build a stack of depth hypotheses *around the backbone's own depth*,
   not over the whole range. The backbone gets the depth roughly right
   after anchoring; what it lacks is precision. Sweeping +/-``range_frac``
   around its estimate needs a fraction of the hypotheses a blind sweep
   would, and cannot wander off to a different surface.
2. For each hypothesis, project every reference pixel into each source
   view and sample it (``grid_sample``).
3. Score agreement by zero-mean normalised cross-correlation over a small
   patch. Zero-mean NCC is used rather than raw difference because it is
   invariant to the per-frame exposure drift this footage has.
4. Take the best-scoring hypothesis per pixel, refine it to sub-pixel by
   fitting a parabola through the three costs around the minimum.
5. Reject where the evidence is weak: low NCC, flat cost curve (no
   distinct minimum, i.e. a textureless patch), or too few source views
   saw the pixel.

Rejected pixels keep the backbone's depth and are reported, rather than
being filled with a guess -- the caller decides what an unrefined pixel is
worth.

What it deliberately does not do
---------------------------------
No smoothing, no regularisation, no plane fitting. Those improve
appearance by moving geometry nobody measured, which is the one thing a
metric pipeline must not do (see ``fusion.photometric``'s module
docstring). This only ever *replaces a depth with a better-measured
depth*, or leaves it alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = ["PlaneSweepResult", "refine_depth_by_plane_sweep"]

#: Depth hypotheses swept per pixel. The sweep is centred on the
#: backbone's own depth over a narrow band, so this buys precision rather
#: than range: 48 samples over +/-15% at 120 m is a ~0.75 m step before
#: sub-pixel refinement, which the parabola fit then takes well below.
_N_HYPOTHESES = 48

#: Half-width of the sweep as a fraction of the backbone's depth. Wide
#: enough to cover the regression error measured on real footage
#: (0.38 m planarity, a few metres of gross error at worst), narrow enough
#: that the sweep cannot lock onto a different surface entirely.
_RANGE_FRACTION = 0.15

#: Patch half-width for NCC. 3 (a 7x7 window) is the usual MVS trade:
#: large enough to be discriminative on aerial texture, small enough not
#: to smear across depth discontinuities at roof edges.
_PATCH_RADIUS = 3

#: Minimum NCC for a match to count. Below this the "best" hypothesis is
#: not meaningfully better than any other and the pixel is left alone.
_MIN_NCC = 0.5

#: Minimum source views that must see a pixel before its refined depth is
#: accepted. Two views can agree by coincidence far more easily than three.
_MIN_VIEWS = 2

#: Minimum contrast between the best cost and the median cost across
#: hypotheses. A flat cost curve means the patch is textureless: every
#: depth explains it equally well, so the argmin is noise.
_MIN_COST_CONTRAST = 0.05


@dataclass
class PlaneSweepResult:
    """Refined depth plus an honest account of what was and was not measured."""

    depth: np.ndarray
    """``(V, H, W)`` depth. Refined where ``refined`` is True, the
    backbone's own value elsewhere."""

    refined: np.ndarray
    """``(V, H, W)`` bool: True where matching produced a usable depth."""

    ncc: np.ndarray
    """``(V, H, W)`` best NCC score, for callers that want to weight by it."""

    stats: dict


def _to_torch(array, device, dtype):
    import torch

    return torch.as_tensor(np.ascontiguousarray(array), device=device, dtype=dtype)


def _sweep_one_view(
    ref_idx: int,
    images_t,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    depth_ref,
    *,
    device,
    n_hypotheses: int,
    range_fraction: float,
    patch_radius: int,
    max_sources: int,
):
    """Plane-sweep one reference view against its neighbours. Returns (depth, ncc, n_views)."""
    import torch
    import torch.nn.functional as F

    _n_views, _, height, width = images_t.shape
    ref_intr = intrinsics[ref_idx]
    ref_pose = poses[ref_idx]

    # Nearest neighbours by camera centre: the views most likely to see
    # the same ground. Sweeping against a view on the far side of the
    # flight costs the same and contributes nothing.
    centres = np.array([p.t for p in poses])
    order = np.argsort(np.linalg.norm(centres - centres[ref_idx], axis=1))
    sources = [int(i) for i in order if int(i) != ref_idx][:max_sources]
    if not sources:
        return None

    # Reference pixel rays in camera frame, as directions at unit depth.
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rays = torch.stack(
        [(xs + 0.5 - ref_intr.cx) / ref_intr.fx, (ys + 0.5 - ref_intr.cy) / ref_intr.fy, torch.ones_like(xs)],
        dim=-1,
    )  # (H, W, 3)

    # Hypotheses centred on the backbone's depth: a multiplicative band,
    # so the sampling is uniform in relative error rather than absolute.
    valid_ref = depth_ref > 1e-6
    base = torch.where(valid_ref, depth_ref, torch.ones_like(depth_ref))
    scales = torch.linspace(1.0 - range_fraction, 1.0 + range_fraction, n_hypotheses, device=device)

    ref_gray = images_t[ref_idx].mean(dim=0, keepdim=True)[None]  # (1,1,H,W)
    k = 2 * patch_radius + 1
    ref_patches = F.unfold(ref_gray, k, padding=patch_radius)  # (1, k*k, H*W)
    ref_centred = ref_patches - ref_patches.mean(dim=1, keepdim=True)
    ref_norm = ref_centred.norm(dim=1, keepdim=True).clamp_min(1e-6)

    # Patch offsets, applied to the PROJECTED CENTRE of each patch.
    #
    # This is what makes it a plane sweep rather than a per-pixel guess.
    # Each pixel here carries its own depth hypothesis (the sweep is
    # centred on the backbone's own depth), so warping the whole image
    # once and then unfolding it compares a coherent reference patch
    # against source pixels gathered from incoherent depths -- neighbours
    # in the warped image came from unrelated places. Measured: NCC
    # collapsed to ~0.59 and the refined depth was worse than the input.
    #
    # Instead each patch is warped RIGIDLY: project the patch centre at
    # its hypothesis, then sample the source at the same pixel offsets
    # around that projected point. This is the standard fronto-parallel
    # patch assumption -- exact for a surface parallel to the image plane,
    # and a good approximation over a 7x7 window on terrain.
    offs = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32)
    off_y, off_x = torch.meshgrid(offs, offs, indexing="ij")
    off_x = off_x.reshape(-1)  # (k*k,)
    off_y = off_y.reshape(-1)

    best_ncc = torch.full((height, width), -1.0, device=device)
    best_scale = torch.ones((height, width), device=device)
    best_index = torch.full((height, width), -1, device=device, dtype=torch.int64)
    # Cost curve statistics, for the flat-curve (textureless) rejection.
    ncc_sum = torch.zeros((height, width), device=device)
    ncc_count = torch.zeros((height, width), device=device)
    view_count = torch.zeros((height, width), device=device)
    # Neighbouring costs around the running best, for sub-pixel refinement.
    prev_ncc = torch.full((height, width), -1.0, device=device)
    best_prev = torch.full((height, width), -1.0, device=device)
    best_next = torch.full((height, width), -1.0, device=device)

    r_ref = _to_torch(ref_pose.R, device, torch.float32)
    t_ref = _to_torch(ref_pose.t, device, torch.float32)

    for h_i, scale in enumerate(scales):
        depth_h = base * scale
        # Reference pixels -> world at this hypothesis.
        pts_cam = rays * depth_h[..., None]
        pts_world = pts_cam @ r_ref.T + t_ref

        ncc_accum = torch.zeros((height, width), device=device)
        seen = torch.zeros((height, width), device=device)

        for src_idx in sources:
            src_pose, src_intr = poses[src_idx], intrinsics[src_idx]
            r_src = _to_torch(src_pose.R, device, torch.float32)
            t_src = _to_torch(src_pose.t, device, torch.float32)
            q = (pts_world - t_src) @ r_src  # world -> source camera frame
            z = q[..., 2]
            in_front = z > 1e-6
            u = src_intr.fx * q[..., 0] / z.clamp_min(1e-6) + src_intr.cx
            v = src_intr.fy * q[..., 1] / z.clamp_min(1e-6) + src_intr.cy
            inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not bool(inside.any()):
                continue

            src_gray = images_t[src_idx].mean(dim=0, keepdim=True)[None]
            # Warp each patch on its reference fronto-parallel depth plane.
            # Copying image offsets into the source is only valid for equal
            # intrinsics and parallel cameras; yaw/tilt otherwise breaks NCC.
            relative_R = r_ref.T @ r_src
            offsets = torch.stack(
                [off_x / ref_intr.fx, off_y / ref_intr.fy, torch.zeros_like(off_x)], dim=-1
            ) @ relative_R
            qx = q[..., 0, None] + depth_h[..., None] * offsets[:, 0]
            qy = q[..., 1, None] + depth_h[..., None] * offsets[:, 1]
            qz = q[..., 2, None] + depth_h[..., None] * offsets[:, 2]
            pu = src_intr.fx * qx / qz.clamp_min(1e-6) + src_intr.cx
            pv = src_intr.fy * qy / qz.clamp_min(1e-6) + src_intr.cy
            inside &= ((qz > 1e-6) & (pu >= 0.5) & (pu <= width - 0.5)
                       & (pv >= 0.5) & (pv <= height - 0.5)).all(dim=-1)
            inside &= (xs >= patch_radius) & (xs < width - patch_radius) & (ys >= patch_radius) & (ys < height - patch_radius)
            grid = torch.stack(
                [2.0 * pu / width - 1.0, 2.0 * pv / height - 1.0], dim=-1
            ).reshape(1, height, width * k * k, 2)
            sampled = F.grid_sample(src_gray, grid, align_corners=False, padding_mode="zeros")
            # -> (1, k*k, H*W), matching unfold's layout for the reference.
            warped_patches = (
                sampled.reshape(1, height, width, k * k).permute(0, 3, 1, 2).reshape(1, k * k, height * width)
            )
            warped_centred = warped_patches - warped_patches.mean(dim=1, keepdim=True)
            warped_norm = warped_centred.norm(dim=1, keepdim=True).clamp_min(1e-6)
            # Zero-mean NCC: invariant to per-frame exposure drift, which
            # a raw intensity difference is not.
            ncc = ((ref_centred * warped_centred).sum(dim=1) / (ref_norm * warped_norm).squeeze(1)).reshape(height, width)
            ncc = torch.where(inside, ncc, torch.zeros_like(ncc))
            ncc_accum += ncc
            seen += inside.float()

        mean_ncc = ncc_accum / seen.clamp_min(1.0)
        mean_ncc = torch.where(seen > 0, mean_ncc, torch.full_like(mean_ncc, -1.0))
        ncc_sum += torch.where(seen > 0, mean_ncc, torch.zeros_like(mean_ncc))
        ncc_count += (seen > 0).float()

        improved = mean_ncc > best_ncc
        # Support must belong to the selected depth, not some other hypothesis.
        view_count = torch.where(improved, seen, view_count)
        best_prev = torch.where(improved, prev_ncc, best_prev)
        best_next = torch.where(improved, torch.full_like(best_next, -1.0), best_next)
        # The sample right after the current best completes its triple.
        just_after = (best_index == h_i - 1) & (~improved) & (h_i > 0)
        best_next = torch.where(just_after, mean_ncc, best_next)
        best_scale = torch.where(improved, torch.full_like(best_scale, float(scale)), best_scale)
        best_ncc = torch.where(improved, mean_ncc, best_ncc)
        best_index = torch.where(improved, torch.full_like(best_index, h_i), best_index)
        prev_ncc = mean_ncc

    # Sub-pixel: parabola through (prev, best, next) in hypothesis index.
    step = (2.0 * range_fraction) / max(n_hypotheses - 1, 1)
    denom = best_prev - 2.0 * best_ncc + best_next
    offset = torch.where(
        (denom.abs() > 1e-6) & (best_prev > -0.5) & (best_next > -0.5),
        0.5 * (best_prev - best_next) / denom,
        torch.zeros_like(denom),
    ).clamp(-1.0, 1.0)
    refined_scale = best_scale + offset * step

    mean_all = ncc_sum / ncc_count.clamp_min(1.0)
    contrast = best_ncc - mean_all
    return refined_scale * base, best_ncc, view_count, contrast


def refine_depth_by_plane_sweep(
    images: np.ndarray,
    depth: np.ndarray,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    *,
    device: str = "cpu",
    n_hypotheses: int = _N_HYPOTHESES,
    range_fraction: float = _RANGE_FRACTION,
    patch_radius: int = _PATCH_RADIUS,
    min_ncc: float = _MIN_NCC,
    min_views: int = _MIN_VIEWS,
    min_contrast: float = _MIN_COST_CONTRAST,
    max_sources: int = 4,
) -> PlaneSweepResult:
    """Refine ``depth`` by photometric matching. ``images`` is ``(V, H, W, 3)`` uint8.

    Every input is what a ``BackboneResult`` already carries, so this
    slots in directly after a backbone call with nothing recomputed.
    """
    import torch

    images = np.asarray(images)
    depth = np.asarray(depth, dtype=np.float64)
    n_views = images.shape[0]
    if n_views < 2:
        return PlaneSweepResult(
            depth=depth,
            refined=np.zeros_like(depth, dtype=bool),
            ncc=np.zeros_like(depth, dtype=np.float32),
            stats={"views": int(n_views), "failure": "plane sweep needs >= 2 views"},
        )

    torch_device = torch.device(device)
    images_t = _to_torch(images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0, torch_device, torch.float32)

    out_depth = depth.copy()
    out_refined = np.zeros(depth.shape, dtype=bool)
    out_ncc = np.zeros(depth.shape, dtype=np.float32)

    for v in range(n_views):
        depth_ref = _to_torch(depth[v], torch_device, torch.float32)
        swept = _sweep_one_view(
            v,
            images_t,
            poses,
            intrinsics,
            depth_ref,
            device=torch_device,
            n_hypotheses=n_hypotheses,
            range_fraction=range_fraction,
            patch_radius=patch_radius,
            max_sources=max_sources,
        )
        if swept is None:
            continue
        new_depth, ncc, view_count, contrast = swept
        accept = (
            (ncc >= min_ncc)
            & (view_count >= min_views)
            & (contrast >= min_contrast)
            & (depth_ref > 1e-6)
            & (new_depth > 1e-6)
        )
        out_depth[v] = torch.where(accept, new_depth, depth_ref).detach().cpu().numpy()
        out_refined[v] = accept.detach().cpu().numpy()
        out_ncc[v] = ncc.detach().cpu().numpy()

    valid = depth > 1e-6
    refined_frac = float(out_refined.sum() / max(valid.sum(), 1))
    changed = np.abs(out_depth - depth)[out_refined]
    stats = {
        "views": int(n_views),
        "hypotheses": int(n_hypotheses),
        "refined_pct": round(100.0 * refined_frac, 2),
        "median_depth_change_m": round(float(np.median(changed)), 4) if changed.size else None,
        "p90_depth_change_m": round(float(np.percentile(changed, 90)), 4) if changed.size else None,
        "median_ncc": round(float(np.median(out_ncc[out_refined])), 3) if out_refined.any() else None,
    }
    logger.info(
        "plane sweep: refined %.1f%% of valid pixels over %d views (median depth change %s m, median NCC %s)",
        stats["refined_pct"],
        n_views,
        stats["median_depth_change_m"],
        stats["median_ncc"],
    )
    return PlaneSweepResult(depth=out_depth, refined=out_refined, ncc=out_ncc, stats=stats)
