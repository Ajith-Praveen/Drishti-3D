"""Pipeline configuration: a dataclass tree, loadable from / savable to YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path

import yaml


@dataclass
class IngestConfig:
    """Video ingest / decoding settings."""

    target_fps: float = 5.0
    resize_max_dim: int | None = None
    telemetry_path: str | None = None
    time_offset_sec: float = 0.0
    # Known camera calibration (the problem statement's optional "camera
    # intrinsics" input). When camera_fx or camera_hfov_deg is set it
    # replaces the metadata/camera-DB/HFOV guess in ingest.intrinsics and
    # is recorded with provenance "user", which focal-from-flow and BA
    # focal refinement both treat as measured. camera_fy/cx/cy default to
    # fx and the image centre. camera_dist_coeffs is OpenCV order
    # (k1, k2, p1, p2[, k3...]) for the RAW video pixels. When the
    # calibration was made at another resolution, camera_calibration_width
    # rescales fx/fy/cx/cy to the video's width.
    camera_fx: float | None = None
    camera_fy: float | None = None
    camera_cx: float | None = None
    camera_cy: float | None = None
    camera_hfov_deg: float | None = None
    camera_dist_coeffs: list[float] | None = None
    camera_calibration_width: int | None = None
    # Camera-database key (e.g. "dji mavic 3") when the video metadata does
    # not name the camera itself.
    camera_model: str | None = None
    # Image-motion clock check (ingest.timesync, TimeSyncStage). "auto"
    # replaces an unmeasured (assumed-zero) video-start offset with the one
    # the keyframes' image rotation implies, and only REPORTS a confident
    # disagreement with an explicit or auto-detected offset, which wins;
    # "correct" applies the measurement over those too; "off" skips it.
    # Measured on PinPoint flight01 it lands within ~0.25 s of the offset
    # that fits the surveyed points best -- enough to catch the 1.1 s
    # error that corrupted earlier runs, not to fine-tune below that.
    auto_sync: str = "auto"
    auto_sync_search_s: float = 10.0
    auto_sync_warn_s: float = 0.75


@dataclass
class TriageConfig:
    """Keyframe selection settings.

    Blur gating is relative, not absolute (see ``relative_blur_threshold``):
    ``blur_score`` mixes variance-of-Laplacian and Tenengrad terms that have
    different magnitudes and units and shift with camera, resolution, and
    scene texture, so a fixed absolute number like the old
    ``min_blur_score = 100.0`` default does not transfer across footage.
    The selector instead tracks a running median/percentile of blur scores
    seen so far in *this* video and rejects candidates below
    ``relative_blur_threshold`` times that running value.  ``min_blur_score``
    is kept only as an absolute hard floor for pathological cases (e.g. a
    fully black or corrupt frame) and defaults low so it essentially never
    fires on its own.

    Keyframe spacing prefers a GPS-driven metric baseline over vision-only
    parallax when telemetry is available (see
    ``triage.selector.select_keyframes`` and
    ``triage.metrics.estimate_parallax_detailed``): translation over a
    near-planar scene (the common nadir/near-nadir drone case) is fully
    explained by a homography just like pure rotation is, so a
    homography-residual-only parallax estimate starves on flat terrain.
    ``min_baseline_m`` is the minimum acceptable metric baseline between
    keyframes; ``baseline_to_altitude_ratio`` adapts that target to height
    above ground using the actual base/height (B/H) ratio needed to resolve
    small structures -- see below for why ``0.02`` (this field's old
    default) was wrong by more than an order of magnitude.

    Why ``baseline_to_altitude_ratio`` was raised from 0.02 to 0.25
    -------------------------------------------------------------------
    Stereo depth error follows ``dz = z**2 / (B * fx)`` (``z`` = range/
    altitude, ``B`` = baseline, ``fx`` = focal length in pixels at the
    working resolution): a fixed 1px disparity error turns into a metric
    depth error that shrinks linearly with baseline. At 120 m AGL with the
    old ``0.02`` ratio, ``B`` = 2.4 m (measured on real footage: 7.7 m
    median, because the floor below also bound it up) -- combined with the
    old working resolution's fx (~288px, see ``GeometryConfig
    .max_image_size``), that is a ~6.5 m depth error per 1px of disparity
    noise, which is exactly the ~1.9 m local planar-fit residual measured
    on a real reconstruction (a 5 m house cannot resolve when the depth
    noise floor is itself metres wide). Standard aerial photogrammetry
    practice targets a B/H ratio of **0.2-0.3** (equivalently 70-80%
    forward overlap) specifically because it is the ratio at which stereo
    intersection geometry is well-conditioned without losing so much
    overlap that matching fails; ``0.02`` under-shot that by more than an
    order of magnitude. ``0.25`` (24-30 m spacing at a typical 120 m AGL
    survey altitude) lands in that standard range and, combined with
    ``GeometryConfig.max_image_size`` raised to a working fx of ~513px
    (the "balanced" profile -- see that dataclass's docstring for why
    924px, not the larger 1288px this was originally going to be, is the
    default on measured wall-clock grounds), brings depth error per 1px
    disparity down to ~0.94-1.17 m at this ratio's 24-30 m spacing --
    combined with ``max_image_size=1288`` (the "accurate" profile,
    fx ~715px) it reaches ~0.71-0.84 m. Forward overlap at 24-30 m spacing
    over this drone's ~84 deg HFOV footprint is still ~80%, the survey
    standard, so nothing is lost on the matching side by widening the
    baseline this much.

    A second, independent benefit: widening the baseline also *reduces*
    keyframe count for the same flight distance (fewer, more widely-spaced
    keyframes cover the same ground), which means fewer submap windows and
    fewer backbone invocations -- this fix is faster as well as more
    accurate, not a trade-off between the two.

    The actual target baseline is ``max(min_baseline_m,
    baseline_to_altitude_ratio * relative_altitude_m)`` whenever relative
    altitude is known, and ``min_baseline_m`` alone otherwise.

    Why ``baseline_to_altitude_ratio`` was walked back from 0.25 to **0.10**
    -------------------------------------------------------------------------
    ``0.25`` fixed per-frame depth precision (the ~1.9 m local planar-fit
    residual documented above) but broke something the docstring above did
    not anticipate: it fragmented the reconstruction into roughly a dozen
    disconnected patches strung along the flight path instead of one
    continuous surface (top-down renders showed square-ish patches with
    clear gaps between consecutive keyframes' own reconstructed footprints,
    not a continuous ribbon). A fragmented model is worse than a coarse
    one -- there is nothing to inspect between the patches -- so this had
    to be fixed even at the cost of walking back some of the depth-precision
    gain above.

    Root cause (measured, not guessed): the *theoretical* along-track
    footprint at 120 m AGL for this drone's ~84 deg HFOV / 3840x2160 sensor
    is ~121 m, which at 0.25's 24-30 m spacing implies a comfortable ~75%
    forward overlap on paper. But ``MapAnything``'s own per-keyframe usable
    (raw confidence > 0.3) reconstructed footprint on real footage measured
    only ~13-25 m across -- 5-9x narrower than the theoretical HFOV-derived
    footprint, almost certainly because its own non-ambiguous/edge/
    confidence masking (see ``geometry.mapanything``'s ``conf``/``mask``
    handling) discards most of a wide-FOV frame's periphery at this
    altitude/resolution. At 0.25's 24-30 m spacing, that ~13-25 m usable
    footprint does not overlap its neighbour at all (measured per-keyframe
    gaps of +2 m to +30 m between *consecutive* keyframes' own reconstructed
    extents, growing along a submap) -- the 75% *theoretical* overlap figure
    was real but irrelevant, because it was computed from a footprint size
    the backbone never actually delivered.

    A 5-point sweep on real 16-keyframe footage (same video/telemetry,
    ``--backbone mapanything``, everything else at this project's defaults)
    measured achieved median baseline, forward overlap (against the
    theoretical 121 m footprint), a rasterized-cloud continuity metric
    (0.5 m grid; largest-connected-component fraction of occupied cells,
    8-connectivity; count of connected components), the local 1x1 m
    planar-fit residual, and wall-clock time:

    =====  ==========  =========  ============  ================  ==========  ========
    ratio  baseline_m  overlap %  n_components  largest_frac (%)  residual_m  time (s)
    =====  ==========  =========  ============  ================  ==========  ========
    0.06     11.34       90.6          5              96.2           0.278      267
    0.10     13.14       89.2          5              96.7           0.277      275
    0.15     18.90       84.4          5              91.8           0.271      270
    0.20     25.00       79.3          5              75.8           0.161      281
    0.25     30.18       75.1         16              18.9           0.150      325
    =====  ==========  =========  ============  ================  ==========  ========

    (``0.25``'s row is measured *after* the submap-merge rotation fix below
    -- see ``geometry.submap._gps_full_anchor_transform``'s docstring --
    which independently corrected 2 of 3 submaps whose own GPS-anchored
    Sim(3) rotation fit was degenerate; before that fix ``0.25`` measured 15
    components / 24.5% largest-fraction, i.e. the merge bug was a real but
    *secondary* contributor next to the dominant per-keyframe-footprint gap
    above. Per-submap, pre-merge continuity checks at every ratio confirmed
    the gaps are already present in a single window's own local-frame
    output, before any cross-window alignment -- this is a backbone/
    keyframe-spacing issue, not a submap-merge artifact.)

    The trend is monotonic and the trade-off is clean: residual degrades
    gracefully (0.15-0.28 m throughout, 7-13x better than the pre-Fix-1
    1.9 m baseline this whole effort started from) while continuity
    collapses sharply above ~0.15. ``0.10`` is chosen as the new default:
    it sits in the flat, safe part of the continuity curve (96.7%, tied
    with ``0.06`` for the best measured value, 5 components = one dominant
    component plus a handful of minor ones) while using the *widest*
    baseline that still comfortably closes the gap -- not the narrowest
    that happens to work, per this project's general preference for the
    widest baseline the data will tolerate (see the original 0.25
    reasoning above, which the same logic now re-derives at a smaller
    number once the backbone's real usable footprint is accounted for).
    Residual at 0.10 (0.277 m) is still comfortably under the 1 m target
    this task set, just short of what 0.20-0.25 could reach if continuity
    were not the overriding priority. No wider ratio was viable: 0.15
    already loses ~5 points of largest-fraction, and 0.20-0.25 fragment
    badly enough to fail the "one dominant component" bar outright. This
    also did not need Task-1b's decoupled-windows escape hatch (denser
    *selection*, wider reconstruction *windows*): a single ratio (0.10)
    satisfies both the sub-metre residual bar and the single-dominant-
    component bar on this footage, so keyframe selection and reconstruction
    windowing stay coupled as before.

    ``min_baseline_m`` moves with the ratio (``ratio * 20 m``, a
    representative close-range inspection altitude) for the same reason it
    always has: it is the floor for low-altitude/no-altitude-known passes,
    and should stay calibrated to whatever the ratio currently is rather
    than silently drifting out of sync with it.
    """

    target_keyframes: int = 150
    # Ceiling for the adaptive GPS spacing threshold, metres. Without it a
    # takeoff/hover stretch projects a huge keyframe count and the
    # threshold ratchets up for good (68 m on DJI_1001: only 80 keyframes,
    # consecutive frames barely overlapping). MapAnything's measured usable
    # footprint is 13-25 m, so 25 m keeps neighbours overlapping.
    max_baseline_m: float = 25.0
    # Overlap-driven spacing (triage.footprint): keyframes are spaced for
    # this fraction of forward overlap of the MEASURED along-track
    # footprint. 0.70 means every ground point is seen by ~3.3 consecutive
    # keyframes -- the minimum for a triangulated (MEASURED) point; more
    # views mostly add disagreeing regressed depth that stacks in fusion.
    # 0 disables it (altitude rule + max_baseline_m cap, the old behaviour).
    forward_overlap: float = 0.70
    # GPS mode: defer a keyframe while the course is changing faster than
    # this (deg/s) -- a banking, blurring turn. None disables.
    max_turn_rate_deg_s: float | None = 12.0
    min_blur_score: float = 1.0
    relative_blur_threshold: float = 0.6
    min_parallax_px: float = 15.0
    max_frames_scanned: int = 20000
    # Decode every selected keyframe once, during triage, and share the
    # images with yaw refinement, matching and geometry. Without this each
    # of those re-decodes all keyframes from the 4K source -- measured at
    # 144 s per pass (1.87 s/frame) on the sample flight, paid 3-4 times.
    cache_keyframe_images: bool = True
    # Longest side stored. Above every consumer's working resolution
    # (matching at 1920 is the highest); 4K would cost 1.9 GB resident.
    # Texture baking is not a consumer -- it reads full-resolution frames
    # itself, because its whole purpose is resolution.
    keyframe_cache_max_size: int = 1920
    min_baseline_m: float = 2.0
    use_gps_baseline: bool = True
    # GPS-baseline mode: decode every frame but convert and blur-score only
    # the frames a keyframe window search reads (triage.selector._LazyEntry),
    # and keep each chosen keyframe's full-resolution pixels so the
    # keyframe cache needs no second decode. Measured: 139 -> ~600 fps.
    lazy_decode: bool = True
    baseline_to_altitude_ratio: float = 0.10


@dataclass
class GeometryConfig:
    """3D reconstruction backbone settings.

    ``max_image_size``: Fix 2 (see ``TriageConfig.baseline_to_altitude_ratio``'s
    docstring for Fix 1, the companion change). Working focal length in
    pixels scales linearly with this, and depth error per 1px of disparity
    noise scales as ``z**2 / (B * fx)`` -- so the old default (518px, giving
    fx ~= 288px at this project's ~84 deg-HFOV/3840px-wide source video) was
    a second, independent contributor to the ~1.9 m local planar-fit
    residual this fix targets, on top of Fix 1's too-narrow baseline.

    Raised to **924px**, not the textbook-larger 1288px, on *measured*
    grounds, not guessed ones: real ``MapAnythingBackbone`` inference on
    this project's own dev machine (Apple Silicon MPS, real
    ``facebook/map-anything-apache`` weights) timed an 8-view window at
    7.64s/66.42s/(1288px 4-view: 71.62s, extrapolated 8-view ~170s) for
    518/924/1288px respectively. A 16-keyframe run plans ~3 windows of 8
    views each (see ``geometry.windows.plan_window_size``'s memory-budget
    floor): at 1288px that is ~510s (8.5 min) of backbone time alone --
    before ingest/triage/BA/fusion/export -- which blows this project's
    "no slower than before" requirement on this hardware. At 924px it is
    ~200s (3.3 min), comfortably inside the same budget, while still
    tripling the old fx (288px -> ~513px) and bringing depth error at
    Fix 1's new baseline down to roughly 0.9-1.2 m -- most of the
    achievable improvement for a fraction of 1288px's wall-clock cost.
    1288px (fx ~715px, ~0.7-0.8 m depth error) is kept as the "accurate"
    quality profile for callers who have the time budget for it (see
    ``QUALITY_PROFILES``/``apply_quality_profile`` below, and their
    ``GEOMETRY_QUALITY_PROFILES`` sibling); "fast" keeps the old 518px for
    callers who need the old speed and can tolerate the old accuracy.

    Also see the important fix this depends on: previously,
    ``geometry.backbone.get_backbone`` always constructed
    ``MapAnythingBackbone`` with its own hardcoded ``max_image_size=518``
    default (ignoring this field entirely for the backbone's own *internal*
    resize -- see ``geometry.mapanything.MapAnythingBackbone.predict``'s
    ``resize_preserving_aspect`` call), so raising this field alone would
    have silently done nothing: ``pipeline.stages.GeometryStage`` pre-resizes
    to this value, but the backbone would then immediately resize a second
    time back down to its own hardcoded 518px. ``get_backbone`` now accepts
    and forwards ``max_image_size`` (and any other kwarg the target
    backbone's constructor accepts) so this field actually reaches the
    model call it's meant to control.
    """

    backbone: str = "mapanything"

    # --- parallax depth anchoring (geometry.depth_anchor) -----------------
    # Rescale each view's backbone depth to agree with feature parallax
    # triangulated from the telemetry poses. This is the telemetry/SRT
    # alignment step of the pipeline, applied where the error lives: per
    # view, before merging. Measured need: MapAnything's depth on 120 m AGL
    # footage came out 0.15-0.59x true, inconsistent between windows.
    depth_anchor: bool = True
    depth_anchor_max_features: int = 3000
    # Fewer parallax samples than this and a view inherits the window
    # median; a whole window below this is left untouched and reported.
    depth_anchor_min_samples: int = 30
    # Corrections beyond this are refused as backbone failure, not scale.
    depth_anchor_max_ratio: float = 20.0
    # Second, per-view fit (geometry.depth_fit): z_ba = a * z_pred + b
    # against the pose-prior bundle adjustment's sparse points, applied
    # before the window becomes a submap. Supersedes the parallax anchor
    # for every view with enough BA points; views without are left as the
    # anchor produced them.
    ba_depth_fit: bool = True
    ba_depth_fit_min_samples: int = 15
    # Upgrade each view's (a, b) to a quadratic scale field over the image
    # when it fits the BA points better (geometry.depth_fit). The affine
    # fit left ~6.5 m RMS at 280 m range on DJI_1001.
    ba_depth_fit_spatial: bool = True
    # Rebuild each window's points with the bundle-adjusted world poses
    # (after the BA depth fit), so every submap is born in the world frame
    # and the merge is the identity ("world_frame" strategy). Removes the
    # per-window Sim(3) fits that made multi-window runs 4-35x rougher.
    # Only engages when the pose prior produced a pose for every keyframe.
    ba_world_frame: bool = True
    # Telemetry with GPS but NO gimbal orientation at all: condition on a
    # nadir camera yawed along the GPS course instead of no pose. Only
    # correct for straight-down footage; flow-yaw and the pose-prior BA
    # refine it. Turn off for oblique video.
    assume_nadir_without_gimbal: bool = True
    # World-frame mode: points kept across ALL windows for the final merge,
    # placement check and TSDF (the live model keeps full density).
    world_frame_total_points: int = 12_000_000
    # Dense-model view selection (world-frame mode): keep a keyframe's depth
    # only if it adds ground seen by fewer than this many selected views.
    # Bundle adjustment still uses every keyframe. 0 disables.
    dense_target_views: int = 3
    # ...and only if at least this fraction of its footprint is such
    # under-covered ground. Measured on flight01 (301 keyframes, lawnmower):
    # 0.4 kept 90 views, median 5 per ground point (was 15), 96% of area.
    dense_min_new_fraction: float = 0.4
    # After every window has been anchored, force the outliers onto one
    # flight-wide depth scale (geometry.scale_consensus). The backbone's
    # metric error is a property of the backbone and the footage, not of
    # the window, so windows that measured it very differently measured
    # it wrong -- and windows at different scales put the same ground at
    # different heights, which the merge cannot undo. Leave on unless
    # deliberately diagnosing the per-window anchor itself.
    scale_consensus: bool = True
    # Cap on how far the camera may travel inside one reconstruction
    # window, in metres. See geometry.windows.plan_windows' `max_extent_m`
    # for the measurement: the same 8 views spanning 48 m returned 78 m of
    # depth against a 120 m truth, and spanning 148 m returned 16 m. Depth
    # that comes back 7x shallow must be scaled 7x to become metric, which
    # scales its error 7x too -- that is what made the ground 12 m thick
    # inside a 2 m cell. 60 m keeps the backbone in its usable regime with
    # margin. Set to null to plan windows purely by keyframe count.
    max_window_extent_m: float | None = None

    # Reconstruct in as FEW backbone calls as possible -- ideally one --
    # rather than many small windows that then have to be merged.
    #
    # Measured on the sample flight, surface roughness inside a 2 m cell
    # (the ground should be flat to a few cm there):
    #
    #     one inference,  6 views                 2.05 m
    #     one inference, 32 views (627 m of it)   2.40 m
    #     19 windows merged (v17)                 8-12 m
    #     64 windows merged (v18)                 72 m
    #
    # Every window boundary is a Sim(3) fit, and on a short straight run
    # of cameras that fit's rotation is ill-conditioned; the errors it
    # injects dominate the backbone's own noise by 3-30x. A single call
    # has no boundaries, so there is nothing to fit and nothing to get
    # wrong -- and it was ~50x faster (37 s against 30 min).
    single_inference: bool = True
    # Hard ceiling on views per backbone call. None means "try them all";
    # attention is quadratic in views, so a long flight will still hit
    # memory and fall back (see _predict_with_oom_backoff).
    max_views_per_inference: int | None = None
    # Refinement pass: after anchoring, run the backbone a second time with
    # the anchored depth supplied as its own `depth_z` prior (MapAnything's
    # depth-completion mode). The network then regresses detail on top of a
    # correctly-scaled field instead of guessing the scale itself. The
    # anchor check is re-run on the result and reported: a ratio near 1.0
    # means the prior was honoured. Doubles backbone inference time, so
    # off by default; enable on a GPU box after A/B against a run without.
    depth_prior_refine: bool = False

    # Replace telemetry gimbal yaw with yaw measured from image motion +
    # GPS displacement (geometry.yaw_from_flow) before any window runs.
    # The sample flight's gimbal heading was stale through turns (10
    # distinct values over 77 keyframes, up to 87 deg wrong) and sat a
    # constant 15 deg from the compass on straight legs; a rotation that
    # wrong corrupts MapAnything's conditioning, the parallax anchor and
    # the merge. Pitch/roll still come from telemetry.
    # Measure focal length from GPS baseline + telemetry altitude + image
    # motion when the intrinsics are only a generic-HFOV guess. Drone video
    # carries no intrinsics, and bundle adjustment cannot recover the focal
    # here because its points were triangulated with the seed -- see
    # geometry.focal_from_flow. Measured 2698 px against a 2132 px guess on
    # the sample flight: a 27% error in the one number that scales every
    # horizontal distance and every triangulated depth.
    # Plane-sweep MVS refinement (geometry.plane_sweep): replace the
    # backbone's REGRESSED depth with depth MEASURED by matching pixels
    # between the views of a window.
    #
    # This is the only step in the pipeline that improves depth PRECISION.
    # depth_anchor and reanchor both correct scale; photometric
    # verification only deletes. A pixel wrong by 0.4 m consistently with
    # its neighbours passes all of them -- and that residual is the 0.38 m
    # planarity noise that forces a coarse voxel and leaves roofs as bumps.
    #
    # Triangulated depth error is Z^2/(fx*B) per pixel of disparity error:
    # 0.23 m per pixel on this footage, ~0.05 m at sub-pixel. The sweep
    # runs on the GPU and only over a narrow band around the anchored
    # depth, so it buys precision rather than range.
    # Fall back to GPS altitude when parallax anchoring cannot measure a
    # window. Parallax is preferred (it assumes nothing about terrain
    # shape) but needs feature matching to succeed; measured on real
    # footage, 15 of 19 windows produced zero parallax samples and kept
    # depth 5-6x too shallow. Every keyframe has an altitude, so this
    # cannot fail the same way.
    gps_altitude_anchor: bool = True

    plane_sweep: bool = True
    # "auto" (default) and "mvs3d": measured per-camera stereo with
    # cross-view validation and volumetric fusion, for every flight direction.
    # "full3d": learned depth with volumetric fusion (legacy optional path).
    # "heightfield" forces 2.5D. "mapanything" is a legacy alias of full3d.
    dense_method: str = "auto"
    heightfield_cell_m: float = 0.5
    plane_sweep_hypotheses: int = 48
    plane_sweep_range_fraction: float = 0.15
    # NCC below which a match is not believed. Pixels that fail keep the
    # backbone's depth rather than being filled with a guess.
    plane_sweep_min_ncc: float = 0.5

    focal_from_flow: bool = True

    yaw_from_flow: bool = True
    # Long side the keyframes are downscaled to for the flow measurement.
    # Direction is all that is needed, and it is stable well below 4K.
    yaw_flow_max_size: int = 640

    # Cap on points kept per window (uniform random subsample above it).
    # Bounds fusion memory independently of resolution and of how much of
    # each frame the backbone keeps. 2.5M x 19 windows fits a 16 GB
    # machine with the float32 photometric accumulators; the pre-TSDF voxel cap is 1.5M for the whole cloud,
    # so this never limits what reaches the mesh. 0 disables.
    max_points_per_window: int = 2_500_000

    # MapAnything's edge mask: drop pixels at depth discontinuities. On
    # depth that is noisy AND at the wrong scale, the relative threshold
    # flags most of the frame as "edge" -- the likely source of the
    # measured 13-25 m usable footprint from a 216 m theoretical one.
    # Off by default: the pipeline has its own evidence-based rejection
    # (photometric verification), which deletes on disagreement between
    # photographs rather than on the backbone's own smoothness.
    mapanything_mask_edges: bool = False
    # Score each pixel by CROSS-VIEW DEPTH AGREEMENT instead of the
    # network's own confidence head. MapAnything projects every pixel's
    # depth into the other views of the window and measures whether they
    # concur -- evidence, not self-assessment.
    #
    # This is the same idea as fusion.photometric, computed where it
    # actually works. Downstream, 96% of points come back UNVERIFIABLE
    # because the 40 sampled keyframes are spread over a 755 x 907 m site
    # so few points get the 3 views needed to judge them. Inside a window
    # all 8 views are present and pixel-aligned.
    # OFF by default, on measurement. Cross-view depth agreement is the
    # right KIND of evidence, but it is far too strict for single-pass
    # survey geometry: at ~23 m keyframe spacing most pixels are not seen
    # consistently by enough views to pass, so they are discarded rather
    # than kept as uncertain. Measured on the sample flight, enabling it
    # took occupied ground from 36.27 ha to 6.15 ha and left only 39 of 77
    # cameras with any reconstructed surface near them -- an 84% loss of
    # coverage, which is 20% of the problem statement's score.
    #
    # Worth revisiting with denser keyframes, where the views exist to
    # make the agreement test meaningful.
    mapanything_multiview_confidence: bool = False
    # Zero the lowest-confidence percentile at the source. None keeps every
    # pixel and lets the confidence tiers downstream decide. 10 discards
    # the worst tenth, which is where the depth noise that inflates the
    # mesh surface area lives.
    # None: keep every pixel and let the confidence TIERS downstream carry
    # the uncertainty, rather than deleting at the source. Discarding a
    # percentile here is irreversible and, on this footage, removed ground
    # that nothing downstream could recover.
    mapanything_confidence_percentile: float | None = None

    # Target window size before geometry.windows.plan_window_size's memory
    # -budget/keyframe-count capping (see that function's docstring for why
    # 5 -- the old default -- gave needlessly many, poorly-conditioned
    # submap junctions; Task 3 of the submap-merge rotation fix).
    window_size: int = 14
    max_image_size: int = 924

    # Window-level parallelism across GPUs.
    #
    # Windows are independent by construction: each one is a self-contained
    # backbone call whose output is aligned to the others afterwards, in
    # `geometry.submap.merge_submaps`. So on a machine with N GPUs, N
    # windows can be in flight at once, each on its own device.
    #
    # `max_devices = 0` means "use every GPU present" -- one on a laptop or
    # the 6 GB RTX 4060 target, two on a Kaggle T4 x2 node, more on a
    # workstation. Set it to 1 to force the historical single-device path
    # (useful for reproducing a run exactly, since floating-point reduction
    # order inside the merge can differ with completion order), or to a
    # smaller number than you have when another process already holds VRAM.
    #
    # This is NOT model parallelism: the backbone is replicated per device,
    # not sharded, so each GPU needs enough VRAM for a whole window on its
    # own. Two 16 GB cards run two 16 GB-class jobs; they do not add up to
    # one 32 GB card.
    max_devices: int = 0

    # Decode the next window's frames on a background thread while the GPU
    # works on the current one. Video decode is CPU/IO-bound and inference
    # is GPU-bound, so without this the GPU idles through every H.264 seek
    # and decode. Helps on every machine including single-GPU and CPU-only,
    # and changes no result -- only when the bytes arrive.
    prefetch_frames: bool = True


@dataclass
class MatchingConfig:
    """Feature matching / bundle-adjustment tuning.

    See ``geometry.features``, ``geometry.tracks``, ``geometry.triangulate``,
    and ``geometry.bundle`` module docstrings for what each knob actually
    controls; this dataclass exists so a single ``Config.quality_profile``
    choice (see ``QUALITY_PROFILES`` / ``apply_quality_profile`` below) can
    retune feature count, detection resolution, point budget, and iteration
    caps together as one coherent speed/accuracy trade-off, instead of
    editing several hardcoded constants scattered across the pipeline.

    This lives in ``config.py`` (this workstream's file) rather than in
    ``pipeline.stages`` (a concurrently-developed workstream, which
    currently hardcodes its own equivalent module constants --
    ``_MATCH_METHOD``, ``_MATCH_MAX_FEATURES``, etc. -- see that module's
    docstring) precisely so both workstreams can coordinate through this
    one file: the field names below intentionally mirror those constants
    1:1 so wiring ``MatchingStage``/``BundleAdjustmentStage`` to read from
    here instead is a mechanical substitution, not a redesign.

    method / max_features / ratio / window / gps_radius_m /
    min_verified_inliers / min_track_length / min_triangulation_angle_deg /
    max_reprojection_px / min_tracks_for_ba:
        Equivalent to ``pipeline.stages``' ``_MATCH_METHOD``,
        ``_MATCH_MAX_FEATURES``, ``_MATCH_RATIO``, ``_MATCH_WINDOW``,
        ``_MATCH_GPS_RADIUS_M``, ``_MIN_VERIFIED_INLIERS``,
        ``_MIN_TRACK_LENGTH``, ``_MIN_TRIANGULATION_ANGLE_DEG``,
        ``_MAX_REPROJECTION_PX``, ``_MIN_TRACKS_FOR_BA`` respectively --
        same defaults, so picking ``quality_profile="balanced"`` (the
        ``Config`` default) reproduces today's behaviour exactly.
    detect_scale:
        Passed straight through to ``geometry.features.detect_and_describe``.
        ``1.0`` (default, "balanced"/"accurate") detects at full
        resolution; a "fast" profile lowers this (see that function's
        docstring for the measured speedup and the accuracy tradeoff it
        documents -- report before/after reprojection error before
        shipping a non-default value).
    max_points_in_ba:
        Passed straight through to
        ``geometry.triangulate.filter_by_reprojection``'s ``max_points``.
        ``None`` (default, "accurate") keeps every track that survives
        the reprojection cut; "fast"/"balanced" cap it, trading the
        long tail of marginal, barely-3-view points (which cost bundle
        adjustment optimizer parameters and residual rows without adding
        much pose-constraining information -- see that function's
        docstring) for a smaller, cheaper problem at ~unchanged pose
        accuracy.
    ba_max_iterations / ba_max_nfev:
        Passed straight through to ``geometry.bundle.BAConfig.max_iterations``
        / ``max_nfev``. See that class's docstring for why ``max_nfev``
        should stay a small constant multiple of ``ba_max_iterations``
        (``None`` derives it that way) rather than scaling with the
        number of points/cameras.
    """

    # "disk" = DISK features + LightGlue (geometry.learned_matching; needs
    # the `matching` extra, falls back to SIFT without it). Measured on
    # flight01: 690 vs 60 verified inliers on the same pair.
    method: str = "disk"
    max_features: int = 4000
    detect_scale: float = 1.0
    # DISK/LightGlue budget: half-resolution, 1024 keypoints -- ~0.2 s per
    # frame and ~0.15 s per pair on an M-series Mac.
    learned_max_features: int = 1024
    learned_detect_scale: float = 0.5
    # Drop a verified pair whose relative rotation misses the IMU/gimbal
    # gravity direction by more than this (degrees). None disables it.
    gravity_check_deg: float | None = 3.0
    ratio: float = 0.8
    window: int = 3
    gps_radius_m: float = 15.0
    # Loop pairs within this share of the measured along-track footprint
    # (triage.footprint), tying a mapping grid's neighbouring strips
    # together; 0 = off (see pipeline.stages._MATCH_FOOTPRINT_FRACTION).
    strip_tie_footprint_fraction: float = 0.0
    # GPS altitude sigma as a multiple of the horizontal one in bundle
    # adjustment (geometry.bundle.BAConfig.gps_vertical_sigma_factor).
    gps_vertical_sigma_factor: float = 1.0
    min_verified_inliers: int = 8
    min_track_length: int = 3
    min_triangulation_angle_deg: float = 1.5
    max_reprojection_px: float = 15.0
    min_tracks_for_ba: int = 20
    max_points_in_ba: int | None = None
    ba_max_iterations: int = 100
    # Re-run bundle adjustment AFTER geometry even though PosePriorStage
    # already refined the poses before it. Off by default: geometry
    # anchors its submap merge on the prior's poses, so the second solve
    # re-derives an answer it already has. Measured: prior converged to
    # 0.63 px in 168 s; the second pass then ran 66 minutes without
    # finishing. Turn on to diagnose a merge suspected of moving cameras.
    redundant_ba_after_geometry: bool = False
    # Let bundle adjustment solve for focal length instead of trusting the
    # intrinsics prior. Drone video rarely carries real intrinsics, so that
    # prior is a generic HFOV guess -- measured 1.57x wrong on the sample
    # footage (2132 px assumed, 3355 px actual), which scales every
    # horizontal measurement and every triangulated depth by the same
    # factor. Only applied where GPS camera priors exist to break the
    # focal/depth ambiguity (see PosePriorStage).
    refine_intrinsics: bool = True
    # Solve one shared radial lens distortion (k1, k2) in the pose-prior
    # bundle adjustment when the lens has no known distortion, then
    # undistort every frame used downstream. An unmodelled lens bends a
    # nadir strip into a bowl: flight01 (k1 ~ -0.2) solved to cameras
    # tilted up to 33 deg and 20-28 m off GPS at 0.4 px reprojection.
    refine_distortion: bool = True
    ba_max_nfev: int | None = None


# ---------------------------------------------------------------------------
# Quality/speed profiles
# ---------------------------------------------------------------------------
#
# A named preset of ``MatchingConfig`` overrides, applied on top of its
# defaults by ``apply_quality_profile``. "balanced" is deliberately every
# ``MatchingConfig`` field's own default (i.e. a no-op override dict) so it
# reproduces the pipeline's original, un-tuned behaviour exactly -- "fast"
# and "accurate" are the two directions away from that baseline.
#
# Numbers below are grounded in profiling real 4K drone footage (8/16/32
# keyframes, MacBook Air M5, see the bundle-adjustment/matching profiling
# notes) rather than guessed: "fast"'s ``detect_scale=0.5`` measures at
# roughly a 4x per-frame SIFT-detection speedup (pixel count scales with
# the square of linear resolution), and ``max_points_in_ba`` caps are sized
# well above what a single flight's BA problem needed in practice to reach
# its converged reprojection error, not picked arbitrarily.
QUALITY_PROFILES: dict[str, dict] = {
    "fast": {
        "max_features": 2000,
        "detect_scale": 0.5,
        # 6000, not 2000: on flight01 the smaller budget left the median
        # camera 15 observations and headings 8 deg off; 6000 gave 58 and
        # halved the heading error against surveyed points, at ~2 min.
        "max_points_in_ba": 6000,
        "ba_max_iterations": 50,
    },
    "balanced": {},
    "accurate": {
        "max_features": 6000,
        "detect_scale": 1.0,
        "max_points_in_ba": None,
        "ba_max_iterations": 150,
    },
}


def _apply_profile(instance: object, profiles: dict[str, dict], profile: str) -> object:
    """Generic ``dataclass_replace(instance, **profiles[profile])`` with a loud unknown-profile error.

    Shared by ``apply_quality_profile`` (``MatchingConfig``),
    ``apply_triage_quality_profile`` (``TriageConfig``), and
    ``apply_geometry_quality_profile`` (``GeometryConfig``) -- same
    "balanced is a no-op, fast/accurate are named override dicts" pattern
    for all three, so a single ``Config.quality_profile`` choice retunes
    keyframe spacing and backbone resolution alongside matching/BA, not
    just the latter.
    """
    if profile not in profiles:
        raise ValueError(f"unknown quality_profile {profile!r}; expected one of {sorted(profiles)}")
    overrides = profiles[profile]
    return dataclass_replace(instance, **overrides)


def apply_quality_profile(matching: MatchingConfig, profile: str) -> MatchingConfig:
    """Return a new ``MatchingConfig`` with ``QUALITY_PROFILES[profile]``'s overrides applied.

    Raises ``ValueError`` for an unknown profile name rather than silently
    falling back to "balanced" -- a typo'd ``quality_profile`` in a saved
    YAML config should fail loudly, not quietly run at different settings
    than the operator asked for.
    """
    return _apply_profile(matching, QUALITY_PROFILES, profile)  # type: ignore[return-value]


# Fix 1 (see TriageConfig.baseline_to_altitude_ratio's docstring): "balanced"
# is deliberately every TriageConfig field's own default (0.10 ratio / 2.0m
# floor -- the fragmentation-fix value, chosen from a measured continuity
# sweep, not the earlier 0.25 depth-precision-only value), same no-op-preset
# pattern as QUALITY_PROFILES above. Both "fast" and "accurate" are grounded
# in the same sweep table (see the docstring): "fast" (0.15) is the widest
# ratio the sweep measured before continuity meaningfully degrades (still
# ~92% largest-component fraction, fewer/wider-spaced keyframes -> fewer
# submap windows -> fewer backbone calls); "accurate" (0.06) is the
# narrowest ratio swept, trading some speed (more, closer-spaced keyframes)
# for the sweep's single best-measured continuity figure (~96% largest
# -component fraction, tied with "balanced").
TRIAGE_QUALITY_PROFILES: dict[str, dict] = {
    "fast": {"baseline_to_altitude_ratio": 0.15, "min_baseline_m": 3.0},
    "balanced": {},
    "accurate": {"baseline_to_altitude_ratio": 0.06, "min_baseline_m": 1.2},
}


def apply_triage_quality_profile(triage: TriageConfig, profile: str) -> TriageConfig:
    """Return a new ``TriageConfig`` with ``TRIAGE_QUALITY_PROFILES[profile]``'s overrides applied."""
    return _apply_profile(triage, TRIAGE_QUALITY_PROFILES, profile)  # type: ignore[return-value]


# Fix 2 (see GeometryConfig.max_image_size's docstring for the measured
# wall-clock numbers behind these three choices): "balanced" is again a
# no-op (924px, this project's measured sweet spot on the dev machine);
# "fast" keeps the pre-fix 518px for callers who need the old speed;
# "accurate" is the textbook-larger 1288px for callers with the time
# budget for the better depth precision it measures out to.
GEOMETRY_QUALITY_PROFILES: dict[str, dict] = {
    # Plane sweep off: on DJI_1001 (~280 m range) it cost ~50 s of each
    # ~65 s window, passed its NCC gate on 0.3-3.8% of pixels and moved
    # those by a median 39 m -- slower AND noisier than the BA-fitted depth.
    "fast": {"max_image_size": 518, "plane_sweep": False},
    "balanced": {},
    "accurate": {"max_image_size": 1288},
}


def apply_geometry_quality_profile(geometry: GeometryConfig, profile: str) -> GeometryConfig:
    """Return a new ``GeometryConfig`` with ``GEOMETRY_QUALITY_PROFILES[profile]``'s overrides applied."""
    return _apply_profile(geometry, GEOMETRY_QUALITY_PROFILES, profile)  # type: ignore[return-value]


@dataclass
class FusionConfig:
    """Multi-view fusion / point-cloud merging settings.

    ``voxel_size`` defaults to ``None`` ("auto"): ``fusion.tsdf.fuse_submaps``
    derives it from the actual ground sample distance (GSD) -- see that
    module's ``_derive_voxel_size`` -- rather than the scene's overall
    bounding-box extent. The old bounding-box heuristic (``max_extent /
    200``, still kept as ``_auto_voxel_size``, only used as a last-resort
    fallback when GSD can't be computed at all) is scale-blind: on a real
    259 m-extent suburban survey it picked voxel=1.30 m -- wider than an
    entire house -- and erased every building into a smooth blob. GSD ties
    voxel size to what the camera/altitude combination can actually
    resolve instead, which is invariant to how large an area the whole
    flight happened to cover. Set an explicit value here to override
    GSD-derived auto-sizing.

    ``voxel_size_gsd_multiplier``:
        How many ground-sample-distances wide one voxel should be. ``3.0``
        by default -- fine enough to keep roof edges/kerbs/driveways
        (features a few GSDs wide) as distinct structure, coarse enough to
        stay robust to per-point depth noise (a voxel exactly 1 GSD wide
        would be as noisy as the raw depth estimate itself).
    ``voxel_count_budget``:
        Hard cap on how many voxels the dense TSDF grid may contain,
        checked against the *ideal* GSD-derived size's implied grid
        dimensions for this cloud's own bounding box. A GSD-correct voxel
        size for a wide-area (hundreds of metres) flight can imply tens to
        hundreds of millions of voxels (measured on a real 258x280x57 m
        flight extent: the true GSD-ideal voxel, ~0.17 m, implies roughly
        800M+ voxels for that scene) -- far more than even this project's
        vectorized numpy TSDF fallback (no open3d in this environment; see
        ``fusion.tsdf.TSDFVolume.integrate_point_cloud``'s docstring) can
        integrate *and mesh-extract* within any sane time budget. When the
        ideal size would exceed this budget, ``_derive_voxel_size`` coarsens
        just enough to fit and logs a prominent warning naming both the
        ideal and actual voxel size -- detail is sacrificed for memory/time,
        on purpose and loudly, rather than the pipeline hanging or OOMing.

        Fix 4 (mesh speed): raised from 150,000 to **2,000,000** on the
        strength of ``integrate_point_cloud``'s vectorization (chunked
        pair-flattening + ``np.bincount`` instead of a per-voxel Python
        loop + ``np.add.at``, plus ``workers=-1`` on the underlying
        ``cKDTree.query_ball_point`` call -- see that method's docstring).
        Measured on this project's own real merged point cloud (57,580
        points from a real 16-keyframe run) at the *new* budget's scale
        (~2.6M actual voxels after margin padding): integration takes ~5.3s
        and marching-tetrahedra mesh extraction ~8.4s, both comfortably
        inside the pipeline's time budget -- versus the old 150,000-voxel
        budget's ~4-6s combined at a 13x coarser voxel size. That is roughly
        a 2.4x finer voxel (cube root of the 13.3x budget increase) for the
        *same* scene -- real, measured progress toward the GSD ideal, but
        an honest one: at this scene's actual (unusually large, 258x280 m)
        extent, even 2,000,000 voxels is still far short of the true
        GSD-ideal ~0.17 m/800M-voxel size, which would need on the order of
        40+ minutes of marching-tetrahedra mesh extraction alone at this
        implementation's measured per-voxel cost (extraction time scales
        with *total* grid cells, not just occupied ones, and was not itself
        vectorized further by this fix). Smaller, more typical scene
        extents reach proportionally finer voxels within the same budget.
        Raise this further (measure first -- ``mesh extraction`` time
        scales roughly linearly with actual voxel count on this backend)
        if more time budget is available; the old, much slower per-voxel
        loop measured well under 2,000 voxels/second on real, dense drone
        footage, which is what originally calibrated 150,000 as the ceiling
        a 10-minute total pipeline run could afford.
    ``pre_tsdf_downsample_voxel_fraction``:
        Fix 3: the cleaned cloud is always downsampled to
        ``voxel_size * pre_tsdf_downsample_voxel_fraction`` (default 0.25,
        i.e. 1/4 of the final TSDF voxel size) before TSDF integration --
        decoupled from (always a small fraction of, never equal to)
        ``voxel_size`` itself, so this can never discard information the
        final mesh could have represented anyway, but bounds
        ``TSDFVolume.integrate_point_cloud``'s per-voxel KD-tree query cost
        (which scales with *local point density*, not just voxel count)
        regardless of how dense the raw cleaned cloud is. Measured on real
        footage: a single 16-keyframe run produced over 1.1 million cleaned
        points, which made per-voxel ball queries take minutes without
        this step. Set to ``0`` to disable (feed the TSDF every cleaned
        point directly) -- only advisable for small/synthetic clouds.
    ``pre_tsdf_max_points``:
        Hard backstop on top of the fraction-based rule above (default
        1,500,000): only fires when even ``voxel_size *
        pre_tsdf_downsample_voxel_fraction`` still leaves more points than
        this, and unlike that rule, coarsening here *does* trade away some
        resolvable detail for a bounded TSDF integration time -- logged
        loudly when it fires, same spirit as ``voxel_count_budget``'s
        clamp. The exported raw ``point_cloud.ply``/``.las`` (see
        ``fusion.tsdf.fuse_submaps``'s ``raw_point_cloud`` return value) is
        captured *before* either of these downsampling steps, so it always
        carries the full cleaned density regardless.
    ``outlier_k``:
        Neighbour count ``fusion.filters.statistical_outlier_removal``
        uses (default 20, the classic PCL/Open3D default) -- exposed here
        (previously hardcoded in ``fuse_submaps``, ignoring any config)
        so it can be tuned alongside ``outlier_std_ratio`` below.

    Confidence-tier thresholds
    ----------------------------
    See ``fusion.tsdf`` module docstring for the full "combined rule" this
    reconciles: a per-point raw ``[0, 1]`` backbone confidence is first
    quantized to the tiered ``Confidence`` enum (``raw_confidence_*_min``
    below), *then* used as that point's TSDF integration weight, and
    finally the TSDF's own accumulated-weight/observation-count bookkeeping
    (``measured_min_*`` below) decides per-*voxel* MEASURED-ness from
    however many of those already-tiered points actually landed there.

    ``raw_confidence_measured_min`` / ``raw_confidence_low_min``:
        Thresholds mirror ``geometry.backbone.NullBackbone``'s own
        reference confidence values (0.95 solid-surface hit / 0.75 ground
        hit / 0.05 miss): a hit confident enough to be a real, un-occluded
        surface sample (>= 0.8) quantizes to MEASURED, a plausible but
        weaker hit (>= 0.3) to LOW_CONFIDENCE, everything else to
        INFERRED. A real backbone's confidence calibration will differ, but
        these are the best default available without per-backbone tuning.
    ``measured_min_views``:
        Minimum number of distinct point *observations* (not integration
        calls -- see ``fusion.tsdf.TSDFVolume.integrate_point_cloud``'s
        docstring) that must have actually contributed nonzero weight to a
        voxel before it can be called MEASURED. A single grazing or
        one-off observation should never be enough, no matter how
        confident it was -- MEASURED is meant to mean "corroborated by
        several independent views," not "one high-confidence guess." 3 is
        the smallest number that is still meaningfully "several" (more
        than a simple pair, which could still be an alignment coincidence)
        while remaining reachable for real overlap ratios.
    ``measured_min_weight``:
        Minimum *total* accumulated integration weight (sum of each
        contributing point's tiered weight -- MEASURED=2, LOW_CONFIDENCE=1,
        INFERRED=0 -- times its distance falloff) for a voxel to be called
        MEASURED. Set to ``1.5 * measured_min_views`` by default: with
        falloff near 1 for nearby points, ``measured_min_views`` points
        that are *mostly* raw-quantized LOW_CONFIDENCE (weight <= 1 each)
        cap out at ``measured_min_views * 1`` and fail this bar, while
        ``measured_min_views`` points that are *mostly* raw-quantized
        MEASURED (weight <= 2 each) clear ``measured_min_views * 2`` and
        pass it comfortably -- i.e. a voxel needs both enough corroborating
        views *and* those views to have themselves mostly been
        individually high-confidence, not merely dense. This is the actual
        reconciliation between the two confidence signals; see the
        ``fusion.tsdf`` module docstring for the failure mode this
        prevents (dense-but-low-confidence geometry silently promoted to
        MEASURED by weight alone).
    ``covariance_measured_max_m`` / ``covariance_low_confidence_max_m``:
        Metre thresholds for the preferred, more principled confidence
        source when available: BA covariance (see
        ``geometry.covariance.confidence_from_covariance``). Mirror that
        module's own ``ConfidenceThresholds`` defaults (5 cm / 50 cm
        largest 1-sigma semi-axis) so overriding them here doesn't require
        reaching into ``geometry.covariance`` directly.
    """

    voxel_size: float | None = None
    # Terrain/auto modes only (full3d never collapses ground columns): a
    # nadir flight that reached the backbone path -- no solved camera for
    # every keyframe, so no height-field stereo -- is fused by median height
    # map, as before full-3D mode existed.
    heightmap_for_nadir: bool = True
    voxel_size_gsd_multiplier: float = 3.0
    # 20M, up from 2M. The 2M figure was calibrated for the pure-numpy
    # dense TSDF; with open3d installed (core dependency) integration is
    # block-sparse and tiled, and the measured throughput (65k-1.16M
    # voxels/s) makes tens of millions affordable. At true survey scale
    # the sample flight covers ~250k m^2; 2M voxels forced a 1.72 m voxel
    # against a 0.14 m GSD-ideal, and the mesh looked like it.
    voxel_count_budget: int = 20_000_000
    pre_tsdf_downsample_voxel_fraction: float = 0.25
    # Raised with voxel_count_budget: the pre-TSDF point ceiling must not
    # be coarser than the voxel grid it feeds (1.5M points over 250k m^2
    # is one point per 0.4 m, coarser than a 0.3 m voxel).
    pre_tsdf_max_points: int = 6_000_000
    outlier_k: int = 20
    outlier_std_ratio: float = 2.0
    min_confidence: int = 1

    raw_confidence_measured_min: float = 0.8
    raw_confidence_low_min: float = 0.3
    measured_min_views: int = 3
    measured_min_weight: float = 4.5
    # --- mesh cleanup (see fusion.mesh.clean_mesh) -------------------
    # Poisson meshing of a single-pass aerial cloud leaves spike slivers
    # across depth discontinuities and speck islands behind them. These
    # dominate how noisy the model LOOKS, independently of how accurately
    # it measures. Cleanup only ever deletes faces -- it never smooths or
    # moves a vertex, because smoothing improves appearance at the direct
    # cost of every measurement taken off the surface.
    # --- photometric verification (see fusion.photometric) -------------
    # Re-grade confidence against the SOURCE FRAMES instead of trusting the
    # backbone's self-reported confidence. A point several views agree
    # about has been checked against several photographs; a point they
    # disagree about is wrong however sure the network was. Costs one
    # projection pass per keyframe and is the only independent evidence
    # this pipeline has about whether its geometry is real.
    photometric_verify: bool = True
    # Views must agree to within this mean absolute colour deviation
    # (0-255). ~12 tolerates JPEG noise and mild exposure drift while
    # still rejecting a point that projects onto different surfaces.
    photometric_agree_threshold: float = 12.0
    # Minimum views before agreement means anything. Two views agree by
    # coincidence far more easily than four.
    photometric_min_views: int = 3
    # Texture contrast floor. A blank road agrees across every view
    # regardless of whether its geometry is right, so below this the point
    # is reported UNVERIFIABLE rather than verified.
    photometric_min_contrast: float = 6.0
    # Reject photometrically inconsistent points BEFORE meshing, instead of
    # labelling them afterwards.
    #
    # This is the difference between a check that grades the output and one
    # that shapes it. With this off, verification still runs, still
    # correctly identifies the points the source frames disagree about, and
    # still writes LOW_CONFIDENCE onto them -- but the mesh was already
    # built from those same points, so the surface carries every error the
    # check just found. Measured on the 956 px run: 59.2% of points
    # photometrically inconsistent, all of them meshed, producing a surface
    # whose median vertex sat 5.5 m above its own local ground on terrain
    # that is mostly flat.
    #
    # Only points with enough views AND enough texture to judge are
    # rejected. UNVERIFIABLE points -- too few views, or a blank surface
    # that agrees regardless of whether its geometry is right -- are kept,
    # because absence of evidence is not evidence of error and deleting on
    # that basis would silently punch holes in every featureless road.
    photometric_reject_before_mesh: bool = True

    # Pin every submap's merge scale to 1.0 instead of re-fitting it from
    # camera centres. Correct whenever the geometry backbone is metric
    # (MapAnything conditioned with real intrinsics is); wrong for a
    # scale-free backbone like plain VGGT/Pi3, which genuinely needs a
    # scale recovered from GPS.
    #
    # The failure this prevents is not subtle. Scale is constrained by the
    # horizontal extent of a window's camera track (tens of metres); it
    # multiplies the camera-to-ground distance (~120 m on a survey). So a
    # scale error far inside the camera fit's own tolerance relocates the
    # ground by tens of metres. Measured on a 77-keyframe flight: ground
    # reconstructed at two elevations 54 m apart, mesh split into 68
    # disconnected components, while every junction RMSE stayed at
    # 0.1-0.35 m and camera centres tracked GPS exactly -- the cameras were
    # right and the ground was wrong.
    #
    # Default False, after measurement. MapAnything reports is_metric=True,
    # but on real 120 m AGL footage its camera-to-ground distance came out
    # at 0.15-0.59x the GPS-confirmed altitude -- so trusting the claim and
    # locking scale left the ground split 50 m. Re-fitting scale from GPS
    # is the safer default; flip this on only for a backbone whose metric
    # output has been verified on the footage at hand.
    lock_metric_scale: bool = False

    # Rescale each submap so its reconstructed flying height matches
    # telemetry `alt_rel` (see geometry.submap.altitude_anchor_scale).
    #
    # OFF by default because `alt_rel` is height above the TAKEOFF POINT,
    # not above the ground being photographed. On a terrain-following
    # flight over sloping ground -- the sample footage descends 38 m of
    # alt_rel while holding altitude -- the two differ by the terrain
    # offset, and "correcting" scale to match alt_rel would bake that
    # offset into the model as a scale error. The measurement is still
    # computed and reported per submap as a diagnostic; it just does not
    # move anything unless an operator who knows the flight was flat
    # turns it on.
    altitude_anchor: bool = False

    # Second depth-anchoring pass against the bundle adjustment's sparse
    # points, per view, before merging (fusion.reanchor). The first pass
    # (geometry.depth_anchor) triangulates its own feature pairs and is
    # noisy where the backbone's depth is badly compressed; BA's multi-
    # view points at ~1.7 px are the better reference once they exist.
    # Refine each submap's translation against the geometry it shares
    # with its predecessor (geometry.overlap_align). The camera-centre
    # Sim(3) is nearly unconstrained vertically because survey camera
    # centres are a near-flat sheet; the shared ground is strongest
    # exactly there. Measured without it: surfaces duplicated across a
    # 14.8 m-thick slab where a single ground should be ~0.1 m.
    overlap_align: bool = True

    # Voxel size of the running model that world-frame windows are fused
    # into as they finish (fusion.incremental). Preview-grade: the final
    # surface still comes from this stage's own fusion.
    incremental_voxel_m: float = 0.3
    # Cell size keyframe agreement is counted on for the live model's
    # confidence tiers: >= 3 keyframes within one cell is MEASURED. One
    # metre is the problem statement's accuracy requirement.
    incremental_tier_voxel_m: float = 1.0
    # Mesh even when the frame placement check FAILED. Off by default:
    # a failed placement meshes several copies of the same surface.
    allow_failed_placement: bool = False
    ba_reanchor: bool = True

    # Upper bound on mesh faces after extraction and cleanup. 4M is a
    # ~200 MB PLY / ~140 MB GLB, opens in any viewer, and still holds a
    # 0.3-0.5 m surface over a survey-scale site. 0 disables.
    max_mesh_faces: int = 4_000_000

    mesh_cleanup: bool = True
    # Drop faces whose longest edge exceeds this multiple of the mesh's own
    # median edge. Scale-free, so one value works from a courtyard to a
    # corridor. Below ~3 it starts eating legitimate steep geometry.
    mesh_max_edge_factor: float = 6.0
    # Floor on the speck-component threshold, in faces.
    mesh_min_component_faces: int = 64

    covariance_measured_max_m: float = 0.05
    covariance_low_confidence_max_m: float = 0.5


@dataclass
class ExportConfig:
    """Output export settings.

    ``scale_warning_tolerance``:
        Fractional deviation of the recovered georeferencing scale factor
        (``geometry.georef.Sim3.scale``) from ``1.0`` above which
        ``pipeline.stages.ExportStage`` logs a prominent warning (and
        records it in the accuracy report card) instead of silently
        exporting geometry whose metric scale still disagrees with GPS.
        ``0.05`` (5%) by default: bundle adjustment + GPS priors should get
        a healthy reconstruction close enough to metric-correct that
        georeferencing's own Sim(3) fit only needs a small residual
        correction; a larger correction than that means something upstream
        (backbone conditioning, BA priors, ...) left a real scale error
        that this fit is papering over, and that is exactly the silent
        failure mode this project cannot afford -- see
        ``pipeline.stages._apply_georeferencing``.
    """

    output_dir: str = "output"
    export_las: bool = True
    export_ply: bool = True
    crs: str = "EPSG:4978"
    scale_warning_tolerance: float = 0.05


@dataclass
class TextureConfig:
    """Photographic texture baking for the fused mesh.

    The distinction this section exists to make: per-vertex colour (which
    the pipeline already produced) is limited by mesh density, while a
    baked texture atlas is limited by the source video. At TSDF vertex
    spacing, that is the difference between a roof that is uniformly grey
    and a roof whose vents, markings and edges are legible. See
    ``fusion.texture`` for the baking method.

    ``enabled``:
        ``True`` attempts a bake whenever a mesh exists. Falls back to
        per-vertex colour -- silently in terms of correctness, loudly in
        the log -- when ``xatlas`` is not installed.

    ``texture_size``:
        Atlas edge length in texels. Memory is ``3 * size^2`` bytes for
        the atlas plus ``~16 * size^2`` for the position/normal buffers
        during the bake: 4096 needs roughly 300 MB transiently, 8192 needs
        1.2 GB. Above 8192 most viewers and GPUs start refusing the
        texture, so that is the practical ceiling regardless of RAM.

    ``blend_views``:
        How many views may contribute to one texel. 1 is a hard argmax and
        produces visible seams wherever the winning view changes; 4 fades
        between them. Higher values keep softening and eventually average
        genuine detail away.

    ``occlusion_tolerance_m``:
        Depth slack for the visibility test, same meaning and the same
        shared implementation as ``SemanticsConfig.occlusion_tolerance_m``.

    ``max_views``:
        Cap on keyframes considered. Bake cost is linear in views, and
        beyond a few dozen well-distributed views additional ones mostly
        re-observe surfaces already covered by a better view. ``0`` means
        no cap.
    """

    enabled: bool = True
    texture_size: int = 4096
    blend_views: int = 4
    occlusion_tolerance_m: float = 0.5
    max_views: int = 120


@dataclass
class SemanticsConfig:
    """Per-frame semantic segmentation and 3D label propagation.

    This section is what turns a geometrically-correct point cloud into a
    *classified* one, and -- via ``remove_dynamic`` -- what keeps vehicles
    and people out of the reconstruction entirely. See the
    ``drishti3d.semantics`` package docstring for the reasoning behind the
    taxonomy and the multi-view voting scheme; only the knobs are
    documented here.

    ``enabled``:
        Master switch. ``False`` skips ``SemanticsStage`` outright (it
        records itself ``"skipped"``, never ``"failed"``), producing the
        same geometry this pipeline produced before semantics existed.
        Default ``True``, because every required reconstruction target
        past bare terrain is a *semantic* category, not a geometric one.

    ``model``:
        A name registered in ``semantics.segmenter`` -- ``"segformer"``
        (real) or ``"null"`` (labels everything ``UNLABELLED``, always
        available, invents nothing). An unavailable model degrades to
        ``"null"`` with a loud log line rather than failing the run.

    ``checkpoint``:
        HuggingFace id for ``"segformer"``. The default is ADE20K-trained,
        i.e. ground-level photography -- out of domain for nadir aerial
        frames. That domain gap, not the checkpoint's benchmark mIoU, is
        what dominates classification error here; ``min_vote_ratio`` below
        is the mitigation.

    ``max_image_size``:
        Long-side resolution the segmenter runs at. Independent of
        ``geometry.max_image_size``: segmentation is far cheaper per pixel
        than depth regression, and class boundaries benefit from the extra
        resolution, so this defaults *higher* than the geometry backbone's
        working size.

    ``batch_size``:
        Frames per forward pass -- the dominant VRAM term in this stage.
        Safe at 4 for a 6 GB target because ``SemanticsStage`` completes
        and unloads before ``GeometryStage`` loads its own model; the two
        never coexist on the device.

    ``remove_dynamic``:
        Mask ``semantics.classes.EXCLUDED_CLASSES`` (vehicles, people,
        sky) out of every frame *before* geometry runs, so those pixels
        never become 3D points at all. Strictly stronger than filtering
        them out afterwards: a car that never generates points also never
        pollutes the TSDF's weight accumulation, never drags a roof edge
        downward, and never contributes a spurious junction residual. Note
        it removes *parked* vehicles too -- see ``semantics.classes``'
        docstring for why that is deliberate rather than a bug.

    ``dilate_dynamic_px``:
        Grow the dynamic mask by this many pixels before applying it.
        Segmentation boundaries are systematically tight around thin
        structures (a car's roof rack, a person's limbs), and a few
        leftover boundary pixels produce exactly the floating specks that
        make a cloud look unprofessional. Cheap insurance; costs a thin
        rim of genuine ground around each masked object.

    ``min_vote_ratio``:
        A point whose winning class holds less than this share of total
        vote weight is demoted to ``UNLABELLED`` rather than committed to.
        ``0.5`` means "an outright majority, not merely a plurality".
        Raise it when the report card shows a high disputed-point count;
        lowering it below ~0.35 mostly manufactures confident-looking
        labels out of noise.

    ``min_views``:
        Minimum number of views that must have seen a point for it to be
        labelled at all. ``2`` rejects points visible in a single frame,
        which are also the points whose geometry is least trustworthy --
        the two failure modes correlate, so one threshold handles both.

    ``occlusion_tolerance_m``:
        Depth slack in the per-view z-buffer visibility test. Must exceed
        the cloud's own depth noise or a surface will occlude *itself*;
        see ``semantics.labelling.visible_mask``.

    ``use_occlusion``:
        Disable only for debugging. With it off, ground points behind a
        building collect that building's labels.
    """

    enabled: bool = True
    model: str = "segformer"
    checkpoint: str = "nvidia/segformer-b4-finetuned-ade-512-512"
    local_weights: str | None = None
    # Hugging Face token for the CHECKPOINT only. Leave None: the default
    # checkpoint is public and is fetched anonymously, which is what you
    # want even -- especially -- when an HF token is present in the
    # environment for other reasons. A fine-grained token scoped to a
    # different namespace makes the Hub answer 401 for a public model
    # rather than serving it anonymously. Set this only for a private or
    # gated checkpoint.
    hf_token: str | None = None
    max_image_size: int = 1024
    batch_size: int = 4

    remove_dynamic: bool = True
    dilate_dynamic_px: int = 9
    # Skip the stage on downward-looking flights while the checkpoint is a
    # ground-level (ADE20K/Cityscapes) model: on DJI_1001's nadir frames it
    # labelled every pixel building or vegetation and found 0% vehicles,
    # while costing ~0.6 s a frame. Moving objects there are removed by the
    # measured-3D consistency test and the texture's median instead.
    # Forward/oblique footage (in domain: sky, people, cars) still runs it.
    skip_nadir_ground_level: bool = True

    min_vote_ratio: float = 0.5
    min_views: int = 2
    occlusion_tolerance_m: float = 0.5
    use_occlusion: bool = True


# "balanced" is a no-op preset (every SemanticsConfig field's own default),
# same convention as QUALITY_PROFILES/TRIAGE_QUALITY_PROFILES above.
# "fast" drops segmentation resolution and relaxes the agreement threshold
# -- more labelled points, more of them wrong. "accurate" raises both and
# demands three views before committing to any label, which on a
# single-pass flight measurably shrinks the labelled fraction: that is the
# trade, not a defect.
SEMANTICS_QUALITY_PROFILES: dict[str, dict] = {
    "fast": {"max_image_size": 640, "min_vote_ratio": 0.4, "min_views": 1},
    "balanced": {},
    "accurate": {"max_image_size": 1280, "min_vote_ratio": 0.6, "min_views": 3},
}


def apply_semantics_quality_profile(semantics: SemanticsConfig, profile: str) -> SemanticsConfig:
    """Return a new ``SemanticsConfig`` with ``SEMANTICS_QUALITY_PROFILES[profile]``'s overrides applied."""
    return _apply_profile(semantics, SEMANTICS_QUALITY_PROFILES, profile)  # type: ignore[return-value]


@dataclass
class ReferenceConfig:
    """Absolute alignment to a reference orthophoto/DEM (geometry.reference_align).

    Off unless ``ortho_path`` is set: the reference is survey data the
    operator supplies (or fetches with scripts/fetch_reference.py), never
    something the pipeline downloads on its own -- the field deployment is
    air-gapped.
    """

    ortho_path: str | None = None
    dem_path: str | None = None
    gsd_m: float = 0.5
    min_inliers: int = 25
    max_shift_m: float = 30.0
    # "translation" (GPS error is a shift; bare-ground matches when a DEM is
    # given) or "similarity" (4-DoF; scale drifted 1.3% between runs of one
    # model on flight01).
    fit: str = "translation"


@dataclass
class AccuracyConfig:
    """Independent survey validation settings.

    ``control_points_path`` uses the explicit JSON schema documented in
    ``docs/control-points.md``. Controls fit the georeferencing transform;
    checkpoints are withheld and score named reconstructed positions.
    """

    control_points_path: str | None = None
    target_m: float = 1.0
    minimum_checkpoints: int = 3
    history_path: str | None = None
    run_id: str | None = None


@dataclass
class Config:
    """Top-level pipeline configuration.

    quality_profile:
        Selects one of ``QUALITY_PROFILES`` ("fast" / "balanced" /
        "accurate") as the *base* for ``matching``, and (see
        ``TRIAGE_QUALITY_PROFILES``/``GEOMETRY_QUALITY_PROFILES``) for
        ``triage``'s keyframe-baseline spacing and ``geometry``'s backbone
        working resolution too -- see each dataclass's own docstring for
        the per-field reasoning. "balanced" (default) is a no-op preset
        for all three, so leaving every field at its own default
        reproduces this project's current tuned defaults exactly (not the
        pre-Fix-1/2 behaviour -- those fixes changed the defaults
        themselves, not just the profile deltas away from them). Explicit
        fields set directly on ``matching``/``triage``/``geometry`` (e.g.
        via a saved YAML's own section for each) override whatever the
        profile picked -- ``load_config`` applies the profile first, then
        layers explicit per-section overrides on top, so a config can say
        "mostly fast, but keep 6000 features" without needing a fourth
        named profile. Note: only ``load_config`` (YAML path) applies
        ``quality_profile`` this way -- constructing ``Config()`` directly
        in Python does not consult ``quality_profile`` at all and just
        uses each dataclass's own field defaults (which already equal the
        "balanced" profile, by the no-op-preset convention above).
    """

    ingest: IngestConfig = field(default_factory=IngestConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    quality_profile: str = "balanced"
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    semantics: SemanticsConfig = field(default_factory=SemanticsConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    reference: ReferenceConfig = field(default_factory=lambda: ReferenceConfig())
    accuracy: AccuracyConfig = field(default_factory=AccuracyConfig)


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file, filling in defaults for missing keys.

    ``quality_profile`` now retunes ``triage``/``geometry`` (baseline
    spacing / working resolution -- Fixes 1 and 2) the same way it already
    retuned ``matching``: the named profile's overrides are applied first,
    then any explicit ``triage:``/``geometry:`` section in the YAML
    overrides those on top -- same "mostly accurate, but keep this one
    field" layering ``matching`` already supported.
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    quality_profile = raw.get("quality_profile", "balanced")
    matching = apply_quality_profile(MatchingConfig(), quality_profile)
    matching = dataclass_replace(matching, **raw.get("matching", {}))

    triage = apply_triage_quality_profile(TriageConfig(), quality_profile)
    triage = dataclass_replace(triage, **raw.get("triage", {}))

    geometry = apply_geometry_quality_profile(GeometryConfig(), quality_profile)
    geometry = dataclass_replace(geometry, **raw.get("geometry", {}))

    semantics = apply_semantics_quality_profile(SemanticsConfig(), quality_profile)
    semantics = dataclass_replace(semantics, **raw.get("semantics", {}))

    return Config(
        ingest=IngestConfig(**raw.get("ingest", {})),
        triage=triage,
        geometry=geometry,
        quality_profile=quality_profile,
        matching=matching,
        semantics=semantics,
        texture=TextureConfig(**raw.get("texture", {})),
        fusion=FusionConfig(**raw.get("fusion", {})),
        export=ExportConfig(**raw.get("export", {})),
        reference=ReferenceConfig(**raw.get("reference", {})),
        accuracy=AccuracyConfig(**raw.get("accuracy", {})),
    )


def save_config(cfg: Config, path: str | Path) -> None:
    """Save a Config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
