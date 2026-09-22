"""Chain pairwise 2D matches into multi-view feature tracks.

A ``Track`` is the atomic unit of correspondence bundle adjustment needs
(see ``bundle.BAProblem``): a set of ``(camera, pixel)`` observations that
all supposedly look at the *same* 3D point. ``features.py`` only gives us
pairwise correspondences between two images at a time; a point seen across
five keyframes shows up as (up to) ten separate pairwise matches (one per
frame pair) that all have to be recognized as "the same point" and merged
into one five-observation track before triangulation (``triangulate.py``)
or bundle adjustment can use it. That merge is a union-find over
``(frame_idx, keypoint_idx)`` nodes, unioned by accepted matches -- see
``build_tracks`` for the one subtlety that makes a naive version of this
silently wrong.

Why not exhaustive all-pairs matching
----------------------------------------
Running ``features.match_features`` on every pair of keyframes is
``O(n^2)`` in the number of keyframes, which is fine for a handful of
frames but becomes the dominant cost of the whole pipeline once a flight
has hundreds of keyframes (each match call itself is not free: descriptor
matching plus RANSAC per pair). A single-pass drone flight has a specific
structure this project can exploit instead of paying that cost blindly:
consecutive keyframes overlap heavily (that is what makes them
reconstructible at all -- see ``geometry.windows``' module docstring on
overlap), so almost all *useful* correspondence lives within a small
window of temporally-nearby frames. The one thing windowed matching alone
misses is a flight that loops back near its own earlier track (a
lawnmower or racetrack pattern) -- ``select_pairs``'s ``gps_radius_m``
covers that case by adding pairs between keyframes that are far apart in
time but geographically close, without paying the full all-pairs cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.geometry.features import Features, Matches
from drishti3d.ingest.telemetry import telemetry_to_enu
from drishti3d.types import Keyframe

# Frames further apart than this in a track's observation set are still
# just "more evidence for the same point" -- there is no upper bound in
# principle, this constant only documents that ``track_statistics``'s
# per-frame observation counting treats every frame index in
# ``[0, n_frames)`` uniformly regardless of how the track spans them.
_MIN_TRACK_OBSERVATIONS = 2

# Below this many observed tracks, a frame is flagged by
# ``track_statistics`` as a reconstruction-failure early warning (see its
# docstring) -- deliberately generous (a real bundle adjustment wants many
# more than this per frame) since the point is to catch *catastrophic*
# under-coverage (a frame that matched almost nothing), not to grade
# marginal ones.
_MIN_OBS_PER_FRAME_WARNING = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """One multi-view correspondence: the same (believed) 3D point, observed in several frames.

    observations:
        ``[(frame_idx, keypoint_idx, uv), ...]``, sorted by ``frame_idx``,
        with **at most one observation per frame** -- see ``build_tracks``
        for why that invariant is actively enforced rather than assumed.
        ``uv`` is a ``(2,)`` float64 pixel coordinate, copied out of the
        owning frame's ``features.Features.keypoints`` at track-build time
        so downstream code (``triangulate.py``, ``bundle.build_ba_problem``)
        never needs the original ``Features`` objects again.
    color:
        Optional ``(3,)`` uint8 BGR color sampled from the first
        observation's image location, for callers that want to carry
        point-cloud color through triangulation without a second pass over
        the source images.
    reprojection_error_px:
        Populated by ``triangulate.filter_by_reprojection`` (``None``
        before that runs) -- the maximum per-observation reprojection
        error, in pixels, against this track's triangulated 3D point.
        ``filter_tracks``' ``max_reprojection_px`` argument reads this
        field, so reprojection-based filtering only works after
        triangulation has populated it.
    """

    observations: list[tuple[int, int, np.ndarray]]
    color: np.ndarray | None = None
    reprojection_error_px: float | None = None

    def __len__(self) -> int:
        return len(self.observations)

    def frame_indices(self) -> list[int]:
        return [obs[0] for obs in self.observations]


@dataclass
class TrackSet:
    """A collection of ``Track``s, in no particular order."""

    tracks: list[Track] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def select_pairs(
    keyframes: list[Keyframe],
    strategy: str = "sequential+loop",
    window: int = 3,
    gps_radius_m: float | None = None,
) -> list[tuple[int, int]]:
    """Choose which keyframe pairs to attempt matching on.

    See the module docstring for why this is not exhaustive all-pairs
    matching. Two independent sources of pairs, both ``O(n log n)`` or
    better (never ``O(n^2)`` in the number of keyframes):

    - **Sequential** (always): keyframe ``i`` is paired with each of its
      next ``window`` neighbours. This is the backbone of correspondence
      for a single-pass flight -- exactly ``O(n * window)`` pairs.
    - **Loop** (``strategy="sequential+loop"`` and ``gps_radius_m`` given):
      additionally pairs any two keyframes whose GPS fixes lie within
      ``gps_radius_m`` of each other but are further apart than ``window``
      in the sequential ordering -- the "flight passes near its own
      earlier track" case. Found via a KD-tree radius query
      (``scipy.spatial.cKDTree.query_pairs``) over the GPS-fitted
      keyframes' local ENU positions, which is ``O(k log k)`` in the
      number of GPS-tagged keyframes ``k``, not ``O(k^2)`` -- a real
      spatial index, not a nested loop.

    Keyframes without a GPS fix are simply excluded from the loop-closure
    search (they can still participate in sequential pairs); if fewer than
    two keyframes have a fix, the loop-closure pass is skipped entirely
    since there is nothing to radius-query.
    """
    if strategy not in ("sequential", "sequential+loop"):
        raise ValueError(f"unknown pair-selection strategy {strategy!r}; expected 'sequential' or 'sequential+loop'")
    if window < 1:
        raise ValueError("window must be >= 1")

    n = len(keyframes)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            pairs.add((i, j))

    if strategy == "sequential+loop" and gps_radius_m is not None and n >= 2:
        samples = [kf.telemetry for kf in keyframes]
        geo_idx = [i for i, s in enumerate(samples) if s is not None and s.geo is not None]
        if len(geo_idx) >= 2:
            enu, _origin = telemetry_to_enu([samples[i] for i in geo_idx])
            tree = cKDTree(enu)
            close_pairs = tree.query_pairs(r=gps_radius_m)
            for a, b in close_pairs:
                gi, gj = geo_idx[a], geo_idx[b]
                lo, hi = (gi, gj) if gi < gj else (gj, gi)
                if hi - lo > window:  # already covered by the sequential pass otherwise
                    pairs.add((lo, hi))

    return sorted(pairs)


# ---------------------------------------------------------------------------
# Track building (union-find over (frame_idx, keypoint_idx) nodes)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Path-compressing union-find over arbitrary hashable nodes, created lazily on first touch."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def __contains__(self, node: tuple[int, int]) -> bool:
        return node in self._parent

    def touch(self, node: tuple[int, int]) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        self.touch(node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        # Path halving: point every visited node directly at the final root
        # so future finds are cheap.
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union_into(self, keep: tuple[int, int], absorb: tuple[int, int]) -> None:
        """Merge ``absorb``'s group into ``keep``'s, making ``keep``'s root the survivor."""
        self._parent[self.find(absorb)] = self.find(keep)

    def nodes(self):
        return self._parent.keys()


def build_tracks(
    keyframe_images: list[np.ndarray],
    features_per_frame: list[Features],
    pair_list: list[tuple[int, int, Matches]],
) -> TrackSet:
    """Chain verified pairwise matches into multi-view tracks.

    Parameters
    ----------
    keyframe_images:
        Decoded BGR images, one per frame, same order/length as
        ``features_per_frame`` -- used only to sample ``Track.color`` and
        to sanity-check that the caller passed a consistent frame count
        (a mismatch here means a caller bug upstream, not a recoverable
        matching failure, so it raises rather than silently truncating).
    features_per_frame:
        One ``features.Features`` per frame, in frame-index order.
    pair_list:
        ``[(frame_i, frame_j, matches), ...]`` -- the *already
        geometrically-verified* ``features.Matches`` for each attempted
        pair (see ``features.geometric_verify`` and ``select_pairs``).
        Only matches with ``inlier_mask`` set are trusted; a pair whose
        ``Matches.inlier_mask`` is ``None`` (never verified) is treated as
        having zero accepted matches rather than guessed at.

    The classic union-find track-corruption bug (read before changing this function)
    -------------------------------------------------------------------------------
    Chaining pairwise matches transitively can accidentally merge two
    *different* keypoints of the *same* frame into one "track" -- e.g.
    frame A's keypoint 0 matches frame B's keypoint 5 (pair 1), frame B's
    keypoint 5 also matches frame C's keypoint 2 (pair 2, chaining A.0 and
    C.2 together via B.5), and *separately* frame A's keypoint 1 matches
    frame C's keypoint 2 (pair 3, a plausible outcome of a repetitive
    texture or a borderline ratio-test pass). A naive union-find just
    unions every edge it's given and only *afterwards* -- if anyone bothers
    to check at all -- discovers the resulting group contains both A.0 and
    A.1: two different pixels in the same image claimed as the same 3D
    point, which is geometrically nonsensical and, if it reaches
    triangulation, produces silent garbage (whichever observation happens
    to be processed is arbitrarily used or overwritten).

    This function instead validates **before** every union: a proposed
    merge of node ``a``'s group and node ``b``'s group is only accepted if
    the two groups do not already disagree about some frame's keypoint
    (i.e. no frame index maps to two different keypoint indices across the
    two groups). An edge that would violate this is dropped -- the
    surviving parts of both groups (from whichever earlier, non-conflicting
    edges built them) are kept untouched, which is the "split" outcome:
    the bad edge is excised rather than corrupting an otherwise-good chain
    or discarding it wholesale.
    """
    if len(keyframe_images) != len(features_per_frame):
        raise ValueError(
            f"build_tracks: keyframe_images ({len(keyframe_images)}) and features_per_frame "
            f"({len(features_per_frame)}) must have the same length"
        )

    uf = _UnionFind()
    # group_frames[root] maps frame_idx -> keypoint_idx for every
    # observation currently merged under `root`. This is the bookkeeping
    # the pre-union consistency check reads and updates; see docstring.
    group_frames: dict[tuple[int, int], dict[int, int]] = {}

    def _ensure(node: tuple[int, int]) -> None:
        uf.touch(node)
        group_frames.setdefault(uf.find(node), {node[0]: node[1]})

    for frame_i, frame_j, matches in pair_list:
        if matches.inlier_mask is None:
            continue
        for qi, ti, keep in zip(matches.query_idx, matches.train_idx, matches.inlier_mask, strict=True):
            if not keep:
                continue
            node_a = (frame_i, int(qi))
            node_b = (frame_j, int(ti))
            _ensure(node_a)
            _ensure(node_b)

            root_a, root_b = uf.find(node_a), uf.find(node_b)
            if root_a == root_b:
                continue

            frames_a = group_frames[root_a]
            frames_b = group_frames[root_b]
            conflict = any(f in frames_a and frames_a[f] != kp for f, kp in frames_b.items())
            if conflict:
                continue  # reject this edge; both groups keep their existing (consistent) members

            uf.union_into(keep=root_a, absorb=root_b)
            frames_a.update(frames_b)
            del group_frames[root_b]

    tracks: list[Track] = []
    emitted_roots: set[tuple[int, int]] = set()
    for node in uf.nodes():
        root = uf.find(node)
        if root in emitted_roots:
            continue
        emitted_roots.add(root)

        frames = group_frames.get(root, {})
        if len(frames) < _MIN_TRACK_OBSERVATIONS:
            continue

        observations = []
        for frame_idx, kp_idx in sorted(frames.items()):
            uv = features_per_frame[frame_idx].keypoints[kp_idx].copy()
            observations.append((frame_idx, kp_idx, uv))

        anchor_frame, _anchor_kp, anchor_uv = observations[0]
        color = _sample_color(keyframe_images[anchor_frame], anchor_uv)

        tracks.append(Track(observations=observations, color=color))

    return TrackSet(tracks=tracks)


def _sample_color(image: np.ndarray, uv: np.ndarray) -> np.ndarray | None:
    if image is None:
        return None
    h, w = image.shape[0], image.shape[1]
    x = int(np.clip(round(float(uv[0])), 0, w - 1))
    y = int(np.clip(round(float(uv[1])), 0, h - 1))
    pixel = image[y, x]
    return np.asarray(pixel, dtype=np.uint8).reshape(-1)[:3].copy()


# ---------------------------------------------------------------------------
# Filtering and diagnostics
# ---------------------------------------------------------------------------


def filter_tracks(trackset: TrackSet, min_length: int = 3, max_reprojection_px: float | None = None) -> TrackSet:
    """Drop poorly-constrained or geometrically-inconsistent tracks.

    min_length:
        A track seen in only 1-2 frames is poorly constrained for
        triangulation: 2 views give exactly one triangulated point with no
        redundancy to detect a bad match (any two rays that aren't
        perfectly parallel "triangulate" to *some* point, correct or not),
        and bundle adjustment gets no leverage on such a point either --
        with only 2 observations its reprojection residuals can always be
        driven to zero by moving the point, contributing no information
        about camera poses. Requiring >= 3 views (the default) is the
        standard SfM floor for a point to meaningfully constrain anything.
    max_reprojection_px:
        If given, also drops tracks whose ``Track.reprojection_error_px``
        exceeds this. That field is only populated by
        ``triangulate.filter_by_reprojection`` (triangulation has to
        happen first to know where a track's point actually is), so
        passing this before triangulation raises rather than silently
        matching everything.
    """
    kept: list[Track] = []
    for track in trackset.tracks:
        if len(track) < min_length:
            continue
        if max_reprojection_px is not None:
            if track.reprojection_error_px is None:
                raise ValueError(
                    "filter_tracks: max_reprojection_px requires Track.reprojection_error_px to be "
                    "populated first -- run triangulate.filter_by_reprojection before filtering on it"
                )
            if track.reprojection_error_px > max_reprojection_px:
                continue
        kept.append(track)
    return TrackSet(tracks=kept)


def track_statistics(trackset: TrackSet, n_frames: int | None = None) -> dict:
    """Summary statistics used to sanity-check a matching run before triangulation/BA.

    ``frac_frames_too_few_tracks`` is a reconstruction-failure early
    warning: a frame that shares almost no tracks with its neighbours will
    end up with almost no reprojection constraints in the resulting
    ``BAProblem``, meaning its pose is essentially unconstrained by this
    stage no matter how good bundle adjustment's optimizer is -- better to
    surface that here than have it show up later as a wild outlier pose.
    ``n_frames`` should be the total number of frames matching was
    attempted over (not just the ones that ended up with an observation);
    if omitted, it's inferred as ``max(observed frame index) + 1``, which
    *undercounts* a frame that matched literally nothing at the tail end
    of the sequence -- pass it explicitly when that distinction matters.
    """
    lengths = np.array([len(t) for t in trackset.tracks], dtype=np.float64)

    observations_per_frame: dict[int, int] = {}
    for track in trackset.tracks:
        for frame_idx in track.frame_indices():
            observations_per_frame[frame_idx] = observations_per_frame.get(frame_idx, 0) + 1

    if n_frames is None:
        n_frames = (max(observations_per_frame) + 1) if observations_per_frame else 0

    if n_frames > 0:
        thin_frames = sum(1 for f in range(n_frames) if observations_per_frame.get(f, 0) < _MIN_OBS_PER_FRAME_WARNING)
        frac_thin = thin_frames / n_frames
    else:
        frac_thin = 1.0

    return {
        "count": len(trackset.tracks),
        "mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "median_length": float(np.median(lengths)) if lengths.size else 0.0,
        "max_length": float(lengths.max()) if lengths.size else 0.0,
        "observations_per_frame": observations_per_frame,
        "n_frames": n_frames,
        "frac_frames_too_few_tracks": frac_thin,
    }
