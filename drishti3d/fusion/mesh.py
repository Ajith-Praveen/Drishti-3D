"""Mesh-level operations: normal estimation, Poisson reconstruction, decimation, stats.

Everything here consumes/produces plain numpy arrays (``vertices: (V, 3)``,
``faces: (F, 3)`` int, optional ``colors: (V, 3)`` uint8, ``confidence:
(V,)`` uint8 of ``types.Confidence`` values) rather than an open3d mesh
object, so the rest of the pipeline (and every test) never needs open3d
installed to hold a mesh in memory -- only ``poisson_reconstruct`` actually
reaches for open3d, and it has a documented, working numpy fallback.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.types import Confidence, PointCloud

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover - exercised only where open3d is installed
    o3d = None
    _HAS_OPEN3D = False

__all__ = [
    "decimate_to_cap",
    "clean_mesh",
    "compute_mesh_stats",
    "decimate_mesh",
    "estimate_normals",
    "poisson_reconstruct",
]

# Below this many neighbours, a PCA-plane normal estimate is meaningless
# (need >= 3 non-collinear points to define a plane at all); such points
# get an arbitrary but deterministic up-normal instead of failing outright.
_MIN_NEIGHBORS_FOR_PCA = 3

# The fraction of lowest-density vertices a Poisson reconstruction marks
# INFERRED by default -- see poisson_reconstruct's docstring for why this
# exists at all.
_DEFAULT_DENSITY_QUANTILE = 0.1


def estimate_normals(
    pc: PointCloud, k: int = 30, camera_positions: np.ndarray | None = None
) -> np.ndarray:
    """Per-point normals via local PCA, oriented toward the nearest camera when known.

    For each point, fits a plane to its ``k`` nearest neighbours (PCA: the
    normal is the eigenvector of the neighbourhood's covariance matrix with
    the *smallest* eigenvalue, i.e. the direction the neighbourhood is
    flattest along). PCA alone cannot tell "up" from "down" -- the sign is
    arbitrary -- so orientation is resolved afterwards:

    - If ``camera_positions`` is given (typically a submap's/window's
      keyframe camera centres), each normal is flipped, if necessary, to
      point toward its nearest camera. This is the geometrically correct
      thing to do for a surface actually photographed from that camera.
    - Otherwise, normals are flipped to point away from the point cloud's
      centroid -- a reasonable default for a single, roughly convex object
      or terrain patch, but not something to trust for a concave/interior
      scene; callers that care should supply ``camera_positions``.
    """
    n = pc.xyz.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    k_eff = min(k, n - 1)
    if k_eff < _MIN_NEIGHBORS_FOR_PCA:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    tree = cKDTree(pc.xyz)
    _, idx = tree.query(pc.xyz, k=k_eff + 1)
    if idx.ndim == 1:
        idx = idx[:, None]

    neighbors = pc.xyz[idx]  # (n, k+1, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    del eigvals
    normals = eigvecs[:, :, 0]  # smallest-eigenvalue eigenvector, per point

    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm_len > 1e-12, norm_len, 1.0)

    if camera_positions is not None and len(camera_positions) > 0:
        cam_positions = np.asarray(camera_positions, dtype=np.float64)
        cam_tree = cKDTree(cam_positions)
        _, cam_idx = cam_tree.query(pc.xyz, k=1)
        to_cam = cam_positions[cam_idx] - pc.xyz
    else:
        centroid = pc.xyz.mean(axis=0)
        to_cam = pc.xyz - centroid  # "outward" stand-in when no camera info exists

    flip = np.einsum("ij,ij->i", normals, to_cam) < 0
    normals[flip] *= -1
    return normals


def _propagate_confidence(pc: PointCloud, vertices: np.ndarray) -> np.ndarray:
    """Nearest-input-point confidence tier for each output mesh vertex."""
    if pc.confidence is None or pc.xyz.shape[0] == 0 or vertices.shape[0] == 0:
        return np.full(vertices.shape[0], Confidence.LOW_CONFIDENCE, dtype=np.uint8)
    tree = cKDTree(pc.xyz)
    _, idx = tree.query(vertices, k=1)
    return np.asarray(pc.confidence)[idx].astype(np.uint8)


def poisson_reconstruct(
    pc: PointCloud,
    depth: int = 9,
    normals: np.ndarray | None = None,
    density_quantile: float = _DEFAULT_DENSITY_QUANTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Screened Poisson surface reconstruction, with fabricated-geometry vertices flagged.

    Poisson reconstruction fits a smooth, typically *watertight* implicit
    surface to an oriented point cloud. Watertight is exactly the problem:
    any gap in the input coverage (an occluded side, a spot the drone never
    overflew) gets silently bridged with a plausible-looking surface that
    nobody ever measured. That is precisely the "silently faked geometry"
    failure mode this whole project exists to catch, so this function never
    returns a mesh without also returning a confidence array that flags
    the least-supported vertices as ``Confidence.INFERRED`` -- via open3d's
    own per-vertex density output when available (low density = the
    implicit function had to extrapolate further from real data to close
    the surface there), or via the numpy fallback's accumulated-weight
    equivalent otherwise.

    Returns ``(vertices, faces, colors, confidence)``.
    """
    n = pc.xyz.shape[0]
    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
        )

    if normals is None:
        normals = estimate_normals(pc, k=30)

    if _HAS_OPEN3D:  # pragma: no cover - requires the optional open3d dependency
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        if pc.rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(pc.rgb.astype(np.float64) / 255.0)

        o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        densities = np.asarray(densities)
        if o3d_mesh.has_vertex_colors():
            colors = np.clip(np.asarray(o3d_mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.full((vertices.shape[0], 3), 128, dtype=np.uint8)

        confidence = _propagate_confidence(pc, vertices)
        if densities.size:
            threshold = float(np.quantile(densities, density_quantile))
            confidence = confidence.copy()
            confidence[densities <= threshold] = Confidence.INFERRED
        return vertices, faces, colors, confidence

    # -- numpy/scipy fallback -------------------------------------------
    #
    # True screened Poisson solves a global Poisson equation over an
    # adaptive octree; reproducing that without open3d is out of scope.
    # Instead we reuse fusion.tsdf's oriented-point-splat volumetric
    # fusion (local tangent-plane signed distance, splatted into a grid,
    # marching-tetrahedra-extracted) as a much simpler stand-in that is
    # still a genuine implicit-surface method and, usefully, produces its
    # confidence array (MEASURED/LOW_CONFIDENCE/INFERRED from accumulated
    # splat weight) via exactly the same "did we actually observe this"
    # bookkeeping the rest of this project relies on -- an unobserved
    # region that Poisson would silently bridge shows up here as a
    # zero-weight gap the volume bridges the same way, and is labelled
    # INFERRED for the same reason.
    from drishti3d.fusion.tsdf import TSDFVolume

    tree = cKDTree(pc.xyz)
    k_eff = min(8, n - 1)
    if k_eff >= 1:
        nn_dist, _ = tree.query(pc.xyz, k=k_eff + 1)
        median_spacing = float(np.median(nn_dist[:, 1:])) if k_eff >= 1 else 0.1
    else:
        median_spacing = 0.1
    median_spacing = max(median_spacing, 1e-4)

    bbox_diag = float(np.linalg.norm(pc.xyz.max(axis=0) - pc.xyz.min(axis=0)))
    depth_clamped = max(1, min(depth, 9))
    resolution_voxel = bbox_diag / (2**depth_clamped) if bbox_diag > 0 else median_spacing
    voxel_size = max(resolution_voxel, median_spacing * 0.5)
    sdf_trunc = 3.0 * voxel_size

    bounds_min = pc.xyz.min(axis=0) - sdf_trunc - voxel_size
    bounds_max = pc.xyz.max(axis=0) + sdf_trunc + voxel_size
    dims = np.maximum(np.ceil((bounds_max - bounds_min) / voxel_size).astype(np.int64), 1) + 1

    volume = TSDFVolume(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        origin=bounds_min,
        dims=tuple(int(d) for d in dims),
        use_open3d=False,
    )
    confidence_in = pc.confidence.astype(np.float64) if pc.confidence is not None else np.ones(n)
    volume.integrate_point_cloud(pc.xyz, normals, colors=pc.rgb, confidence=confidence_in)
    vertices, faces, colors, confidence = volume.extract_triangle_mesh()
    return vertices, faces, colors, confidence


def _cluster_decimate(vertices: np.ndarray, faces: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """One pass of vertex-clustering decimation at a given grid cell size."""
    origin = vertices.min(axis=0)
    idx = np.floor((vertices - origin) / cell_size).astype(np.int64)
    uniq, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_clusters = uniq.shape[0]

    sums = np.zeros((n_clusters, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_clusters)
    np.add.at(sums, inverse, vertices)
    new_vertices = sums / counts[:, None]

    new_faces = inverse[faces]
    degenerate = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 0] == new_faces[:, 2])
    )
    new_faces = new_faces[~degenerate]

    if new_faces.shape[0] > 0:
        sorted_faces = np.sort(new_faces, axis=1)
        _, first_seen = np.unique(sorted_faces, axis=0, return_index=True)
        new_faces = new_faces[np.sort(first_seen)]

    return new_vertices, new_faces


#: Face count above which quadric decimation is preceded by a cheap
#: vertex-clustering pass. Measured: open3d's quadric solver did not
#: finish 68.7M faces in 28 minutes; 8M completes in about a minute.
_QUADRIC_FACE_LIMIT = 8_000_000


def decimate_to_cap(
    vertices: np.ndarray, faces: np.ndarray, max_faces: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate to at most ``max_faces``; also return, per new vertex, the nearest original vertex index.

    Quadric-error decimation through open3d when it is importable (shape-
    preserving), the numpy vertex-clustering ``decimate_mesh`` otherwise.
    The returned index lets a caller carry every per-vertex channel
    (colour, confidence tier, semantic class) across the decimation by a
    plain gather, instead of each channel needing its own interpolation
    rule -- a nearest original vertex is the honest value for a categorical
    channel and a fine approximation for the others at these densities.
    """
    from scipy.spatial import cKDTree

    if faces.shape[0] <= max_faces or vertices.shape[0] == 0:
        return vertices, faces, np.arange(vertices.shape[0])

    # Quadric decimation is shape-preserving but its cost explodes: open3d
    # ran for 28+ minutes on a 68.7M-face mesh without finishing, because
    # it maintains a priority queue over every edge. Vertex clustering is
    # O(n) and finishes in seconds. So: cluster first whenever the mesh is
    # far above the cap (that is where the bulk of the reduction happens
    # and where shape fidelity matters least -- the input at that size is
    # dominated by noise-generated surface area anyway), then hand the
    # much smaller result to the quadric pass for the final, quality-
    # sensitive step.
    # Keep a handle on the ORIGINAL vertices. The returned index must map
    # each output vertex to an index in THIS array, because the caller
    # gathers original-length per-vertex channels (colour, confidence,
    # semantic class) through it. Rebinding `vertices` during the
    # pre-pass below and then querying against the rebound array yields an
    # identity map into the decimated set, which silently mis-pairs every
    # channel with the wrong geometry.
    original_vertices = np.asarray(vertices, dtype=np.float64)

    working_v, working_f = vertices, faces
    if faces.shape[0] > _QUADRIC_FACE_LIMIT:
        pre_target = max(max_faces * 2, _QUADRIC_FACE_LIMIT // 2)
        working_v, working_f = decimate_mesh(working_v, working_f, pre_target)
        if working_f.shape[0] <= max_faces:
            _, nn0 = cKDTree(original_vertices).query(np.asarray(working_v, dtype=np.float64), k=1)
            return working_v, working_f, np.asarray(nn0, dtype=np.int64)

    try:
        import open3d as o3d

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(working_v, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(working_f, dtype=np.int32)),
        )
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        new_v = np.asarray(m.vertices, dtype=np.float64)
        new_f = np.asarray(m.triangles, dtype=np.int64)
    except ImportError:
        new_v, new_f = decimate_mesh(working_v, working_f, int(max_faces))
    # Against the ORIGINAL vertices, not the pre-pass output.
    _, nn = cKDTree(original_vertices).query(np.asarray(new_v, dtype=np.float64), k=1)
    return new_v, new_f, np.asarray(nn, dtype=np.int64)


def decimate_mesh(
    vertices: np.ndarray, faces: np.ndarray, target_triangles: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``faces`` toward ``target_triangles`` via vertex-clustering decimation.

    Vertex clustering (snap vertices to a coarse grid, collapse each
    occupied cell to one averaged vertex, drop faces that degenerate to
    zero area or duplicate another face) is a simple, dependency-free
    decimation method -- it will not exactly hit ``target_triangles`` and
    is not as shape-preserving as quadric-error decimation, but it is
    correct, fast, and needs nothing beyond numpy. Binary search over the
    grid cell size gets within a small margin of the requested count.
    """
    n_faces = faces.shape[0]
    if n_faces <= target_triangles or n_faces == 0 or vertices.shape[0] == 0:
        return vertices, faces

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0:
        return vertices, faces

    lo, hi = diag / 2000.0, diag
    best_vertices, best_faces = vertices, faces
    for _ in range(24):
        mid = (lo + hi) / 2.0
        v2, f2 = _cluster_decimate(vertices, faces, mid)
        if f2.shape[0] == 0:
            hi = mid
            continue
        if f2.shape[0] > target_triangles:
            lo = mid
        else:
            hi = mid
            best_vertices, best_faces = v2, f2
        if abs(f2.shape[0] - target_triangles) <= max(1, int(target_triangles * 0.05)):
            best_vertices, best_faces = v2, f2
            break

    return best_vertices, best_faces


def _connected_components(n_vertices: int, faces: np.ndarray) -> tuple[int, np.ndarray]:
    """``(count, per-vertex labels)`` over the triangle-adjacency graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    data = np.ones(len(edges), dtype=np.uint8)
    graph = coo_matrix(
        (data, (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)
    ).tocsr()
    return _cc(graph, directed=False, return_labels=True)


def _compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no face references any more, and reindex."""
    if faces.size == 0:
        return vertices[:0], faces
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_edge_factor: float = 6.0,
    min_component_faces: int = 64,
    min_component_ratio: float = 0.01,
    component_face_cap: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove Poisson spikes and speck islands. Only ever removes geometry.

    Why this is needed at all
    -------------------------
    Poisson reconstruction over a single-pass aerial cloud produces two
    artifacts that dominate how "noisy" the model *looks*, independently of
    how accurate it measures:

    1. **Spike slivers** -- long thin triangles thrown across depth
       discontinuities (roof edge to ground) and through low-confidence
       regions. The surface underneath is fine; these are the visible mess.
    2. **Speck islands** -- tiny disconnected blobs, mostly revealed once
       the slivers joining them to the surface are gone.

    Both are judged against the mesh's **own statistics** -- median edge
    length, largest-component size -- so the behaviour is scale-free: the
    same thresholds work on a 10 m courtyard and a 2 km corridor without
    retuning.

    What it deliberately does NOT do
    --------------------------------
    Nothing here adds, smooths or moves a vertex. Smoothing a reconstructed
    surface makes it look better while making every measurement taken off
    it worse, which for a system whose deliverable is metric accuracy is
    exactly the wrong trade. Cleanup only deletes faces it can justify.

    Large disconnected regions are also KEPT. A single UAV pass genuinely
    produces several unconnected surfaces -- a courtyard seen from one
    side, a block occluded from the flight line -- and discarding them
    would destroy real structure. Only specks go.

    ``component_face_cap`` is the subtle one. A pure ratio does not scale:
    1% of a 6.5M-face mesh is 65,000 faces, which is an entire building.
    Capping the derived threshold keeps the rule aimed at the tens-to-
    hundreds-of-faces specks Poisson actually leaves behind.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    stats: dict = {"faces_in": len(faces), "vertices_in": len(vertices)}

    if faces.size == 0:
        stats.update(faces_out=0, vertices_out=0, removed_spike_faces=0, removed_component_faces=0)
        return vertices, faces, stats

    # --- 1. spike slivers, judged against this mesh's own median edge ---
    removed_spikes = 0
    if max_edge_factor and max_edge_factor > 0:
        edge_len = np.stack(
            [
                np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1),
                np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 1]], axis=1),
                np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=1),
            ],
            axis=1,
        )
        median_edge = float(np.median(edge_len))
        stats["median_edge_m"] = round(median_edge, 6)
        if median_edge > 0:
            keep = edge_len.max(axis=1) <= max_edge_factor * median_edge
            removed_spikes = int((~keep).sum())
            # Never delete everything: a mesh whose median edge is itself
            # degenerate would otherwise be wiped out entirely.
            if removed_spikes and keep.any():
                faces = faces[keep]
            else:
                removed_spikes = 0
    stats["removed_spike_faces"] = removed_spikes

    # --- 2. speck components, measured AFTER spike removal ---
    # Order matters: the slivers are what tether specks to the main
    # surface, so components counted before their removal look connected.
    removed_components = 0
    if len(faces):
        n_before, labels = _connected_components(len(vertices), faces)
        face_labels = labels[faces[:, 0]]
        counts = np.bincount(face_labels, minlength=int(labels.max()) + 1)
        threshold = max(int(min_component_faces), int(min_component_ratio * counts.max()))
        if component_face_cap:
            threshold = min(threshold, int(component_face_cap))

        keep = counts[face_labels] >= threshold
        removed_components = int((~keep).sum())
        if removed_components and keep.any():
            faces = faces[keep]
        else:
            removed_components = 0

        stats["component_face_threshold"] = int(threshold)
        stats["components_before"] = int(n_before)
        stats["components_kept"] = int((counts >= threshold).sum())
    stats["removed_component_faces"] = removed_components

    # `kept_vertices` indexes the ORIGINAL vertex array. Callers carrying
    # parallel per-vertex channels (colour, confidence, semantic class)
    # must gather them through it, or those channels silently mis-pair
    # with geometry after compaction.
    kept_vertices = np.unique(faces) if faces.size else np.zeros(0, dtype=np.int64)
    vertices, faces = _compact(vertices, faces)
    stats["kept_vertices"] = kept_vertices
    stats["faces_out"] = len(faces)
    stats["vertices_out"] = len(vertices)
    stats["removed_faces_pct"] = (
        round(100.0 * (stats["faces_in"] - stats["faces_out"]) / stats["faces_in"], 3)
        if stats["faces_in"]
        else 0.0
    )
    return vertices, faces, stats


def _is_watertight(faces: np.ndarray) -> bool:
    """A mesh is (edge-)watertight iff every undirected edge borders exactly two triangles."""
    if faces.shape[0] == 0:
        return False
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges_sorted = np.sort(edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def compute_mesh_stats(
    vertices: np.ndarray, faces: np.ndarray, confidence: np.ndarray | None = None
) -> dict:
    """Triangle count, surface area, bounding box, watertightness, and confidence breakdown.

    ``confidence`` (per-vertex ``types.Confidence`` values) is optional --
    when given, the returned dict includes the percentage of vertices at
    each tier, which is exactly what the accuracy report card
    (``export.report.build_report``) surfaces to an analyst.
    """
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    if n_faces > 0:
        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        surface_area = float(np.linalg.norm(cross, axis=1).sum() / 2.0)
    else:
        surface_area = 0.0

    if n_vertices > 0:
        bbox_min = vertices.min(axis=0).tolist()
        bbox_max = vertices.max(axis=0).tolist()
    else:
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]

    stats: dict = {
        "triangle_count": int(n_faces),
        "vertex_count": int(n_vertices),
        "surface_area_m2": surface_area,
        "bounding_box_min": bbox_min,
        "bounding_box_max": bbox_max,
        "watertight": _is_watertight(faces),
    }

    if confidence is not None and n_vertices > 0:
        confidence = np.asarray(confidence)
        total = confidence.shape[0]
        stats["confidence_breakdown_pct"] = {
            "measured": float(100.0 * np.sum(confidence == Confidence.MEASURED) / total),
            "low_confidence": float(100.0 * np.sum(confidence == Confidence.LOW_CONFIDENCE) / total),
            "inferred": float(100.0 * np.sum(confidence == Confidence.INFERRED) / total),
        }
    else:
        stats["confidence_breakdown_pct"] = None

    return stats
