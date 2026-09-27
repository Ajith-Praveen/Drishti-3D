"""``export_all``: the one-call deliverable façade ``pipeline.stages.ExportStage`` drives.

``export.formats``/``export.geotiff``/``export.report`` each do one job well
(write a PLY, rasterize a DSM, render a report card) but none of them know
about each other or about "here is everything this run should hand to an
analyst." ``export_all`` is that missing façade: given whatever the pipeline
ended up with (a plain point cloud, or a ``(vertices, faces, colors,
confidence)`` mesh tuple -- see ``export.formats.as_geometry``), it writes
every deliverable format that input actually supports, the DSM/orthomosaic/
confidence-raster GeoTIFFs the point cloud can support (orthomosaic needs
``rgb``; the others just need points), and the text+HTML accuracy report
card -- and returns exactly which files it wrote (never fabricating a path
for a format it skipped), so ``ExportStage`` has something real to put in
``StageResult.artifacts``.

Two format-level inconsistencies this module papers over rather than
letting leak into the pipeline stage:

- ``export_ply`` accepts a ``PointCloud`` *or* a mesh tuple; ``export_glb``
  only ever accepts raw ``vertices``/``faces``/``colors``/``confidence``
  arrays. ``export_all`` always has both forms on hand (via
  ``as_geometry``) and hands each writer whichever it wants.
- Confidence may still arrive as raw continuous ``[0, 1]`` backbone
  confidence rather than the tiered ``Confidence`` enum a point cloud is
  documented to carry (this can happen when fusion was skipped and
  ``PipelineState.point_cloud`` is still ``geometry``'s directly-merged
  cloud -- see ``fusion.tsdf``'s identical, independently-necessary fix for
  why this distinction matters). Every uint8-confidence writer here
  (PLY/LAS/GLB) would otherwise silently truncate e.g. ``0.95`` to ``0``;
  ``_ensure_tiered_confidence`` defensively re-quantizes before any of them
  run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from drishti3d.export.formats import (
    MeshLike,
    as_geometry,
    export_fbx,
    export_glb,
    export_las,
    export_obj,
    export_ply,
    export_xyz,
    semantic_of,
)
from drishti3d.export.geotiff import (
    point_cloud_to_confidence_raster,
    point_cloud_to_dsm,
    point_cloud_to_orthomosaic,
    write_geotiff,
)
from drishti3d.export.report import build_report, render_report_html, render_report_text
from drishti3d.export.terrain import dtm_from_point_cloud, height_above_ground
from drishti3d.fusion.completion import complete_facades
from drishti3d.types import Confidence, PointCloud, Pose

logger = logging.getLogger(__name__)

#: DSM holes up to this many cells from measured data are filled (and marked); farther is nodata.
_DSM_PINHOLE_CELLS = 2

__all__ = ["export_all"]

_ALL_FORMATS = frozenset({"ply", "las", "glb", "obj", "fbx", "xyz"})

# See module docstring: raw backbone confidence tops out well under 1.0, so
# thresholds mirror fusion.tsdf's identical quantization (kept in sync by
# hand -- these two modules can't share a helper without either owning the
# other's file, and both are small/self-contained enough that duplication
# is cheaper than a forced coupling).
_CONF_TIER_MEASURED_MIN = 0.8
_CONF_TIER_LOW_MIN = 0.3

# Target ~this many raster cells along the point cloud's longest horizontal
# axis when the caller doesn't ask for a specific ground sample distance --
# enough detail to be useful without producing an unreasonably large raster
# for a wide flight strip.
_TARGET_RASTER_CELLS = 512
_AUTO_RESOLUTION_MIN_M = 0.01


def _ensure_tiered_confidence(confidence: np.ndarray | None) -> np.ndarray | None:
    """Defensively map raw ``[0, 1]`` confidence onto the tiered ``Confidence`` enum; a no-op if already tiered."""
    if confidence is None:
        return None
    arr = np.asarray(confidence)
    if arr.size == 0:
        return arr.astype(np.uint8)
    already_tiered = np.all(np.isin(arr, (Confidence.INFERRED, Confidence.LOW_CONFIDENCE, Confidence.MEASURED)))
    if already_tiered:
        return arr.astype(np.uint8)

    tier = np.full(arr.shape, Confidence.INFERRED, dtype=np.uint8)
    tier[arr >= _CONF_TIER_LOW_MIN] = Confidence.LOW_CONFIDENCE
    tier[arr >= _CONF_TIER_MEASURED_MIN] = Confidence.MEASURED
    return tier


def _auto_resolution_m(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return _AUTO_RESOLUTION_MIN_M
    extent = xyz[:, :2].max(axis=0) - xyz[:, :2].min(axis=0)
    max_extent = float(extent.max())
    if max_extent <= 0.0:
        return _AUTO_RESOLUTION_MIN_M
    return max(max_extent / _TARGET_RASTER_CELLS, _AUTO_RESOLUTION_MIN_M)


def _true_ortho_raster(map_xyz: np.ndarray, texture: np.ndarray, uv: np.ndarray, max_px: int = 12000):
    """Resample a height-field true-ortho texture onto the map grid. Returns (rgb HxWx3, GeoTransform).

    The texture's planar UVs belong to the vertices; georeferencing and
    reference alignment moved the vertices by similarity transforms and the
    map projection is conformal, so vertex map XY -> texture pixel is an
    affine map, fitted here by least squares and checked. Texels keep their
    ground size (~0.17 m on flight01) instead of the vertex rasterization's
    extent-derived cell (1.3 m).
    """
    import cv2

    from drishti3d.export.geotiff import GeoTransform

    xy = np.asarray(map_xyz, dtype=np.float64)[:, :2]
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] != uv.shape[0] or xy.shape[0] < 10:
        raise ValueError("texture UVs do not pair with the exported vertices")
    th, tw = texture.shape[:2]
    px = np.c_[uv[:, 0] * tw - 0.5, (1.0 - uv[:, 1]) * th - 0.5]  # texel coordinates (col, row)
    step = max(1, xy.shape[0] // 50000)
    # Centred: map coordinates are ~1e6 m, and an uncentred [x, y, 1] design
    # loses the texel-level precision this fit needs.
    centre = xy.mean(axis=0)
    A_in = np.c_[xy[::step] - centre, np.ones(xy[::step].shape[0])]
    coef, *_ = np.linalg.lstsq(A_in, px[::step], rcond=None)  # (3, 2): [x-cx, y-cy, 1] -> (col, row)
    resid = np.abs(A_in @ coef - px[::step]).max()
    if resid > 2.0:
        raise ValueError(f"vertex-to-texture map is not affine (max residual {resid:.1f} texels)")
    lin = coef[:2].T  # 2x2: d(col,row)/d(x,y)
    res = float(1.0 / np.sqrt(abs(np.linalg.det(lin))))  # map units per texel
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    res = max(res, float(max(x1 - x0, y1 - y0)) / max_px)
    ncols, nrows = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    gx = x0 + (np.arange(ncols) + 0.5) * res
    gy = y1 - (np.arange(nrows) + 0.5) * res
    GX, GY = np.meshgrid(gx - centre[0], gy - centre[1])
    map_c = (coef[0, 0] * GX + coef[1, 0] * GY + coef[2, 0]).astype(np.float32)
    map_r = (coef[0, 1] * GX + coef[1, 1] * GY + coef[2, 1]).astype(np.float32)
    rgb = cv2.remap(np.ascontiguousarray(texture), map_c, map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # GeoTransform origins are the CENTRE of the top-left pixel (world-file order).
    return rgb, GeoTransform(
        pixel_size_x=res,
        rotation_y=0.0,
        rotation_x=0.0,
        pixel_size_y=-res,
        x_origin=float(x0) + res / 2.0,
        y_origin=float(y1) - res / 2.0,
    )


def _enu_to_map_fn(origin, crs: str):
    """ENU (about ``origin``) -> ``crs`` easting/northing/ellipsoidal height, exact per point."""
    import pyproj

    from drishti3d.geometry.georef import enu_to_wgs84

    to_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def fn(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if xyz.shape[0] == 0:
            return xyz.copy()
        llh = enu_to_wgs84(xyz, origin)  # (lon, lat, alt) per georef's convention
        e, n = to_crs.transform(llh[:, 0], llh[:, 1])
        return np.stack([e, n, llh[:, 2]], axis=1)

    return fn


def _mapped(pc: PointCloud, to_map) -> PointCloud:
    if to_map is None or pc is None:
        return pc
    import dataclasses

    return dataclasses.replace(pc, xyz=to_map(pc.xyz), covariance=pc.covariance)


def _write_georef_sidecar(path: Path, origin, crs: str) -> None:
    """How to take the local-frame mesh deliverables to map coordinates."""
    import json

    try:
        path.write_text(
            json.dumps(
                {
                    "mesh_frame": "local ENU metres (x east, y north, z up) about origin",
                    "origin": {"lat": origin.lat, "lon": origin.lon, "alt_msl": origin.alt_msl},
                    "map_crs": crs,
                    "map_deliverables": "model.las, point_cloud.las, model.xyz and every .tif are in map_crs",
                },
                indent=2,
            )
        )
    except OSError:
        logger.warning("export_all: could not write %s", path, exc_info=True)


def export_all(
    result_or_pointcloud: PointCloud | MeshLike,
    out_dir: str | Path,
    formats: set[str] | None = None,
    crs: str | None = None,
    *,
    poses: list[Pose] | None = None,
    resolution_m: float | None = None,
    report_artifacts: dict[str, Any] | None = None,
    raw_point_cloud: PointCloud | None = None,
    report_out: dict[str, Any] | None = None,
    geo_origin=None,
    ortho_texture: tuple[np.ndarray, np.ndarray] | None = None,
    uncertainty_m: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write every deliverable ``result_or_pointcloud`` supports into ``out_dir``. Returns ``{name: path}`` actually written.

    ``result_or_pointcloud`` is anything ``export.formats.as_geometry``
    accepts: a ``PointCloud``, or a ``(vertices, faces)`` / ``(vertices,
    faces, colors, confidence)`` mesh tuple. ``formats`` restricts which of
    ``{"ply", "las", "glb", "obj", "xyz"}`` are attempted (default: all
    five); ``"obj"`` is silently skipped for a points-only input (OBJ has no
    point-cloud-only convention worth writing). Rasters (DSM/confidence,
    plus orthomosaic when colour is available) and the text+HTML report are
    always attempted whenever there are points to rasterize/report on.

    ``report_out``, when given, is updated in place with the accuracy
    report card this function builds for ``report.html``/``report.txt``
    (``export.report.build_report``). It is an out-parameter rather than
    a second return value so that every existing caller keeps working
    unchanged; without it the report card is only ever written to disk,
    which is why the app's Report panel showed "not computed" for every
    metric after a real run.

    ``raw_point_cloud``, when given and non-empty, is written as
    ``point_cloud.ply``/``point_cloud.las`` -- *independently* of
    ``result_or_pointcloud`` and unconditional on ``formats`` (meshing is
    lossy: this is the full-detail, never-voxelised dense cloud a caller
    like ``fusion.tsdf.fuse_submaps`` produces alongside, but before, its
    own TSDF mesh -- see that function's docstring). Every run that has one
    gets both files, so an operator always has a full-detail fallback on
    disk regardless of how the mesh/point-cloud deliverable above turned
    out.

    Never raises for one format's failure -- each writer is wrapped
    individually and logged, so one broken exporter (e.g. a LAS CRS string
    ``pyproj`` rejects) doesn't cost every other deliverable. Returns only
    the entries that actually got written; check the returned dict's keys
    rather than assuming every requested format landed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map-coordinate deliverables. The model arrives in local ENU about
    # `geo_origin`; LAS, XYZ and every GeoTIFF are written in `crs` (the UTM
    # zone) by transforming each point exactly -- ENU and UTM grid north
    # differ by the meridian convergence (~1.5 deg at flight01), so a plain
    # offset would be metres wrong at the edges. Mesh formats (PLY/GLB/OBJ/
    # FBX) stay local for viewers; georef.json says how to map them.
    to_map = None
    if geo_origin is not None and crs:
        to_map = _enu_to_map_fn(geo_origin, crs)
        _write_georef_sidecar(out_dir / "georef.json", geo_origin, crs)
    written: dict[str, Path] = {}
    # Collected from the DTM/facade passes below and folded into the report
    # card, so terrain provenance ("was this ground classified semantically
    # or morphologically?") reaches the operator rather than only the log.
    terrain_stats: dict[str, Any] = {}

    xyz, faces, rgb, confidence = as_geometry(result_or_pointcloud)
    semantic_class, semantic_conf = semantic_of(result_or_pointcloud)
    confidence = _ensure_tiered_confidence(confidence)
    has_faces = faces is not None and len(faces) > 0

    pc = PointCloud(
        xyz=xyz,
        rgb=rgb,
        confidence=confidence,
        semantic_class=semantic_class,
        semantic_confidence=semantic_conf,
        # Per-vertex height uncertainty rides separately (the mesh tuple has
        # no slot for it) and reaches model.las as ``height_uncertainty_m``.
        uncertainty_m=(
            uncertainty_m
            if uncertainty_m is not None and len(uncertainty_m) == len(xyz)
            else getattr(result_or_pointcloud, "uncertainty_m", None)
        ),
    )
    # The 6-tuple form (see formats.as_geometry) carries the semantic pair
    # alongside the mesh, so a meshed export keeps its per-vertex classes
    # instead of silently dropping them on the faces path.
    mesh_or_pc: PointCloud | MeshLike = (
        (xyz, faces, rgb, confidence, semantic_class, semantic_conf) if has_faces else pc
    )
    pc_map = _mapped(pc, to_map)

    wanted = _ALL_FORMATS if formats is None else (set(formats) & _ALL_FORMATS)
    n = int(xyz.shape[0])

    if n == 0:
        logger.warning("export_all: empty point cloud/mesh -- only the report card will be written")

    if n > 0:
        if "ply" in wanted:
            try:
                path = out_dir / "model.ply"
                export_ply(path, mesh_or_pc)
                written["ply"] = path
            except Exception:
                logger.exception("export_all: PLY export failed")

        if "las" in wanted:
            try:
                path = out_dir / "model.las"
                export_las(path, pc_map, crs=crs)
                written["las"] = path
            except Exception:
                logger.exception("export_all: LAS export failed")

        if "glb" in wanted:
            try:
                path = out_dir / "model.glb"
                export_glb(path, xyz, faces=faces, colors=rgb, confidence=confidence)
                written["glb"] = path
            except Exception:
                logger.exception("export_all: GLB export failed")

        if "obj" in wanted:
            if has_faces:
                try:
                    path = out_dir / "model.obj"
                    export_obj(path, xyz, faces, colors=rgb)
                    written["obj"] = path
                except Exception:
                    logger.exception("export_all: OBJ export failed")
            else:
                logger.info("export_all: skipping OBJ -- no mesh faces available (point-cloud-only result)")

        if "fbx" in wanted and has_faces:
            try:
                path = out_dir / "model.fbx"
                export_fbx(path, xyz, faces, colors=rgb)
                written["fbx"] = path
            except Exception:
                logger.exception("export_all: FBX export failed")

        if "xyz" in wanted:
            try:
                path = out_dir / "model.xyz"
                export_xyz(path, pc_map)
                written["xyz"] = path
            except Exception:
                logger.exception("export_all: XYZ export failed")

        res = resolution_m if resolution_m is not None else _auto_resolution_m(pc_map.xyz)

    if raw_point_cloud is not None and raw_point_cloud.xyz.shape[0] > 0:
        raw_pc = PointCloud(
            xyz=raw_point_cloud.xyz,
            rgb=raw_point_cloud.rgb,
            covariance=raw_point_cloud.covariance,
            confidence=_ensure_tiered_confidence(raw_point_cloud.confidence),
            semantic_class=raw_point_cloud.semantic_class,
            semantic_confidence=raw_point_cloud.semantic_confidence,
        )
        try:
            path = out_dir / "point_cloud.ply"
            export_ply(path, raw_pc)
            written["point_cloud_ply"] = path
        except Exception:
            logger.exception("export_all: raw fused point-cloud PLY export failed")

        try:
            path = out_dir / "point_cloud.las"
            export_las(path, _mapped(raw_pc, to_map), crs=crs)
            written["point_cloud_las"] = path
        except Exception:
            logger.exception("export_all: raw fused point-cloud LAS export failed")

    if n > 0:
        dsm = None
        try:
            dsm, transform, filled = point_cloud_to_dsm(pc_map, resolution_m=res)
            # Cells the model never observed are nodata beyond pinhole range
            # of real data: a nearest-neighbour surface stretched across
            # ground the camera did not see is invented height (it was ~half
            # of the raster on flight01). Pinholes stay filled and, like the
            # DTM's, are marked in dsm_interpolated.tif.
            from scipy import ndimage

            far = filled & (ndimage.distance_transform_edt(filled) > _DSM_PINHOLE_CELLS)
            dsm = np.where(far, np.nan, dsm)
            path = out_dir / "dsm.tif"
            write_geotiff(path, dsm.astype(np.float32), transform, crs=crs, nodata=float("nan"))
            written["dsm"] = path
            path = out_dir / "dsm_interpolated.tif"
            write_geotiff(path, (filled & ~far).astype(np.uint8), transform, crs=crs)
            written["dsm_interpolated"] = path
        except Exception:
            logger.info("export_all: DSM raster not written", exc_info=True)

        # Bare-earth DTM, the height model above it, and the mask saying
        # which DTM cells were interpolated rather than observed. The mask
        # is a deliverable in its own right: under every building the
        # ground elevation is a guess, and a user measuring a cutting depth
        # has to be able to see that.
        dtm_source = _mapped(raw_pc, to_map) if raw_point_cloud is not None and raw_pc.semantic_class is not None else pc_map
        try:
            dtm_result = dtm_from_point_cloud(dtm_source, resolution_m=res)
            path = out_dir / "dtm.tif"
            write_geotiff(path, dtm_result.dtm.astype(np.float32), dtm_result.transform, crs=crs, nodata=float("nan"))
            written["dtm"] = path
            terrain_stats.update(dtm_result.stats)

            path = out_dir / "dtm_interpolated.tif"
            write_geotiff(path, dtm_result.interpolated_mask.astype(np.uint8), dtm_result.transform, crs=crs)
            written["dtm_interpolated"] = path

            if dsm is not None and dsm.shape == dtm_result.dtm.shape:
                path = out_dir / "height_above_ground.tif"
                write_geotiff(
                    path,
                    height_above_ground(dsm, dtm_result.dtm).astype(np.float32),
                    dtm_result.transform,
                    crs=crs,
                    nodata=float("nan"),
                )
                written["height_above_ground"] = path
        except Exception:
            logger.info("export_all: DTM/height rasters not written", exc_info=True)
            dtm_result = None

        # Inferred facades, written as their OWN files. Never merged into
        # point_cloud.las -- see fusion.completion's docstring on why that
        # separation is what makes extruded geometry defensible at all.
        if dtm_result is not None:
            try:
                completed = complete_facades(
                    dtm_source, dtm_result.dtm, dtm_result.transform, resolution_m=res
                )
                if completed is not None:
                    terrain_stats.update(completed.stats)
                    path = out_dir / "facades_inferred.las"
                    export_las(path, completed.facade_points, crs=crs)
                    written["facades_inferred_las"] = path
                    path = out_dir / "facades_inferred.ply"
                    export_ply(path, completed.facade_points)
                    written["facades_inferred_ply"] = path
            except Exception:
                logger.info("export_all: facade completion not written", exc_info=True)

        wrote_true_ortho = False
        if ortho_texture is not None:
            try:
                ortho, transform = _true_ortho_raster(pc_map.xyz, *ortho_texture)
                path = out_dir / "orthomosaic.tif"
                write_geotiff(path, ortho, transform, crs=crs, nodata=0)
                written["orthomosaic"] = path
                wrote_true_ortho = True
            except Exception:
                logger.warning("export_all: true-ortho raster failed; rasterizing vertex colour instead", exc_info=True)
        if rgb is not None and not wrote_true_ortho:
            try:
                ortho, transform, _filled = point_cloud_to_orthomosaic(pc_map, resolution_m=res)
                path = out_dir / "orthomosaic.tif"
                write_geotiff(path, ortho, transform, crs=crs)
                written["orthomosaic"] = path
            except Exception:
                logger.info("export_all: orthomosaic raster not written", exc_info=True)

        if confidence is not None:
            try:
                conf_raster, transform, _filled = point_cloud_to_confidence_raster(pc_map, resolution_m=res)
                path = out_dir / "confidence.tif"
                write_geotiff(path, conf_raster.astype(np.float32), transform, crs=crs)
                written["confidence_raster"] = path
            except Exception:
                logger.info("export_all: confidence raster not written", exc_info=True)

    try:
        artifacts: dict[str, Any] = dict(report_artifacts or {})
        artifacts.setdefault("confidence", confidence)
        artifacts.update(terrain_stats)
        if poses is not None:
            artifacts.setdefault("keyframe_count", len(poses))
        report = build_report(artifacts)
        if report_out is not None:
            # The same dict that becomes report.html/report.txt, handed
            # back to the caller. Without this the GUI's accuracy report
            # card had no source at all: every metric it displays is
            # computed here and was previously written only to disk.
            report_out.update(report)

        text_path = out_dir / "report.txt"
        text_path.write_text(render_report_text(report))
        written["report_txt"] = text_path

        html_path = out_dir / "report.html"
        html_path.write_text(render_report_html(report))
        written["report_html"] = html_path
    except Exception:
        logger.exception("export_all: report generation failed")

    return written
