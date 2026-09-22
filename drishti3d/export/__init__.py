"""Export stage: fused meshes/point clouds -> deliverable files + the accuracy report card.

``export.formats`` writes the actual point-cloud/mesh files (PLY, OBJ,
GLB, LAS, XYZ), keeping per-point/per-vertex confidence as a first-class
exported channel everywhere the format allows it. ``export.geotiff``
rasterizes to a DSM/orthomosaic/confidence raster and writes them as
georeferenced files, with an explicit fallback chain when ``rasterio``/
``tifffile`` aren't installed. ``export.report`` builds and renders the
accuracy report card -- the analyst-facing summary of what was actually
measured, with "not computed" standing in for anything that wasn't.
"""

from drishti3d.export.bundle_export import export_all
from drishti3d.export.formats import (
    as_geometry,
    export_glb,
    export_las,
    export_obj,
    export_ply,
    export_xyz,
    read_glb,
    read_ply,
)
from drishti3d.export.geotiff import (
    GeoTransform,
    point_cloud_to_confidence_raster,
    point_cloud_to_dsm,
    point_cloud_to_orthomosaic,
    write_geotiff,
)
from drishti3d.export.report import (
    NOT_COMPUTED,
    build_report,
    check_point_residuals,
    render_report_html,
    render_report_text,
)

__all__ = [
    "NOT_COMPUTED",
    "GeoTransform",
    "as_geometry",
    "build_report",
    "check_point_residuals",
    "export_all",
    "export_glb",
    "export_las",
    "export_obj",
    "export_ply",
    "export_xyz",
    "point_cloud_to_confidence_raster",
    "point_cloud_to_dsm",
    "point_cloud_to_orthomosaic",
    "read_glb",
    "read_ply",
    "render_report_html",
    "render_report_text",
    "write_geotiff",
]
