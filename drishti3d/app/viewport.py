"""The native 3D viewport: VTK embedded in a Qt widget.

No web technology is involved anywhere in this module -- rendering is done
by VTK's OpenGL pipeline, embedded into the Qt widget tree via
``QVTKRenderWindowInteractor``. Imports are taken from specific
``vtkmodules.*`` submodules (never the monolithic ``import vtk``) so
application startup stays fast and binary size / import time is not paid
for unused VTK modules (e.g. plotting, imaging, MPI parallelism).

What lives in the scene
-----------------------
The viewport is the app's main display, so it holds the whole mission,
not just the final cloud. Layers, in the order they become available
during a run:

1. **Flight path** -- the drone's track, coloured dark-to-bright with
   time, with start/end markers. Available from telemetry alone.
2. **Cameras** -- a frustum per keyframe pose, so a pose error is visible
   as a frustum pointing somewhere the drone never looked.
3. **Point cloud** -- grows window by window while geometry runs.
4. **Mesh** -- the fused TSDF surface, once fusion finishes.

Plus permanent furniture: an auto-scaled ground grid, orientation axes, a
scale bar, and a corner HUD.

Everything is built with vectorised NumPy -> VTK conversion
-----------------------------------------------------------
This matters more than it sounds. The previous implementation filled the
per-point colour array with a Python ``for i in range(n): SetTuple3(...)``
loop. On this project's own 3.97M-vertex result that is ~4 million
interpreter-level VTK calls on the GUI thread -- minutes of a frozen
window, which is exactly what "the live preview doesn't work" looked
like. Every array here is converted in one ``numpy_to_vtk`` call, and
the vertex cell array is built from NumPy offset/connectivity buffers
rather than by ``vtkVertexGlyphFilter``.

Display decimation
------------------
Interactive framerate, not fidelity, is what a 4M-point cloud costs. The
viewport shows a strided subsample above
:data:`DISPLAY_POINT_BUDGET` and keeps the full cloud for measurement and
export. ``point_count()`` reports the true count; ``displayed_point_count()``
reports what is on screen, and the HUD says so when they differ -- a
viewer that silently drops 80% of the data would be lying.
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import (
    vtkAxesActor,
    vtkLegendScaleActor,
    vtkScalarBarActor,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
    vtkTextActor,
)

# vtkRenderingOpenGL2 / vtkInteractionStyle register factory overrides
# (OpenGL render window, trackball interactor style, ...) purely as a side
# effect of import -- required even though nothing here is referenced by
# name.
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain

from drishti3d.app import theme
from drishti3d.types import Confidence, PointCloud

COLOR_MODES = ("rgb", "confidence", "uncertainty", "height")

# View-up is always +Z: navigation orbits about the vertical (see
# Viewport.__init__), so "top" looks down from 3.4 deg south of the zenith:
# VTK discards a view-up within ~2.6 deg of the viewing direction.
VIEW_PRESETS = {
    "top": (0, -0.06, 1, 0, 0, 1),
    "front": (0, -1, 0, 0, 0, 1),
    "side": (1, 0, 0, 0, 0, 1),
    "iso": (1, -1, 1, 0, 0, 1),
}

#: Above this many points the viewport renders a strided subsample. Tuned
#: so a mid-range integrated GPU still orbits at interactive rates; the
#: full cloud is always retained for measurement and export.
DISPLAY_POINT_BUDGET = 1_500_000

#: Meshes above this many triangles get a vertex-clustered proxy of about
#: ``MESH_PROXY_VERTICES`` vertices that is drawn instead while the camera
#: moves (a 0.5 m height field over DJI_1001 is 9.7 M triangles).
MESH_PROXY_FACES = 1_500_000
MESH_PROXY_VERTICES = 400_000

#: Height-uncertainty heatmap span, metres: green 0, amber 1 m (the
#: problem statement's accuracy target), red at and beyond 2 m.
UNCERTAINTY_MAX_M = 2.0

_INFERRED_WARNING = (
    "Measurement touches INFERRED geometry: this surface was never "
    "directly observed by the drone and was filled in by the model's "
    "prior. Measuring it is disallowed by default — treat this value as "
    "illustrative only, not surveyed truth."
)


class _RenderOnDemandQVTK(QVTKRenderWindowInteractor):
    """QVTK widget that leaves rendering to explicit viewport calls.

    On macOS with this PySide6/VTK combination, QVTK's default paintEvent
    enters vtkRenderWindowInteractor.Render() and can spin forever during
    startup. The render window's own Render() path is fast, so Viewport
    renders when the scene changes and Qt paint events only acknowledge the
    exposed native surface.
    """

    _repaint_pending = False

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.accept()
        # Without a compositor (X11: Linux desktops, remote/VNC sessions) an
        # exposed native window keeps nothing -- a closed dialog's pixels stay
        # on the 3D view -- so redraw it once this paint event has returned.
        if sys.platform != "darwin" and not self._repaint_pending:
            if getattr(self.parent(), "_started", False):
                self._repaint_pending = True
                QTimer.singleShot(0, self._redraw_after_expose)

    def _redraw_after_expose(self) -> None:
        self._repaint_pending = False
        owner = self.parent()
        if getattr(owner, "_started", False):
            owner._render()


# ---------------------------------------------------------------------------
# NumPy -> VTK helpers (all vectorised; see module docstring)
# ---------------------------------------------------------------------------


def _vtk_points(xyz: np.ndarray) -> vtkPoints:
    from vtkmodules.util import numpy_support

    points = vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(xyz, dtype=np.float64), deep=True))
    return points


def _vtk_scalars(values: np.ndarray, name: str):
    from vtkmodules.util import numpy_support

    array = numpy_support.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    array.SetName(name)
    return array


def _vtk_colors(rgb: np.ndarray, name: str = "rgb") -> vtkUnsignedCharArray:
    """An (N, 3) uint8 array as a VTK colour array, in one conversion."""
    from vtkmodules.util import numpy_support

    array = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(rgb, dtype=np.uint8), deep=True, array_type=3  # VTK_UNSIGNED_CHAR
    )
    array.SetName(name)
    return array


def _vertex_cells(count: int) -> vtkCellArray:
    """A vertex cell per point, built from NumPy buffers.

    ``vtkVertexGlyphFilter`` does this too, but it walks the points in
    C++ and reallocates; building the offset/connectivity pair directly
    is a single pair of array conversions and measurably faster on the
    multi-million-point clouds this app actually produces.
    """
    from vtkmodules.util import numpy_support

    offsets = np.arange(count + 1, dtype=np.int64)
    connectivity = np.arange(count, dtype=np.int64)

    cells = vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    return cells


def _triangle_cells(faces: np.ndarray) -> vtkCellArray:
    """(F, 3) integer faces as a VTK triangle cell array, vectorised."""
    from vtkmodules.util import numpy_support

    faces = np.ascontiguousarray(faces, dtype=np.int64)
    offsets = np.arange(faces.shape[0] + 1, dtype=np.int64) * 3
    connectivity = faces.reshape(-1)

    cells = vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(connectivity), deep=True),
    )
    return cells


def _polyline_cells(counts: list[int]) -> vtkCellArray:
    """One polyline per entry in ``counts``, over consecutive point ids."""
    from vtkmodules.util import numpy_support

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    connectivity = np.arange(int(offsets[-1]), dtype=np.int64)

    cells = vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    return cells


def _confidence_lookup_table() -> vtkLookupTable:
    """Green (MEASURED) / amber (LOW_CONFIDENCE) / red (INFERRED)."""
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(3)
    lut.SetTableRange(0, 2)
    lut.SetTableValue(int(Confidence.INFERRED), *theme.rgb_f(theme.CONF_INFERRED), 1.0)
    lut.SetTableValue(int(Confidence.LOW_CONFIDENCE), *theme.rgb_f(theme.CONF_LOW), 1.0)
    lut.SetTableValue(int(Confidence.MEASURED), *theme.rgb_f(theme.CONF_MEASURED), 1.0)
    lut.Build()
    return lut


def _uncertainty_lookup_table() -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(3)
    lut.SetTableRange(0, 2)
    lut.SetTableValue(0, *theme.rgb_f(theme.CONF_MEASURED), 1.0)
    lut.SetTableValue(1, *theme.rgb_f(theme.CONF_LOW), 1.0)
    lut.SetTableValue(2, *theme.rgb_f(theme.CONF_INFERRED), 1.0)
    lut.Build()
    return lut


def _uncertainty_metric_lookup_table() -> vtkLookupTable:
    """Continuous height uncertainty in metres: green -> amber (1 m) -> red; grey where not measured."""
    stops = np.array([theme.rgb_f(theme.CONF_MEASURED), theme.rgb_f(theme.CONF_LOW), theme.rgb_f(theme.CONF_INFERRED)])
    n = 256
    t = np.linspace(0.0, 2.0, n)
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetTableRange(0.0, UNCERTAINTY_MAX_M)
    for i, x in enumerate(t):
        k = min(int(x), 1)
        c = stops[k] + (stops[k + 1] - stops[k]) * (x - k)
        lut.SetTableValue(i, float(c[0]), float(c[1]), float(c[2]), 1.0)
    lut.SetNanColor(*theme.rgb_f("#8d939b"), 1.0)
    lut.Build()
    return lut


def _cluster_mesh(vertices: np.ndarray, faces: np.ndarray, target_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation: (kept vertex indices, triangles over them).

    Every vertex in one cubic bin collapses onto the bin's lowest-index
    vertex, so the proxy's colours and heatmaps are the full mesh's own
    per-vertex values looked up by index -- nothing is resampled.
    """
    v = np.asarray(vertices, dtype=np.float64)
    lo = v.min(axis=0)
    sample = faces[:: max(1, len(faces) // 20000)]
    edge = float(np.median(np.linalg.norm(v[sample[:, 0]] - v[sample[:, 1]], axis=1)))
    if not np.isfinite(edge) or edge <= 0:
        edge = float(np.ptp(v, axis=0).max()) / 1000.0 or 1.0
    # Occupied bins on a surface scale with 1/edge^2.
    b = edge * np.sqrt(max(1.0, len(v) / float(target_vertices)))
    cells = np.floor((v - lo) / b).astype(np.int64)
    dims = cells.max(axis=0) + 1
    key = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    f = inverse.reshape(-1)[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return first, f[keep]


def _height_lookup_table(z_min: float, z_max: float) -> vtkLookupTable:
    """A perceptually-reasonable blue -> green -> yellow -> red height ramp."""
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(z_min, z_max)
    lut.SetHueRange(0.667, 0.0)  # blue -> red
    lut.SetSaturationRange(0.85, 0.9)
    lut.SetValueRange(0.85, 0.95)
    lut.Build()
    return lut


def _time_ramp(n: int) -> np.ndarray:
    """(n, 3) uint8: the flight-path colour ramp, dark violet -> hot amber.

    Deliberately not the height ramp: the two are on screen together and
    a viewer must never have to ask whether a colour means "later" or
    "higher".
    """
    t = np.linspace(0.0, 1.0, max(n, 1))[:, None]
    cold = np.array([[70, 60, 150]], dtype=np.float64)
    mid = np.array([[70, 190, 210]], dtype=np.float64)
    hot = np.array([[255, 190, 70]], dtype=np.float64)
    first = cold + (mid - cold) * np.clip(t / 0.5, 0, 1)
    second = mid + (hot - mid) * np.clip((t - 0.5) / 0.5, 0, 1)
    ramp = np.where(t < 0.5, first, second)
    return ramp.astype(np.uint8)


class Viewport(QWidget):
    """The central 3D view. Embeds a VTK render window inside a QWidget."""

    measurementMade = Signal(str, float, str)  # kind, value, warning ("" if none)
    measurementCompleted = Signal(object)  # app.measure.Measurement, with its points and extras
    sceneChanged = Signal()  # emitted whenever a layer's contents change

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Give QVTK its final Qt parent from construction. On macOS, creating
        # it unparented and then reparenting can leave the first paint stuck
        # inside vtkCocoaRenderWindow::Render(), so the app process starts but
        # never presents a usable UI.
        self.vtk_widget = _RenderOnDemandQVTK(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        # Flat, not a gradient. A graded backdrop competes with the
        # reconstruction's own shading and makes a point cloud's density
        # look like it varies with screen position.
        self.renderer.SetBackground(*theme.rgb_f(theme.BG_VOID))
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        # Measurements draw in a layer on top that shares the camera, so an
        # outline or label is never hidden behind a tree crown or a roof.
        # Created with the first measurement, not here: a second layer on
        # the window before its first paint stalls vtkCocoaRenderWindow on
        # macOS the same way reparenting does (see above).
        self._overlay = None

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # Terrain navigation: orbit about the vertical through the focal
        # point, so the horizon never rolls (a trackball tumbles a map
        # upside down within a few drags). Left drag orbits, Shift+left or
        # middle drag pans, Ctrl/Cmd+left or right drag and the wheel /
        # two-finger scroll zoom, double-click re-centres the orbit on the
        # surface under the cursor.
        style = vtkInteractorStyleTerrain()
        self.interactor.SetInteractorStyle(style)
        self._style = style
        self._left_mode: str | None = None
        self.interactor.EnableRenderOff()
        # EnableRenderOff keeps QVTK's startup paint out of
        # vtkRenderWindowInteractor::Render() (see _RenderOnDemandQVTK),
        # but the interactor still announces every camera move as a
        # RenderEvent. Nothing listened, so dragging moved the camera and
        # the picture stayed frozen until some other layer changed.
        self.interactor.AddObserver("RenderEvent", lambda *_: self._render())
        style.AddObserver("LeftButtonPressEvent", self._on_left_press)
        style.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        style.AddObserver("RightButtonPressEvent", self._on_right_press)
        style.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        self.interactor.AddObserver("KeyPressEvent", self._on_key_press)
        style.AddObserver("MouseWheelForwardEvent", lambda *_: self._wheel_zoom(1.1))
        style.AddObserver("MouseWheelBackwardEvent", lambda *_: self._wheel_zoom(1.0 / 1.1))
        style.AddObserver("StartInteractionEvent", lambda *_: self._proxy_begin())
        style.AddObserver("EndInteractionEvent", lambda *_: self._proxy_end(render=False))
        self._picker = vtkPropPicker()
        # Wheel zoom has no end event: fall back to the full mesh once the
        # wheel has been still this long.
        self._proxy_timer = QTimer(self)
        self._proxy_timer.setSingleShot(True)
        self._proxy_timer.setInterval(250)
        self._proxy_timer.timeout.connect(self._proxy_end)

        # --- state -------------------------------------------------------
        self._point_cloud: PointCloud | None = None
        self._display_stride = 1
        self._cloud_updates = 0
        self._color_mode = "rgb"
        # Added to z for the height colouring: the local frame's origin
        # altitude, so heights read as metres above sea level.
        self._height_datum_m = 0.0
        self._height_title = "Height (m)"
        self._point_actor: vtkActor | None = None
        self._point_mapper: vtkPolyDataMapper | None = None
        self._point_polydata: vtkPolyData | None = None
        self._point_size = 3

        self._mesh_actor: vtkActor | None = None
        self._mesh_mapper: vtkPolyDataMapper | None = None
        self._mesh_polydata: vtkPolyData | None = None
        self._mesh_face_count = 0
        self._mesh_visible = True
        # Decimated stand-in drawn while the camera moves (large meshes only).
        self._proxy_actor: vtkActor | None = None
        self._proxy_mapper: vtkPolyDataMapper | None = None
        self._proxy_polydata: vtkPolyData | None = None
        self._proxy_active = False
        # Whether a layer's "uncertainty" array is metres (height-field
        # stereo) or only the inverted confidence tier.
        self._uncertainty_metric = {"points": False, "mesh": False}
        self._uncertainty_summary: str | None = None

        self._path_actor: vtkActor | None = None
        self._path_marker_actor: vtkActor | None = None
        self._camera_actor: vtkActor | None = None
        self._camera_count = 0
        self._camera_scale = 5.0
        self._highlight_actor: vtkActor | None = None

        self._scalar_bar: vtkScalarBarActor | None = None
        self._grid_actor: vtkActor | None = None
        self._grid_extent = 0.0
        # Bounds of the DATA layers only, keyed by layer. The grid and the
        # camera framing are both derived from this rather than from
        # ``renderer.ComputeVisiblePropBounds()`` -- that includes the grid
        # itself, so the grid grew to fit the grid and ran away (measured:
        # a 650 m survey produced a 5 km grid).
        self._layer_bounds: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._axes_widget: vtkOrientationMarkerWidget | None = None
        self._scale_actor: vtkLegendScaleActor | None = None
        # Measurement mode (start_measurement): the kind being drawn, its
        # surface points so far, and every finished measurement's actors.
        self._measure_kind: str | None = None
        self._measure_pts: list[np.ndarray] = []
        self._measure_actors: list[list] = []
        self._measurements: list = []
        self._draft_actor = None
        self._press_xy: tuple[int, int] | None = None
        self._surface_xyz: np.ndarray | None = None
        self._height_grid = None
        self._point_locator = None
        self._started = False

        self._hud_actor = self._build_hud()
        self._hint_actor = self._build_hint()

        self._build_grid()
        self._build_axes_widget()
        self._build_scale_bar()

        self._update_hud()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the VTK interactor's render loop hookup. Call after show()."""
        if self._started:
            return
        self._started = True
        self.interactor.Initialize()
        self.renderer.ResetCamera()
        self.set_view("iso")
        self.vtk_widget.GetRenderWindow().Render()

    def close(self) -> bool:
        self.vtk_widget.Finalize()
        return super().close()

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------
    def set_point_cloud(self, pc: PointCloud, reset_camera: bool | None = None) -> None:
        """Render a PointCloud. Vectorised; safe to call on a 4M-point cloud.

        ``reset_camera`` decides whether the view re-frames. Left as
        ``None`` it follows the same policy the headless snapshot writer
        already uses (``export.preview.ProgressSnapshotWriter``'s
        ``lock_bounds_after``): re-frame while the cloud is still a small
        fraction of the eventual scene, then stop, because an operator
        who has zoomed in to watch one building reconstruct should not
        have the camera yanked back every time another window lands.
        """
        first_updates = self._cloud_updates < 3
        if reset_camera is None:
            reset_camera = first_updates
        self._cloud_updates += 1

        self._point_cloud = pc
        self._point_locator = None
        if self._surface_xyz is None:
            self._height_grid = None

        total = int(pc.xyz.shape[0])
        stride = max(1, int(np.ceil(total / DISPLAY_POINT_BUDGET))) if total else 1
        self._display_stride = stride

        xyz = pc.xyz[::stride]
        shown = int(xyz.shape[0])

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(xyz))
        polydata.SetVerts(_vertex_cells(shown))

        point_data = polydata.GetPointData()
        if pc.rgb is not None:
            point_data.AddArray(_vtk_colors(pc.rgb[::stride]))
        self._add_confidence_arrays(point_data, "points", pc.confidence, pc.uncertainty_m, total, stride)
        if self._mesh_actor is None:
            self._uncertainty_summary = _summarize_uncertainty(pc.uncertainty_m)
        point_data.AddArray(_vtk_scalars(xyz[:, 2].astype(np.float64) + self._height_datum_m, "height"))

        self._point_polydata = polydata

        if self._point_mapper is None:
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(self._point_size)
            actor.GetProperty().SetLighting(False)
            self.renderer.AddActor(actor)
            self._point_mapper = mapper
            self._point_actor = actor
        else:
            # Reuse the actor across partial updates so a growing cloud
            # does not churn the renderer's actor list once per window.
            self._point_mapper.SetInputData(polydata)

        self.set_color_mode(self._color_mode)
        self._track_bounds("cloud", xyz)
        if reset_camera:
            self._frame_data()
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def clear_point_cloud(self) -> None:
        if self._point_actor is not None:
            self.renderer.RemoveActor(self._point_actor)
        self._point_actor = None
        self._point_mapper = None
        self._point_polydata = None
        self._point_cloud = None
        self._point_locator = None
        self._cloud_updates = 0
        self._remove_scalar_bar()
        self._track_bounds("cloud", None)
        self._update_hud()
        self._render()

    def point_count(self) -> int:
        """Points in the loaded cloud (not the number drawn -- see module docs)."""
        return 0 if self._point_cloud is None else int(self._point_cloud.xyz.shape[0])

    def displayed_point_count(self) -> int:
        """Points actually on screen after display decimation."""
        if self._point_polydata is None:
            return 0
        return int(self._point_polydata.GetNumberOfPoints())

    def set_point_size(self, size: int) -> None:
        self._point_size = int(size)
        if self._point_actor is not None:
            self._point_actor.GetProperty().SetPointSize(self._point_size)
            self._render()

    def set_point_cloud_visible(self, visible: bool) -> None:
        if self._point_actor is not None:
            self._point_actor.SetVisibility(bool(visible))
            self._render()

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    def set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        uncertainty_m: np.ndarray | None = None,
    ) -> None:
        """Render a triangle mesh given (V, 3) vertices and (F, 3) face indices.

        ``confidence`` (tiers) and ``uncertainty_m`` (metres, NaN = not
        measured) are per-vertex, so the colour modes paint the SURFACE as
        a heatmap, not only the points.
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        n = int(vertices.shape[0])
        self._surface_xyz = vertices  # heights for volume / profile measurements
        self._height_grid = None

        def build(index: np.ndarray | None, tri: np.ndarray) -> vtkPolyData:
            pick = (lambda a: a) if index is None else (lambda a: a[index])
            poly = vtkPolyData()
            poly.SetPoints(_vtk_points(pick(vertices)))
            poly.SetPolys(_triangle_cells(tri))
            data = poly.GetPointData()
            if vertex_colors is not None and len(vertex_colors) == n:
                data.AddArray(_vtk_colors(pick(np.asarray(vertex_colors))))
            conf = None if confidence is None or len(confidence) != n else pick(np.asarray(confidence))
            sigma = None if uncertainty_m is None or len(uncertainty_m) != n else pick(np.asarray(uncertainty_m))
            self._add_confidence_arrays(data, "mesh", conf, sigma, None, 1)
            data.AddArray(_vtk_scalars(np.asarray(pick(vertices)[:, 2], dtype=np.float64) + self._height_datum_m, "height"))
            return poly

        def actor_for(poly: vtkPolyData) -> tuple[vtkActor, vtkPolyDataMapper]:
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.StaticOn()  # the data never changes after upload
            actor = vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.12)
            prop.SetSpecularPower(22)
            prop.SetColor(*theme.rgb_f("#8d939b"))
            self.renderer.AddActor(actor)
            return actor, mapper

        self._remove_mesh_actors()
        self._mesh_polydata = build(None, faces)
        self._mesh_actor, self._mesh_mapper = actor_for(self._mesh_polydata)
        self._mesh_actor.SetVisibility(self._mesh_visible)
        if faces.shape[0] > MESH_PROXY_FACES:
            keep, proxy_faces = _cluster_mesh(vertices, faces, MESH_PROXY_VERTICES)
            self._proxy_polydata = build(keep, proxy_faces)
            self._proxy_actor, self._proxy_mapper = actor_for(self._proxy_polydata)
            self._proxy_actor.SetVisibility(False)
        self._mesh_face_count = int(faces.shape[0])
        self._uncertainty_summary = _summarize_uncertainty(uncertainty_m)

        self.set_color_mode(self._color_mode)
        self._track_bounds("mesh", vertices)
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def _remove_mesh_actors(self) -> None:
        for actor in (self._mesh_actor, self._proxy_actor):
            if actor is not None:
                self.renderer.RemoveActor(actor)
        self._mesh_actor = self._proxy_actor = None
        self._mesh_mapper = self._proxy_mapper = None
        self._mesh_polydata = self._proxy_polydata = None
        self._proxy_active = False
        self._uncertainty_metric["mesh"] = False

    def _add_confidence_arrays(self, data, layer: str, confidence, sigma_m, total: int | None, stride: int) -> None:
        """"confidence" (tier) and "uncertainty" arrays: metres when measured, else the inverted tier."""
        if total is not None:
            confidence = None if confidence is None else confidence[::stride]
            sigma_m = None if sigma_m is None or len(sigma_m) != total else sigma_m[::stride]
        if confidence is not None:
            data.AddArray(_vtk_scalars(np.asarray(confidence, dtype=np.float64), "confidence"))
        metric = sigma_m is not None
        self._uncertainty_metric[layer] = metric
        if metric:
            data.AddArray(_vtk_scalars(np.asarray(sigma_m, dtype=np.float64), "uncertainty"))
        elif confidence is not None:
            # Inverse framing of the tier: 0 = certain, 2 = uncertain.
            data.AddArray(_vtk_scalars(2.0 - np.asarray(confidence, dtype=np.float64), "uncertainty"))

    def mesh_face_count(self) -> int:
        return self._mesh_face_count

    def set_mesh_visible(self, visible: bool) -> None:
        self._mesh_visible = bool(visible)
        if self._mesh_actor is not None:
            self._proxy_end(render=False)
            self._mesh_actor.SetVisibility(self._mesh_visible)
            self._render()

    def has_mesh(self) -> bool:
        return self._mesh_actor is not None

    # ------------------------------------------------------------------
    # Flight path
    # ------------------------------------------------------------------
    def set_flight_path(self, positions: np.ndarray) -> None:
        """Draw the drone track through ``positions`` ((N, 3), world/ENU).

        Coloured dark-to-bright with time so the direction of travel is
        readable without arrows, with a marker at each end.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 2:
            self.clear_flight_path()
            return

        n = int(positions.shape[0])

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(positions))
        polydata.SetLines(_polyline_cells([n]))
        polydata.GetPointData().SetScalars(_vtk_colors(_time_ramp(n), "time"))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetColorModeToDirectScalars()

        if self._path_actor is not None:
            self.renderer.RemoveActor(self._path_actor)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2.5)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetRenderLinesAsTubes(True)
        self.renderer.AddActor(actor)
        self._path_actor = actor

        self._set_path_markers(positions[0], positions[-1])
        self._track_bounds("path", positions)
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def _set_path_markers(self, start: np.ndarray, end: np.ndarray) -> None:
        """Two filled squares: green at the take-off end, red at the last frame."""
        points = np.vstack([start, end])

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(points))
        polydata.SetVerts(_vertex_cells(2))
        colors = np.array(
            [
                [int(c * 255) for c in theme.rgb_f(theme.OK)],
                [int(c * 255) for c in theme.rgb_f(theme.ERR)],
            ],
            dtype=np.uint8,
        )
        polydata.GetPointData().SetScalars(_vtk_colors(colors, "marker"))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetColorModeToDirectScalars()

        if self._path_marker_actor is not None:
            self.renderer.RemoveActor(self._path_marker_actor)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(11)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().SetRenderPointsAsSpheres(True)
        self.renderer.AddActor(actor)
        self._path_marker_actor = actor

    def clear_flight_path(self) -> None:
        for attr in ("_path_actor", "_path_marker_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.renderer.RemoveActor(actor)
                setattr(self, attr, None)
        self._track_bounds("path", None)
        self._render()

    def set_flight_path_visible(self, visible: bool) -> None:
        for actor in (self._path_actor, self._path_marker_actor):
            if actor is not None:
                actor.SetVisibility(bool(visible))
        self._render()

    def has_flight_path(self) -> bool:
        return self._path_actor is not None

    # ------------------------------------------------------------------
    # Camera frustums
    # ------------------------------------------------------------------
    def set_cameras(
        self,
        poses,
        intrinsics=None,
        scale: float | None = None,
    ) -> None:
        """Draw one wireframe frustum per pose.

        ``poses`` is a sequence of :class:`drishti3d.types.Pose` (``R``
        world-from-camera, ``t`` the camera centre in world/ENU). When
        ``intrinsics`` is given the frustum has the camera's true aspect
        and field of view, so a wrong focal length is visible as a
        footprint that does not match the ground; without it a generic
        4:3 / 60-degree pyramid is drawn instead.

        All frustums go into a single polydata -- one actor, not N -- so
        77 cameras cost one draw call rather than 77.
        """
        poses = list(poses or [])
        if not poses:
            self.clear_cameras()
            return

        centres = np.array([np.asarray(p.t, dtype=np.float64).reshape(3) for p in poses])

        if scale is None:
            if len(centres) > 1:
                spacing = np.linalg.norm(np.diff(centres, axis=0), axis=1)
                spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
                scale = float(np.median(spacing)) * 0.45 if spacing.size else 5.0
            else:
                scale = 5.0
            scale = float(np.clip(scale, 0.5, 60.0))
        self._camera_scale = float(scale)

        # Frustum corners in the camera frame (OpenCV: X right, Y down,
        # Z forward), at depth ``scale``.
        if intrinsics is not None:
            half_w = scale * (intrinsics.width / 2.0) / float(intrinsics.fx)
            half_h = scale * (intrinsics.height / 2.0) / float(intrinsics.fy)
        else:
            half_w = scale * 0.577  # ~60 deg horizontal
            half_h = half_w * 0.75

        local = np.array(
            [
                [0.0, 0.0, 0.0],            # 0 apex (camera centre)
                [-half_w, -half_h, scale],  # 1 top-left
                [half_w, -half_h, scale],   # 2 top-right
                [half_w, half_h, scale],    # 3 bottom-right
                [-half_w, half_h, scale],   # 4 bottom-left
            ]
        )
        #: closed image rectangle, then the four rays back to the apex,
        #: then a short "up" tick so the camera's roll is readable.
        segments = [
            (1, 2), (2, 3), (3, 4), (4, 1),
            (0, 1), (0, 2), (0, 3), (0, 4),
        ]

        rotations = np.array([np.asarray(p.R, dtype=np.float64).reshape(3, 3) for p in poses])
        # (N, 5, 3): rotate every local corner into the world frame at once.
        world = np.einsum("nij,kj->nki", rotations, local) + centres[:, None, :]

        n = len(poses)
        vertices = world.reshape(-1, 3)
        segment_array = np.array(segments, dtype=np.int64)
        offsets_per_camera = (np.arange(n, dtype=np.int64) * 5)[:, None, None]
        lines = (segment_array[None, :, :] + offsets_per_camera).reshape(-1, 2)

        # Explicit connectivity: the segments index shared corners (four
        # rays all start at the apex), so the consecutive-id shortcut
        # ``_polyline_cells`` uses does not apply here.
        from vtkmodules.util import numpy_support

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(vertices))
        cells = vtkCellArray()
        cells.SetData(
            numpy_support.numpy_to_vtkIdTypeArray(
                np.arange(lines.shape[0] + 1, dtype=np.int64) * 2, deep=True
            ),
            numpy_support.numpy_to_vtkIdTypeArray(
                np.ascontiguousarray(lines.reshape(-1), dtype=np.int64), deep=True
            ),
        )
        polydata.SetLines(cells)

        # Colour each frustum by its position in the flight, matching the
        # flight-path ramp exactly so a camera and the track agree.
        ramp = _time_ramp(n)
        polydata.GetPointData().SetScalars(_vtk_colors(np.repeat(ramp, 5, axis=0), "time"))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetColorModeToDirectScalars()

        if self._camera_actor is not None:
            self.renderer.RemoveActor(self._camera_actor)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(1.2)
        actor.GetProperty().SetOpacity(0.85)
        actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(actor)

        self._camera_actor = actor
        self._camera_count = n

        self._track_bounds("cameras", vertices)
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def clear_cameras(self) -> None:
        self.clear_camera_highlight()
        if self._camera_actor is not None:
            self.renderer.RemoveActor(self._camera_actor)
        self._camera_actor = None
        self._camera_count = 0
        self._track_bounds("cameras", None)
        self._render()

    def set_cameras_visible(self, visible: bool) -> None:
        if self._camera_actor is not None:
            self._camera_actor.SetVisibility(bool(visible))
            self._render()

    def camera_count(self) -> int:
        return self._camera_count

    def has_cameras(self) -> bool:
        return self._camera_actor is not None

    def highlight_camera(self, pose, intrinsics=None) -> None:
        """Draw one camera large and bright: the one whose frame the video pane shows."""
        self.clear_camera_highlight(render=False)
        if pose is None:
            self._render()
            return
        scale = self._camera_scale * 1.8
        if intrinsics is not None:
            half_w = scale * (intrinsics.width / 2.0) / float(intrinsics.fx)
            half_h = scale * (intrinsics.height / 2.0) / float(intrinsics.fy)
        else:
            half_w = scale * 0.577
            half_h = half_w * 0.75
        local = np.array(
            [
                [0.0, 0.0, 0.0],
                [-half_w, -half_h, scale],
                [half_w, -half_h, scale],
                [half_w, half_h, scale],
                [-half_w, half_h, scale],
                [0.0, -half_h * 1.35, scale],  # "up" tick: which image edge is the top
            ]
        )
        R = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
        centre = np.asarray(pose.t, dtype=np.float64).reshape(3)
        world = local @ R.T + centre
        segments = [(1, 2), (2, 3), (3, 4), (4, 1), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (5, 2)]
        cells = vtkCellArray()
        for a, b in segments:
            cells.InsertNextCell(2)
            cells.InsertCellPoint(a)
            cells.InsertCellPoint(b)
        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(world))
        polydata.SetLines(cells)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*theme.rgb_f(theme.ACCENT_BRIGHT))
        prop.SetLineWidth(3.0)
        prop.SetLighting(False)
        self.renderer.AddActor(actor)
        self._highlight_actor = actor
        self._render()

    def clear_camera_highlight(self, render: bool = True) -> None:
        if self._highlight_actor is not None:
            self.renderer.RemoveActor(self._highlight_actor)
            self._highlight_actor = None
            if render:
                self._render()

    def look_through(self, pose, intrinsics=None) -> None:
        """Put the 3D view at a camera, looking where it looked, with its field of view."""
        if pose is None:
            return
        R = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
        centre = np.asarray(pose.t, dtype=np.float64).reshape(3)
        forward, up = R[:, 2], -R[:, 1]
        bounds = self.renderer.ComputeVisiblePropBounds()
        ground = 0.5 * (bounds[4] + bounds[5]) if bounds[0] <= bounds[1] else centre[2] - 100.0
        distance = max(10.0, abs(centre[2] - ground) / max(0.2, abs(forward[2])))
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*centre)
        camera.SetFocalPoint(*(centre + forward * distance))
        camera.SetViewUp(*up)
        if intrinsics is not None:
            # Fit the whole frame: a view narrower than the image matches its
            # horizontal field of view, a wider one its vertical.
            view_w, view_h = self.vtk_widget.GetRenderWindow().GetSize()
            image_aspect = intrinsics.width / float(intrinsics.height)
            if view_h and view_w / float(view_h) < image_aspect:
                camera.UseHorizontalViewAngleOn()
                camera.SetViewAngle(float(np.degrees(2.0 * np.arctan(intrinsics.width / 2.0 / float(intrinsics.fx)))))
            else:
                camera.UseHorizontalViewAngleOff()
                camera.SetViewAngle(float(np.degrees(2.0 * np.arctan(intrinsics.height / 2.0 / float(intrinsics.fy)))))
        self.renderer.ResetCameraClippingRange()
        self._render()

    # ------------------------------------------------------------------
    # Whole-scene
    # ------------------------------------------------------------------
    def clear_scene(self) -> None:
        """Drop every data layer, keeping the grid/axes furniture."""
        self.clear_point_cloud()
        self.clear_cameras()
        self.clear_flight_path()
        self._remove_mesh_actors()
        self._mesh_face_count = 0
        self._uncertainty_summary = None
        self._track_bounds("mesh", None)
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def actor_count(self) -> int:
        return self.renderer.GetActors().GetNumberOfItems()

    # ------------------------------------------------------------------
    # Color modes
    # ------------------------------------------------------------------
    def set_color_mode(self, mode: str) -> None:
        """Colour every layer that carries the mode's array: points, mesh and its motion proxy."""
        if mode not in COLOR_MODES:
            raise ValueError(f"unknown color mode: {mode!r} (expected one of {COLOR_MODES})")
        self._color_mode = mode
        self._remove_scalar_bar()
        bar = None
        for mapper, poly, layer in (
            (self._point_mapper, self._point_polydata, "points"),
            (self._mesh_mapper, self._mesh_polydata, "mesh"),
            (self._proxy_mapper, self._proxy_polydata, "mesh"),
        ):
            if mapper is None or poly is None:
                continue
            legend = self._apply_color_mode(mapper, poly, mode, self._uncertainty_metric[layer])
            if legend is not None and mode == "height":
                legend = (legend[0], self._height_title, legend[2])
            # One legend: the surface's when there is one, it is what is looked at.
            if legend is not None and (bar is None or layer == "mesh"):
                bar = legend
        if bar is not None:
            self._add_scalar_bar(*bar)
        self._render()

    @staticmethod
    def _apply_color_mode(mapper, poly, mode: str, metric: bool):
        """Point ``mapper`` at ``poly``'s array for ``mode``; returns ``(lut, title, labels)`` or None."""
        point_data = poly.GetPointData()
        if mode != "rgb" and point_data.GetArray(mode) is None:
            return None  # this layer has no such channel: leave it as it is
        if mode == "rgb":
            if point_data.GetArray("rgb") is None:
                mapper.ScalarVisibilityOff()
                return None
            point_data.SetActiveScalars("rgb")
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            return None
        point_data.SetActiveScalars(mode)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(mode)
        mapper.SetColorModeToMapScalars()
        mapper.ScalarVisibilityOn()
        if mode == "uncertainty" and metric:
            lut, lo, hi = _uncertainty_metric_lookup_table(), 0.0, UNCERTAINTY_MAX_M
            title, labels = "Height \u00b1 (m)", None
        elif mode in ("confidence", "uncertainty"):
            lut = _confidence_lookup_table() if mode == "confidence" else _uncertainty_lookup_table()
            lo, hi = 0.0, 2.0
            title = mode.capitalize()
            labels = ["INFERRED", "LOW", "MEASURED"] if mode == "confidence" else ["LOW", "MEDIUM", "HIGH"]
        else:  # height
            z = np.asarray(point_data.GetArray("height").GetRange(), dtype=np.float64)
            lo, hi = float(z[0]), float(z[1])
            if hi <= lo:
                hi = lo + 1.0
            lut, title, labels = _height_lookup_table(lo, hi), "Height (m)", None
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(lo, hi)
        return lut, title, labels

    def color_mode(self) -> str:
        return self._color_mode

    def set_height_datum(self, offset_m: float, title: str = "Height (m above sea level)") -> None:
        """Label heights against a datum: ``offset_m`` is added to z (the local frame's origin altitude).

        Applies to geometry set after this call; the local-frame coordinates
        themselves (picking, measurements, exports) are unchanged.
        """
        self._height_datum_m = float(offset_m) if np.isfinite(offset_m) else 0.0
        self._height_title = title if self._height_datum_m else "Height (m)"

    def active_scalar_name(self) -> str | None:
        """Name of the point-data array currently driving color, if any."""
        if self._point_polydata is None:
            return None
        active = self._point_polydata.GetPointData().GetScalars()
        return active.GetName() if active is not None else None

    def _add_scalar_bar(self, lut: vtkLookupTable, title: str, labels: list[str] | None) -> None:
        bar = vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle(title)
        bar.SetNumberOfLabels(len(labels) if labels else 5)
        bar.SetWidth(0.035)
        bar.SetHeight(0.34)
        bar.SetPosition(0.945, 0.10)
        bar.SetBarRatio(0.28)
        bar.UnconstrainedFontSizeOn()

        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetColor(*theme.rgb_f(theme.TEXT_SECONDARY))
            prop.SetFontFamilyToArial()
            prop.SetFontSize(11)
            prop.BoldOff()
            prop.ItalicOff()
            prop.ShadowOff()
        bar.GetTitleTextProperty().SetFontSize(11)

        # vtkScalarBarActor is a vtkActor2D; this VTK build's Python
        # bindings expose 2D-prop management on vtkViewport only via the
        # generic AddViewProp/RemoveViewProp (no AddActor2D wrapper).
        self.renderer.AddViewProp(bar)
        self._scalar_bar = bar

    def _remove_scalar_bar(self) -> None:
        if self._scalar_bar is not None:
            self.renderer.RemoveViewProp(self._scalar_bar)
            self._scalar_bar = None

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def _build_hud(self) -> vtkTextActor:
        # Top-left. The bottom edge belongs to the scale ruler and the
        # bottom-right corner to the orientation gizmo; overlapping them
        # was the first thing that looked amateur in the render.
        actor = vtkTextActor()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.GetPositionCoordinate().SetValue(0.012, 0.975)
        prop = actor.GetTextProperty()
        prop.SetVerticalJustificationToTop()
        prop.SetFontFamilyToCourier()
        prop.SetFontSize(12)
        prop.SetColor(*theme.rgb_f(theme.TEXT_MUTED))
        prop.SetLineSpacing(1.25)
        prop.ShadowOff()
        self.renderer.AddViewProp(actor)
        return actor

    def _build_hint(self) -> vtkTextActor:
        actor = vtkTextActor()
        actor.SetInput(
            "No reconstruction loaded\n\n"
            "Open a drone video and press Run,\n"
            "or load a previous run from File > Open Run\n\n"
            "Drag: orbit  \u00b7  Shift+drag: pan  \u00b7  Scroll: zoom  \u00b7  Double-click: pivot"
        )
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(15)
        prop.SetColor(*theme.rgb_f(theme.TEXT_FAINT))
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToCentered()
        prop.SetLineSpacing(1.4)
        prop.ShadowOff()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.GetPositionCoordinate().SetValue(0.5, 0.5)
        self.renderer.AddViewProp(actor)
        return actor

    def _update_hud(self) -> None:
        """Corner readout: what is in the scene, and what is being hidden."""
        lines: list[str] = []
        if self._camera_count:
            lines.append(f"cameras    {self._camera_count:>12,}")
        total = self.point_count()
        if total:
            shown = self.displayed_point_count()
            lines.append(f"points     {total:>12,}")
            if shown < total:
                pct = 100.0 * shown / total
                lines.append(f"  drawn    {shown:>12,}  ({pct:.0f}%, 1:{self._display_stride})")
        if self._mesh_face_count:
            lines.append(f"triangles  {self._mesh_face_count:>12,}")
        if self._uncertainty_summary:
            lines.append(self._uncertainty_summary)

        self._hud_actor.SetInput("\n".join(lines))

        empty = not (self._camera_count or total or self._mesh_face_count)
        self._hint_actor.SetVisibility(1 if empty else 0)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _frame_data(self) -> None:
        """Fit the camera to the DATA, not to every visible prop.

        ``renderer.ResetCamera()`` with no argument frames the ground
        grid too, which is deliberately larger than the survey -- so the
        default framing pushed a 650 m site into the middle sixth of the
        window.
        """
        # Undo a "look through this camera" field of view (VTK's default is 30 deg).
        self.renderer.GetActiveCamera().UseHorizontalViewAngleOff()
        self.renderer.GetActiveCamera().SetViewAngle(30.0)
        bounds = self.scene_bounds()
        if bounds is None:
            self.renderer.ResetCamera()
        else:
            # Pad degenerate axes (a perfectly flat cloud, a single
            # camera) so ResetCamera has a non-zero extent to work with.
            padded = list(bounds)
            for axis in range(3):
                low, high = padded[2 * axis], padded[2 * axis + 1]
                if high - low < 1e-6:
                    padded[2 * axis] = low - 0.5
                    padded[2 * axis + 1] = high + 0.5
            self.renderer.ResetCamera(*padded)
        self.renderer.ResetCameraClippingRange()

    def reset_camera(self) -> None:
        self._frame_data()
        self._render()

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            raise ValueError(f"unknown view preset: {name!r} (expected one of {tuple(VIEW_PRESETS)})")
        px, py, pz, ux, uy, uz = VIEW_PRESETS[name]

        # Frame the data first so "distance" means something, then place
        # the camera on the requested axis at that distance and re-frame.
        self._frame_data()
        camera = self.renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        distance = camera.GetDistance() or 1.0

        camera.SetPosition(
            focal[0] + px * distance,
            focal[1] + py * distance,
            focal[2] + pz * distance,
        )
        camera.SetViewUp(ux, uy, uz)
        self._frame_data()
        self._render()

    # ------------------------------------------------------------------
    # Grid / axes / scale
    # ------------------------------------------------------------------
    def _nice_spacing(self, extent: float) -> float:
        """A 1/2/5 x 10^n grid step giving roughly 20 divisions across."""
        if extent <= 0:
            return 5.0
        raw = extent / 20.0
        magnitude = 10.0 ** np.floor(np.log10(raw))
        for step in (1.0, 2.0, 5.0, 10.0):
            if raw <= step * magnitude:
                return step * magnitude
        return 10.0 * magnitude

    def _build_grid(self, size: float = 100.0, spacing: float = 5.0) -> None:
        """A ground grid at z=0, with every fifth line emphasised."""
        n = int(size / spacing)
        coords = np.arange(-n, n + 1, dtype=np.float64) * spacing

        # Two segments per coordinate (one N-S, one E-W).
        along_y = np.stack(
            [
                np.stack([coords, np.full_like(coords, -size), np.zeros_like(coords)], axis=1),
                np.stack([coords, np.full_like(coords, size), np.zeros_like(coords)], axis=1),
            ],
            axis=1,
        )
        along_x = np.stack(
            [
                np.stack([np.full_like(coords, -size), coords, np.zeros_like(coords)], axis=1),
                np.stack([np.full_like(coords, size), coords, np.zeros_like(coords)], axis=1),
            ],
            axis=1,
        )
        segments = np.concatenate([along_y, along_x], axis=0)
        vertices = segments.reshape(-1, 3)

        # Emphasis: axis lines brightest, every fifth line a step up.
        index = np.concatenate([np.arange(-n, n + 1), np.arange(-n, n + 1)])
        faint = np.array(theme.rgb_f("#191c20")) * 255
        minor = np.array(theme.rgb_f("#22262b")) * 255
        major = np.array(theme.rgb_f("#333940")) * 255
        colors = np.where((index % 5 == 0)[:, None], minor, faint)
        colors = np.where((index == 0)[:, None], major, colors)
        vertex_colors = np.repeat(colors, 2, axis=0).astype(np.uint8)

        from vtkmodules.util import numpy_support

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(vertices))
        cells = vtkCellArray()
        count = segments.shape[0]
        cells.SetData(
            numpy_support.numpy_to_vtkIdTypeArray(
                np.arange(count + 1, dtype=np.int64) * 2, deep=True
            ),
            numpy_support.numpy_to_vtkIdTypeArray(np.arange(count * 2, dtype=np.int64), deep=True),
        )
        polydata.SetLines(cells)
        polydata.GetPointData().SetScalars(_vtk_colors(vertex_colors, "grid"))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetColorModeToDirectScalars()

        if self._grid_actor is not None:
            visible = bool(self._grid_actor.GetVisibility())
            self.renderer.RemoveActor(self._grid_actor)
        else:
            visible = True

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(1)
        actor.GetProperty().SetLighting(False)
        actor.SetVisibility(visible)
        self.renderer.AddActor(actor)

        self._grid_actor = actor
        self._grid_extent = size

    def _track_bounds(self, layer: str, xyz: np.ndarray | None) -> None:
        """Record a data layer's extent (or drop it when the layer goes away)."""
        if xyz is None or len(xyz) == 0:
            self._layer_bounds.pop(layer, None)
        else:
            data = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            finite = data[np.all(np.isfinite(data), axis=1)]
            if finite.size == 0:
                self._layer_bounds.pop(layer, None)
            else:
                low = finite.min(axis=0)
                high = finite.max(axis=0)
                self._layer_bounds[layer] = (
                    float(low[0]), float(high[0]),
                    float(low[1]), float(high[1]),
                    float(low[2]), float(high[2]),
                )
        self._fit_grid_to_scene()

    def scene_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """(xmin, xmax, ymin, ymax, zmin, zmax) over the data layers, or None."""
        if not self._layer_bounds:
            return None
        all_bounds = np.array(list(self._layer_bounds.values()), dtype=np.float64)
        return (
            float(all_bounds[:, 0].min()), float(all_bounds[:, 1].max()),
            float(all_bounds[:, 2].min()), float(all_bounds[:, 3].max()),
            float(all_bounds[:, 4].min()), float(all_bounds[:, 5].max()),
        )

    def _fit_grid_to_scene(self) -> None:
        """Rebuild the grid when the data outgrows it.

        A fixed 100 m grid under a 650 m survey is worse than no grid: it
        reads as the extent of the site. Sized from the DATA bounds only
        -- see ``_layer_bounds``.
        """
        bounds = self.scene_bounds()
        if bounds is None:
            return
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
        if extent <= 0:
            return

        # Half-extent, rounded up to a round number, with ~15% margin.
        target = float(np.ceil(extent * 0.58 / 25.0) * 25.0)
        target = max(target, 25.0)
        if abs(target - self._grid_extent) < 1e-6:
            return

        self._build_grid(size=target, spacing=self._nice_spacing(target * 2))

    def set_grid_visible(self, visible: bool) -> None:
        if self._grid_actor is not None:
            self._grid_actor.SetVisibility(bool(visible))
            self._render()

    def _build_axes_widget(self) -> None:
        axes = vtkAxesActor()
        axes.SetXAxisLabelText("E")
        axes.SetYAxisLabelText("N")
        axes.SetZAxisLabelText("U")
        for caption in (
            axes.GetXAxisCaptionActor2D(),
            axes.GetYAxisCaptionActor2D(),
            axes.GetZAxisCaptionActor2D(),
        ):
            prop = caption.GetCaptionTextProperty()
            prop.SetColor(*theme.rgb_f(theme.TEXT_SECONDARY))
            prop.ShadowOff()
            prop.ItalicOff()
            prop.BoldOn()

        widget = vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.86, 0.02, 1.0, 0.20)
        widget.EnabledOn()
        widget.InteractiveOff()
        self._axes_widget = widget

    def set_axes_visible(self, visible: bool) -> None:
        if self._axes_widget is not None:
            self._axes_widget.SetEnabled(1 if visible else 0)
            self._render()

    def _build_scale_bar(self) -> None:
        """A metric scale ruler along the viewport edge.

        This is a metric survey product; a 3D view with no scale reference
        invites exactly the wrong reading of it.
        """
        actor = vtkLegendScaleActor()
        actor.SetLabelModeToDistance()
        actor.TopAxisVisibilityOff()
        actor.LeftAxisVisibilityOff()
        actor.RightAxisVisibilityOff()
        actor.BottomAxisVisibilityOn()
        actor.LegendVisibilityOff()
        try:
            axis = actor.GetBottomAxis()
            axis.GetProperty().SetColor(*theme.rgb_f(theme.TEXT_FAINT))
            axis.GetLabelTextProperty().SetColor(*theme.rgb_f(theme.TEXT_MUTED))
            axis.GetLabelTextProperty().ShadowOff()
            axis.GetLabelTextProperty().SetFontSize(10)
        except AttributeError:  # pragma: no cover - VTK build dependent
            pass
        self.renderer.AddViewProp(actor)
        self._scale_actor = actor

    def set_scale_bar_visible(self, visible: bool) -> None:
        if self._scale_actor is not None:
            self._scale_actor.SetVisibility(1 if visible else 0)
            self._render()

    # ------------------------------------------------------------------
    # Measurement tools
    # ------------------------------------------------------------------
    #: Points each kind takes before it finishes by itself (None: until
    #: right-click / double-click / Enter), and the least it needs.
    _MEASURE_POINTS = {"point": (1, 1), "distance": (2, 2), "area": (None, 3), "volume": (None, 3), "profile": (None, 2)}
    _MEASURE_COLOURS = {
        "point": (1.0, 0.84, 0.0), "distance": (1.0, 0.62, 0.17), "area": (0.2, 0.8, 0.4),
        "volume": (1.0, 0.4, 0.2), "profile": (0.8, 0.4, 1.0),
    }

    def start_measurement(self, kind: str) -> None:
        """Click points on the SURFACE (a click, not a drag, which still orbits); right-click,
        double-click or Enter finishes a polygon or profile, Esc cancels."""
        if kind not in self._MEASURE_POINTS:
            raise ValueError(f"unknown measurement {kind!r}")
        self._clear_draft()
        self._measure_kind = kind
        self._measure_pts = []

    def start_measure_distance(self) -> None:
        self.start_measurement("distance")

    def start_measure_area(self) -> None:
        self.start_measurement("area")

    def measuring(self) -> str | None:
        return self._measure_kind

    def cancel_measurement(self) -> None:
        self._measure_kind = None
        self._measure_pts = []
        self._clear_draft()
        self._render()

    def measurements(self) -> list:
        """Finished measurements (``app.measure.Measurement``), oldest first."""
        return list(self._measurements)

    def clear_measurements(self) -> None:
        self.cancel_measurement()
        for actors in self._measure_actors:
            for actor in actors:
                self._overlay_renderer().RemoveViewProp(actor)
        self._measure_actors.clear()
        self._measurements.clear()
        self._render()

    def set_measurements_visible(self, visible: bool) -> None:
        for actors in self._measure_actors:
            for actor in actors:
                actor.SetVisibility(1 if visible else 0)
        self._render()

    def measurement_count(self) -> int:
        return len(self._measurements)

    def _surface_pick(self, x: int, y: int) -> np.ndarray | None:
        """World point on the model (mesh, else points) under display pixel (x, y), or None."""
        from vtkmodules.vtkRenderingCore import vtkPropCollection

        props = vtkPropCollection()
        for actor in (self._mesh_actor, self._proxy_actor, self._point_actor):
            if actor is not None and actor.GetVisibility():
                props.AddItem(actor)
        if props.GetNumberOfItems() == 0 or not self._picker.PickProp(x, y, self.renderer, props):
            return None
        return np.asarray(self._picker.GetPickPosition(), dtype=np.float64)

    def add_measurement_point(self, world_xyz) -> None:
        """Add a surface point (local metres) to the measurement being drawn, as a click would."""
        if self._measure_kind is None:
            raise RuntimeError("start_measurement() first")
        self._measure_pts.append(np.asarray(world_xyz, dtype=np.float64).reshape(3))
        auto, _ = self._MEASURE_POINTS[self._measure_kind]
        if auto is not None and len(self._measure_pts) >= auto:
            self.finish_measurement()
        else:
            self._draw_draft()

    def _add_measure_point(self, x: int, y: int) -> None:
        p = self._surface_pick(x, y)
        if p is None:
            return
        self._measure_pts.append(p)
        auto, _ = self._MEASURE_POINTS[self._measure_kind]
        if auto is not None and len(self._measure_pts) >= auto:
            self.finish_measurement()
        else:
            self._draw_draft()

    def finish_measurement(self) -> None:
        """Compute and keep the measurement being drawn (if it has enough points)."""
        kind = self._measure_kind
        if kind is None:
            return
        _, need = self._MEASURE_POINTS[kind]
        if len(self._measure_pts) < need:
            return
        pts = np.asarray(self._measure_pts, dtype=np.float64)
        try:
            m = self._compute_measurement(kind, pts)
        except ValueError as exc:
            self._measure_kind, self._measure_pts = None, []
            self._clear_draft()
            self.measurementMade.emit(kind, float("nan"), str(exc))
            return
        self._measure_kind, self._measure_pts = None, []
        self._clear_draft()
        self._measurements.append(m)
        self._measure_actors.append(self._measurement_actors(m))
        self._render()
        self.measurementMade.emit(kind, float(m.value), m.warning)
        self.measurementCompleted.emit(m)

    def _height_grid_for_measuring(self):
        from drishti3d.app.measure import HeightGrid

        if self._height_grid is None:
            xyz = self._surface_xyz if self._surface_xyz is not None else (
                self._point_cloud.xyz if self._point_cloud is not None else None)
            if xyz is None or len(xyz) == 0:
                raise ValueError("no surface loaded to measure on")
            self._height_grid = HeightGrid.from_points(xyz)
        return self._height_grid

    def _compute_measurement(self, kind: str, pts: np.ndarray):
        from drishti3d.app.measure import (
            Measurement,
            cut_fill_volume,
            elevation_profile,
            polygon_area_xy,
            polyline_length,
        )

        n = sum(1 for m in self._measurements if m.kind == kind) + 1
        warning = self._measurement_confidence_guard([list(p) for p in pts])
        datum = self._height_datum_m
        if kind == "point":
            elev = float(pts[0, 2] + datum)
            return Measurement(kind, pts, elev, "m", f"Point {n}", warning, {"elevation_m": elev})
        if kind == "distance":
            d3 = polyline_length(pts)
            dxy = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
            dz = float(pts[-1, 2] - pts[0, 2])
            return Measurement(kind, pts, d3, "m", f"Distance {n}", warning,
                               {"horizontal_m": dxy, "height_difference_m": dz})
        if kind == "area":
            area = polygon_area_xy(pts)
            perim = polyline_length(np.vstack([pts, pts[:1]]))
            return Measurement(kind, pts, area, "m²", f"Area {n}", warning, {"perimeter_m": perim})
        grid = self._height_grid_for_measuring()
        if kind == "volume":
            v = cut_fill_volume(grid, pts)
            return Measurement(kind, pts, v["above_m3"], "m³", f"Volume {n}", warning,
                               {k: v[k] for k in ("above_m3", "below_m3", "net_m3", "area_m2", "max_above_m",
                                                  "max_below_m", "coverage")})
        prof = elevation_profile(grid, pts)
        extra = {k: prof[k] for k in ("length_m", "climb_m", "descent_m", "max_slope_deg", "valid_fraction")}
        extra.update({"min_elevation_m": prof["min_m"] + datum, "max_elevation_m": prof["max_m"] + datum,
                      "_xy": prof["xy"], "_z": prof["elevation_m"], "_distance": prof["distance_m"]})
        return Measurement(kind, pts, prof["length_m"], "m", f"Profile {n}", warning, extra)

    def _polyline_actor(self, pts: np.ndarray, closed: bool, colour, width: float = 3.0, opacity: float = 1.0):
        poly = vtkPolyData()
        poly.SetPoints(_vtk_points(pts))
        lines = vtkCellArray()
        idx = list(range(len(pts))) + ([0] if closed and len(pts) > 2 else [])
        if len(idx) >= 2:
            lines.InsertNextCell(len(idx))
            for i in idx:
                lines.InsertCellPoint(i)
        poly.SetLines(lines)
        poly.SetVerts(_vertex_cells(len(pts)))
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*colour)
        prop.SetLineWidth(width)
        prop.SetPointSize(10)
        prop.SetRenderPointsAsSpheres(True)
        prop.SetOpacity(opacity)
        prop.LightingOff()
        return actor

    def _measurement_actors(self, m) -> list:
        from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D

        colour = self._MEASURE_COLOURS[m.kind]
        pts = m.points
        if m.kind == "profile" and m.extra.get("_xy") is not None:
            prof = np.c_[m.extra["_xy"], m.extra["_z"]]
            prof = prof[np.isfinite(prof).all(axis=1)]
            line = self._polyline_actor(prof if len(prof) >= 2 else pts, False, colour)
        else:
            line = self._polyline_actor(pts, m.kind in ("area", "volume"), colour)
        label = vtkBillboardTextActor3D()
        label.SetInput(f"{m.label}: {_measure_text(m)}")
        anchor = pts.mean(axis=0) if m.kind in ("area", "volume") else pts[-1]
        label.SetPosition(*anchor)
        tp = label.GetTextProperty()
        tp.SetColor(*colour)
        tp.SetFontSize(14)
        tp.SetBold(True)
        tp.SetBackgroundColor(0.05, 0.05, 0.07)
        tp.SetBackgroundOpacity(0.7)
        overlay = self._overlay_renderer()
        for actor in (line, label):
            overlay.AddViewProp(actor)
        return [line, label]

    def _draw_draft(self) -> None:
        self._clear_draft()
        if not self._measure_pts or self._measure_kind is None:
            return
        pts = np.asarray(self._measure_pts, dtype=np.float64)
        closed = self._measure_kind in ("area", "volume")
        self._draft_actor = self._polyline_actor(pts, closed, self._MEASURE_COLOURS[self._measure_kind], 2.0, 0.85)
        self._overlay_renderer().AddViewProp(self._draft_actor)
        self._render()

    def _clear_draft(self) -> None:
        if getattr(self, "_draft_actor", None) is not None:
            self._overlay_renderer().RemoveViewProp(self._draft_actor)
            self._draft_actor = None

    def _on_right_press(self, style, _event) -> None:
        if self._measure_kind is not None:
            self.finish_measurement()
            return
        style.OnRightButtonDown()

    def _on_right_release(self, style, _event) -> None:
        if self._measure_kind is None:
            style.OnRightButtonUp()

    def _on_key_press(self, _obj, _event) -> None:
        if self._measure_kind is None:
            return
        key = self.interactor.GetKeySym()
        if key == "Escape":
            self.cancel_measurement()
        elif key in ("Return", "KP_Enter"):
            self.finish_measurement()
        elif key == "BackSpace" and self._measure_pts:
            self._measure_pts.pop()
            self._draw_draft()

    def _nearest_point_index(self, world_point: np.ndarray) -> int | None:
        """Nearest cloud point, via a VTK locator built once and reused.

        The brute-force version scanned all N points per endpoint. On the
        3.97M-point result that is a ~100 MB temporary and a visible stall
        every time the operator finishes a measurement; a locator makes it
        a tree descent.
        """
        if self._point_polydata is None or self._point_polydata.GetNumberOfPoints() == 0:
            return None
        if self._point_locator is None:
            from vtkmodules.vtkCommonDataModel import vtkStaticPointLocator

            locator = vtkStaticPointLocator()
            locator.SetDataSet(self._point_polydata)
            locator.BuildLocator()
            self._point_locator = locator
        return int(self._point_locator.FindClosestPoint(list(map(float, world_point))))

    def _measurement_confidence_guard(self, world_points: list[list[float]]) -> str:
        """Flag a measurement if it falls near INFERRED geometry.

        Measuring inferred geometry is disallowed by default because that
        geometry was never directly observed by any camera -- it was
        filled in by the reconstruction model's prior, so a distance/area
        computed against it is not a surveyed quantity and must not be
        reported as one without a caveat.

        Returns a non-empty warning string when any measurement endpoint's
        nearest point-cloud neighbour is INFERRED; otherwise "".
        """
        if self._point_cloud is None or self._point_cloud.confidence is None:
            return ""

        # The locator indexes the DISPLAYED (strided) cloud, so map its
        # index back onto the full confidence array.
        confidence = self._point_cloud.confidence
        stride = max(1, self._display_stride)
        for world_point in world_points:
            index = self._nearest_point_index(np.asarray(world_point, dtype=np.float64))
            if index is None:
                continue
            full_index = min(index * stride, confidence.shape[0] - 1)
            if confidence[full_index] == int(Confidence.INFERRED):
                return _INFERRED_WARNING
        return ""

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def screenshot(self, path: str, scale: int = 2) -> None:
        """Write a PNG of the current view, at ``scale`` x the on-screen size."""
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        w2i.SetInputBufferTypeToRGB()
        if self._overlay is not None:
            # With the measurement layer present the filter's own re-render
            # crashes vtkCocoaRenderWindow (VTK 9, macOS); read the frame
            # just drawn instead. Device pixels already carry the display's
            # scale (2x on Retina).
            self._render()
            w2i.ShouldRerenderOff()
            w2i.ReadFrontBufferOn()
        else:
            w2i.SetScale(max(1, int(scale)))
            w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _on_left_press(self, style, _event) -> None:
        x, y = self.interactor.GetEventPosition()
        style.FindPokedRenderer(x, y)
        if style.GetCurrentRenderer() is None:
            return
        self._press_xy = (x, y)
        if self.interactor.GetRepeatCount() and self._measure_kind is not None:
            self.finish_measurement()  # double-click closes the polygon / profile
            self._press_xy = None
            return
        if self.interactor.GetRepeatCount() and self._pivot_at(x, y):
            return
        if self.interactor.GetShiftKey():
            self._left_mode = "pan"
            style.StartPan()
        elif self.interactor.GetControlKey():
            self._left_mode = "dolly"
            style.StartDolly()
        else:
            self._left_mode = "rotate"
            style.StartRotate()

    def _on_left_release(self, style, _event) -> None:
        mode, self._left_mode = self._left_mode, None
        if mode == "pan":
            style.EndPan()
        elif mode == "dolly":
            style.EndDolly()
        elif mode == "rotate":
            style.EndRotate()
        # Measuring: a click (not a drag, which orbited) places a point.
        press, self._press_xy = self._press_xy, None
        if self._measure_kind is not None and press is not None:
            x, y = self.interactor.GetEventPosition()
            if abs(x - press[0]) <= 4 and abs(y - press[1]) <= 4:
                self._add_measure_point(x, y)

    def _pivot_at(self, x: int, y: int) -> bool:
        """Double-click: orbit about the surface point under the cursor from now on."""
        if not self._picker.Pick(x, y, 0.0, self.renderer):
            return False
        camera = self.renderer.GetActiveCamera()
        target = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
        shift = target - np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        camera.SetFocalPoint(*target)
        camera.SetPosition(*(np.asarray(camera.GetPosition(), dtype=np.float64) + shift))
        self.renderer.ResetCameraClippingRange()
        self._render()
        return True

    def _wheel_zoom(self, factor: float) -> None:
        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self._proxy_begin()
        self._proxy_timer.start()
        self._render()

    def _proxy_begin(self) -> None:
        """Camera starts moving: draw the decimated stand-in instead of a huge mesh."""
        if self._proxy_actor is None or self._proxy_active or not self._mesh_visible:
            return
        self._proxy_active = True
        self._mesh_actor.SetVisibility(False)
        self._proxy_actor.SetVisibility(True)

    def _proxy_end(self, render: bool = True) -> None:
        self._proxy_timer.stop()
        if not self._proxy_active:
            return
        self._proxy_active = False
        if self._proxy_actor is not None:
            self._proxy_actor.SetVisibility(False)
        if self._mesh_actor is not None:
            self._mesh_actor.SetVisibility(self._mesh_visible)
        if render:
            self._render()

    def _render(self) -> None:
        if self._started:
            if self._overlay is not None:
                camera = self.renderer.GetActiveCamera()
                if self._overlay.GetActiveCamera() is not camera:
                    self._overlay.SetActiveCamera(camera)
            self.vtk_widget.GetRenderWindow().Render()

    def _overlay_renderer(self):
        """The measurement layer, created on first use (see ``__init__``)."""
        if self._overlay is None:
            overlay = vtkRenderer()
            overlay.SetLayer(1)
            overlay.InteractiveOff()
            overlay.SetActiveCamera(self.renderer.GetActiveCamera())
            window = self.vtk_widget.GetRenderWindow()
            window.SetNumberOfLayers(2)
            window.AddRenderer(overlay)
            self._overlay = overlay
        return self._overlay


def _summarize_uncertainty(sigma_m: np.ndarray | None) -> str | None:
    """HUD line: median height uncertainty and the share within 1 m."""
    if sigma_m is None:
        return None
    sigma = np.asarray(sigma_m, dtype=np.float64)
    sigma = sigma[np.isfinite(sigma)]
    if not sigma.size:
        return None
    return f"height \u00b1    {np.median(sigma):>9.2f} m  (<=1 m: {100.0 * np.mean(sigma <= 1.0):.0f}%)"


def _measure_text(m) -> str:
    """Short on-screen reading of a measurement."""
    if m.kind == "point":
        return f"{m.value:,.1f} m"
    if m.kind == "distance":
        return f"{m.value:,.2f} m (Δh {m.extra.get('height_difference_m', 0.0):+.2f} m)"
    if m.kind == "area":
        return f"{m.value:,.1f} m²"
    if m.kind == "volume":
        return f"{m.extra.get('above_m3', 0.0):,.1f} m³ above, {m.extra.get('below_m3', 0.0):,.1f} m³ below base"
    return f"{m.value:,.1f} m, climb {m.extra.get('climb_m', 0.0):,.1f} m"


def _polygon_area_xy(points: list[list[float]]) -> float:
    """Shoelace formula, projected onto the XY (horizontal) plane."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1, _ = points[i]
        x2, y2, _ = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
