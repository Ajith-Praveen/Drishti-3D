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

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkInteractionWidgets import (
    vtkDistanceWidget,
    vtkOrientationMarkerWidget,
)
from vtkmodules.vtkRenderingAnnotation import (
    vtkAxesActor,
    vtkLegendScaleActor,
    vtkScalarBarActor,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkTextActor,
)

# vtkRenderingOpenGL2 / vtkInteractionStyle register factory overrides
# (OpenGL render window, trackball interactor style, ...) purely as a side
# effect of import -- required even though nothing here is referenced by
# name.
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

from drishti3d.app import theme
from drishti3d.types import Confidence, PointCloud

COLOR_MODES = ("rgb", "confidence", "uncertainty", "height")

VIEW_PRESETS = {
    "top": (0, 0, 1, 0, 1, 0),
    "front": (0, -1, 0, 0, 0, 1),
    "side": (1, 0, 0, 0, 0, 1),
    "iso": (1, -1, 1, 0, 0, 1),
}

#: Above this many points the viewport renders a strided subsample. Tuned
#: so a mid-range integrated GPU still orbits at interactive rates; the
#: full cloud is always retained for measurement and export.
DISPLAY_POINT_BUDGET = 1_500_000

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

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.accept()


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

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.interactor.EnableRenderOff()

        # --- state -------------------------------------------------------
        self._point_cloud: PointCloud | None = None
        self._display_stride = 1
        self._cloud_updates = 0
        self._color_mode = "rgb"
        self._point_actor: vtkActor | None = None
        self._point_mapper: vtkPolyDataMapper | None = None
        self._point_polydata: vtkPolyData | None = None
        self._point_size = 3

        self._mesh_actor: vtkActor | None = None
        self._mesh_face_count = 0

        self._path_actor: vtkActor | None = None
        self._path_marker_actor: vtkActor | None = None
        self._camera_actor: vtkActor | None = None
        self._camera_count = 0

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
        self._measure_widgets: list = []
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
        if pc.confidence is not None:
            confidence = pc.confidence[::stride].astype(np.float64)
            point_data.AddArray(_vtk_scalars(confidence, "confidence"))
            # "uncertainty" is the inverse framing of confidence:
            # 0 = certain, 2 = uncertain.
            point_data.AddArray(_vtk_scalars(2.0 - confidence, "uncertainty"))
        point_data.AddArray(_vtk_scalars(xyz[:, 2].astype(np.float64), "height"))

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
    ) -> None:
        """Render a triangle mesh given (V, 3) vertices and (F, 3) face indices."""
        faces = np.ascontiguousarray(faces, dtype=np.int64)

        polydata = vtkPolyData()
        polydata.SetPoints(_vtk_points(np.ascontiguousarray(vertices, dtype=np.float64)))
        polydata.SetPolys(_triangle_cells(faces))

        if vertex_colors is not None:
            polydata.GetPointData().SetScalars(_vtk_colors(vertex_colors))

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        if self._mesh_actor is not None:
            self.renderer.RemoveActor(self._mesh_actor)

        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.12)
        prop.SetSpecularPower(22)
        if vertex_colors is None:
            prop.SetColor(*theme.rgb_f("#8d939b"))
        self.renderer.AddActor(actor)

        self._mesh_actor = actor
        self._mesh_face_count = int(faces.shape[0])

        self._track_bounds("mesh", vertices)
        self._update_hud()
        self.sceneChanged.emit()
        self._render()

    def mesh_face_count(self) -> int:
        return self._mesh_face_count

    def set_mesh_visible(self, visible: bool) -> None:
        if self._mesh_actor is not None:
            self._mesh_actor.SetVisibility(bool(visible))
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

    # ------------------------------------------------------------------
    # Whole-scene
    # ------------------------------------------------------------------
    def clear_scene(self) -> None:
        """Drop every data layer, keeping the grid/axes furniture."""
        self.clear_point_cloud()
        self.clear_cameras()
        self.clear_flight_path()
        if self._mesh_actor is not None:
            self.renderer.RemoveActor(self._mesh_actor)
        self._mesh_actor = None
        self._mesh_face_count = 0
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
        if mode not in COLOR_MODES:
            raise ValueError(f"unknown color mode: {mode!r} (expected one of {COLOR_MODES})")
        self._color_mode = mode

        if self._point_mapper is None or self._point_polydata is None:
            return

        poly = self._point_polydata
        mapper = self._point_mapper
        point_data = poly.GetPointData()

        self._remove_scalar_bar()

        if mode == "rgb":
            array = point_data.GetArray("rgb")
            if array is not None:
                point_data.SetScalars(array)
                mapper.SetScalarModeToUsePointData()
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
            else:
                mapper.ScalarVisibilityOff()
        elif mode in ("confidence", "uncertainty"):
            array = point_data.GetArray(mode)
            if array is not None:
                point_data.SetActiveScalars(mode)
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(mode)
                lut = (
                    _confidence_lookup_table()
                    if mode == "confidence"
                    else _uncertainty_lookup_table()
                )
                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(0, 2)
                mapper.SetColorModeToMapScalars()
                mapper.ScalarVisibilityOn()
                labels = (
                    ["INFERRED", "LOW", "MEASURED"]
                    if mode == "confidence"
                    else ["LOW", "MEDIUM", "HIGH"]
                )
                self._add_scalar_bar(lut, mode.capitalize(), labels)
        elif mode == "height":
            array = point_data.GetArray("height")
            if array is not None:
                point_data.SetActiveScalars("height")
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray("height")
                z = np.asarray(array.GetRange(), dtype=np.float64)
                z_min, z_max = float(z[0]), float(z[1])
                if z_max <= z_min:
                    z_max = z_min + 1.0
                lut = _height_lookup_table(z_min, z_max)
                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(z_min, z_max)
                mapper.SetColorModeToMapScalars()
                mapper.ScalarVisibilityOn()
                self._add_scalar_bar(lut, "Height (m)", None)

        self._render()

    def color_mode(self) -> str:
        return self._color_mode

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
            "or load a previous run from File > Open Run"
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
    def start_measure_distance(self) -> None:
        widget = vtkDistanceWidget()
        widget.SetInteractor(self.interactor)
        widget.CreateDefaultRepresentation()
        widget.On()

        def _on_end_interaction(obj, _event) -> None:
            rep = obj.GetDistanceRepresentation()
            value = rep.GetDistance()
            p1 = [0.0, 0.0, 0.0]
            p2 = [0.0, 0.0, 0.0]
            rep.GetPoint1WorldPosition(p1)
            rep.GetPoint2WorldPosition(p2)
            warning = self._measurement_confidence_guard([p1, p2])
            self.measurementMade.emit("distance", float(value), warning)

        widget.AddObserver("EndInteractionEvent", _on_end_interaction)
        self._measure_widgets.append(widget)

    def start_measure_area(self) -> None:
        """Start an area measurement.

        VTK does not ship a dedicated "area widget" the way it does a
        distance widget, so area is measured via a closed contour: the
        operator lays down points with a ``vtkContourWidget`` and the
        polygon area (shoelace formula, projected onto the horizontal
        plane) is reported when the contour is closed.
        """
        from vtkmodules.vtkInteractionWidgets import (
            vtkContourWidget,
            vtkOrientedGlyphContourRepresentation,
        )

        rep = vtkOrientedGlyphContourRepresentation()
        widget = vtkContourWidget()
        widget.SetInteractor(self.interactor)
        widget.SetRepresentation(rep)
        widget.On()

        def _on_end_interaction(obj, _event) -> None:
            contour_rep = obj.GetContourRepresentation()
            n = contour_rep.GetNumberOfNodes()
            if n < 3:
                return
            pts = []
            for i in range(n):
                p = [0.0, 0.0, 0.0]
                contour_rep.GetNthNodeWorldPosition(i, p)
                pts.append(p)
            area = _polygon_area_xy(pts)
            warning = self._measurement_confidence_guard(pts)
            self.measurementMade.emit("area", float(area), warning)

        widget.AddObserver("EndInteractionEvent", _on_end_interaction)
        self._measure_widgets.append(widget)

    def clear_measurements(self) -> None:
        for widget in self._measure_widgets:
            widget.Off()
        self._measure_widgets.clear()
        self._render()

    def set_measurements_visible(self, visible: bool) -> None:
        for widget in self._measure_widgets:
            widget.SetEnabled(1 if visible else 0)
        self._render()

    def measurement_count(self) -> int:
        return len(self._measure_widgets)

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
        w2i.SetScale(max(1, int(scale)))
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    def _render(self) -> None:
        if self._started:
            self.vtk_widget.GetRenderWindow().Render()


def _polygon_area_xy(points: list[list[float]]) -> float:
    """Shoelace formula, projected onto the XY (horizontal) plane."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1, _ = points[i]
        x2, y2, _ = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
