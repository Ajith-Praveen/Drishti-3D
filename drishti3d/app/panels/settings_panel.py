"""Reconstruction settings the operator actually needs to choose.

Scope
-----
Not every config field belongs in a UI. This panel exposes the handful
whose values an operator genuinely trades off per flight, each labelled
with what it costs and what it buys -- measured on this project's own
footage, not guessed:

- **Working resolution.** The single biggest cost in the pipeline.
  Measured on an 8-view window: 5.3 s at 518 px against ~52 s at 956 px
  with bf16, because a ViT's attention cost grows with the square of the
  token count. Geometry is ~60% of a run, so this knob roughly decides
  the runtime.
- **Frames per window.** How many views the backbone reasons over at
  once. More views means better-conditioned geometry within a window and
  fewer windows to align afterwards, but memory grows with the square of
  the view count. The pipeline may lower this to fit the device's memory
  budget (``geometry.windows.plan_window_size``), so it is a REQUEST, and
  the panel says so rather than implying it is final.
- **Quality profile.** Retunes keyframe spacing, matching and bundle
  adjustment together as one coherent trade-off.
- **Semantics.** Costs ~144 s and, on nadir aerial imagery, returns
  70-92% UNLABELLED because ADE20K does not transfer to that viewpoint.
  Off by default until that domain gap is closed.

Everything else stays in the YAML config, where it can be set without
cluttering the screen an operator uses under time pressure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from drishti3d.config import Config

#: Working resolutions offered, with the measured per-window cost of each
#: (8 views, bf16, Apple Silicon). Not a free-form field: the backbone
#: crops to a multiple of its patch size anyway, so arbitrary values buy
#: nothing and invite confusion about what actually ran.
_RESOLUTIONS: list[tuple[str, int]] = [
    ("518 px  — fastest", 518),
    ("700 px  — balanced", 700),
    ("924 px  — finer detail", 924),
    ("1288 px — maximum, slow", 1288),
]


def _default_output_root() -> Path:
    # Finder may launch a frozen app with cwd=/; /output is not writable.
    if getattr(sys, "frozen", False) or Path.cwd() == Path(Path.cwd().anchor):
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        return (Path(documents) if documents else Path.home()) / "DRISHTI-3D" / "output"
    return Path("output").resolve()

#: Mesh detail presets: (label, max_mesh_faces, voxel_count_budget).
#: Face and voxel budgets are paired deliberately -- a 776 x 853 m survey
#: needs ~5.3M faces and ~15.9M voxels to hold a 0.5 m voxel, and setting
#: either alone lets the other silently coarsen the result.
_MESH_DETAIL: tuple[tuple[str, int, int], ...] = (
    ("Draft - fastest, ~1.5 m surface", 2_000_000, 4_000_000),
    ("Balanced - ~0.8 m surface", 8_000_000, 16_000_000),
    ("Detailed - ~0.5 m surface, slow", 20_000_000, 40_000_000),
    ("Maximum - finest, very slow", 40_000_000, 80_000_000),
)

_PROFILES = ["fast", "balanced", "accurate"]

# Coefficient counts OpenCV's distortion models accept (k1, k2, p1, p2[, k3...]).
_DIST_LENGTHS = (4, 5, 8, 12, 14)


def _parse_distortion(text: str) -> list[float] | None:
    """Comma/space-separated OpenCV distortion coefficients; None when empty or malformed."""
    parts = text.replace(",", " ").split()
    if not parts:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) not in _DIST_LENGTHS or not all(math.isfinite(v) for v in values):
        return None
    return values


class SettingsPanel(QWidget):
    """Operator-facing reconstruction settings. Emits ``settingsChanged``."""

    settingsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Reconstruction")
        form = QFormLayout(box)

        self.resolution_combo = QComboBox()
        for label, value in _RESOLUTIONS:
            self.resolution_combo.addItem(label, value)
        self.resolution_combo.setCurrentIndex(0)
        self.resolution_combo.setToolTip(
            "Longest image side the geometry backbone runs at.\n"
            "Cost grows with the square of the token count: measured 5.3 s per\n"
            "8-view window at 518 px against ~52 s at 956 px."
        )
        form.addRow("Working resolution:", self.resolution_combo)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(4, 32)
        self.window_spin.setValue(8)
        self.window_spin.setToolTip(
            "Keyframes the backbone reconstructs together.\n"
            "Larger windows condition the geometry better and leave fewer\n"
            "junctions to align, but memory grows with the square of the\n"
            "view count. This is a REQUEST: the pipeline lowers it if the\n"
            "device cannot hold a window that size."
        )
        form.addRow("Frames per window:", self.window_spin)

        self.window_note = QLabel("request — lowered automatically if memory is short")
        self.window_note.setObjectName("mutedNote")
        self.window_note.setWordWrap(True)
        form.addRow("", self.window_note)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-3600.0, 3600.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setSpecialValueText("auto")
        self.offset_spin.setMinimum(-3600.01)
        self.offset_spin.setValue(-3600.01)
        self.offset_spin.setToolTip(
            "Seconds of flight-log time at video frame 0. 'auto' lets the\n"
            "pipeline detect it (DJI isVideo flag or a per-frame video clock).\n"
            "Set it when the offset was measured, e.g. 1.2 for PinPoint flight01's\n"
            "full video. For a trimmed clip add the clip's start time: 121.2 for\n"
            "flight01_120_480.mp4."
        )
        form.addRow("Telemetry offset (s):", self.offset_spin)

        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(0.0, 50000.0)
        self.focal_spin.setDecimals(1)
        self.focal_spin.setSingleStep(10.0)
        self.focal_spin.setSpecialValueText("auto")
        self.focal_spin.setValue(0.0)
        self.focal_spin.setToolTip(
            "Known focal length in pixels at the video's resolution (the optional\n"
            "camera intrinsics input). 'auto' estimates it from GPS baseline and\n"
            "image motion. A known value is kept fixed during reconstruction."
        )
        form.addRow("Camera focal (px):", self.focal_spin)

        self.dist_edit = QLineEdit()
        self.dist_edit.setPlaceholderText("auto -- k1, k2, p1, p2[, k3] (OpenCV)")
        self.dist_edit.setToolTip(
            "Lens distortion of the raw video, OpenCV order. Leave empty when unknown."
        )
        form.addRow("Lens distortion:", self.dist_edit)
        self.dist_note = QLabel("")
        self.dist_note.setWordWrap(True)
        self.dist_note.setProperty("error", True)
        self.dist_note.hide()
        form.addRow("", self.dist_note)

        self.ref_ortho_edit = QLineEdit()
        self.ref_ortho_edit.setPlaceholderText("off -- GeoTIFF orthophoto for absolute alignment")
        self.ref_ortho_edit.setToolTip(
            "Reference orthophoto (GeoTIFF). When set, the GPS-georeferenced model\n"
            "is matched against it to remove the GPS offset (geometry.reference_align)."
        )
        form.addRow("Reference orthophoto:", self.ref_ortho_edit)
        self.ref_dem_edit = QLineEdit()
        self.ref_dem_edit.setPlaceholderText("optional -- GeoTIFF elevation model")
        form.addRow("Reference DEM:", self.ref_dem_edit)

        self.reconstruction_combo = QComboBox()
        self.reconstruction_combo.addItem("Automatic — measured 3D (stereo + volumetric fusion)", "auto")
        self.reconstruction_combo.addItem("Measured 3D — stereo only; fails if cameras cannot be solved", "mvs3d")
        self.reconstruction_combo.addItem("Learned 3D — depth model with volumetric fusion", "full3d")
        self.reconstruction_combo.addItem("Terrain 2.5D — one height per location, walls inferred", "heightfield")
        self.reconstruction_combo.setToolTip(
            "Automatic measures depth in every camera (plane-sweep stereo with semi-global matching),\n"
            "keeps only depths neighbouring cameras agree on, and fuses them in 3D: trees, roofs and\n"
            "visible walls stay 3D. Nothing unseen is invented; downward flights close small gaps under\n"
            "canopy rims (up to 8 m, reported). If the cameras cannot be solved, Automatic falls back\n"
            "to the learned depth model. Resolution controls stereo detail. Oblique views are needed\n"
            "to measure façades. Terrain 2.5D is an explicit alternative for a single-height map."
        )
        form.addRow("Reconstruction:", self.reconstruction_combo)

        self.live_windows_check = QCheckBox("Live multi-window (fuse each window as it finishes)")
        self.live_windows_check.setChecked(True)
        self.live_windows_check.setToolTip(
            "Reconstruct in windows of the size above and fuse each one into\n"
            "the model the moment it finishes, so the model grows along the\n"
            "flight path. Windows are placed with the bundle-adjusted poses,\n"
            "so no window-to-window fit is needed. Off: one backbone call\n"
            "over the whole flight, nothing visible until it returns."
        )
        form.addRow("", self.live_windows_check)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(_PROFILES)
        self.profile_combo.setCurrentText("fast")
        self.profile_combo.setToolTip(
            "Retunes keyframe spacing, feature matching and bundle\n"
            "adjustment together as one speed/accuracy trade-off."
        )
        form.addRow("Quality profile:", self.profile_combo)

        # Mesh detail. Exposed because it is the sharpest speed/accuracy
        # lever in the pipeline and the two are weighted almost equally
        # in the evaluation, so the operator -- not a hardcoded default --
        # should choose where to sit on it.
        #
        # Measured on a 776 x 853 m survey (4-minute 4K video, Apple M-series):
        #
        #   faces   voxel     mesh roughness   fusion
        #     2M    1.55 m          5.69 m      190 s
        #     8M    0.77 m          3.66 m     1189 s
        #
        # The voxel budget has to move with it: a face cap that is too
        # low silently coarsens the voxel straight back, which is exactly
        # what made an earlier fix look like it had done nothing. So both
        # are set from one control rather than left to disagree.
        self.mesh_combo = QComboBox()
        for label, faces, voxels in _MESH_DETAIL:
            self.mesh_combo.addItem(label, (faces, voxels))
        self.mesh_combo.setCurrentIndex(1)
        self.mesh_combo.setToolTip(
            "Triangle budget for the exported mesh, and the TSDF voxel\n"
            "budget that has to match it.\n\n"
            "Higher = finer surface detail (pool edges, driveways, roof\n"
            "lines) and a correspondingly longer fusion stage. The point\n"
            "cloud is unaffected -- it is always written at full detail."
        )
        form.addRow("Mesh detail:", self.mesh_combo)

        self.mesh_note = QLabel("")
        self.mesh_note.setWordWrap(True)
        self.mesh_note.setStyleSheet("color: #9aa0a6;")
        form.addRow("", self.mesh_note)

        self.semantics_check = QCheckBox("Semantic classification")
        self.semantics_check.setChecked(False)
        self.semantics_check.setToolTip(
            "Per-point terrain/building/road/vegetation labels.\n"
            "Costs ~144 s and currently returns 70-92% unlabelled on nadir\n"
            "aerial imagery, because the segmentation model is trained on\n"
            "ground-level scenes."
        )
        form.addRow("", self.semantics_check)

        layout.addWidget(box)

        # --- output ---------------------------------------------------
        # Previously unreachable from the GUI at all: every run wrote to
        # ``./output`` relative to the launch directory, so two runs
        # overwrote each other's deliverables and a double-clicked .app
        # (whose working directory is ``/``) could not write at all.
        output_box = QGroupBox("Output")
        output_layout = QVBoxLayout(output_box)
        output_layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.output_edit = QLineEdit(str(_default_output_root()))
        self.output_edit.setToolTip(
            "Where this run writes its deliverables, stage renders and\n"
            "placement checks. The point clouds, meshes, GeoTIFFs and the\n"
            "accuracy report all land under here."
        )
        row.addWidget(self.output_edit, 1)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse_output)
        row.addWidget(browse)
        output_layout.addLayout(row)

        self.timestamp_check = QCheckBox("New sub-folder per run")
        self.timestamp_check.setChecked(True)
        self.timestamp_check.setToolTip(
            "Write each run into its own timestamped directory.\n"
            "Off, consecutive runs overwrite one another's deliverables\n"
            "and the stage renders of the previous flight linger on disk."
        )
        output_layout.addWidget(self.timestamp_check)

        layout.addWidget(output_box)
        layout.addStretch(1)

        # Seeded now so ``resolve_run_dir`` is valid before the first
        # run (the Settings panel shows the path, and Reveal Output
        # Folder works, without needing a run to have started).
        self.begin_run()

        self.output_edit.textChanged.connect(self.settingsChanged)
        self.timestamp_check.toggled.connect(self.settingsChanged)
        self.resolution_combo.currentIndexChanged.connect(self.settingsChanged)
        self.window_spin.valueChanged.connect(self.settingsChanged)
        self.profile_combo.currentTextChanged.connect(self.settingsChanged)
        self.reconstruction_combo.currentIndexChanged.connect(self.settingsChanged)
        self.mesh_combo.currentIndexChanged.connect(self.settingsChanged)
        self.mesh_combo.currentIndexChanged.connect(self._update_mesh_note)
        self.semantics_check.toggled.connect(self.settingsChanged)
        self.focal_spin.valueChanged.connect(self.settingsChanged)
        self.focal_spin.valueChanged.connect(self._validate_distortion)
        self.dist_edit.textChanged.connect(self._validate_distortion)
        self.dist_edit.textChanged.connect(self.settingsChanged)
        self._update_mesh_note()

    # -- values ---------------------------------------------------------

    @property
    def max_image_size(self) -> int:
        return int(self.resolution_combo.currentData())

    @property
    def window_size(self) -> int:
        return int(self.window_spin.value())

    @property
    def max_mesh_faces(self) -> int:
        return int(self.mesh_combo.currentData()[0])

    @property
    def voxel_count_budget(self) -> int:
        return int(self.mesh_combo.currentData()[1])

    def _update_mesh_note(self) -> None:
        faces, voxels = self.mesh_combo.currentData()
        self.mesh_note.setText(
            f"{faces / 1e6:g}M faces, {voxels / 1e6:g}M voxels. "
            "Fusion time scales roughly with the face count; the exported "
            "point cloud keeps full detail regardless."
        )

    @property
    def quality_profile(self) -> str:
        return self.profile_combo.currentText()

    @property
    def semantics_enabled(self) -> bool:
        return self.semantics_check.isChecked()

    @property
    def telemetry_offset_s(self) -> float | None:
        """The explicit offset, or ``None`` when the field is on 'auto'."""
        v = float(self.offset_spin.value())
        return None if v <= self.offset_spin.minimum() else v

    def set_telemetry_offset(self, seconds: float | None) -> None:
        self.offset_spin.setValue(self.offset_spin.minimum() if seconds is None else float(seconds))

    @property
    def live_windows(self) -> bool:
        return self.live_windows_check.isChecked()

    @property
    def camera_fx(self) -> float | None:
        """The known focal length, or ``None`` when the field is on 'auto'."""
        v = float(self.focal_spin.value())
        return None if v <= self.focal_spin.minimum() else v

    @property
    def dist_coeffs(self) -> list[float] | None:
        """The typed distortion coefficients; ``None`` when empty, malformed, or without a focal.

        Distortion coefficients are defined relative to a focal length, so
        they are only passed on together with a known one.
        """
        if self.camera_fx is None:
            return None
        return _parse_distortion(self.dist_edit.text())

    def _validate_distortion(self) -> None:
        text = self.dist_edit.text().strip()
        if not text:
            self.dist_note.hide()
            return
        if _parse_distortion(text) is None:
            self.dist_note.setText("ignored: needs 4, 5, 8, 12 or 14 numbers (k1, k2, p1, p2[, k3...])")
        elif self.camera_fx is None:
            self.dist_note.setText("ignored: set Camera focal too -- distortion is relative to a focal length")
        else:
            self.dist_note.hide()
            return
        self.dist_note.show()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    @property
    def output_root(self) -> Path:
        text = self.output_edit.text().strip()
        return Path(text).expanduser() if text else _default_output_root()

    def begin_run(self) -> None:
        """Stamp a fresh run directory. Call once, when a run starts.

        The stamp is cached rather than generated inside
        ``resolve_run_dir`` so that every call during one run -- building
        the config, pointing the renders panel, revealing the folder --
        names the SAME directory. A freshly-generated timestamp per call
        would send the pipeline and the viewer to different places.
        """
        from time import strftime

        self._run_stamp = strftime("%Y%m%d_%H%M%S")

    def resolve_run_dir(self) -> Path:
        """The directory this run writes into.

        With "New sub-folder per run" on, ``<root>/run_<stamp>``;
        otherwise the root itself.
        """
        root = self.output_root
        if self.timestamp_check.isChecked():
            return root / f"run_{self._run_stamp}"
        return root

    def build_config(self) -> Config:
        """A ``Config`` reflecting the current selections.

        Each profile table is applied EXPLICITLY, mirroring what
        ``config.load_config`` does for a YAML file.

        ``Config(quality_profile="fast")`` does NOT retune anything -- it
        only stores the string. Verified: with that constructor alone,
        ``matching.max_points_in_ba`` stays None and ``detect_scale``
        stays 1.0 for every profile, so the operator's choice would have
        been inert. That is the same class of bug that left MatchingConfig
        dead in the CLI for the whole project, so it is spelled out here
        rather than trusted to the constructor.
        """
        from dataclasses import replace

        from drishti3d.config import (
            GeometryConfig,
            MatchingConfig,
            SemanticsConfig,
            TriageConfig,
            apply_geometry_quality_profile,
            apply_quality_profile,
            apply_semantics_quality_profile,
            apply_triage_quality_profile,
        )

        profile = self.quality_profile
        config = Config(quality_profile=profile)
        config.matching = apply_quality_profile(MatchingConfig(), profile)
        config.triage = apply_triage_quality_profile(TriageConfig(), profile)
        config.semantics = apply_semantics_quality_profile(SemanticsConfig(), profile)

        # Profile first, then the operator's explicit choices on top --
        # the same layering order load_config uses.
        geometry = apply_geometry_quality_profile(GeometryConfig(), profile)
        config.geometry = replace(
            geometry,
            dense_method=self.reconstruction_combo.currentData(),
            max_image_size=self.max_image_size,
            window_size=self.window_size,
            single_inference=not self.live_windows,
        )
        config.semantics = replace(config.semantics, enabled=self.semantics_enabled)

        # Fusion budgets, from the one Mesh detail control. FusionConfig
        # has no quality-profile function of its own, so these are set
        # directly -- and set TOGETHER, because either one alone is
        # overridden by the other (see the Mesh detail comment above).
        config.fusion = replace(
            config.fusion,
            max_mesh_faces=self.max_mesh_faces,
            voxel_count_budget=self.voxel_count_budget,
        )

        # ``<run>/output`` -- the stage renders and placement checks are
        # written to ``<run>/preview`` and ``<run>/placement``, which
        # ``pipeline.stages._preview_dir`` derives as siblings of the
        # export directory. Anchoring the deliverables one level down is
        # what makes that layout come out right.
        config.export = replace(config.export, output_dir=str(self.resolve_run_dir() / "output"))
        ortho = self.ref_ortho_edit.text().strip()
        dem = self.ref_dem_edit.text().strip()
        config.reference = replace(config.reference, ortho_path=ortho or None, dem_path=dem or None)
        config.ingest = replace(config.ingest, camera_fx=self.camera_fx, camera_dist_coeffs=self.dist_coeffs)
        return config

    def apply_config(self, config: Config) -> None:
        """Reflect a loaded ``Config`` in the controls.

        Only the fields this panel actually exposes are adopted; the
        rest of the config stays where it is, in the YAML.
        """
        method = config.geometry.dense_method
        if method == "mapanything":
            method = "full3d"
        mode_index = self.reconstruction_combo.findData(method)
        if mode_index >= 0:
            self.reconstruction_combo.setCurrentIndex(mode_index)
        index = self.resolution_combo.findData(int(config.geometry.max_image_size))
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

        self.window_spin.setValue(int(config.geometry.window_size))
        self.live_windows_check.setChecked(not bool(getattr(config.geometry, "single_inference", True)))

        if config.quality_profile in _PROFILES:
            self.profile_combo.setCurrentText(config.quality_profile)

        faces = int(getattr(config.fusion, "max_mesh_faces", 0) or 0)
        if faces:
            # Nearest preset at or below the configured budget, so a
            # hand-tuned YAML never silently gets a finer mesh than it
            # asked for.
            best = max(
                (i for i in range(self.mesh_combo.count()) if self.mesh_combo.itemData(i)[0] <= faces),
                default=0,
            )
            self.mesh_combo.setCurrentIndex(best)

        self.semantics_check.setChecked(bool(getattr(config.semantics, "enabled", False)))

        fx = getattr(config.ingest, "camera_fx", None)
        self.focal_spin.setValue(self.focal_spin.minimum() if fx is None else float(fx))
        dist = getattr(config.ingest, "camera_dist_coeffs", None)
        self.dist_edit.setText(", ".join(f"{v:g}" for v in dist) if dist else "")

        output_dir = getattr(config.export, "output_dir", "") or ""
        if output_dir:
            # The YAML names the EXPORT directory; this panel edits its
            # parent, the run root (see build_config).
            self.output_edit.setText(str(Path(output_dir).resolve().parent))
            self.timestamp_check.setChecked(False)

    def set_effective_window_size(self, planned: int) -> None:
        """Report what the pipeline actually used, once it has decided.

        The requested size is an upper bound; showing only the request
        would let an operator believe a run used more views than it did.
        """
        requested = self.window_size
        if planned == requested:
            self.window_note.setText(f"using {planned} frames per window, as requested")
        else:
            self.window_note.setText(
                f"using {planned} frames per window — lowered from {requested} to fit the memory budget"
            )
