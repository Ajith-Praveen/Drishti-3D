"""Raw video beside the model.

Scrub through the keyframes: this pane shows the drone's own video frame,
the 3D view highlights the camera that took it, and "Look through this
camera" puts the 3D view at that camera with its field of view, so the
frame and the reconstruction can be compared directly.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import theme

_CACHE_FRAMES = 24
_MAX_WIDTH = 1280  # decoded frames are shrunk to this; the pane is never wider


class VideoSyncPanel(QWidget):
    """Video frame of the selected keyframe, with a slider over all posed keyframes."""

    #: (pose, intrinsics) of the selected camera; (None, None) when cleared.
    cameraSelected = Signal(object, object)
    lookThroughChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video_path: str | None = None
        self._entries: list[tuple[float, object, object, object]] = []
        self._capture = None
        self._cache: OrderedDict[float, QImage] = OrderedDict()
        self._image: QImage | None = None

        title = QLabel("Video beside model")
        title.setFont(theme.ui_font(12, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color:{theme.TEXT_PRIMARY};")

        self.frame_label = QLabel("Run or open a reconstruction to compare its video with the model.")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setWordWrap(True)
        self.frame_label.setMinimumSize(240, 150)
        self.frame_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.frame_label.setStyleSheet(f"background:{theme.BG_VOID}; color:{theme.TEXT_SECONDARY};")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.prev_button = QPushButton("◀")
        self.next_button = QPushButton("▶")
        for button in (self.prev_button, self.next_button):
            button.setFixedWidth(34)
            button.setEnabled(False)
        self.prev_button.setToolTip("Previous camera")
        self.next_button.setToolTip("Next camera")

        self.look_through = QCheckBox("Look through this camera")
        self.look_through.setToolTip("Put the 3D view at this camera, with its field of view")
        self.look_through.setEnabled(False)

        self.info_label = QLabel("")
        self.info_label.setFont(theme.mono_font(10))
        self.info_label.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
        self.info_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.next_button)

        column = QVBoxLayout(self)
        column.setContentsMargins(10, 8, 10, 8)
        column.setSpacing(6)
        column.addWidget(title)
        column.addWidget(self.frame_label, 1)
        column.addLayout(controls)
        column.addWidget(self.look_through)
        column.addWidget(self.info_label)

        # Decode only where the slider settles, not every position it passes.
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(60)
        self._settle.timeout.connect(self._show_current)
        self.slider.valueChanged.connect(lambda _value: self._settle.start())
        self.prev_button.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.next_button.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.look_through.toggled.connect(self._on_look_through)

    # ------------------------------------------------------------------
    def set_run(self, video_path: str | None, keyframes, poses) -> None:
        """Take a finished run's keyframes and camera poses (index-aligned lists)."""
        self.clear()
        keyframes = list(keyframes or [])
        poses = list(poses or [])
        entries = []
        for index, keyframe in enumerate(keyframes):
            pose = poses[index] if index < len(poses) else getattr(keyframe, "pose", None)
            timestamp = getattr(keyframe, "timestamp", None)
            if pose is None or timestamp is None or not math.isfinite(float(timestamp)):
                continue
            entries.append((float(timestamp), pose, getattr(keyframe, "intrinsics", None), getattr(keyframe, "telemetry", None)))
        if not entries:
            return
        if not video_path or not Path(video_path).exists():
            self.frame_label.setText(f"Video not found:\n{video_path or '(none recorded)'}")
            return
        self._video_path = str(video_path)
        self._entries = entries
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(entries) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        for widget in (self.slider, self.prev_button, self.next_button, self.look_through):
            widget.setEnabled(True)
        if self.isVisible():
            self._show_current()

    def clear(self) -> None:
        self._settle.stop()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._cache.clear()
        self._entries = []
        self._video_path = None
        self._image = None
        self.slider.setEnabled(False)
        for widget in (self.prev_button, self.next_button, self.look_through):
            widget.setEnabled(False)
        self.frame_label.setPixmap(QPixmap())
        self.frame_label.setText("Run or open a reconstruction to compare its video with the model.")
        self.info_label.setText("")

    def has_data(self) -> bool:
        return bool(self._entries)

    def refresh(self) -> None:
        """Show the selected camera again (the pane was just made visible)."""
        if self._entries:
            self._show_current()

    # ------------------------------------------------------------------
    def _show_current(self) -> None:
        if not self._entries:
            return
        index = self.slider.value()
        timestamp, pose, intrinsics, telemetry = self._entries[index]
        image = self._frame_at(timestamp)
        if image is None:
            self.frame_label.setText(f"Could not decode the video at {timestamp:.2f} s")
        else:
            self._image = image
            self._rescale()
        minutes, seconds = divmod(timestamp, 60.0)
        text = f"camera {index + 1} of {len(self._entries)}   video {int(minutes):02d}:{seconds:04.1f}"
        geo = getattr(telemetry, "geo", None) if telemetry is not None else None
        if geo is not None:
            text += f"\nGPS {geo.lat:.6f}, {geo.lon:.6f}   {geo.alt_msl:.1f} m"
        self.info_label.setText(text)
        self.cameraSelected.emit(pose, intrinsics)

    def _frame_at(self, timestamp: float) -> QImage | None:
        key = round(timestamp, 3)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        import cv2

        if self._capture is None:
            self._capture = cv2.VideoCapture(self._video_path)
        self._capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            return None
        height, width = bgr.shape[:2]
        if width > _MAX_WIDTH:
            bgr = cv2.resize(bgr, (_MAX_WIDTH, round(height * _MAX_WIDTH / width)), interpolation=cv2.INTER_AREA)
            height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self._cache[key] = image
        while len(self._cache) > _CACHE_FRAMES:
            self._cache.popitem(last=False)
        return image

    def _rescale(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(pixmap)

    def _on_look_through(self, checked: bool) -> None:
        self.lookThroughChanged.emit(bool(checked))
        if checked and self._entries:
            self._show_current()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()
