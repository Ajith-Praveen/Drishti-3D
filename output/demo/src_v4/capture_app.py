"""Lossless full-resolution screenshot of the real DRISHTI-3D window with the DJI_1001 result loaded."""
import sys

sys.path.insert(0, "/tmp/demo2")
import cv2  # noqa: E402
from PySide6.QtCore import QEvent, QObject, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QSplitter  # noqa: E402

from appcap import grab_window  # noqa: E402
from drishti3d.app.main_window import MainWindow  # noqa: E402
from drishti3d.app.theme import apply_theme  # noqa: E402

RESULT = "/Users/ajith/sih/output/final_2026-09-27/DJI_1001_measured3d"
OUT = "/tmp/demo4/app_full.png"

app = QApplication(sys.argv[:1]); apply_theme(app)
avail = app.primaryScreen().availableGeometry()
W = min(1344, int(avail.width() * 0.96)); H = W * 9 // 16
class _NoInput(QObject):
    # the capture window must never react to the user's typing or clicks (their keys hit app shortcuts)
    BLOCK = {QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride, QEvent.Shortcut, QEvent.MouseButtonPress,
             QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.Wheel}

    def eventFilter(self, obj, ev):  # noqa: N802
        return ev.type() in self.BLOCK


_guard = _NoInput(); app.installEventFilter(_guard)
win = MainWindow(); win.resize(W, H); win.move(avail.x() + 10, avail.y() + 10)
win.setAttribute(Qt.WA_ShowWithoutActivating, True); win.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
win.show(); win.viewport.start()


def close_modals():
    w = QApplication.activeModalWidget()
    if isinstance(w, QDialog):
        print("closed modal:", w.windowTitle(), flush=True); w.accept()


def load():
    win._load_result(RESULT)
    QTimer.singleShot(5000, arrange)


def arrange():
    close_modals()
    win.inspector.setCurrentIndex(0)
    split = next(sp for sp in win.findChildren(QSplitter) if sp.orientation() == Qt.Vertical)
    tot = sum(split.sizes()); split.setSizes([int(tot * 0.86), tot - int(tot * 0.86)])
    for _ in range(6):
        app.processEvents()
    import math
    import numpy as np
    cam = win.viewport.renderer.GetActiveCamera(); cam.Dolly(1.25); cam.OrthogonalizeViewUp()
    up = np.array(cam.GetViewUp()); pos = np.array(cam.GetPosition()); fp = np.array(cam.GetFocalPoint())
    d = np.linalg.norm(pos - fp); h_world = 2 * d * math.tan(math.radians(cam.GetViewAngle()) / 2)
    shift = 0.0 * h_world * up          # no pan: model centred
    cam.SetPosition(*(pos + shift)); cam.SetFocalPoint(*(fp + shift))
    win.viewport.renderer.ResetCameraClippingRange()
    QTimer.singleShot(2500, shot)


def shot():
    close_modals()
    res = win._last_result
    label = getattr(res, "outcome_label", None)
    if win._thread is not None or label != "Valid" or len([q for q in res.poses if q is not None]) != 52:
        print("ABORT: app is not showing the DJI_1001 result:", label, win._thread, flush=True)
        app.exit(3); return
    f = grab_window(win)
    cv2.imwrite(OUT, cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print("result:", label, "cameras 52", flush=True)
    print("frame", f.shape, "dpr", win.devicePixelRatioF(), flush=True)
    app.quit()


modal_timer = QTimer(); modal_timer.timeout.connect(close_modals); modal_timer.start(500)
QTimer.singleShot(1200, load)
app.exec()
