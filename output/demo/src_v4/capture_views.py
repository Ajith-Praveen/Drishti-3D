"""Slide 4 app views, lossless at device resolution: rgb / confidence / height (measured 3D), measurements with the
elevation-profile window, uncertainty (Terrain 2.5D run of the same flight). Input to the window is blocked."""
import json
import os
import sys

sys.path.insert(0, "/tmp/demo2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QSplitter  # noqa: E402

from appcap import grab_window, qimage_to_rgb  # noqa: E402
from drishti3d.app.main_window import MainWindow  # noqa: E402
from drishti3d.app.panels.profile_dialog import ProfileDialog  # noqa: E402
from drishti3d.app.theme import apply_theme  # noqa: E402

R3D = "/Users/ajith/sih/output/final_2026-09-27/DJI_1001_measured3d"
R25 = "/Users/ajith/sih/output/final_2026-09-26/DJI_1001_complete"
OUT = "/tmp/demo4/views"
os.makedirs(OUT, exist_ok=True)
PICKS = json.load(open("/tmp/demo3/picks3.json"))

app = QApplication(sys.argv[:1]); apply_theme(app)


class _NoInput(QObject):
    BLOCK = {QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride, QEvent.Shortcut, QEvent.MouseButtonPress,
             QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.Wheel}

    def eventFilter(self, obj, ev):  # noqa: N802
        return ev.type() in self.BLOCK


_guard = _NoInput(); app.installEventFilter(_guard)
avail = app.primaryScreen().availableGeometry()
W = min(1344, int(avail.width() * 0.96)); H = W * 9 // 16
win = MainWindow(); win.resize(W, H); win.move(avail.x() + 10, avail.y() + 10)
win.setAttribute(Qt.WA_ShowWithoutActivating, True); win.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
win.show(); win.viewport.start()
meta = {"shots": {}}


def close_modals():
    w = QApplication.activeModalWidget()
    if isinstance(w, QDialog):
        meta.setdefault("modals", []).append(w.windowTitle()); w.accept()


def vp_rect():
    vp = win.viewport.vtk_widget; dpr = win.devicePixelRatioF(); tl = vp.mapTo(win, QPoint(0, 0))
    return [int(round(tl.x() * dpr)), int(round(tl.y() * dpr)), int(round(vp.width() * dpr)), int(round(vp.height() * dpr))]


def save(name):
    f = grab_window(win)
    cv2.imwrite(f"{OUT}/{name}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    meta["shots"][name] = {"viewport": vp_rect(), "size": list(f.shape[:2])}
    print("saved", name, f.shape, flush=True)


def arrange(dolly=1.25):
    close_modals(); win.inspector.setCurrentIndex(0)
    split = next(sp for sp in win.findChildren(QSplitter) if sp.orientation() == Qt.Vertical)
    tot = sum(split.sizes()); split.setSizes([int(tot * 0.86), tot - int(tot * 0.86)])
    for _ in range(6):
        app.processEvents()
    cam = win.viewport.renderer.GetActiveCamera(); cam.Dolly(dolly)
    win.viewport.renderer.ResetCameraClippingRange(); win.viewport._render()


def cam_set(pos, fp, angle):
    c = win.viewport.renderer.GetActiveCamera(); c.SetFocalPoint(*fp); c.SetPosition(*pos); c.SetViewUp(0, 0, 1)
    c.SetViewAngle(angle); win.viewport.renderer.ResetCameraClippingRange(); win.viewport._render()


def check(label, cameras):
    res = win._last_result
    ok = getattr(res, "outcome_label", None) == label and len([q for q in res.poses if q is not None]) == cameras
    if win._thread is not None or not ok:
        raise SystemExit(f"ABORT: unexpected app state ({getattr(res, 'outcome_label', None)})")


def script():
    win._load_result(R3D); yield 5000
    check("Valid", 52); arrange(); yield 2000
    save("rgb")
    for mode in ("confidence", "height"):
        win._color_mode_actions[mode].trigger(); yield 1500
        save(mode)
    win._color_mode_actions["rgb"].trigger(); yield 800

    # measurement close-up, as in the demo video
    v, res = win.viewport, win._last_result
    C = np.array([np.asarray(q.t).reshape(3) for q in res.poses if q is not None]); c = C[int(0.3 * (len(C) - 1))]
    g = v._height_grid_for_measuring()
    zc = float(np.nanmedian(g.sample(np.array([c[:2]])))); tgt = np.array([c[0], c[1], zc])
    pc = tgt + np.array([60, -70, 70]) * 1.8
    cam_set(pc, tgt, 40); yield 400
    rw = v.vtk_widget.GetRenderWindow()

    def pick(fx, fy):
        w, h = rw.GetSize()
        for dx, dy in ((0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01), (0.02, 0.02), (-0.02, -0.02)):
            q = v._surface_pick(int((fx + dx) * w), int((1 - fy - dy) * h))
            if q is not None:
                return q
        return None

    P = {k: [q for q in (pick(fx, fy) for fx, fy in xy) if q is not None] for k, xy in PICKS.items()}
    allp = np.vstack([np.asarray(q) for qs in P.values() for q in qs])
    shift = np.r_[0.5 * (allp[:, :2].min(0) + allp[:, :2].max(0)) - tgt[:2], 0.0]
    cam_set(pc + shift, tgt + shift, 40); yield 400
    for kind in ("point", "distance", "area", "volume", "profile"):
        v.start_measurement(kind)
        for q in P[kind]:
            v.add_measurement_point(q)
        if v.measuring():
            v.finish_measurement()
        yield 300
    yield 1500
    dlgs = [w for w in QApplication.topLevelWidgets() if isinstance(w, ProfileDialog) and w.isVisible()]
    for d in dlgs:
        d.setAttribute(Qt.WA_ShowWithoutActivating, True)
    save("measure")
    if dlgs:
        d = dlgs[-1]; d.resize(620, 330); yield 800
        img = qimage_to_rgb(d.grab().toImage())
        cv2.imwrite(f"{OUT}/profile_dialog.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        for d in dlgs:
            d.close()
        yield 400
    meta["measurements"] = [{"kind": m.kind, "label": m.label, "value": m.value, "unit": m.unit} for m in v.measurements()]
    print("measurements", meta["measurements"], flush=True)

    # the Terrain 2.5D run of the same flight carries per-point uncertainty
    win._clear_measurements(); yield 300
    win._load_result(R25); yield 6000
    arrange(); yield 2000
    win._color_mode_actions["uncertainty"].trigger(); yield 1500
    save("uncertainty_25d")
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1, default=str)


gen = script()


def step():
    close_modals()
    try:
        delay = next(gen)
    except StopIteration:
        app.quit(); return
    except SystemExit as exc:
        print(exc, flush=True); app.exit(3); return
    QTimer.singleShot(int(delay), step)


modal_timer = QTimer(); modal_timer.timeout.connect(close_modals); modal_timer.start(500)
QTimer.singleShot(1200, step)
app.exec()
