"""Slide 4 screenshots: native-resolution 1.6:1 crops of the app's 3D view, plus the profile window and the maps."""
import glob
import json
import os

import cv2
import numpy as np
import tifffile

V = "/tmp/demo4/views"
OUT = "/Users/ajith/sih/output/slides/slide4/assets/screens"
os.makedirs(OUT, exist_ok=True)
ASPECT = 1.6
meta = json.load(open(f"{V}/meta.json"))


def crop(img, rect, top_frac):
    x, y, w, h = rect
    v = img[y:y + h, x:x + w]
    ch = int(round(w / ASPECT))
    if ch <= h:
        y0 = min(max(0, int(round(top_frac * h))), h - ch)
        return v[y0:y0 + ch].copy()
    cw = int(round(h * ASPECT)); x0 = (w - cw) // 2
    return v[:, x0:x0 + cw].copy()


def save(name, rgb):
    cv2.imwrite(f"{OUT}/{name}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(name, rgb.shape[1], "x", rgb.shape[0])


def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# 1 live preview: a frame of the real live run (the model building, cameras solved)
live = load("/tmp/demo2/run/r0250.jpg")
save("1_live_preview", live[208:208 + 500, 503:503 + 800].copy())     # 3D area only (tab bar starts lower)

# 2-4 colour modes, 5 measurements (+ the elevation-profile window)
for name, key, top in (("2_confidence", "confidence", 0.2), ("3_uncertainty_2.5D", "uncertainty_25d", 0.24), ("4_elevation", "height", 0.2)):
    save(name, crop(load(f"{V}/{key}.png"), meta["shots"][key]["viewport"], top))
m = crop(load(f"{V}/measure.png"), meta["shots"]["measure"]["viewport"], 0.18)
dlg = load(f"{V}/profile_dialog.png")
dw = int(m.shape[1] * 0.38); dh = int(dlg.shape[0] * dw / dlg.shape[1])
d = cv2.resize(dlg, (dw, dh), interpolation=cv2.INTER_AREA)
x0, y0 = 20, m.shape[0] - dh - 20                               # bottom-left: keeps every label visible
m[y0 - 4:y0 + dh + 4, x0 - 4:x0 + dw + 4] = (70, 86, 110)       # frame around the profile window
m[y0:y0 + dh, x0:x0 + dw] = d
save("5_measure_profile", m)

# 6 GeoTIFF outputs: orthomosaic | elevation (DSM, hill-shaded), same area
D = glob.glob("/tmp/demo2/app_run/run_*/output")[0]
ortho = tifffile.imread(f"{D}/orthomosaic.tif")[..., :3]
dsm = tifffile.imread(f"{D}/dsm.tif").astype(np.float64)
valid = ortho.max(-1) > 8
ys, xs = np.nonzero(valid)
bw = xs.max() - xs.min()
w = 0.42 * bw; h = w / ASPECT
f = 8                                                         # search the fullest window on a coarse grid
vm = valid[::f, ::f].astype(np.float64); ii = np.pad(vm.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
ww, hh = int(w / f), int(h / f)
best, X0, Y0 = -1.0, 0, 0
for yy in range(0, vm.shape[0] - hh, 4):
    for xx in range(0, vm.shape[1] - ww, 4):
        frac = (ii[yy + hh, xx + ww] - ii[yy, xx + ww] - ii[yy + hh, xx] + ii[yy, xx]) / (ww * hh)
        if frac > best:
            best, X0, Y0 = frac, xx * f, yy * f
print("maps window valid fraction", round(best, 3))
oc = ortho[Y0:Y0 + int(h), X0:X0 + int(w)]
sx, sy = dsm.shape[1] / ortho.shape[1], dsm.shape[0] / ortho.shape[0]
z = dsm[int(Y0 * sy):int((Y0 + h) * sy), int(X0 * sx):int((X0 + w) * sx)]
zf = np.where(np.isfinite(z), z, np.nanmedian(z))
gy, gx = np.gradient(zf * 2.0, 0.38)
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy); az, alt = np.radians(315), np.radians(40)
shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
lo, hi = np.nanpercentile(z, [2, 99]); u = np.clip((z - lo) / (hi - lo), 0, 1)
stops = np.array([(30, 70, 110), (40, 140, 120), (150, 190, 90), (235, 200, 90), (225, 120, 70), (250, 245, 240)], float)
xsu = np.linspace(0, 1, len(stops))
col = np.stack([np.interp(u, xsu, stops[:, k]) for k in range(3)], -1) * (0.3 + 0.7 * shade)[..., None]
col[~np.isfinite(z)] = (14, 17, 19)
OW = 1314; OH = int(round(OW / ASPECT))
left = cv2.resize(oc, (OW, OH), interpolation=cv2.INTER_AREA)
right = cv2.resize(col.astype(np.uint8), (OW, OH), interpolation=cv2.INTER_CUBIC)
out = left.copy(); out[:, OW // 2:] = right[:, OW // 2:]
out[~(cv2.resize(valid[Y0:Y0 + int(h), X0:X0 + int(w)].astype(np.uint8), (OW, OH)) > 0)] = (14, 17, 19)
out[:, OW // 2 - 3:OW // 2 + 3] = (255, 255, 255)
save("6_ortho_dsm", out)
