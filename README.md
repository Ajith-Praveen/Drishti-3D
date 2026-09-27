# DRISHTI-3D

Desktop app that turns a single-pass drone video into a georeferenced 3D model.
Smart India Hackathon 2026, problem statement SIH26158 (NTRO).

## Results at a glance

| | Measured on real flights |
|---|---|
| **Speed** | 11.4-minute DJI video -> finished model in **6.9 min** on an M-series laptop (target: < 15 min per 10 min of video); PinPoint flight01 6-min video in ~7.8 min |
| **Accuracy** | flight01 (Spain): surveyed features sit a median **0.29 m** from the national orthophoto (IGN PNOA, 13 features, all <= 1 m) in measured 3D, 0.85 m (25 features) in Terrain 2.5D; camera track vs COLMAP 1.6 m |
| **Model** | true 3D mesh from per-camera stereo: DJI_1001 **8.0 M vertices, 84 % directly measured**, confidence per point; unseen areas are not invented |
| **Outputs** | OBJ, PLY, LAS, GeoTIFF (DSM, DTM, orthomosaic, confidence), glTF/GLB, FBX, textured OBJ/GLB, HTML/TXT report, cameras.json |
| **Viewer** | desktop app with live preview, heatmaps, and point / distance / area / volume / profile tools exporting GeoJSON and KML |

Limits, stated plainly: absolute accuracy has not been checked against RTK
checkpoints (none were available); on flight01, strips flown in opposite
directions disagree by ~3 m in depth, so measured 3D covers only part of that
flight (Terrain 2.5D covers it fully); a downward camera cannot see facades.

## Pipeline stages

1. **Ingest** — decode the drone video and align it with flight telemetry.
2. **Triage** — select a well-distributed, sharp, low-blur set of keyframes.
3. **Clock sync** — check the telemetry clock against the keyframes' own image
   rotation (see below).
4. **Pose prior** — bundle-adjust the keyframe cameras against GPS and gimbal priors.
5. **Geometry** — `auto` (default) measures depth per camera with multi-view
   stereo, validates it across views, and reconstructs volumetrically in 3D.
6. **Fusion** — merge per-frame geometry into a single confidence-weighted surface.
7. **Export** — georeference and write the final model (e.g. LAS/PLY) to disk.

## Reconstruction modes

Settings > Reconstruction (or `--dense-method`):

- **Automatic / Measured 3D** (`auto`, default, or `mvs3d`;
  `drishti3d/geometry/mvs3d.py`). A true 3D mesh, measured, for downward,
  forward and oblique footage alike:
  - depth per camera by GPU plane-sweep stereo (NCC, best 2 of 4 source
    views) with semi-global matching over the cost volume, so the depth maps
    are dense and smooth instead of speckled;
  - on dense mapping flights, depth maps for a subset of reference views
    chosen so every part of the ground is still covered five times (all
    keyframes remain stereo sources);
  - a depth is kept only if neighbouring depth maps agree with it in depth
    and after a round-trip reprojection (1 px);
  - the survivors are fused in an Open3D TSDF volume and meshed; walls,
    tree crowns and overhangs stay 3D;
  - downward flights also get a true-ortho texture (orthomosaic.tif,
    textured OBJ/glTF), and enclosed gaps up to 8 m that a downward camera
    cannot see (under canopy rims, beside walls) are closed from their rims
    without new vertices; the closed area is reported.

  Nothing unseen is invented: the outer border and large unseen areas stay
  open. If the cameras cannot be solved, Automatic falls back to Learned 3D;
  explicit Measured 3D stops with an error instead. The resolution setting
  sets the stereo resolution (at least 768 px).
- **Learned 3D** (`full3d`). Optional learned multi-view depth followed by
  volumetric TSDF meshing. Neither nadir height-map fusion nor ground-footprint
  view culling is used. This is the legacy depth-model path, not the default.
- **Terrain 2.5D** (`heightfield`). One height per ground cell, walls and
  unseen areas filled and tiered INFERRED. Complete DSM coverage, 2.5D shape.

Measured on this project's footage (MPS, M-series Mac):

| Flight | Measured 3D | Notes |
|---|---|---|
| DJI_1001 (nadir, ~280 m, 11.4 min) | 50 views (49 depth maps), 87 % of depths confirmed across views, 8.0 M vertices at 0.5 m, 84 % MEASURED; geometry 143 s, run ~455 s | trees as crowns, houses as blocks; Terrain 2.5D extruded both into prisms |
| PinPoint flight01 (nadir grid, ~116 m) | 185 views (84 depth maps), 2.2 M vertices at 0.28 m; geometry 174 s, run ~470 s (budget 540 s) | stable features vs the IGN orthophoto 0.29 m median (13 features, all <= 1 m; Terrain 2.5D 0.85 m) but 26/64 survey points on measured surface (Terrain 2.5D 61/64): strips flown in opposite directions disagree by ~3 m in depth, so that ground stays open |
| Front_View_Light (vineyard, ~1 m up, forward) | 11 views, 184 k vertices at 0.06 m; geometry 26 s | vine rows and trees; the near ground is seen only at grazing angles |

The flight01 strip disagreement is a camera-solve limit, not a stereo one:
matching across strips (`_MATCH_FOOTPRINT_FRACTION` in
`pipeline/stages.py`) makes the strips agree to 0.3 m but bends the block
against IGN (rolling shutter flips its skew with flight direction and the
camera model is rigid), so it stays off until the camera model handles it.

```bash
.venv/bin/python -m drishti3d.pipeline.runner VIDEO.mp4 \
  --telemetry LOG.csv --config full_3d.yaml --out output/full3d_run
```

`--dense-method full3d` overrides a loaded configuration. `mapanything` is
accepted as a legacy alias. Full 3D costs more time and memory than terrain
MVS, and does not by itself establish metric accuracy or recover unseen
walls. Use footage with oblique views for façades; if the log lacks camera
orientation and the video is oblique, set
`geometry.assume_nadir_without_gimbal: false`. Reports record the actual
representation; GeoTIFF DSM/DTM exports remain 2.5D derived products.

Launch the updated source app with `Run DRISHTI-3D (from source).command`;
previously built application bundles need rebuilding to include changes.

## Optional terrain surface: height-field multi-view stereo

With explicit `geometry.dense_method: heightfield`, a nadir flight whose keyframes
all have a bundle-adjusted pose gets its surface measured directly
(`drishti3d/geometry/heightfield.py`):

- It sweeps candidate heights for every ground cell, projecting through the
  solved lens into every view that sees the cell.
- It keeps the height where the views agree.
- It colours each cell with the median across the views: a true orthophoto.
- The mesh is single-layer by construction, and it is photo-textured through
  planar UVs (`model_textured.obj/.png`, `model_textured.glb`), so no texture
  atlas or `xatlas` is needed.

On PinPoint flight01, with the same cameras, this gives DSM-vs-DEM MAD 1.4-1.7 m
in ~30 s, against 7.5 m in ~20 min for backbone depth fusion. See
`evidence/flight01-benchmark.md`. `dense_method: mapanything` forces the
backbone with full-3D fusion; `heightfield` reports an error when the flight is not nadir.

## Known camera calibration and telemetry clock

- The problem statement lists camera intrinsics as an optional input:
  - `--fx/--fy/--cx/--cy`, `--hfov`, `--dist "k1,k2,p1,p2[,k3]"` and
    `--calibration-width` on `drishti3d-run`;
  - `ingest.camera_*` in a config;
  - Camera focal / Lens distortion in Settings.

  All of these supply a calibration that is kept fixed (provenance `user`).
- `TimeSyncStage` measures the video-to-log clock offset from keyframe rotation
  against the logged heading:
  - an unmeasured offset is replaced;
  - an explicit one is kept, but a disagreement over `ingest.auto_sync_warn_s`
    is reported on the report card;
  - `ingest.auto_sync: correct` applies the measurement anyway, and `off` skips it.
- RTK / PPK GPS (the problem statement's optional corrections input) is read from
  CSV logs:
  - accuracy columns in metres (`hAcc`/`vAcc`, `eph`/`epv`,
    `horizontal_accuracy`, DJI `RtkStdLat`/`RtkStdLon`/`RtkStdHgt`; `(mm)` and
    `(cm)` units converted);
  - or a fix state (`fix_type`/`gps_status` 6 = RTK fixed, 5 = float; DJI
    `RtkFlag` 50/34; `rtk_status` text). A fix state with no accuracy counts as
    5 cm fixed / 50 cm float.

  Each camera's GPS weight then follows the log's accuracy plus the video-to-log
  timing error (speed × clock uncertainty), never below 0.25 m for the
  camera-to-antenna offset. The report claims centimetre-level absolute accuracy
  only when the log itself reports RTK-grade accuracy. Logs without these columns
  behave exactly as before (2.5 m prior).

## Benchmark

The flight01 accuracy numbers above come from the PinPoint validation dataset
([doi:10.5281/zenodo.22671839](https://doi.org/10.5281/zenodo.22671839)); the
method and full results are in [evidence/flight01-benchmark.md](evidence/flight01-benchmark.md).
flight01 runs need `--telemetry-offset 121.66`: the clip's frame 0 is mkv PTS
120.464 s, and the log runs 1.2 s ahead of the mkv clock.

## Setup

```bash
# Install uv if you haven't already: https://docs.astral.sh/uv/
uv sync

# Optional extras
uv sync --extra gui   # PySide6 desktop UI + VTK viewport
uv sync --extra ml    # torch / torchvision (CUDA or MPS acceleration)
uv sync --extra semantics  # SegFormer semantic classification + dynamic-object masking
uv sync --extra texture    # xatlas UV unwrapping for the photographic texture atlas
uv sync --extra reference  # rasterio, for reference-orthophoto/DEM alignment
```

The `ml` extra also installs the pinned MapAnything implementation. Its weights
are downloaded on first use, or loaded from `DRISHTI3D_MAPANYTHING_WEIGHTS` for
offline runs. A missing reconstruction model now fails the geometry stage;
`--backbone null` is explicitly synthetic and is only for demos.
It also installs kornia for the default DISK + LightGlue feature matching
(weights downloaded once on first use); without kornia, matching logs a
warning and falls back to SIFT.

For video-based depth refinement, keep `geometry.plane_sweep: true`. It matches
source image patches after metric depth fitting, including rotated camera views.
Nadir height-map meshes also respect `fusion.photometric_reject_before_mesh`:
vertices contradicted by textured source views and their incident triangles are
removed, leaving gaps rather than inventing a surface. This can increase runtime
and reduce mesh coverage. Neither image agreement nor a completed export alone
establishes absolute metric accuracy; check camera alignment and withheld survey
points before using measurements.

## Measuring on the model

Tools menu, on any loaded or finished run (points snap to the model surface;
a click places a point, a drag still orbits; right-click, double-click or
Enter finishes; Backspace undoes a point; Esc cancels):

- **Point** (P): latitude, longitude and elevation above sea level.
- **Distance** (M): 3D length, horizontal length and height difference.
- **Area** (Shift+M): planimetric area and perimeter of a polygon.
- **Volume** (V): cut and fill against a base plane fitted through the
  outline's own corners (a stockpile, debris, a pit).
- **Elevation profile** (L): heights along a route, with climb, descent and
  the steepest slope, as a chart (exportable to CSV).
- **Export Measurements** writes them all as GeoJSON (QGIS/ArcGIS) or KML
  (Google Earth), in WGS84 with sea-level elevations.

A measurement touching INFERRED surface carries a warning.

## Moving objects and semantic masking

Measured 3D keeps only depths that neighbouring cameras agree on, and the
orthophoto takes the median colour across views, so moving vehicles and
people drop out of the geometry and the texture without any model (the
highway traffic on DJI_1001 is absent from its orthomosaic). The optional
`semantics` extra (SegFormer, `transformers`) additionally masks vehicles,
people and sky before depth is measured. Its default ADE20K checkpoint is
trained on ground-level photos: on downward flights it found no vehicles
(0% on DJI_1001), so it is skipped there (`semantics.skip_nadir_ground_level`)
and runs on forward/oblique footage, where it also masks sky. For nadir
footage set `semantics.checkpoint` to an aerial model. Model weights (256 MB)
download on first use.

## Windows and Linux (NVIDIA GPU)

The pipeline picks CUDA automatically when an NVIDIA GPU is present (then
Apple MPS, then CPU). Builds:

- **Windows 10/11 x64**: `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1`
  (installs the CUDA build of the locked torch from PyTorch's index; `-Cuda cu126`
  for older drivers, `-CpuOnly` without a GPU) -> `dist\DRISHTI-3D\DRISHTI-3D.exe`.
- **Linux x86_64** (Ubuntu 22.04+): `./packaging/build_linux.sh` -> `dist/DRISHTI-3D/`
  with a `.desktop` entry. PyPI's Linux torch already carries CUDA.
- **Docker, Linux + NVIDIA** (NVIDIA Container Toolkit):
  `docker build -f packaging/Dockerfile -t drishti3d .`, then
  `docker run --rm --gpus all -v "$PWD/data:/data" drishti3d run /data/flight.mp4 --telemetry /data/flight.csv --out /data/out`
  (the desktop app over X11: see the Dockerfile header).
- **CI**: `.github/workflows/build.yml` builds the Windows, Linux and macOS
  apps and the Docker image (run it from the Actions tab or with a `v*` tag).

Open3D ships Linux wheels for x86_64 only, so Linux ARM is not supported.

## Desktop app build

`./packaging/build_app.sh` bundles torch and MapAnything whenever they are
installed, so the built app can reconstruct as well as open files. The build
ends with the same offline runtime preflight every run performs. Set
`DRISHTI3D_BUILD_WEIGHTS_DIR` to the verified checkpoint
(`python -m drishti3d.runtime locate`) to bundle the 4.9 GB weights for an
air-gapped machine. Headless use of the built app:
`DRISHTI-3D.app/Contents/MacOS/DRISHTI-3D run VIDEO --telemetry LOG --out DIR`.

## Coordinate conventions

- **World frame**: ENU (East-North-Up), units in metres, Z-up. X points East,
  Y points North, Z points Up. All georeferenced/reconstructed geometry
  (poses, point clouds) is expressed in this frame unless otherwise noted.
- **Camera frame**: OpenCV convention. X points right, Y points down, Z
  points forward (out of the lens, into the scene).
- **Geographic coordinates**: latitude/longitude in decimal degrees (WGS84),
  altitude in metres. Distinct from the local ENU world frame; conversion
  between the two is handled during georeferencing (e.g. via `pyproj`).
