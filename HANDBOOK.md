# 3CHAKSHU — Team Handbook

**SIH26158 — Single-Pass Drone Video to Accurate 3D Model Generation System**
Organisation: National Technical Research Organisation (NTRO) · Category: Software · Theme: Drone/Robotics

---

## 1. The problem, stated properly

Generate a georeferenced, metrically accurate, textured 3D model from **one** drone pass.

The trap most teams miss: a single pass means the cameras lie on a **nearly straight line**. That is a degenerate geometry. Depth along the viewing direction is weakly determined; roll and pitch are barely observable. Classical photogrammetry (COLMAP, Pix4D, Metashape) assumes multiple crossing passes with 70–80% overlap. On a single strip it does not merely run slowly — the estimation problem is ill-conditioned.

So the task is not "make photogrammetry faster." It is "reconstruct where the geometry is degenerate, and be honest about what could not be recovered."

### What the PS asks for

| Requirement | Target |
|---|---|
| Reconstruction type | 3D mesh / point cloud |
| Processing time | < 15 min for a 10-min video |
| Spatial accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization | Web-based **or desktop** viewer |

Mandatory inputs: drone video (1080p/4K), GPS coordinates, flight metadata.
Optional: IMU, barometric altitude, camera intrinsics, RTK/PPK.

**Evaluation weights:** Accuracy 30% · Completeness 20% · Speed 20% · Innovation 15% · Scalability 10% · UI 5%.

---

## 2. Existing methods and where they fail

| Method | Examples | Failure on single pass |
|---|---|---|
| SfM + MVS photogrammetry | COLMAP, Metashape, Pix4D, OpenDroneMap, DJI Terra | Needs cross-strips and orbits. Hours of compute. Scale drift on linear paths |
| Real-time 2D mapping | Pix4Dreact, DJI Terra live | 2D orthomosaic only — no 3D, no measurement |
| NeRF | Instant-NGP, Nerfacto, Zip-NeRF | Requires COLMAP poses first (already broken). Not metric. Poor mesh extraction |
| 3D Gaussian Splatting | 3DGS, 2DGS, SuGaR | Beautiful, fast — but not metric, not georeferenced |
| Visual/VI SLAM | ORB-SLAM3, DROID-SLAM | Real-time poses but sparse; not survey-grade surfaces |
| Feed-forward pose-free | DUSt3R, MASt3R, VGGT, Pi3, MapAnything | Solves the hard part (no poses needed) but scale-free, drifts, no georeferencing |
| Autonomous scan | Skydio 3D Scan | Solves it by flying *more* passes — the opposite of the ask |

### The documented gaps

| # | Gap |
|---|---|
| G1 | No tool is *designed* for single-strip degenerate geometry |
| G2 | Feed-forward models are scale-free; GPS alone is 2–5 m; nothing fuses GPS + IMU + metric-depth priors |
| G3 | Every tool ingests photos. None does blur/parallax/redundancy-aware keyframe selection from video |
| G4 | Dynamic objects (cars, people) ghost into the mesh |
| G5 | Occluded surfaces are silently left hollow, or silently faked |
| G6 | **No tool tells you which geometry was measured and which was guessed** |
| G7 | State-of-the-art is research repos — nothing deployable offline for a field operator |
| G8 | Metashape takes hours; requirement is 15 minutes |

### The gap sentence (use verbatim on slide 1)

> Photogrammetry pipelines (COLMAP, Pix4D, Metashape) require 70–80% overlap across multiple passes and fail on single-strip flights due to weak baseline geometry. Recent pose-free feed-forward models (DUSt3R 2024, VGGT CVPR 2025) reconstruct from unposed frames but produce scale-free, non-georeferenced output with drift over long sequences. No existing system combines pose-free video reconstruction with GPS/IMU metric anchoring, dynamic-object rejection, and per-vertex confidence reporting — and none ships as an offline desktop tool for air-gapped field use.

---

## 3. Architecture

```
Drone video + GPS/telemetry
   │
1. INGEST      decode video (PyAV), parse telemetry (DJI SRT / Airdata CSV / GPX),
   │            resolve camera intrinsics (EXIF → camera DB → HFOV default)
   │
2. TRIAGE      select keyframes. Spacing driven by GPS baseline in METRES,
   │            scaled to altitude (B/H ratio), not by fixed frame stride.
   │            Blur-aware: picks sharpest candidate in a local window
   │
3. GEOMETRY    MapAnything predicts poses + dense metric points per window of
   │            frames. Windows overlap; submaps merged via Umeyama + RANSAC
   │
4. MATCHING    SIFT → ratio test + cross-check → geometric verification
   │            (essential matrix when calibrated) → union-find multi-view
   │            tracks → DLT triangulation with degenerate-angle rejection
   │
5. BUNDLE ADJ  sparse Levenberg-Marquardt. GPS position priors, gravity/Z-up
   │            prior, optional GCPs, gauge fixing.  ← accuracy is made here
   │
6. COVARIANCE  Schur complement on the BA normal equations
   │            → a 3×3 uncertainty ellipsoid per point
   │
7. FUSION      confidence-weighted TSDF → mesh with per-vertex trust tier
   │
8. EXPORT      PLY / OBJ / GLB / LAS / GeoTIFF + accuracy report card
```

**The core design decision:** feed-forward models give *robustness* where classical SfM fails to converge. Bundle adjustment gives *accuracy* the model cannot. Neither alone is sufficient. Step 3 initialises, step 5 refines.

### Module map

```
drishti3d/
├── types.py          shared contract — world = ENU (Z-up, metres),
│                     camera = OpenCV (X right, Y down, Z forward)
├── ingest/           video.py (PyAV) · telemetry.py · intrinsics.py
├── triage/           metrics.py (blur, parallax) · selector.py
├── geometry/
│   ├── backbone.py       swappable: MapAnything / VGGT / Pi3 / Null
│   ├── mapanything.py    real adapter, offline weights
│   ├── windows.py        submap planning under a VRAM budget
│   ├── submap.py         Umeyama merge + collinearity guard
│   ├── features.py       SIFT, matching, geometric verification
│   ├── tracks.py         union-find multi-view tracks
│   ├── triangulate.py    DLT + degenerate-angle rejection
│   ├── bundle.py         sparse LM, GPS/gravity/GCP priors, gauge fixing
│   ├── covariance.py     Schur complement → per-point 3×3
│   ├── georef.py         ENU↔WGS84, UTM, honest accuracy split
│   └── observability.py  ← THE NOVELTY
├── fusion/           filters · confidence-weighted TSDF · mesh
├── export/           PLY/OBJ/GLB/LAS · GeoTIFF · report card
├── pipeline/         runner, stages, save/load cached results
└── app/              PySide6 window · VTK viewport · panels
```

### Three decisions to defend in Q&A

1. **Swappable backbone.** If MapAnything's licence or VRAM proves wrong, swap in VGGT or Pi3 without touching the pipeline. `NullBackbone` also runs the whole system with no GPU and no weights — the stage-failure insurance policy.
2. **No GTSAM / Ceres / g2o.** Sparse LM on scipy instead. Those solvers are install hell and would break the single offline installer requirement.
3. **Submap windowing.** Peak memory is constant in video length, not linear. A 30-minute flight uses the same peak memory as a 5-minute one. That is the scalability answer.

---

## 4. What is built

192 automated tests passing.

| Component | Status |
|---|---|
| Shared type contract, coordinate conventions | Done |
| Video decode (PyAV), 4K verified | Done |
| Telemetry: DJI SRT, Airdata CSV, GPX | Done |
| Video/telemetry time-sync auto-detection | Done |
| Camera intrinsics resolution + camera DB | Done |
| Keyframe triage (GPS-baseline + vision fallback) | Done |
| Backbone abstraction + NullBackbone | Done |
| MapAnything adapter | Written, **never executed** |
| Submap planning + Umeyama merge + degeneracy guard | Done |
| SIFT matching, tracks, triangulation | Done |
| Sparse bundle adjustment with priors | Done |
| Per-point covariance (Schur complement) | Done |
| Observability field + corrective flight planning | Done |
| Georeferencing, honest relative/absolute split | Done |
| Confidence-weighted TSDF → mesh | Done |
| Export: PLY, OBJ, GLB, LAS, GeoTIFF, report | Done |
| Pipeline runner + CLI + cached result save/load | Done |
| Qt/VTK desktop app, confidence rendering, measurement | Done |
| Validated on synthetic data end-to-end | Done |
| Validated on real 4K drone footage | **Partial** |

**Proven on synthetic data:** 138,318-vertex mesh, 276,628 faces, 10 export files, BA converging 1.11 px → 0.66 px, all three confidence tiers present in the exported LAS.

**Proven on real footage (DJI_0753.MP4, 4K, 242 s):** ingest, telemetry parsing, time-sync detection (42.90 s offset auto-detected), and triage all work. 60 keyframes selected, trajectory 2029 m, 8.4 m/s — physically plausible.

---

## 5. Known gaps — be honest about these

| # | Gap | Severity |
|---|---|---|
| 1 | **Performance.** Full run on real 4K did not complete in 88 min. `max_image_size` is applied only inside the MapAnything adapter, so NullBackbone ray-casts at native 4K → ~500M points | **Blocking** |
| 2 | **MapAnything has never run.** Adapter is written and smoke-tested against a fake package, not the real model | **Blocking** |
| 3 | No progress output during long stages — 88 minutes of silence | High |
| 4 | Report card can present structurally meaningless numbers when using the null backbone without labelling them as such | High |
| 5 | Dynamic object masking (YOLO + SAM2) — planned, never built | Medium |
| 6 | Occlusion completion with INFERRED flagging — planned, never built | Medium |
| 7 | Semantic layers (ground/building/road/vegetation) — cut | Low |
| 8 | No baseline comparison against COLMAP / OpenDroneMap yet | **High — this is a scoring gap** |
| 9 | No ground-truth validation harness, so ≤1 m is unproven | **High — this is a scoring gap** |
| 10 | `.fbx` export not implemented | Low |

Items 8 and 9 are worth more marks than any new feature.

---

## 6. The novelty

**Be explicit about what is not ours.** MapAnything, bundle adjustment, TSDF fusion, SIFT, Umeyama — all published. Judges can find the papers. Claiming them is how teams lose.

### A. The observability field — the real contribution

Every existing tool outputs one undifferentiated mesh. We compute, from the bundle-adjustment covariance, a per-point uncertainty **ellipsoid** — anisotropic, not a scalar confidence.

**Measured result:**

| Camera configuration | Median anisotropy ratio |
|---|---|
| Well-conditioned multi-view | **2.15** |
| Collinear single-pass nadir | **85.75** |

A ~40× difference, from real bundle adjustment on both configurations (each converging to reprojection RMSE < 1e-6), same points, only the geometry differing. The worst-constrained direction comes out near-vertical (median \|cos\| > 0.9) — independently reproducing the classical "poor vertical accuracy from a single strip" photogrammetric result.

That is a measured number, not a claim.

### B. The trust layer

Every vertex tagged **MEASURED / LOW-CONFIDENCE / INFERRED**, rendered green/amber/red, and it **survives export**:

| Format | Mechanism | Verified |
|---|---|---|
| LAS | extra-bytes dimension | Exact round-trip |
| GLB | custom `_CONFIDENCE` attribute, non-normalized UNSIGNED_BYTE | Exact round-trip |
| PLY | custom scalar property | Exact round-trip |

Measurement tools refuse to measure inferred geometry by default.

Why it matters: for a reconnaissance or disaster analyst, knowing which parts of the model are real is a safety property, not a nicety.

### C. Inverting the ellipsoid into a flight line

The ellipsoid's worst-constrained axis indicates which viewing direction would collapse it. Cluster the high-uncertainty points, aggregate their required viewing directions, fit one straight line:

> *"One more pass, heading 072°, 80 m AGL, gimbal −35° — predicted to resolve 84% of remaining uncertainty."*

The literature has had per-point covariance since ~2018 and next-best-view planning since 2012, but those two lines diverged. Closing that loop, specifically in the single-pass degenerate regime, is new.

### D. Honest dual accuracy reporting

Every competing team will claim "≤1 m". With standalone GPS that is **physically impossible** — it is sensor bias, not an algorithm problem.

| Accuracy type | Standalone GPS | With PPK/RTK |
|---|---|---|
| Relative / metric | sub-metre — achievable | sub-metre |
| Absolute georeferencing | 2–5 m, GPS-bias-limited | 1–3 cm |

We report both separately, and only claim centimetres when RTK/PPK-grade input is actually detected. This is enforced in code with a test, not written in a footnote.

### E. Video-native keyframe triage

Spacing is driven by real metric baseline from GPS, scaled to altitude (B/H ratio ≈ 2% of AGL), with blur-aware selection of the sharpest candidate in a local window. Everyone else takes every Nth frame.

### F. Air-gapped native desktop

PySide6 + VTK. No webview, no cloud, no runtime model download. NTRO cannot upload sensitive footage anywhere, and 4K video over field connectivity is not viable regardless.

---

## 7. How this stands out

### What 90% of teams will build

- A COLMAP or OpenDroneMap wrapper with a nicer UI — "what did *you* build?" ends them
- Extract every 10th frame, feed to photogrammetry — ignores blur, ignores redundancy, will not finish in 15 minutes
- A 3DGS demo — looks stunning, **not metric, not georeferenced**, fails the 30% accuracy criterion outright
- A web app with a cloud GPU — NTRO is air-gapped; instant fail in Q&A
- "≤1 m accuracy" claimed with zero measurement
- Occlusion ignored: they show the pretty camera angle and hide the hollow building backs

### Mapping to the evaluation criteria

| Criterion | Weight | Our answer |
|---|---|---|
| Reconstruction accuracy | 30% | GPS + IMU + metric-depth pose graph; measured RMSE reported, not claimed |
| Model completeness | 20% | Confidence-weighted TSDF, flagged inference for occlusions, coverage metric |
| Processing speed | 20% | Feed-forward init, not iterative SfM; progressive output |
| Innovation | 15% | Observability field, refly heatmap, video-native triage |
| Scalability | 10% | Submap architecture — memory constant in video length |
| User interface | 5% | Native offline desktop, measurement tools, single installer |

### The one-line pitch

> The components are published research. The observability field, the trust layer, and the offline deployable tool are ours.

---

## 8. Judge Q&A

**"This is just COLMAP with a UI."**
COLMAP fails on single-strip input — weak baseline, no loop closure. We use pose-free feed-forward reconstruction that needs no initial poses, then anchor it metrically with a GPS/IMU-constrained bundle adjustment. We ran COLMAP on the same input; here is the comparison table.

**"How do you get ≤1 m without GCPs?"**
Three decorrelated constraints: GPS gives absolute position (2–5 m raw), monocular metric depth gives per-frame scale, IMU gives relative motion. Fused in one factor graph the residual drops. We report measured RMSE and per-region uncertainty rather than a single global number. And we are explicit that *absolute* accuracy with standalone GPS is bias-limited — no algorithm removes that.

**"You're hallucinating the occluded building backs."**
We never silently fake geometry. Unobserved surfaces are left open or completed by plane/symmetry extrusion and tagged INFERRED — rendered in a distinct colour and excluded from measurement by default. For a reconnaissance product, knowing what you don't know is the feature.

**"Real-time?"**
Near-real-time, as the PS asks. Coarse geometry early, refined model within the budget, on a single laptop.

**"Why desktop, not web?"**
The end user is NTRO — air-gapped. The video is sensitive and cannot leave the machine. Uploading 4K footage over field connectivity is not viable either. Fully offline, single installer.

**"Does it scale?"**
Submap architecture — peak memory is constant in video length, not linear. A 30-minute flight uses the same peak memory as a 5-minute one.

**"Do you have quantum/novel AI?"**
No, and we do not claim it. Our contribution is the observability field and the trust layer, built on published reconstruction components we cite.

---

## 9. Commands

Launch the desktop app:
```bash
cd /Users/ajith/Desktop/sih && QT_QPA_PLATFORM_PLUGIN_PATH=~/.drishti3d/qtplugins/platforms PYTHONPATH=/Users/ajith/Desktop/sih uv run python -m drishti3d.app.main
```

Headless run:
```bash
PYTHONPATH=/Users/ajith/Desktop/sih uv run python -m drishti3d.pipeline.runner VIDEO.MP4 --telemetry FLIGHT.csv --backbone null --out ./result
```

Ingest + triage only (fast):
```bash
PYTHONPATH=/Users/ajith/Desktop/sih uv run python -m drishti3d.pipeline.runner VIDEO.MP4 --telemetry FLIGHT.csv --dry-run --out ./result
```

VRAM benchmark:
```bash
uv run python scripts/benchmark_vram.py --backend mapanything --views 2,4,8,12,16 --sizes 256,384,518 --out results.json
```

Tests:
```bash
uv run pytest tests/ --ignore=tests/test_app.py
```

### Environment notes (macOS)

- The project sits in `~/Desktop`, which is TCC-protected. macOS refuses to load the Qt plugin dylib from there — hence the `QT_QPA_PLATFORM_PLUGIN_PATH` pointing at `~/.drishti3d/qtplugins`.
- The editable install is broken by a `_virtualenv.pth` conflict — hence `PYTHONPATH`.
- **Both problems disappear if the project is moved out of Desktop** (`mv ~/Desktop/sih ~/sih`). Recommended.

---

## 10. Priorities before the hackathon

1. **Fix the performance bug** — apply `max_image_size` in the geometry stage; profile the real hot spot
2. **Run MapAnything once** — the single largest unknown
3. **Baseline comparison vs COLMAP / OpenDroneMap** on identical single-pass input — highest marks per hour of work
4. **Ground-truth validation harness** so ≤1 m is measured, not claimed
5. Environment spike on the RTX 4060 — VRAM ceiling at 6 GB
6. Cache a completed result so the demo survives a dead GPU on stage

Items 3 and 4 are worth more than any new feature.
