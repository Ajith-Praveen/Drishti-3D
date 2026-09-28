# flight01 benchmark: surveyed ground points, same footage, every method

Dataset: PinPoint flight01 (`flight01/`). A fixed-wing grid survey at 100-127 m
AGL over farmland and ravines near 41.77 N, 0.74 W. The 6-minute clip is
`flight01_120_480.mp4` (1280x720, 60 fps) with an ArduPilot log
(`flight01_telemetry.csv`). The truth is 100 surveyed ground features
(`points.csv`: a pixel in a frame plus its surveyed latitude/longitude). A
reference IGN orthophoto (0.3 m) and DEM (10 m) are in `flight01/reference/`.

## Protocol

`scripts/score_flight01.py RUN/output --clip-start 120.46 [--subset nadir]`

For each surveyed pixel, the script:
1. takes the camera at that video time from the run's `cameras.json`;
2. undistorts the pixel with the run's lens;
3. casts the ray onto the exported surface (an exact ray/triangle hit);
4. reports the horizontal distance to the surveyed truth.

This is end to end: camera, lens, surface and georeferencing together, which
is what an operator measuring on the model gets.

- The model is mapped to local ENU by the same camera-to-GPS similarity the
  pipeline had, so the scorer does not flatter it. It keeps gravity (yaw fit)
  on a straight track. `--trust-georef` uses the run's own georeferencing
  instead, which is needed to measure reference alignment.
- `--subset nadir` restricts to PinPoint's 64 `nadir_ok` rows. The flight banks
  24-46 deg in its turns while the logged gimbal stays constant, so rows in
  turns measure camera interpolation, not the model.
- Timing, established 2026-09-25:
  - the clip's frame 0 is mkv PTS 120.464 s, found by frame matching;
  - telemetry = mkv time + 1.2 s;
  - so runs need `--telemetry-offset 121.66` and scoring needs `--clip-start 120.46`.

  The earlier 120 / 121.2 values were each 0.46 s off: ~8 m along-track, which
  every earlier score silently included.

## Results (horizontal error on surveyed points)

| Method | Points hit | Median | RMSE | <= 3 m |
|---|---|---|---|---|
| Naive: logged GPS + attitude projected onto terrain (PinPoint `err_att_m`) | 64/64 nadir | 10.0 m | 12.7 m | - |
| COLMAP 4.2 (720 frames, OPENCV self-calibration, GPS-aligned at 121.66) | 64/64 nadir | 3.29 m | 3.80 m | 42% |
| COLMAP 4.2, all rows | 85/100 | 3.75 m | 10.0 m | 32% |
| DRISHTI-3D, Sep 24 runs (backbone dense, broken camera solve) | 95/100 | 41-65 m | 169-223 m | 0-1% |
| DRISHTI-3D prototype: height-field MVS on BA cameras with the old 121.2 clock | 62/64 nadir | 3.73 m | 6.66 m | 42% |
| DRISHTI-3D main downstream (height-field -> fusion -> export -> IGN alignment) on the same old-clock cameras | 53/64 nadir | 3.30 m | 6.93 m | 43% |
| **DRISHTI-3D main, full pipeline, corrected clock (see below)** | 47/64 nadir | 4.02 m | 6.22 m | 38% |

The COLMAP rows are the rival-level bar: most SIH26158 repositories on GitHub
wrap COLMAP or OpenDroneMap. No rival repository reports surveyed-point accuracy.
The best reported numbers are camera-trajectory residuals.

## Surface vs reference DEM (same bundle-adjusted cameras)

| Dense method | Spread (MAD) | Within 5 m of median offset | Dense time |
|---|---|---|---|
| Backbone depth + fusion (MapAnything, 208 views, one window) | 7.5 m | ~40% | 1225 s |
| Height-field multi-view stereo (`geometry.heightfield`) | 1.4-1.7 m | 74-81% | 23-35 s |

The reference is a 10 m terrain model, so trees and embankments count as error
here. The remaining +4 m offset is the flat-terrain focal/depth ambiguity: BA
focal 847 px vs 877 px from COLMAP/OpenSfM, 3.5% at 110 m. Reference-DEM
alignment removes it.

## Reference alignment (IGN orthophoto + DEM)

- **Old 4-DoF fit.** It fitted a similarity to DISK+LightGlue matches, and
  ~90% of those came from one textured corner (ravine scrub). The scale swung
  1.3% between two runs of one model (0.998 vs 0.987), which is 5 m at 400 m
  from the centre. The two runs' corrections differed by 3.7 m, and the survey
  points confirmed the difference.
- **Now: translation only, bare ground only.** Matches are kept only where our
  DSM agrees with the reference terrain model. The reference ortho is rectified
  on that terrain model, so tree crowns in it are relief-displaced, while ours
  are a true orthophoto. The fit is a deterministic 1-point RANSAC plus the
  inlier mean.
- **Result.** Two runs gave identical corrections (E -1.5, N +0.5, U -6.0 m;
  221 bare-ground matches, 151 inliers). The systematic error left against the
  survey is **E +0.87 m, N -0.24 m**.
- **Remaining scatter.** It is ~3 m, and it comes from the borrowed cameras:
  their bundle adjustment used GPS priors on the 0.46 s-wrong clock, which
  shears alternate legs along-track.

Downstream timings (208 keyframes, Apple M5): height-field MVS 44 s, fusion
0.2 s, export 27 s (20 deliverables including a textured OBJ/GLB).

## End to end in main (2026-09-26)

This is the full 6-min clip at `--telemetry-offset 121.66`, with the camera-solve
patch, the height-field dense stage and bare-ground reference alignment. It ran
on an Apple M5 with nothing else running.

- **Outcome: valid.**
  - 0.35 px reprojection
  - lens k1 -0.241 / k2 0.050 at f 881, the same as COLMAP's self-calibration
  - 19 of 208 banked-turn cameras left unrefined, and excluded from the dense stage
- **Time: 573 s** against a proportional budget of 540 s (15 min per 10 min of video):
  - triage 115 s
  - pose prior 467 s
  - height-field 28 s
  - fusion 0.2 s
  - export 20 s
- **Surface vs reference DEM** (`dsm.tif`, 0.5 m): median +0.10 m, bare-ground
  mode +0.01 m, MAD 2.0 m.
- **Orthomosaic vs IGN orthophoto** (`scripts/ortho_check.py`, 60 m tiles,
  camera-independent): common offset **0.64 m** (E +0.63, N -0.09), median tile
  error 2.06 m, internal distortion 1.82 m. `orthomosaic.tif` is now the
  true-ortho at 0.166 m/px; it was 1.3 m/px.
- **Surveyed points, nadir, same 47 points for both** (DRISHTI leaves unobserved
  ground empty, so it hits 47 of 64; COLMAP's sparse Delaunay spans everything):

| | All 47 | Survey frame < 0.25 s from a DRISHTI keyframe (n=14) | Farther (n=33) |
|---|---|---|---|
| DRISHTI-3D | 4.02 m | **2.97 m** | 4.42 m |
| COLMAP 4.2 | 3.28 m | 3.85 m | 3.17 m |

Where the camera at the survey frame is essentially exact, DRISHTI's surface
and georeferencing beat COLMAP's by 0.9 m. The overall gap comes from
interpolating DRISHTI's cameras across its ~1.7 s keyframe spacing (208
keyframes over 360 s), against COLMAP's 720 cameras at 0.5 s. That is a
property of this video-frame scorer, not of the model: measuring on the model
itself (the ortho check above) involves no cameras. COLMAP took 33 min on this
Mac for a sparse model only (no CUDA, so no dense MVS). DRISHTI took 9.5 min for
a dense, textured, georeferenced, reference-aligned model.

## Second dataset: DJI_1001 (Austin, 1080p, 11.4 min, no gimbal or altitude in the CSV)

- Outcome valid, 0.35 px reprojection, **438 s** (faster than real time).
- 14 end-of-flight cameras were left unrefined: the log and images disagree
  there.
- Height-field MVS: 51 views, 5.1 M cells at 0.5 m, median NCC 0.90.
- Ortho at 0.25 m/px; DSM heights 143-180 m.
- There is no ground truth for this flight.

## Camera-free check: where the surveyed features sit in the orthomosaic (2026-09-26)

`scripts/survey_ortho_check.py RUN/output --clip-start 120.46 [--ortho flight01/reference/ortho.tif]`

The ray scorer above needs a camera at each survey frame and interpolates
one between keyframes. This check does not trust that camera. For each
surveyed pixel it cuts a rectified patch of the survey frame and finds it in
an orthophoto by normalised cross-correlation. The feature's position in the
orthophoto minus its surveyed position is the orthophoto's own error.

- Features on a hedge, road or field edge match anywhere along the line (the
  aperture problem). A visual check confirmed this caused 10 m "errors", so
  matches whose correlation stays within 0.05 of the peak over more than
  1.5 m are dropped.
- The same method run on the IGN orthophoto shows how good the truth itself
  is.

| Orthophoto | Features found (nadir) | Median error vs survey truth |
|---|---|---|
| IGN national orthophoto (PNOA) | 21/64 | 1.47 m |
| DRISHTI-3D `orthomosaic.tif` | 27/64 | 2.42 m |
| DRISHTI-3D vs IGN at the same features (n=17, both matched) | - | **0.88 m** (59% within 1 m) |

What this shows:

- **The survey truth is itself ~1.5 m off the national orthophoto**, so it cannot
  certify 1 m for any method, including the IGN orthophoto.
- **Measured against the official orthophoto at stable features, the model is
  0.88 m median.** IGN's own planimetric accuracy is ~0.5 m RMSE, and
  hedge features carry IGN's relief displacement, so part of that 0.88 m is
  the reference's error.
- **Vertical** could not be verified to 1 m. The reference DEM is Copernicus
  GLO-30, a 30 m radar surface model with a 2-4 m specification. Our bare-ground
  heights agree with it to 1.7 m median.

Tried and rejected: a local (rubber-sheet) correction of the model toward the
IGN orthophoto. It was fitted from bare-ground template matches, because
dense phase correlation over masked tiles is biased toward zero: a model moved
2 m east measured 0.09 m. With each surveyed feature's own 30 m neighbourhood
held out, it made those features worse (0.88 -> 1.42 m against IGN). Fields
change between the IGN date and the flight, so bare-field matches are not
stable enough to steer the model. It is not in the pipeline.

## Faster, more accurate camera solve (2026-09-26)

Bundle adjustment was 70% of the run: three scipy `least_squares(trf, lsmr)`
solves took 139 + 105 + 124 s on flight01. It is now Levenberg-Marquardt
with the points eliminated by Schur complement (`BAConfig.solver="schur"`):
each step solves the ~1,150 camera and lens unknowns exactly. Robust Huber
rows are weighted by plain IRLS. Scipy's second-order correction zeroes
Huber outlier rows, so it stalled at twice scipy's cost.

- **Speed.** The same three solves now take 17 + 8 + 4 s. The pose prior
  dropped from 499 s to 163 s.
- **Rejected alternative.** Capping scipy's inner LSMR iterations was 4x
  faster, but cameras then ended 4.5 m from GPS, not 2.6 m.
- **GPS prior weight.** An exact solver honours the GPS priors fully, so their
  weight matters. Camera-track shape was scored against COLMAP's independent
  720-camera solution:

| GPS prior sigma | Track vs COLMAP (median / p90 / vertical) |
|---|---|
| scipy, 5 m (old) | 2.44 / 5.55 / 1.27 m |
| Schur, 5 m | 2.54 / 6.15 / 1.59 m |
| Schur, 2.5 m (new default) | 2.35 / 4.70 / 1.13 m |
| Schur, 1.25 m | 2.29 / 3.52 / 1.15 m |
| Schur, 10-40 m | 3.55-4.98 / 9.6-14.6 m (the block bends) |

End to end on flight01 (same inputs, Apple M5):

| | Old (scipy, sigma 5 m) | New (Schur, sigma 2.5 m) |
|---|---|---|
| Total time (budget 540 s) | 634 s | **248 s** |
| Orthomosaic vs IGN, 60 m tiles: median / internal distortion | 2.06 / 1.82 m | **1.01 / 0.72 m** |
| Stable surveyed features, ours vs IGN | 0.88 m, 59% <= 1 m (n=17) | **0.85 m, 70% <= 1 m (n=23)** |
| Camera track vs COLMAP (median / vertical) | 2.32 / 1.22 m | **1.61 / 0.72 m** |
| Ray scorer vs survey truth (nadir) | 4.02 m | **3.51 m** |

DJI_1001 (11.4 min, nadir, no gimbal log, no ground-height reference):

- **Time.** 438 s -> **219 s**. Valid, 0.35 px, focal 1,079 px (was 1,077),
  k1 -0.019 (unchanged). DSM median 167 m, p5-p95 151-176 m, which matches the
  terrain there (~150-180 m).
- **Focal.** The exact solver exposed an unobservable focal. At the old
  shared-focal prior (sigma 0.3, log) the images pulled the focal from the
  1,066 px guess to 1,720 px, a 58 deg field of view, and sank the DSM to ~0 m.
  At sigma 0.05 it still went to 1,640 px. Horizontal geometry does not
  depend on focal for a nadir view, so only heights were affected. The prior is
  now 0.01, which holds a focal that only a guess supports. Lens distortion is
  still solved. flight01 measures its focal from image flow and does not use
  this prior.
- **Cameras.** The GPS-outlier test scales with the prior sigma, so it is now
  10 m instead of 20 m. That leaves 6 more end-of-flight cameras unrefined
  (51 -> 45 dense views; grid coverage 59.7% -> 52.9%). Keeping them at 20 m let
  one be pulled 32.9 m off its fix, and the camera gate rejected the whole run.

## Completeness: the whole visible scene (2026-09-26)

`geometry.heightfield` now completes the surface. Every cell a posed camera
saw, but stereo did not measure, is filled and tiered INFERRED with no
measured uncertainty. Typical causes are bare fields, shadow, ground
occluded behind a building, and rejected spikes. Holes the model encloses
completely are filled too, whatever their size: ground seen only from the
banked turns, whose cameras could not be posed. These holes are coloured
by inpainting.

How the fill works:

- **Base surface.** It is the robust coarse-level surface plus the measured
  rim's difference from it. That difference fades over 10 m, so large gaps
  follow the robust surface instead of extrapolating the noisiest
  (fewest-view) edge cells.
- **Ground or roof.** A hole bordered by ground is filled from the ground. A
  hole bordered mostly by roof or canopy is filled from its rim.
- **Height limits.** Within 30 m of measurements a fill stays inside their
  height range (+-2 m), so it neither invents structure nor hangs skirts.
  Nothing may stand 60 m above its local ground or above half the flight
  height.
- **Walls.** Height steps of more than 3 m are closed with walls built from
  duplicated INFERRED vertices. The 2.5D surface used to leave them open, and
  oblique rays passed through.

Found on the way: the small-hole filler divided `uniform_filter` residue by
residue where a cell had no known neighbour. That produced flight01's 540
floating vertices and DJI_1001's 2,663 cells at exactly the camera plane.
Both are now 0.

| flight01 | Before | Now |
|---|---|---|
| Surface cells (DSM) | 0.80 M | **1.04 M** |
| Visible scene covered | 76% (measured only) | **100%** (76% measured + 24% INFERRED) |
| Surveyed points the model covers (nadir) | 52/64 | **61/64** (the other 3 were seen only by unposed cameras) |
| Accuracy on measured surface (ray scorer) | 3.18 m (44 points) | 3.17 m (44 points) |
| Accuracy on INFERRED surface | - | 4.63 m (14 points) |
| Stable features vs IGN | 0.85 m (70% <= 1 m) | 0.85 m (**80% <= 1 m**, n=25) |
| Floating vertices | 540 | **0** |
| Time | 259 s | 284 s |

DJI_1001 surface cells went from 4.5 M to 6.5 M (+45%). Of the visible scene,
64% is measured and 36% INFERRED. The DSM spans 142-179 m (p1-p99, the
terrain) with no floating cells; before the fix, 44k cells reached up to the
camera altitude.

## Measured 3D becomes the default (2026-09-27)

`geometry.dense_method: auto` now runs measured 3D (`geometry/mvs3d.py`):
plane-sweep stereo per camera with semi-global matching, cross-view
consistency (depth and 1 px round trip), TSDF fusion (Open3D tensor
`VoxelBlockGrid`), a true-ortho texture and closure of enclosed gaps up to
8 m on downward flights. Terrain 2.5D (`heightfield`) is explicit only.

What changed on the way, each measured on DJI_1001 or flight01:

- `torch.topk` over the source views was 75% of the sweep on MPS (488 ms
  vs 3.7 ms for an exact pairwise-ranking top-k): 8.4 -> 2.1 s per 924 px view.
- A local refinement that rejected every pixel whose refined peak fell on
  its window edge halved the depth fill (88% -> 46%); it now keeps the swept
  estimate there, and runs only when a hypothesis step is coarse (close range).
- SGM: local depth roughness p90 0.75 -> 0.41 m at 280 m range; neighbouring
  views then agree to 0.2 m median (DJI_1001, 7 views).
- Tensor TSDF: 19.2 -> 1.1 s for 7 views; hypotheses chosen for a 0.5% depth
  step (48 on aerial footage, up to 128 close to the camera); reference
  subset for dense flights (flight01: 84 of 185 views).

| flight01, same cameras | survey points on surface | survey rays, median | stable features vs IGN | geometry |
|---|---|---|---|---|
| Terrain 2.5D (heightfield) | 61/64 | 3.94 m | 0.85 m (n = 25, 80% <= 1 m) | 30 s |
| Measured 3D | 26/64 | 3.38 m | 0.29 m (n = 13, 100% <= 1 m) | 174 s |

Measured 3D is the more accurate surface where it measures, but leaves
~60% of flight01's ground open: strips flown in opposite directions
disagree by ~3 m in depth (|dz| p50 3.0 m, 3.7 px round trip, against
0.25 m within a strip), so the consistency test rejects their overlap.
Tying the strips in the camera solve (loop pairs within 0.45 x the
footprint, 866 pairs) made them agree to 0.3 m, doubled the measured
coverage (55/64 survey points) and improved the camera track against COLMAP
(1.61 -> 1.12 m), but bent the block against IGN (tile internal distortion
0.92 -> 1.99 m; survey rays 3.9 -> 5.3 m with the height field on the same
cameras). The pattern fits rolling shutter -- ~0.7% along-track stretch at
17 m/s, sign flipping with flight direction, ~1.6 m of depth per strip at
115 m -- which the rigid camera model cannot absorb. Cross-strip matching
is therefore off (`_MATCH_FOOTPRINT_FRACTION = 0`) until the camera model
handles rolling shutter.

DJI_1001 (no strips flown against each other): 49 depth maps, 87% of depths
confirmed, 8.0 M vertices at 0.5 m, 84% MEASURED, geometry 143 s, run ~455 s.
Trees come out as crowns and houses as blocks where the height field
extruded both into prisms.
