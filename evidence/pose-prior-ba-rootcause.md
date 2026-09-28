# Pose-prior bundle adjustment: why cameras moved tens of metres

Follow-up to `camera-alignment-criticalfix.md`, which added the physical
camera gate. The gate was right to reject: every real-footage solve was
physically wrong. This note records why, what changed, and the evidence.
Reproduced on flight01 (`output/mesh_validation/source_20s.mp4`, SIFT) with
the pose-prior problem dumped via `DRISHTI_DUMP_BA` and re-solved offline.

## Root causes, in the order they bit

1. **Gauge anchors at bad seeds.** With no explicit fixed cameras,
   `bundle._default_gauge_fix` froze cameras 0 and 1 at their seed poses and
   dropped their priors, although all 12 cameras carried GPS and gravity
   priors. The clip starts in a turn where yaw-from-flow seeded those two
   cameras 29 deg off (telemetry was right). The whole strip rotated onto
   them: 169 m at the far end, 30 deg rotations. Fix: when GPS (spread
   > 4 sigma) plus gravity priors already define position, scale and
   orientation, fix no camera (`_priors_fix_datum`).

2. **Unmodelled lens distortion.** With the gauge fixed, the pinhole solve
   converged to a bowl: tilts 33 -> 2 -> 32 deg along the strip, heights bowed
   +/-20 m, ground 30 m too close, yet 0.39 px reprojection. The lens is a
   strong barrel (k1 -0.22, ~70 px at the frame edges, ~200 px at the
   corners). Fix: one shared radial (k1, k2) per video in BA
   (`BAConfig.refine_distortion`), solved in the pose prior whenever no
   distortion is on record; once validated, every decoded frame and the
   keyframe cache are undistorted with the same K, stored tracks move to
   their undistorted pixels, all intrinsics become pinhole, and the raw lens
   is kept in `state.lens_distortion` / `cameras.json` (`_install_lens`).
   Solved on the 12-camera clip: k1 -0.225, k2 0.044 at f 833; on the
   208-camera flight: -0.243/0.051 at f 881. COLMAP's OPENCV
   self-calibration of the same flight: fx 876, k1 -0.241, k2 0.050.

3. **Gate reference was the flow-yaw seed.** A correct solution moved
   cameras 0/1 by 35 deg from their (wrong) seeds. The rotation check now
   uses the closer of the seed and the logged attitude; the position check
   is unchanged. A lens-model check (no fold-over, corner distortion <= 40%)
   was added.

4. **Georeferencing re-fitted rotation on world-frame poses.**
   `_apply_georeferencing` fixed the rotation only for the
   `telemetry_rotation` merge; the world-frame path re-fitted it by Umeyama
   on a near-collinear strip, rolling the exported model and cameras 14 deg
   about the flight line (dense cloud 12 deg tilted, 21 m sideways vs DEM).

5. **Clip clock.** `test_flight01_visible.mkv` has a 0.486 s container start;
   `flight01_120_480.mp4` t=0 is mkv PTS 120.464 s (frame matching), so the
   runs need `--telemetry-offset 121.66` and scoring `--clip-start 120.46`,
   not 121.2 / 120. The 0.46 s error is ~8 m along-track at 17 m/s: on the
   grid flight BA reconciled opposite legs through cross-leg tie points and
   sat 8.3 m from GPS along each camera's own direction of motion (COLMAP:
   8.9 m). Fitting the BA cameras to GPS(t + dt) gives dt = 121.60 s.

6. **At the correct clock, a few frames the images cannot pose.** Three
   banking end-of-leg frames (2.6 px vs 0.2 px) warm-started the 12-camera
   clip into the bowl; on the 208-camera flight, 17 turn-boundary keyframes
   had 1-14 observations and one camera was pulled 36-38 m off GPS. Cameras
   with < 15 observations, > 4 GPS sigma off, or median residual
   > max(1.5 px, 4x the global median) now keep their telemetry pose, and the
   solve restarts from the seeds without them (restarting matters: continuing
   from the bent solution keeps the bend). They are listed in
   `state.unrefined_keyframes`, skipped for dense views and fit points, and
   the gate judges the posed cameras. More than 1/3 unposed still rejects.

7. **Unobservable focal.** Over flat ground seen straight down, focal and
   ground depth trade off exactly and GPS pins only the cameras. DJI_1001
   (no altitude log) ran the shared focal to 2.2x. A log-space prior
   (sigma 0.3) now decides only that direction. The pipeline no longer
   uses the per-camera fx/fy/cx/cy refinement; the main checkout's run on
   DJI_1001 was rejected for a moving principal point, the same ambiguity.

8. **Runtime.** `ftol` 1e-10 is never met on real tracks: the robust pass
   always ran to 3000 evaluations. 1e-6 stops after ~17 with cameras within
   1.4 cm / 0.008 deg of that answer.

## Verification

Gate results (limits 15/30 m, 15/30 deg), posed cameras:

| run | clock | cameras (unposed) | shift median/max | rotation median/max | reprojection |
|---|---|---|---|---|---|
| old code, 20 s clip | 121.2 | 12 | 60.1 / 168.8 m, rejected | 31.5 / 49.0 deg | |
| 20 s clip | 121.2 | 12 (0) | 1.23 / 3.77 m | 3.8 / 14.1 deg | 0.29 px |
| 20 s clip | 121.68 | 12 (3) | 0.68 / 1.81 m | 2.6 / 6.7 deg | 0.28 px |
| DJI_1001 (dry run) | video clock | 52 (14) | 1.65 / 3.41 m | 0.8 / 3.4 deg | 0.90 px |
| full flight01, main + this patch | 121.66 | 208 (19) | 3.47 / 8.60 m | 6.3 / 15.6 deg | 0.35 px |

The full-flight run (main checkout's height-field dense path + this patch, SIFT matching)
finished with outcome **Valid**, placement PASS, lens k1 -0.241 / k2 0.050
at f 881 (COLMAP OPENCV on the same flight: fx 876, k1 -0.241, k2 0.050),
and TimeSyncStage agreeing with the 121.66 s clock to -0.07 s.

End to end (`scripts/score_flight01.py --clip-start 120.46`, mesh surface):
PinPoint's 64-point nadir subset scores median 5.5 m on the 18 rays that hit
the surface (logged-attitude baseline on the same points 11.6 m); all 100
points: 31 hits, median 12.6 m. The 36 non-nadir points fall in banked turns
(aircraft roll 24-46 deg while the CSV gimbal columns stay constant) where
triage leaves 8-10 s keyframe gaps, so keyframe interpolation cannot score
them. Rays from the solved cameras + lens onto the reference DEM instead of
the model: nadir subset median 5.1-5.4 m (baseline 10.0 m).

DSM vs `flight01/reference/dem.tif`, on DSM pixels backed by the model
(51% of the raster; the rest is gap fill): median +1.1 m, MAD 2.3 m, 74%
within 5 m. The pre-fix full-flight run: median +17.9 m, MAD 21.6 m.

Sparse BA points of the 20 s clip against `flight01/reference/dem.tif`:
constant +8.5 m offset (f 833 from flow vs true ~877: the flat-terrain
focal/depth ambiguity), cross-track slope 0.8 deg, residual MAD 0.59 m.

Tests: new regression tests for each cause in `tests/test_bundle.py`
(datum policy, badly seeded anchors, OpenCV lens projection, doming and
its removal), `tests/test_pose_validation.py` (attitude reference, lens
fold-over, transactional lens commit, world-frame georeferencing,
unposed-camera handling and cap) and `tests/test_ingest.py` (frame and
cache undistortion).
