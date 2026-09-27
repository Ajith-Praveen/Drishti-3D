# Camera alignment and depth-scale rejection

Investigated `output/mesh_validation/rotation_corrected_run/meta.json` and
`arrays.npz`, plus `pose_probe.log` and the current reconstruction pipeline.

## Evidence

The saved 12-camera, 20-second run used an explicit 120-second telemetry offset,
full timeline coverage, and guessed camera intrinsics. It accepted 5.776 px
reprojection RMSE despite median camera movement of 45.807 m / 24.083 degrees.
The maximum position error was 72.287 m. Re-triangulation retained zero tracks.
Convergence and lower image error did not establish physical correctness.

Coverage proves overlap of time ranges, not that the selected video and flight
instants correspond. The correct offset cannot be established from these metrics.

## Confirmed implementation defects and corrections

- Telemetry compass headings were passed directly into a counterclockwise ENU
  rotation function. Convert at the telemetry-to-pose boundary and before flow
  diagnostics; keep internal yaw/course conventions unchanged.
- Linear heading interpolation crossed the long arc at North / +/-180 degrees.
  Unwrap headings before interpolation.
- A global robust loss downweighted GPS/gravity/GCP violations along with image
  outliers. Apply robust loss only to reprojection rows; retain quadratic priors.
  Cap image-derived relaxation of the tilt prior at 5 degrees.
- BA independently refined calibration even for measured lenses; pass two and
  depth-fit triangulation then used stale calibration. Refine only a guessed
  calibration with sufficient GPS support, use the solved intrinsics in pass
  two, and propagate every accepted camera's fx/fy/cx/cy at native resolution.
  Do not replace a solved calibration with a median focal length.
- No physical acceptance gate existed. Validate both BA passes against the
  original seed and raw GPS before committing cameras, calibration or depth-fit
  points. Reject median/max position shifts above 15/30 m, median/max rotation
  changes above 15/30 degrees, and long-baseline median scale outside 0.8–1.25.
  Reject nonfinite poses, invalid rotations/calibration, excessive lens changes,
  >1% nonpositive observed depths, and insufficient re-triangulated tracks.
  These are conservative rejection limits, not accuracy guarantees.
- Failed camera refinement now stops downstream reconstruction and export;
  genuinely unavailable refinement remains a distinct skipped state.
- An unresolved CSV/GPX clock is refused before camera refinement/dense geometry.
  Old geometry caches are invalidated; changed camera conditioning/calibration
  or telemetry-clock evidence prevents stale geometry reuse.

## Saved-data verification

Applying the new gate to the saved keyframe poses rejects the actual artifact:
median position error 45.807 m, maximum 72.287 m.

A fresh sparse-only rerun of `source_20s.mp4` with the saved configuration and
explicit 120-second offset also rejects the candidate: median/max position
35.965/121.946 m, median/max rotation 21.613/40.548 degrees, baseline scale 1.405.
No dense backbone or new geometry export was run in this verification.

This change prevents unsafe reconstruction; it does not certify a replacement
model. The saved clip still needs independently verified synchronization and
camera calibration before a physically accepted reconstruction can be produced.

## Automated verification

- Bundle, ingest, pipeline and initial rejection regression suite: 70 passed.
- Cache, yaw, depth-fit and rejection tests: 44 passed.
- Expanded rejection/calibration tests, including both BA passes, measured-lens
  preservation and transactional rejection: 24 passed.
- Ruff passes for the new validation module/tests, bundle solver and cache
  changes; `git diff --check` passes. Existing unrelated lint findings in the
  larger stages module remain.

Pipeline plumbing tests explicitly skip matching their tiled synthetic video,
which has arbitrary GPS and no consistent physical camera. Camera acceptance is
tested separately; the report plumbing test injects a known measurement.
