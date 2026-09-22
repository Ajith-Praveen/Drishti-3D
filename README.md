# DRISHTI-3D

Desktop app that turns a single-pass drone video into a georeferenced 3D model.

## Pipeline stages

1. **Ingest** — decode the drone video and align it with flight telemetry.
2. **Triage** — select a well-distributed, sharp, low-blur set of keyframes.
3. **Geometry** — recover per-keyframe camera poses and depth via a learned backbone.
4. **Fusion** — merge per-frame geometry into a single confidence-weighted point cloud.
5. **Export** — georeference and write the final model (e.g. LAS/PLY) to disk.

## Setup

```bash
# Install uv if you haven't already: https://docs.astral.sh/uv/
uv sync

# Optional extras
uv sync --extra gui   # PySide6 desktop UI + VTK viewport
uv sync --extra ml    # torch / torchvision (CUDA or MPS acceleration)
uv sync --extra semantics  # SegFormer semantic classification + dynamic-object masking
uv sync --extra texture    # xatlas UV unwrapping for the photographic texture atlas

# Run tests
uv run pytest
```

## Coordinate conventions

- **World frame**: ENU (East-North-Up), units in metres, Z-up. X points East,
  Y points North, Z points Up. All georeferenced/reconstructed geometry
  (poses, point clouds) is expressed in this frame unless otherwise noted.
- **Camera frame**: OpenCV convention. X points right, Y points down, Z
  points forward (out of the lens, into the scene).
- **Geographic coordinates**: latitude/longitude in decimal degrees (WGS84),
  altitude in metres. Distinct from the local ENU world frame; conversion
  between the two is handled during georeferencing (e.g. via `pyproj`).
