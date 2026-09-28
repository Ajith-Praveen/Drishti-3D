# DRISHTI-3D

A native desktop application that reconstructs a geo-referenced 3D model from drone video and a flight telemetry log.

The app selects keyframes, matches features, solves camera poses, estimates multi-view depth, and fuses a textured surface. The viewer supports confidence colouring, point coordinates, distance, area, volume, and elevation-profile measurements. Models and maps export to OBJ, PLY, LAS, glTF/GLB, FBX and GeoTIFF; measurements export to GeoJSON and KML.

## Run from source

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --extra gui --extra ml --extra semantics --extra texture --extra reference
```

On macOS and Linux, launch with `./run.sh`. On Windows:

```powershell
.venv\Scripts\python.exe -m drishti3d.app.main
```

Select a drone video and telemetry log in the app, choose the output folder, then press **Run**. To reconstruct without the GUI:

```bash
.venv/bin/python -m drishti3d.pipeline.runner flight.mp4 --telemetry flight.csv --config full_3d.yaml --out results/flight
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. Run the command with `--help` for calibration, telemetry timing, and reference-data options.

## Build desktop packages

Build on the target operating system; PyInstaller does not cross-compile.

| Platform | Command | Output |
| --- | --- | --- |
| macOS Apple Silicon | `./packaging/build_app.sh` after syncing the environment above | `dist/DRISHTI-3D.app` |
| Windows x64 | `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1` | `dist/DRISHTI-3D-windows-x64.zip` |
| Linux x86_64 | `./packaging/build_linux.sh` | `dist/DRISHTI-3D-linux-x86_64.tar.gz` |

Windows defaults to the CUDA 12.6 wheels; use `-CpuOnly` for a CPU-only package or `-Cuda cu130` for that CUDA wheel variant. Linux x86_64 PyTorch wheels include CUDA runtime dependencies. Acceleration selects NVIDIA CUDA, Apple Metal, or CPU as available. NVIDIA acceleration requires a compatible host driver.

The Linux CI build uses Ubuntu 24.04. Install the Qt/VTK system libraries listed in `packaging/build_linux.sh` before building or running the package. Linux ARM is not supported by the current Open3D dependency.

Every full desktop build must pass `packaging/smoke_test.py` before packaging. The checks exercise native torchvision operators, geospatial libraries in both import orders, the reconstruction modules, and Qt resources. They do not require a display, GPU, or optional model downloads.

## Docker

Build the Linux x86_64 image:

```bash
docker build --platform linux/amd64 -f packaging/Dockerfile -t drishti3d .
docker run --rm drishti3d
docker run --rm --gpus all -v "$PWD/data:/data" drishti3d \
  run /data/flight.mp4 --telemetry /data/flight.csv --out /data/results
```

The default command prints pipeline help. NVIDIA acceleration requires the NVIDIA Container Toolkit; omit `--gpus all` to run on CPU. The image also supports `app` through an X11 display; see the Dockerfile header for the required mounts and environment.

Run the same mandatory smoke checks used by CI:

```bash
docker run --rm --entrypoint python drishti3d:latest packaging/smoke_test.py python packaging/entry.py
```

## Models and offline operation

Measured 3D is the default reconstruction mode. The optional learned-depth fallback uses MapAnything; its large checkpoint is not included in standard CI artifacts. The `runtime check` command specifically validates that optional offline runtime and reports missing assets as errors.

Set `DRISHTI3D_MAPANYTHING_WEIGHTS` to a verified local checkpoint for that fallback. For an offline bundle, set `DRISHTI3D_BUILD_WEIGHTS_DIR` before building. `DRISHTI3D_BUILD_TORCH_HUB` can provide cached DINOv2 code and DISK/LightGlue weights. Learned matching and semantic segmentation may need model caches prepared before offline use; matching can fall back to SIFT.

## Build verification

The GitHub Actions `build` workflow builds and smoke-tests Windows, Linux, macOS, and Docker. Start it through **Actions → build → Run workflow**, or push a version tag beginning with `v`. Desktop archives are attached to the run.

Local regression checks for the build validator:

```bash
.venv/bin/python -m unittest discover -s packaging -p test_smoke_test.py
```

The repository contains application source, bundled UI assets, dependencies, configuration, build infrastructure, and Markdown evidence in `evidence/`. Captured flights, generated output, evidence images and HTML/text reports, presentation material, local experiments, and deployment-demo scripts are excluded.
