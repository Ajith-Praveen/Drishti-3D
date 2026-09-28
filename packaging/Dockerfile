# DRISHTI-3D on Linux with an NVIDIA GPU: the headless pipeline, and the
# desktop app over X11.
#
#   docker build -f packaging/Dockerfile -t drishti3d .
#
#   # reconstruct (needs the NVIDIA Container Toolkit for --gpus)
#   docker run --rm --gpus all -v "$PWD/data:/data" drishti3d \
#       run /data/flight.mp4 --telemetry /data/flight.csv --out /data/out
#
#   # the desktop app on the host's X server
#   xhost +local:docker
#   docker run --rm --gpus all -e DISPLAY -e QT_QPA_PLATFORM=xcb \
#       -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD/data:/data" drishti3d app
#
# BASE can point at any Debian/Ubuntu image; a CPU-only build for machines
# without a GPU: --build-arg BASE=python:3.12-slim-bookworm
ARG BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg \
        libgl1 libegl1 libglib2.0-0 libfontconfig1 libdbus-1-3 \
        libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \
        libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 libxrender1 libxi6 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/drishti3d
# Dependencies first (a cached layer), then the source.
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project --extra gui --extra ml --extra semantics --extra reference
COPY drishti3d ./drishti3d
COPY packaging ./packaging
RUN uv sync --frozen --extra gui --extra ml --extra semantics --extra reference

ENV PATH="/opt/drishti3d/.venv/bin:${PATH}" \
    QT_QPA_PLATFORM=offscreen

# `run ...` / `runtime check` (packaging/entry.py), or `app` for the GUI.
RUN printf '%s\n' '#!/bin/sh' \
        'if [ "$1" = "app" ]; then shift; exec drishti3d "$@"; fi' \
        'exec python /opt/drishti3d/packaging/entry.py "$@"' > /usr/local/bin/drishti3d-entry \
    && chmod +x /usr/local/bin/drishti3d-entry
ENTRYPOINT ["drishti3d-entry"]
CMD ["run", "--help"]
