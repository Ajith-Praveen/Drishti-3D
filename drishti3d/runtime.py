"""Offline reconstruction-runtime discovery and fail-fast validation.

The frozen application is a tested macOS/arm64 runtime: its Python,
torch/torchvision/MapAnything versions and Apache checkpoint are pinned in
``models/mapanything-runtime.json``.  This module checks that contract before
video decoding starts and never downloads a model at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

WEIGHTS_ENV_VAR = "DRISHTI3D_MAPANYTHING_WEIGHTS"
BUILD_WEIGHTS_ENV_VAR = "DRISHTI3D_BUILD_WEIGHTS_DIR"
_MANIFEST = Path(__file__).with_name("models") / "mapanything-runtime.json"


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    ok: bool
    detail: str
    warning: bool = False


@dataclass(frozen=True)
class RuntimeReport:
    backbone: str
    platform_id: str
    backend: str | None
    weights_dir: str | None
    checks: tuple[RuntimeCheck, ...]
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return all(check.ok or check.warning for check in self.checks)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.ok and not check.warning)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def summary(self) -> str:
        if self.skipped:
            return f"runtime preflight skipped for backbone {self.backbone!r}"
        if self.ok:
            return f"runtime preflight passed ({self.platform_id}, {self.backend}, offline weights verified)"
        return "runtime preflight failed: " + "; ".join(self.errors)


class ReconstructionRuntimeError(RuntimeError):
    def __init__(self, report: RuntimeReport) -> None:
        self.report = report
        super().__init__(report.summary())


def load_runtime_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else _MANIFEST
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read reconstruction runtime manifest {manifest_path}: {exc}") from exc


def platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    names = {"darwin": "macos", "windows": "windows", "linux": "linux"}
    return f"{names.get(system, system)}-{machine}"


def _frozen_weights_dir() -> Path | None:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return None
    return Path(bundle_root) / "drishti3d" / "models" / "mapanything"


def _cache_weights_dir(manifest: dict[str, Any]) -> Path:
    revision = manifest["revision"]
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--map-anything-apache"
        / "snapshots"
        / revision
    )


def locate_weights(
    explicit: str | Path | None = None, manifest: dict[str, Any] | None = None
) -> Path | None:
    """Find only local weights, with bundled assets taking priority in a frozen app."""
    manifest = manifest or load_runtime_manifest()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(WEIGHTS_ENV_VAR):
        candidates.append(Path(os.environ[WEIGHTS_ENV_VAR]))
    frozen = _frozen_weights_dir()
    if frozen is not None:
        candidates.append(frozen)
    candidates.append(_cache_weights_dir(manifest))
    required = tuple(manifest["files"])
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weight_files(weights_dir: Path, manifest: dict[str, Any], verify_hash: bool = True) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    for filename, expected in manifest["files"].items():
        path = weights_dir / filename
        if not path.is_file():
            checks.append(RuntimeCheck(f"weights:{filename}", False, f"missing model file: {path}"))
            continue
        size = path.stat().st_size
        if size != int(expected["size"]):
            checks.append(
                RuntimeCheck(
                    f"weights:{filename}",
                    False,
                    f"model file {path} has {size} bytes; expected {expected['size']}",
                )
            )
            continue
        if verify_hash:
            actual = _sha256(path)
            if actual != expected["sha256"]:
                checks.append(
                    RuntimeCheck(
                        f"weights:{filename}",
                        False,
                        f"checksum mismatch for {path}: got {actual}, expected {expected['sha256']}",
                    )
                )
                continue
        checks.append(RuntimeCheck(f"weights:{filename}", True, f"verified {path.name} ({size} bytes)"))
    return checks


def torch_hub_dir() -> Path:
    """The directory torch.hub resolves, without importing torch ($TORCH_HOME/hub)."""
    home = os.environ.get("TORCH_HOME")
    if home:
        return Path(home) / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch" / "hub"


def torch_hub_checks() -> list[RuntimeCheck]:
    """MapAnything builds its encoder from torch-hub CODE, and would fetch it from GitHub if absent."""
    hub = torch_hub_dir()
    code = hub / "facebookresearch_dinov2_main"
    checks = [
        RuntimeCheck(
            "torch-hub:dinov2",
            code.is_dir(),
            f"DINOv2 encoder code found in {code}"
            if code.is_dir()
            else f"DINOv2 encoder code missing from {hub}: MapAnything would try to download it",
        )
    ]
    matcher = [hub / "checkpoints" / n for n in ("depth-save.pth", "disk_lightglue_v0-1_arxiv-pth")]
    have = all(m.is_file() for m in matcher)
    checks.append(
        RuntimeCheck(
            "torch-hub:disk-lightglue",
            have,
            "DISK + LightGlue weights present" if have else "DISK + LightGlue weights missing: matching falls back to SIFT offline",
            warning=True,
        )
    )
    return checks


def _package_check(package: str, expected: str, enforce_version: bool) -> RuntimeCheck:
    try:
        actual = version(package)
        importlib.import_module(package)
    except (PackageNotFoundError, ImportError, OSError, RuntimeError) as exc:
        return RuntimeCheck(f"dependency:{package}", False, f"{package} cannot be loaded: {exc}")
    if enforce_version and actual != expected:
        return RuntimeCheck(
            f"dependency:{package}", False, f"{package} {actual} is installed; tested runtime requires {expected}"
        )
    suffix = " (tested version)" if actual == expected else " (unverified version)"
    return RuntimeCheck(f"dependency:{package}", True, f"{package} {actual}{suffix}")


def check_reconstruction_runtime(
    backbone: str,
    *,
    weights_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    strict_platform: bool | None = None,
    verify_hash: bool = True,
) -> RuntimeReport:
    """Validate dependencies, accelerator and immutable local weights.

    ``null`` and other explicitly selected backbones bypass this MapAnything
    contract.  Frozen applications are strict by default; source checkouts
    still report an unverified platform as a warning so CUDA development is
    not disabled by a macOS-specific desktop release.
    """
    current_platform = platform_id()
    if backbone.lower() != "mapanything":
        return RuntimeReport(backbone, current_platform, None, None, (), skipped=True)

    manifest = load_runtime_manifest(manifest_path)
    supported = manifest["supported_runtime"]
    strict = bool(getattr(sys, "frozen", False)) if strict_platform is None else strict_platform
    matches_platform = platform.system() == supported["system"] and platform.machine() == supported["machine"]
    checks: list[RuntimeCheck] = []
    checks.append(
        RuntimeCheck(
            "platform",
            matches_platform,
            (
                f"platform {current_platform} matches tested runtime"
                if matches_platform
                else f"platform {current_platform} is unverified; shipped runtime is {supported['platform_id']}"
            ),
            warning=not strict,
        )
    )
    py_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    enforce_versions = matches_platform or strict
    checks.append(
        RuntimeCheck(
            "python",
            (not enforce_versions) or py_actual == supported["python"],
            f"Python {py_actual}; tested runtime requires {supported['python']}",
        )
    )
    for package, expected in supported["packages"].items():
        checks.append(_package_check(package, expected, enforce_versions))

    backend: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            backend = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            backend = "mps"
        else:
            backend = "cpu"
        expected_backend = supported["backend"]
        backend_ok = (not enforce_versions) or backend == expected_backend
        checks.append(
            RuntimeCheck(
                "backend",
                backend_ok,
                f"selected backend {backend}; tested {supported['platform_id']} runtime requires {expected_backend}",
            )
        )
    except Exception as exc:  # dependency check already captures the usual cases
        checks.append(RuntimeCheck("backend", False, f"cannot inspect torch accelerator: {exc}"))

    checks.extend(torch_hub_checks())

    located = locate_weights(weights_dir, manifest)
    if located is None:
        checks.append(
            RuntimeCheck(
                "weights",
                False,
                f"offline {manifest['checkpoint']} weights were not found; set {WEIGHTS_ENV_VAR} to a verified snapshot",
            )
        )
    else:
        checks.extend(verify_weight_files(located, manifest, verify_hash=verify_hash))
        # This opens and parses the tensor index without allocating the 4.6 GB model.
        try:
            from safetensors import safe_open

            with safe_open(located / "model.safetensors", framework="pt", device="cpu") as handle:
                tensor_count = len(handle.keys())
            checks.append(
                RuntimeCheck("weights:tensor-index", tensor_count > 0, f"safetensors index contains {tensor_count} tensors")
            )
        except Exception as exc:
            checks.append(RuntimeCheck("weights:tensor-index", False, f"model tensor index cannot be opened: {exc}"))

    return RuntimeReport(backbone, current_platform, backend, str(located) if located else None, tuple(checks))


def require_reconstruction_runtime(backbone: str, **kwargs: Any) -> RuntimeReport:
    report = check_reconstruction_runtime(backbone, **kwargs)
    if not report.ok:
        raise ReconstructionRuntimeError(report)
    if report.weights_dir:
        # Geometry constructs the adapter later.  Pin it to the exact snapshot
        # just verified so from_pretrained cannot select a cache head or network.
        os.environ[WEIGHTS_ENV_VAR] = report.weights_dir
    return report


def smoke_model(weights_dir: Path, *, inference: bool) -> dict[str, Any]:
    """Actually load the checkpoint and optionally execute a two-view prediction."""
    os.environ[WEIGHTS_ENV_VAR] = str(weights_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch

    from drishti3d.geometry.mapanything import MapAnythingBackbone
    from drishti3d.types import CameraIntrinsics, Pose

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    model = MapAnythingBackbone(local_weights_dir=weights_dir, max_image_size=56, mask_edges=False)
    model.load(device=device)
    result: dict[str, Any] = {"loaded": True, "device": device}
    if inference:
        yy, xx = np.mgrid[:56, :56]
        base = np.stack(((xx * 4) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256), axis=-1).astype(np.uint8)
        images = [base, np.roll(base, 2, axis=1)]
        intrinsics = [CameraIntrinsics(50, 50, 27.5, 27.5, 56, 56) for _ in images]
        poses = [Pose(np.eye(3), np.array([float(i), 0.0, 20.0])) for i in range(2)]
        prediction = model.predict(images, intrinsics=intrinsics, poses=poses)
        finite = np.isfinite(prediction.depth) & (prediction.depth > 0)
        poses_finite = all(np.isfinite(p.R).all() and np.isfinite(p.t).all() for p in prediction.poses)
        if not finite.any() or not poses_finite:
            raise RuntimeError("MapAnything smoke inference produced no finite positive depth or a non-finite pose")
        result.update(
            inferred=True,
            views=int(prediction.depth.shape[0]),
            valid_depth_pixels=int(finite.sum()),
            output_shape=list(prediction.depth.shape),
            finite_poses=poses_finite,
        )
    model.unload()
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the offline DRISHTI-3D reconstruction runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    locate = sub.add_parser("locate", help="print the local checkpoint directory")
    locate.add_argument("--weights")
    check = sub.add_parser("check", help="verify runtime dependencies and model files")
    check.add_argument("--weights")
    check.add_argument("--strict-platform", action="store_true")
    check.add_argument("--skip-hash", action="store_true")
    check.add_argument("--smoke", choices=("none", "load", "inference"), default="none")
    check.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "locate":
        path = locate_weights(args.weights)
        if path is None:
            print(f"offline weights not found; set {WEIGHTS_ENV_VAR}", file=sys.stderr)
            return 1
        print(path)
        return 0
    report = check_reconstruction_runtime(
        "mapanything",
        weights_dir=args.weights,
        strict_platform=args.strict_platform,
        verify_hash=not args.skip_hash,
    )
    if not report.ok:
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary(), file=sys.stderr)
        return 1
    smoke = None
    if args.smoke != "none":
        smoke = smoke_model(Path(report.weights_dir), inference=args.smoke == "inference")
    print(json.dumps({"runtime": report.to_dict(), "smoke": smoke}, indent=2) if args.json else report.summary())
    if smoke:
        print(f"model smoke passed: {smoke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
