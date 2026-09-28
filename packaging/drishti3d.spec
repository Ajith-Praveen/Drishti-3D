# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DRISHTI-3D native desktop application.

Build with::

    uv run pyinstaller packaging/drishti3d.spec --noconfirm

Produces ``dist/DRISHTI-3D.app`` on macOS and ``dist/DRISHTI-3D/`` (with
``DRISHTI-3D.exe``) on Windows/Linux. One application, no Python install
required on the target machine, no network access at runtime.

Why onedir and not onefile
--------------------------
``--onefile`` unpacks the entire bundle to a temp directory on every
launch. This application ships VTK and Qt, which together are several
hundred megabytes, so onefile turns a one-second start into a
twenty-second one and re-pays that cost every single run. Onedir (and the
``.app`` bundle that wraps it on macOS) is what a user expects a native
application to behave like.

Why so many hidden imports
--------------------------
Two libraries here defeat static analysis:

- **VTK** builds its public API by importing ``vtkmodules.all`` at runtime
  through ``__getattr__`` indirection, so PyInstaller's import graph sees
  almost none of the ~200 submodules that are actually needed.
- **The geometry/semantics backbones** are imported lazily by design (see
  ``geometry.backbone.create_backbone`` and
  ``semantics.segmenter.create_segmenter``) precisely so the pipeline runs
  without torch installed. That deliberate laziness means PyInstaller
  cannot see them either, so anything optional must be named explicitly or
  excluded on purpose.
"""

import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

APP_NAME = "DRISHTI-3D"
PROJECT_ROOT = Path(SPECPATH).parent  # noqa: F821 -- SPECPATH is injected by PyInstaller

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------

hiddenimports: list[str] = []

# VTK: collect every submodule. Expensive to bundle, but the alternative is
# a viewport that raises AttributeError on a machine we cannot debug.
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += ["vtkmodules.all", "vtkmodules.util.numpy_support", "vtkmodules.qt.QVTKRenderWindowInteractor"]

# Our own stage modules are imported lazily, inside each stage's run(), so
# none of them appear in any static import chain from app.main. Collect the
# whole package: a hand-kept list went stale as soon as new modules landed
# (heightmap, incremental, depth_fit, pose_validation, reference_align...).
hiddenimports += collect_submodules("drishti3d")

# Scientific stack pieces that are reached through dispatch rather than a
# plain import statement.
hiddenimports += [
    "scipy.spatial.transform._rotation_groups",
    "scipy.special._cdflib",
    "scipy._lib.messagestream",
    "laspy.vlrs.known",
    "pyproj.datadir",
]

datas = []
# The pinned reconstruction-runtime manifest drishti3d.runtime reads before
# every run. Without it the frozen app's preflight cannot start at all.
datas += [(str(PROJECT_ROOT / "drishti3d" / "models" / "mapanything-runtime.json"), "drishti3d/models")]
# pyproj ships its own PROJ database; without it every CRS lookup fails at
# runtime with an error that looks nothing like "missing data file".
datas += collect_data_files("pyproj")
datas += collect_data_files("laspy")
# The bundled UI typefaces (Tools > Preferences > Type) and the logo SVG.
# theme.load_bundled_fonts / icons.mark look for them at these same
# package-relative paths inside the frozen app.
datas += [
    (str(PROJECT_ROOT / "drishti3d" / "app" / "fonts"), "drishti3d/app/fonts"),
    (str(PROJECT_ROOT / "drishti3d" / "app" / "assets"), "drishti3d/app/assets"),
]

# ---------------------------------------------------------------------------
# Reconstruction runtime (torch + MapAnything)
# ---------------------------------------------------------------------------
#
# Geometry REQUIRES the MapAnything backbone: without it the stage fails
# rather than silently producing synthetic output (see pipeline.stages), so
# an app built without torch opens but cannot reconstruct real footage.
# The runtime is therefore bundled whenever it is installed in the build
# environment. PyInstaller builds for the machine it runs on, so the torch
# wheel bundled here is the right one for that platform (MPS on Apple
# Silicon); a CUDA build is made on the CUDA machine.
#
#   DRISHTI3D_BUNDLE_ML=auto (default)  bundle when importable, warn if not
#   DRISHTI3D_BUNDLE_ML=1               fail the build when missing
#   DRISHTI3D_BUNDLE_ML=0               slim viewer-only build
#
# drishti3d.runtime checks the pinned package versions through
# importlib.metadata, so their dist-info metadata ships too.
#
# Weights: set DRISHTI3D_BUILD_WEIGHTS_DIR to a verified MapAnything
# snapshot (`python -m drishti3d.runtime locate` prints it) to bundle the
# 4.9 GB checkpoint for an air-gapped machine. Without it the app uses the
# pinned snapshot in ~/.cache/huggingface or $DRISHTI3D_MAPANYTHING_WEIGHTS.

_ML_RUNTIME = ("torch", "torchvision", "mapanything", "uniception", "safetensors")
# Imported by MapAnything's model code at load time, several through hydra
# config instantiation that static analysis cannot see.
_ML_SUPPORT = ("hydra", "omegaconf", "timm", "einops", "huggingface_hub")


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_bundle_ml_setting = os.environ.get("DRISHTI3D_BUNDLE_ML", "auto").strip().lower()
_ml_missing = [name for name in _ML_RUNTIME + _ML_SUPPORT if not _importable(name)]
if _bundle_ml_setting in ("0", "false", "no"):
    BUNDLE_ML = False
elif _ml_missing:
    if _bundle_ml_setting in ("1", "true", "yes"):
        raise SystemExit(
            f"DRISHTI3D_BUNDLE_ML=1 but {', '.join(_ml_missing)} not installed; "
            "run `uv sync --extra gui --extra ml` first"
        )
    print(
        "WARNING: building WITHOUT the reconstruction runtime "
        f"({', '.join(_ml_missing)} missing): the app will open but cannot reconstruct real footage"
    )
    BUNDLE_ML = False
else:
    BUNDLE_ML = True

binaries: list = []
if BUNDLE_ML:
    # torchvision >= 0.29 loads its compiled ops as _C_stable.so/image_stable.so
    # through torch.ops.load_library from its own package folder. The stock
    # hook still names the old torchvision._C module, so without this the
    # frozen app fails with "operator torchvision::nms does not exist".
    # collect_dynamic_libs only matches lib*.so / *.dylib: it picks up the
    # image codecs under .dylibs but not _C_stable.so itself. Windows ships
    # the same dynamically loaded operators as .pyd, not .so.
    binaries += collect_dynamic_libs("torchvision")
    _tv_dir = Path(importlib.util.find_spec("torchvision").submodule_search_locations[0])
    binaries += [(str(_so), "torchvision") for _pattern in ("*.so", "*.pyd") for _so in sorted(_tv_dir.glob(_pattern))]
    for _name in ("mapanything", "uniception", "hydra", "omegaconf", "timm"):
        hiddenimports += collect_submodules(_name)
        datas += collect_data_files(_name)
    for _name in _ML_RUNTIME:
        datas += copy_metadata(_name)
    # DISK + LightGlue matching; the pipeline falls back to SIFT without it.
    if _importable("kornia"):
        hiddenimports += collect_submodules("kornia")
        datas += copy_metadata("kornia")

    # torch hub: MapAnything builds its DINOv2 encoder from hub CODE
    # (facebookresearch_dinov2_main), and kornia keeps the DISK/LightGlue
    # weights under hub/checkpoints. Both are fetched from the network on
    # first use, so an air-gapped machine without this folder cannot build
    # the model. Bundled as torch_home/hub; entry.py points TORCH_HOME at it
    # when the user's own cache lacks the DINOv2 code.
    _hub = Path(os.environ.get("DRISHTI3D_BUILD_TORCH_HUB", Path.home() / ".cache" / "torch" / "hub"))
    if (_hub / "facebookresearch_dinov2_main").is_dir():
        datas += [(str(_hub / "facebookresearch_dinov2_main"), "torch_home/hub/facebookresearch_dinov2_main")]
        for _ckpt in sorted((_hub / "checkpoints").glob("*")) if (_hub / "checkpoints").is_dir() else []:
            datas += [(str(_ckpt), "torch_home/hub/checkpoints")]
    else:
        print(f"WARNING: no DINOv2 torch-hub code in {_hub}: the built app needs network on first model load")

    _weights_dir = os.environ.get("DRISHTI3D_BUILD_WEIGHTS_DIR")
    if _weights_dir:
        import json as _json

        _manifest = _json.loads((PROJECT_ROOT / "drishti3d" / "models" / "mapanything-runtime.json").read_text())
        _missing = [f for f in _manifest["files"] if not (Path(_weights_dir) / f).is_file()]
        if _missing:
            raise SystemExit(f"DRISHTI3D_BUILD_WEIGHTS_DIR={_weights_dir} lacks {', '.join(_missing)}")
        # runtime._frozen_weights_dir looks exactly here.
        datas += [(str((Path(_weights_dir) / f).resolve()), "drishti3d/models/mapanything") for f in _manifest["files"]]

    # Semantic masking (SemanticsStage: vehicles, people, sky) with SegFormer.
    # transformers imports its model families lazily, so the one used is
    # named explicitly. The checkpoint itself downloads on first use.
    if _importable("transformers"):
        hiddenimports += collect_submodules("transformers.models.segformer")
        hiddenimports += ["transformers.image_processing_utils", "transformers.modeling_utils"]
        for _name in ("transformers", "tokenizers", "safetensors", "huggingface_hub", "regex"):
            if _importable(_name):
                datas += copy_metadata(_name)

# Photographic texture atlas (fusion.texture); per-vertex colour without it.
if _importable("xatlas"):
    hiddenimports += ["xatlas"]
# Reference orthophoto/DEM alignment reads GeoTIFFs through rasterio (GDAL).
if _importable("rasterio"):
    hiddenimports += collect_submodules("rasterio")
    datas += collect_data_files("rasterio")

# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
#
# transformers is bundled with the ML stack (above): SemanticsStage skips the
# ground-level ADE20K model on downward flights (70-92% UNLABELLED there) and
# runs it on forward/oblique footage, where it masks people, vehicles and sky.
excludes = [
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
]
if not BUNDLE_ML:
    excludes += ["torch", "torchvision", "mapanything", "uniception", "kornia"]

block_cipher = None

a = Analysis(  # noqa: F821
    [str(PROJECT_ROOT / "packaging" / "entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    # kornia compiles helpers with TorchScript, which reads the .py SOURCE:
    # bytecode-only collection breaks DISK+LightGlue in the frozen app ("Can't
    # get source for sampson_epipolar_distance"). torch's own hook does the
    # same for torch.
    module_collection_mode={"kornia": "pyz+py"} if BUNDLE_ML else {},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821

# Windows taskbar/Explorer icon (packaging/make_icon.py --ico); macOS uses
# the .icns on the bundle below, Linux the .png in the desktop entry.
_ICO = PROJECT_ROOT / "packaging" / "DRISHTI-3D.ico"

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts Qt/VTK shared libraries; never enable it here.
    icon=str(_ICO) if sys.platform == "win32" and _ICO.exists() else None,
    # Preserve CLI stdout/stderr on Windows; hide an owned console when
    # launched from Explorer. Windowed mode replaces those streams with None.
    console=sys.platform == "win32",
    hide_console="hide-early" if sys.platform == "win32" else None,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# Generated from app/assets/logo.svg by packaging/make_icon.py (build_app.sh
# runs it first). Absent on a bare PyInstaller run, which then falls back
# to the generic icon rather than failing.
_ICNS = PROJECT_ROOT / "packaging" / "DRISHTI-3D.icns"

# Windows and Linux ship the COLLECT folder (dist/DRISHTI-3D/) as is.
if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=str(_ICNS) if _ICNS.exists() else None,
        bundle_identifier="in.drishti3d.app",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": "DRISHTI-3D",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            # The app reads drone video the user picks from a file dialog; on
            # recent macOS that needs a declared purpose string or the open
            # dialog silently returns nothing for files in protected folders.
            "NSDesktopFolderUsageDescription": "DRISHTI-3D opens drone video files you select.",
            "NSDocumentsFolderUsageDescription": "DRISHTI-3D opens drone video files you select.",
            "NSDownloadsFolderUsageDescription": "DRISHTI-3D opens drone video files you select.",
        },
    )
