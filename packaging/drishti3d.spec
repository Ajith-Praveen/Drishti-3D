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

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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

# Our own lazily-imported stage modules -- none of these appear in any
# static import chain from app.main.
hiddenimports += [
    "drishti3d.export.bundle_export",
    "drishti3d.export.depthviz",
    "drishti3d.export.formats",
    "drishti3d.export.geotiff",
    "drishti3d.export.report",
    "drishti3d.export.terrain",
    "drishti3d.fusion.completion",
    "drishti3d.fusion.filters",
    "drishti3d.fusion.mesh",
    "drishti3d.fusion.texture",
    "drishti3d.geometry.backbone",
    "drishti3d.ingest.photometric",
    "drishti3d.pipeline.runner",
    "drishti3d.pipeline.stages",
    "drishti3d.semantics.classes",
    "drishti3d.semantics.labelling",
    "drishti3d.semantics.segmenter",
]

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
# pyproj ships its own PROJ database; without it every CRS lookup fails at
# runtime with an error that looks nothing like "missing data file".
datas += collect_data_files("pyproj")
datas += collect_data_files("laspy")

# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
#
# torch/transformers/xatlas are deliberately NOT bundled. Three reasons:
#
# 1. Size. A CUDA-enabled torch is ~2.5 GB; bundling it turns a ~400 MB
#    application into a ~3 GB one.
# 2. Hardware. The wheel that is correct here (CPU/MPS on this Mac) is the
#    wrong one on the deployment target (CUDA on a 4060), so a bundled
#    torch would be actively harmful.
# 3. It is unnecessary. Every stage that needs them degrades cleanly when
#    they are absent (SemanticsStage -> "skipped", texture -> per-vertex
#    colour, geometry -> NullBackbone), so the packaged app runs and
#    produces deliverables with none of them present.
#
# An operator who wants the learned backbones installs them into the same
# environment and runs from source, or uses a GPU build produced by
# `packaging/build_app.py --with-ml`.
excludes = [
    "torch",
    "torchvision",
    "transformers",
    "xatlas",
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
]

block_cipher = None

a = Analysis(  # noqa: F821
    [str(PROJECT_ROOT / "packaging" / "entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821

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
    console=False,
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

app = BUNDLE(  # noqa: F821
    coll,
    name=f"{APP_NAME}.app",
    icon=None,
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
