"""Give colliding bundled dylibs distinct identities (macOS app bundles). Run before signing.

Wheels are built with delocate: every package loads the libraries in its
own ``.dylibs`` folder. PyInstaller rewrites those references to
``@rpath/<name>``, and two things then go wrong when two packages ship
DIFFERENT libraries under the same file name -- pyproj and rasterio both
ship ``libproj.25.9.8.1.dylib``, and rasterio's has its symbols renamed
(``internal_proj_*``):

1. ``@rpath/<name>`` resolves through one symlink at the top of
   ``Contents/Frameworks``, which can only point at one of the copies;
2. even with correct paths, dyld reuses an already-loaded image whose
   install name matches. Whichever package imports first wins, and the
   other fails with "Symbol not found" (measured both ways round: pyproj
   missing ``_proj_context_create``, rasterio's GDAL missing
   ``_internal_geod_init``).

So for every file name shipped by two or more packages with different
contents, each package's copy gets its own install name
(``@rpath/<package>__<name>``) and every reference to it from that
package's binaries is rewritten to the copy's actual path via
``@loader_path`` -- the wheel's original layout, now unambiguous to dyld.
Identical copies are left alone.

Usage: python packaging/fix_dylib_collisions.py dist/DRISHTI-3D.app
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

_MACHO_SUFFIXES = (".so", ".dylib")


def _deps(path: Path) -> list[str]:
    out = subprocess.run(["otool", "-L", str(path)], capture_output=True, text=True, check=False).stdout
    return [line.strip().split(" (")[0] for line in out.splitlines()[1:] if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool(*args: str) -> None:
    subprocess.run(["install_name_tool", *args], check=True, capture_output=True)


def fix(app: Path) -> list[str]:
    frameworks = app / "Contents" / "Frameworks"
    by_name: dict[str, list[tuple[Path, Path]]] = {}
    for pkg in sorted(p for p in frameworks.iterdir() if p.is_dir() and (p / "__dot__dylibs").is_dir()):
        for lib in sorted((pkg / "__dot__dylibs").iterdir()):
            if lib.is_file() and not lib.is_symlink() and lib.suffix == ".dylib":
                by_name.setdefault(lib.name, []).append((pkg, lib))
    colliding = {n: c for n, c in by_name.items() if len(c) > 1 and len({_digest(f) for _, f in c}) > 1}

    changes: list[str] = []
    for name, copies in sorted(colliding.items()):
        for pkg, lib in copies:
            new_id = f"@rpath/{pkg.name}__{name}"
            _tool("-id", new_id, str(lib))
            changes.append(f"{lib.relative_to(frameworks)}: id -> {new_id}")
            for binary in sorted(pkg.rglob("*")):
                if binary.is_symlink() or not binary.is_file() or binary.suffix not in _MACHO_SUFFIXES:
                    continue
                for dep in _deps(binary):
                    if dep.rsplit("/", 1)[-1] != name or not dep.startswith(("@rpath/", "@loader_path/")):
                        continue
                    target = f"@loader_path/{os.path.relpath(lib, binary.parent)}"
                    if dep != target:
                        _tool("-change", dep, target, str(binary))
                        changes.append(f"{binary.relative_to(frameworks)}: {dep} -> {target}")
    return changes


def main() -> int:
    if sys.platform != "darwin":
        return 0
    changes = fix(Path(sys.argv[1]))
    for c in changes:
        print(f"  {c}")
    print(f"  {len(changes)} install-name change(s) for colliding dylibs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
