"""Offline reconstruction-runtime preflight (drishti3d.runtime)."""

from __future__ import annotations


def test_runtime_reports_missing_torch_hub_code(tmp_path, monkeypatch):
    """An air-gapped machine without MapAnything's DINOv2 hub code must fail the preflight, not mid-run."""
    from drishti3d import runtime

    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    checks = {c.name: c for c in runtime.torch_hub_checks()}
    assert not checks["torch-hub:dinov2"].ok and not checks["torch-hub:dinov2"].warning
    assert checks["torch-hub:disk-lightglue"].warning

    (tmp_path / "hub" / "facebookresearch_dinov2_main").mkdir(parents=True)
    assert {c.name: c for c in runtime.torch_hub_checks()}["torch-hub:dinov2"].ok
