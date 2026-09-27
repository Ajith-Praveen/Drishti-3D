"""One conservative reconstruction outcome for pipeline, files and UI."""
from __future__ import annotations

OUTCOMES = ("valid", "unverified", "failed", "cancelled")
OUTCOME_LABELS = {
    "valid": "Valid",
    "unverified": "Unverified — diagnostic output",
    "failed": "Failed — diagnostic output",
    "cancelled": "Cancelled — partial diagnostic output",
}


def assess_quality(report, stages, has_geometry):
    """Return explicit outcome and reasons; missing evidence never implies PASS.

    A saved outcome cannot promote missing or contradictory evidence to valid.
    Failure and cancellation markers survive legacy reloads and manual exports.
    Valid means the implemented checks passed, not survey-grade certification.
    """
    if report.get("cancelled") or report.get("outcome") == "cancelled":
        return "cancelled", ["Run cancelled; any retained geometry is partial."]
    reasons = []
    if report.get("outcome") == "failed":
        reasons.extend(report.get("quality_reasons") or ["Reconstruction previously failed quality checks."])
    evidence = [report]
    statuses = {}
    for stage in stages:
        statuses[stage.name] = stage.status
        evidence.append(stage.artifacts or {})
        if stage.status == "failed":
            reasons.append(f"Stage {stage.name} failed.")
    placements = [e.get("placement") for e in evidence if isinstance(e.get("placement"), dict)]
    if any(p.get("verdict") == "FAIL" or p.get("passed") is False and p.get("verdict") != "UNMEASURED" for p in placements):
        reasons.append("Frame placement failed.")
    if any(e.get("placement_verdict") == "FAIL" for e in evidence):
        reasons.append("Frame placement failed.")
    if reasons:
        return "failed", list(dict.fromkeys(reasons))
    if not has_geometry:
        reasons.append("No reconstructed geometry available.")
    if report.get("backbone") == "null":
        reasons.append("Synthetic demonstration geometry.")
    if not placements or any(p.get("verdict") != "PASS" for p in placements):
        reasons.append("Placement check has no measured PASS.")
    if not any(isinstance(e.get("camera_validation"), dict) and e["camera_validation"] for e in evidence):
        reasons.append("Camera validation evidence missing.")
    for name in ("geometry", "fusion"):
        if statuses.get(name) != "ok":
            reasons.append(f"Stage {name} did not complete successfully.")
    return ("unverified", reasons) if reasons else ("valid", [])


def quality_fields(report, stages, has_geometry):
    outcome, reasons = assess_quality(report, stages, has_geometry)
    return {"outcome": outcome, "quality_reasons": reasons, "diagnostic_output": outcome != "valid"}
