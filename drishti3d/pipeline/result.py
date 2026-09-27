"""The pipeline's output contract: ``StageResult`` + ``PipelineResult``.

``PipelineResult.save``/``.load`` is the cached-demo safety net documented
in the pipeline brief: a completed run must be loadable from disk and
displayable in the GUI in well under a second, with no GPU, no torch, and
no re-running any stage. That means round-trip correctness (point cloud
arrays, poses, stage results, and the report dict all coming back exactly
as they went in) matters more here than compactness or elegance.

Storage layout (a plain directory, not a single archive, so a human can
poke at ``meta.json`` without unzipping anything):

- ``<dir>/arrays.npz``: every numpy array (point cloud xyz/rgb/covariance/
  confidence, pose rotations/translations), via ``numpy.savez_compressed``.
- ``<dir>/meta.json``: everything else (stage results, the report dict, the
  config, keyframes, and the two input paths), plain JSON.

Submaps are deliberately NOT round-tripped: they are per-window
intermediate geometry that has already been folded into ``point_cloud`` /
``poses`` by ``geometry.submap.merge_submaps``, so keeping them out of the
cached artifact keeps it small without losing anything a loaded-from-disk
result actually needs (browsing the model, re-showing the report, redoing
an export). A freshly-run ``PipelineResult`` still carries them in memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drishti3d.config import (
    Config,
    ExportConfig,
    FusionConfig,
    GeometryConfig,
    IngestConfig,
    TriageConfig,
)
from drishti3d.types import (
    CameraIntrinsics,
    Confidence,
    FrameMetrics,
    GeoPoint,
    Keyframe,
    PointCloud,
    Pose,
    Submap,
    TelemetrySample,
)

# Valid StageResult.status values, in rough lifecycle order.
STAGE_STATUSES = ("pending", "running", "ok", "skipped", "failed")


@dataclass
class StageResult:
    """One pipeline stage's outcome, uniform across every stage the runner executes."""

    name: str
    status: str = "pending"
    elapsed_s: float = 0.0
    message: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "message": self.message,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StageResult:
        return cls(
            name=d["name"],
            status=d.get("status", "pending"),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            message=d.get("message", ""),
            artifacts=d.get("artifacts", {}),
        )


# ---------------------------------------------------------------------------
# JSON-safe (de)serialization helpers for the dataclasses nested inside a
# Keyframe. Written by hand (rather than a generic dataclass walker) since
# these types are a small, fixed, documented contract (drishti3d.types) and
# several fields are optional numpy arrays that ``json`` can't handle as-is.
# ---------------------------------------------------------------------------


def _geopoint_to_dict(geo: GeoPoint | None) -> dict | None:
    if geo is None:
        return None
    return {
        "lat": geo.lat,
        "lon": geo.lon,
        "alt_msl": geo.alt_msl,
        "alt_rel": geo.alt_rel,
        "accuracy_h": geo.accuracy_h,
        "accuracy_v": geo.accuracy_v,
    }


def _geopoint_from_dict(d: dict | None) -> GeoPoint | None:
    return None if d is None else GeoPoint(**d)


def _telemetry_to_dict(sample: TelemetrySample | None) -> dict | None:
    if sample is None:
        return None
    return {
        "timestamp": sample.timestamp,
        "geo": _geopoint_to_dict(sample.geo),
        "gimbal_pitch": sample.gimbal_pitch,
        "gimbal_roll": sample.gimbal_roll,
        "gimbal_yaw": sample.gimbal_yaw,
        "imu_accel": sample.imu_accel.tolist() if sample.imu_accel is not None else None,
        "imu_gyro": sample.imu_gyro.tolist() if sample.imu_gyro is not None else None,
        "baro_alt": sample.baro_alt,
    }


def _telemetry_from_dict(d: dict | None) -> TelemetrySample | None:
    if d is None:
        return None
    return TelemetrySample(
        timestamp=d["timestamp"],
        geo=_geopoint_from_dict(d.get("geo")),
        gimbal_pitch=d.get("gimbal_pitch"),
        gimbal_roll=d.get("gimbal_roll"),
        gimbal_yaw=d.get("gimbal_yaw"),
        imu_accel=np.array(d["imu_accel"]) if d.get("imu_accel") is not None else None,
        imu_gyro=np.array(d["imu_gyro"]) if d.get("imu_gyro") is not None else None,
        baro_alt=d.get("baro_alt"),
    )


def _intrinsics_to_dict(intr: CameraIntrinsics | None) -> dict | None:
    if intr is None:
        return None
    return {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "width": intr.width,
        "height": intr.height,
        "dist_coeffs": intr.dist_coeffs.tolist() if intr.dist_coeffs is not None else None,
    }


def _intrinsics_from_dict(d: dict | None) -> CameraIntrinsics | None:
    if d is None:
        return None
    dist = d.get("dist_coeffs")
    return CameraIntrinsics(
        fx=d["fx"],
        fy=d["fy"],
        cx=d["cx"],
        cy=d["cy"],
        width=d["width"],
        height=d["height"],
        dist_coeffs=np.array(dist) if dist is not None else None,
    )


def _pose_to_dict(pose: Pose | None) -> dict | None:
    if pose is None:
        return None
    return {"R": pose.R.tolist(), "t": pose.t.tolist()}


def _pose_from_dict(d: dict | None) -> Pose | None:
    if d is None:
        return None
    return Pose(R=np.array(d["R"], dtype=np.float64), t=np.array(d["t"], dtype=np.float64))


def _metrics_to_dict(metrics: FrameMetrics) -> dict:
    return {
        "index": metrics.index,
        "timestamp": metrics.timestamp,
        "blur_score": metrics.blur_score,
        "exposure_score": metrics.exposure_score,
        "mean_luma": metrics.mean_luma,
        "estimated_parallax": metrics.estimated_parallax,
    }


def _metrics_from_dict(d: dict) -> FrameMetrics:
    return FrameMetrics(**d)


def _keyframe_to_dict(kf: Keyframe) -> dict:
    return {
        "frame_index": kf.frame_index,
        "timestamp": kf.timestamp,
        "metrics": _metrics_to_dict(kf.metrics),
        "telemetry": _telemetry_to_dict(kf.telemetry),
        "intrinsics": _intrinsics_to_dict(kf.intrinsics),
        "pose": _pose_to_dict(kf.pose),
    }


def _keyframe_from_dict(d: dict) -> Keyframe:
    return Keyframe(
        frame_index=d["frame_index"],
        timestamp=d["timestamp"],
        metrics=_metrics_from_dict(d["metrics"]),
        telemetry=_telemetry_from_dict(d.get("telemetry")),
        intrinsics=_intrinsics_from_dict(d.get("intrinsics")),
        pose=_pose_from_dict(d.get("pose")),
    )


def _config_to_dict(config: Config | None) -> dict | None:
    return None if config is None else asdict(config)


def _config_from_dict(d: dict | None) -> Config | None:
    if d is None:
        return None
    return Config(
        ingest=IngestConfig(**d["ingest"]),
        triage=TriageConfig(**d["triage"]),
        geometry=GeometryConfig(**d["geometry"]),
        fusion=FusionConfig(**d["fusion"]),
        export=ExportConfig(**d["export"]),
    )


@dataclass
class PipelineResult:
    """Everything a completed (or partially-completed) pipeline run produced."""

    keyframes: list[Keyframe] = field(default_factory=list)
    submaps: list[Submap] = field(default_factory=list)
    point_cloud: PointCloud | None = None
    # Triangle connectivity for ``point_cloud`` when it came from
    # FusionStage's TSDF mesh extraction (``PipelineState.mesh_faces``).
    # Round-tripped by save/load: without it a reloaded run can only ever
    # be shown as a point smear, even though the run produced an 8M-face
    # surface and wrote it to OBJ/GLB/FBX on disk.
    mesh_faces: np.ndarray | None = None
    poses: list[Pose] = field(default_factory=list)
    stage_results: list[StageResult] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    config: Config | None = None
    video_path: str | None = None
    telemetry_path: str | None = None

    @property
    def quality(self) -> dict:
        from drishti3d.pipeline.quality import quality_fields

        return quality_fields(self.report or {}, self.stage_results,
                              self.point_cloud is not None and len(self.point_cloud.xyz) > 0)

    @property
    def outcome(self) -> str:
        return self.quality["outcome"]

    @property
    def outcome_label(self) -> str:
        from drishti3d.pipeline.quality import OUTCOME_LABELS

        return OUTCOME_LABELS[self.outcome]

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """A human-readable text block: stage table + point/pose counts + report."""
        n_points = int(self.point_cloud.xyz.shape[0]) if self.point_cloud is not None else 0
        n_faces = int(self.mesh_faces.shape[0]) if self.mesh_faces is not None else 0
        lines = [
            "DRISHTI-3D Pipeline Result",
            "=" * 30,
            "",
            f"Video:      {self.video_path or 'n/a'}",
            f"Telemetry:  {self.telemetry_path or 'n/a'}",
            f"Outcome:    {self.outcome_label}",
            f"Keyframes:  {len(self.keyframes)}",
            f"Submaps:    {len(self.submaps)}",
            f"Points:     {n_points}",
            f"Triangles:  {n_faces}",
            f"Poses:      {len(self.poses)}",
            "",
            "Stages:",
        ]
        for sr in self.stage_results:
            first_line = sr.message.splitlines()[0] if sr.message else ""
            lines.append(f"  {sr.name:<20} {sr.status:<8} {sr.elapsed_s:7.2f}s  {first_line}")

        if self.report:
            lines.append("")
            lines.append("Report:")
            for key, value in self.report.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def stage(self, name: str) -> StageResult | None:
        """The ``StageResult`` for ``name``, or ``None`` if it never ran."""
        return next((sr for sr in self.stage_results if sr.name == name), None)

    def artifacts(self, name: str) -> dict:
        """One stage's artifacts dict (empty when the stage is absent)."""
        stage = self.stage(name)
        return dict(stage.artifacts) if stage and stage.artifacts else {}

    def report_card(self) -> dict:
        """The accuracy report card, assembled from the best available source.

        ``ExportStage`` publishes the real card (the same dict that
        becomes ``report.html``) into ``self.report`` and into its own
        artifacts. But a run saved before that existed -- or one that
        never reached export, or was cancelled -- has neither, while the
        underlying measurements are still sitting in other stages'
        artifacts and in the point cloud.

        So: start from the published card, then fill any gap from the
        stage that actually measured it. Nothing here computes a new
        number or guesses one; each fallback is the same quantity read
        from the stage that produced it, and a value that genuinely was
        never measured stays absent so the UI renders "not computed".
        """
        card: dict[str, Any] = {}

        # 1. The published card, if this run has one.
        export_artifacts = self.artifacts("export")
        if isinstance(export_artifacts.get("report_card"), dict):
            card.update(export_artifacts["report_card"])
        card.update({k: v for k, v in (self.report or {}).items() if v is not None})

        def missing(key: str) -> bool:
            value = card.get(key)
            return value is None or value == "not computed"

        # 2. Reprojection error: measured by the pose prior's bundle
        #    adjustment, whose artifacts survive in every saved run.
        if missing("mean_reprojection_error_px"):
            rmse = self.artifacts("pose_prior").get("rmse_after_px")
            if rmse is not None:
                card["mean_reprojection_error_px"] = float(rmse)

        # 3. Coverage: triage's timeline coverage, as a percentage.
        if missing("coverage_pct"):
            fraction = (self.report or {}).get("triage_timeline_coverage_fraction")
            if fraction is None:
                fraction = self.artifacts("triage").get("timeline_coverage_fraction")
            if fraction is not None:
                card["coverage_pct"] = float(fraction) * 100.0

        if missing("keyframe_count") and self.keyframes:
            card["keyframe_count"] = len(self.keyframes)

        # 4. Confidence breakdown, straight off the round-tripped array.
        if missing("confidence_breakdown_pct") and self.point_cloud is not None:
            confidence = self.point_cloud.confidence
            if confidence is not None and confidence.size:
                total = float(confidence.size)
                card["confidence_breakdown_pct"] = {
                    "measured_pct": 100.0 * float(np.count_nonzero(confidence == Confidence.MEASURED)) / total,
                    "low_confidence_pct": 100.0
                    * float(np.count_nonzero(confidence == Confidence.LOW_CONFIDENCE))
                    / total,
                    "inferred_pct": 100.0 * float(np.count_nonzero(confidence == Confidence.INFERRED)) / total,
                }

        # 5. Provenance carried by the stages that determined it.
        fusion = self.artifacts("fusion")
        if missing("confidence_source") and fusion.get("confidence_source"):
            card["confidence_source"] = fusion["confidence_source"]

        geometry = self.artifacts("geometry")
        # The configured neural backbone may not have run: auto uses stereo.
        # Prefer the stage's actual method, including when opening older runs.
        method = geometry.get("backbone")
        if method in ("mvs3d", "heightfield_mvs"):
            card["backbone"] = method
            if self.mesh_faces is not None and len(self.mesh_faces):
                card["reconstruction_representation"] = (
                    "3D volumetric mesh" if method == "mvs3d" else "2.5D height-field surface"
                )
        for key in ("merge_strategy", "merge_strategy_reason", "placement", "depth_anchor", "flight_profile"):
            if missing(key) and geometry.get(key) is not None:
                card[key] = geometry[key]

        card.update(self.quality)
        return card

    # ------------------------------------------------------------------
    def save(self, dir: str | Path) -> None:
        """Save this result to ``dir`` (created if needed) for near-instant reload.

        Round-trips ``point_cloud``, ``mesh_faces``, ``poses``, ``keyframes``,
        ``stage_results``, ``report``, and ``config`` exactly (see
        ``load``). Deliberately does not round-trip ``submaps`` -- see the
        module docstring.
        """
        out = Path(dir)
        out.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        has_point_cloud = self.point_cloud is not None
        if self.point_cloud is not None:
            arrays["pc_xyz"] = np.asarray(self.point_cloud.xyz)
            if self.point_cloud.rgb is not None:
                arrays["pc_rgb"] = np.asarray(self.point_cloud.rgb)
            if self.point_cloud.covariance is not None:
                arrays["pc_covariance"] = np.asarray(self.point_cloud.covariance)
            if self.point_cloud.confidence is not None:
                arrays["pc_confidence"] = np.asarray(self.point_cloud.confidence)
            if self.point_cloud.semantic_class is not None:
                arrays["pc_semantic_class"] = np.asarray(self.point_cloud.semantic_class)
            if self.point_cloud.semantic_confidence is not None:
                arrays["pc_semantic_confidence"] = np.asarray(self.point_cloud.semantic_confidence)
            if self.point_cloud.uncertainty_m is not None:
                arrays["pc_uncertainty_m"] = np.asarray(self.point_cloud.uncertainty_m)

        if self.mesh_faces is not None and len(self.mesh_faces):
            # int32 halves the file for any mesh under 2^31 vertices,
            # which every mesh this pipeline can produce is.
            arrays["mesh_faces"] = np.asarray(self.mesh_faces, dtype=np.int32)

        n_poses = len(self.poses)
        arrays["poses_R"] = (
            np.stack([p.R for p in self.poses], axis=0) if n_poses else np.zeros((0, 3, 3), dtype=np.float64)
        )
        arrays["poses_t"] = (
            np.stack([p.t for p in self.poses], axis=0) if n_poses else np.zeros((0, 3), dtype=np.float64)
        )

        np.savez_compressed(out / "arrays.npz", **arrays)

        meta = {
            "has_point_cloud": has_point_cloud,
            "n_poses": n_poses,
            "keyframes": [_keyframe_to_dict(kf) for kf in self.keyframes],
            "stage_results": [sr.to_dict() for sr in self.stage_results],
            "outcome": self.outcome,
            "report": {**self.report, **self.quality},
            "config": _config_to_dict(self.config),
            "video_path": self.video_path,
            "telemetry_path": self.telemetry_path,
        }
        with (out / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dir: str | Path) -> PipelineResult:
        """Load a result previously written by ``save`` -- the cached-demo path.

        Must work with no GPU, no torch, and no model weights: this only
        ever touches ``numpy``/``json``, matching ``save``.
        """
        d = Path(dir)
        with (d / "meta.json").open() as f:
            meta = json.load(f)

        with np.load(d / "arrays.npz", allow_pickle=False) as arrays:
            point_cloud: PointCloud | None = None
            if meta.get("has_point_cloud"):
                point_cloud = PointCloud(
                    xyz=arrays["pc_xyz"],
                    rgb=arrays.get("pc_rgb", None),
                    covariance=arrays.get("pc_covariance", None),
                    confidence=arrays.get("pc_confidence", None),
                    semantic_class=arrays.get("pc_semantic_class", None),
                    semantic_confidence=arrays.get("pc_semantic_confidence", None),
                    uncertainty_m=arrays.get("pc_uncertainty_m", None),
                )

            mesh_faces = arrays["mesh_faces"] if "mesh_faces" in arrays.files else None

            n_poses = int(meta.get("n_poses", 0))
            poses_R = arrays["poses_R"]
            poses_t = arrays["poses_t"]
            poses = [Pose(R=poses_R[i], t=poses_t[i]) for i in range(n_poses)]

        keyframes = [_keyframe_from_dict(kd) for kd in meta.get("keyframes", [])]
        stage_results = [StageResult.from_dict(sd) for sd in meta.get("stage_results", [])]
        config = _config_from_dict(meta.get("config"))
        report = dict(meta.get("report") or {})
        if meta.get("outcome") in {"failed", "cancelled"}:
            report["outcome"] = meta["outcome"]

        return cls(
            keyframes=keyframes,
            submaps=[],
            point_cloud=point_cloud,
            mesh_faces=mesh_faces,
            poses=poses,
            stage_results=stage_results,
            report=report,
            config=config,
            video_path=meta.get("video_path"),
            telemetry_path=meta.get("telemetry_path"),
        )
