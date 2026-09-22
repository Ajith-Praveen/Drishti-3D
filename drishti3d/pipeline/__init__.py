"""The pipeline package: wires ingest -> triage -> geometry -> bundle -> fusion -> export.

``run_pipeline`` (see ``pipeline.runner``) is the single entry point used
by both the headless ``drishti3d-run`` CLI and the Qt app's
``app.workers.RealPipeline``.
"""

from drishti3d.pipeline.result import PipelineResult, StageResult
from drishti3d.pipeline.runner import run_pipeline
from drishti3d.pipeline.stages import CancelToken, PipelineCancelled, StageUnavailable

__all__ = [
    "CancelToken",
    "PipelineCancelled",
    "PipelineResult",
    "StageResult",
    "StageUnavailable",
    "run_pipeline",
]
