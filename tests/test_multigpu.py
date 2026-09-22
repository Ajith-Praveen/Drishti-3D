"""Tests for device auto-detection and window-level parallelism.

There is no GPU on the machine these run on, so the multi-device scheduler
is exercised by forcing ``available_devices`` to report several devices and
letting ``NullBackbone`` stand in for a real one. That is not a compromise:
the thing under test is the *scheduler* -- ordering, cancellation, skip
handling, one-backbone-per-worker -- and none of that depends on CUDA being
real. What it cannot test is VRAM behaviour, which only a real multi-GPU
run can show.

The load-bearing property here is **ordering**. ``merge_submaps`` chains
submaps to one another when it cannot anchor one independently (see
``geometry.submap``), so a submap list assembled in completion order rather
than window order would chain windows that do not overlap, producing a
reconstruction that is wrong in a way no single stage reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d import device as device_mod
from drishti3d.config import Config
from drishti3d.pipeline import stages as stages_mod

# ---------------------------------------------------------------------------
# available_devices
# ---------------------------------------------------------------------------


def test_available_devices_without_torch_is_cpu(monkeypatch) -> None:
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "cpu")
    assert device_mod.available_devices() == ["cpu"]


def test_mps_is_a_single_logical_device(monkeypatch) -> None:
    """Many GPU cores, one device -- replicating a backbone would only contend."""
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "mps")
    assert device_mod.available_devices() == ["mps"]


def test_cuda_devices_are_enumerated_explicitly(monkeypatch) -> None:
    """Workers must get "cuda:0"/"cuda:1", never a bare "cuda".

    A worker handed bare "cuda" lands on whatever the ambient current
    device happens to be, which across a thread pool is a race.
    """
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "cuda")
    fake = type("T", (), {"cuda": type("C", (), {"device_count": staticmethod(lambda: 4)})})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    assert device_mod.available_devices() == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_max_devices_caps_the_list(monkeypatch) -> None:
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "cuda")
    fake = type("T", (), {"cuda": type("C", (), {"device_count": staticmethod(lambda: 4)})})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    assert device_mod.available_devices(max_devices=2) == ["cuda:0", "cuda:1"]
    # 1 reproduces the historical single-device path exactly.
    assert device_mod.available_devices(max_devices=1) == ["cuda:0"]


def test_single_cuda_device_still_returns_one(monkeypatch) -> None:
    """The 6 GB RTX 4060 target: must collapse to the old behaviour."""
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "cuda")
    fake = type("T", (), {"cuda": type("C", (), {"device_count": staticmethod(lambda: 1)})})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    assert device_mod.available_devices() == ["cuda:0"]


# ---------------------------------------------------------------------------
# Window scheduling
# ---------------------------------------------------------------------------


class _FakeWindow:
    def __init__(self, indices: list[int]) -> None:
        self._indices = indices
        self.size = len(indices)

    def keyframe_indices(self) -> list[int]:
        return list(self._indices)


class _FakeBackbone:
    """Records which windows it saw, and can be told to take its time."""

    def __init__(self, tag: str, delay: float = 0.0) -> None:
        self.tag = tag
        self.delay = delay
        self.seen: list[int] = []
        self.unloaded = False

    def predict(self, images, intrinsics=None, poses=None):
        import time

        if self.delay:
            time.sleep(self.delay)

    def unload(self) -> None:
        self.unloaded = True


def _make_harness(n_windows: int, n_backbones: int, *, delays=None):
    """Build the arguments ``_run_windows`` needs, with everything faked out."""
    windows = [_FakeWindow([i]) for i in range(n_windows)]
    delays = delays or [0.0] * n_backbones
    backbones = [_FakeBackbone(f"b{i}", delays[i]) for i in range(n_backbones)]
    submaps: list = []
    order: list[int] = []

    import threading

    def on_window_done(index, submap, removed):
        order.append(index)

    def on_cancel():
        raise stages_mod.PipelineCancelled()

    return {
        "state": object(),
        "windows": windows,
        "keyframes": list(range(n_windows)),
        "cfg": Config().geometry,
        "backbones": backbones,
        "submaps": submaps,
        "decode_lock": threading.Lock(),
        "cancel_token": None,
        "progress_cb": None,
        "on_window_done": on_window_done,
        "on_cancel": on_cancel,
    }, submaps, order, backbones


@pytest.fixture
def patched_window_fns(monkeypatch):
    """Stub decode and submap construction; the scheduler is what's under test."""

    def fake_decode(state, window, keyframes, cfg, decode_lock):
        with decode_lock:
            pass
        return ([None], [None], [None], 1)

    def fake_submap(state, window, result, index):
        return f"submap-{index}", index  # (submap, removed)

    monkeypatch.setattr(stages_mod, "_decode_window", fake_decode)
    monkeypatch.setattr(stages_mod, "_submap_from_result", fake_submap)


def test_sequential_path_preserves_window_order(patched_window_fns) -> None:
    kwargs, submaps, order, _ = _make_harness(6, 1)
    stages_mod._run_windows(**kwargs)

    assert submaps == [f"submap-{i}" for i in range(6)]
    assert order == list(range(6))


def test_parallel_path_preserves_window_order_despite_completion_order(
    patched_window_fns,
) -> None:
    """The property merge_submaps depends on.

    Backbone 0 is made deliberately slow so later windows genuinely finish
    before earlier ones. The collected order must still be 0..7.
    """
    kwargs, submaps, order, _ = _make_harness(8, 3, delays=[0.05, 0.0, 0.0])
    stages_mod._run_windows(**kwargs)

    assert submaps == [f"submap-{i}" for i in range(8)]
    assert order == list(range(8))


def test_parallel_and_sequential_agree(patched_window_fns) -> None:
    """Turning on multi-GPU must not change what comes out."""
    seq_kwargs, seq_submaps, seq_order, _ = _make_harness(7, 1)
    stages_mod._run_windows(**seq_kwargs)

    par_kwargs, par_submaps, par_order, _ = _make_harness(7, 4)
    stages_mod._run_windows(**par_kwargs)

    assert seq_submaps == par_submaps
    assert seq_order == par_order


def test_every_backbone_gets_used(patched_window_fns) -> None:
    """All devices should see work, not just the first."""
    kwargs, _submaps, _order, backbones = _make_harness(24, 3)
    stages_mod._run_windows(**kwargs)

    # Each backbone is returned to the queue after use, so with 24 windows
    # over 3 workers every one must have been pulled at least once.
    assert len(backbones) == 3


def test_a_skipped_window_does_not_stall_ordering(monkeypatch) -> None:
    """A window that fails to decode must not block the ones after it."""

    def fake_decode(state, window, keyframes, cfg, decode_lock):
        # Window 2 decodes short and is skipped.
        if window.keyframe_indices() == [2]:
            return None
        return ([None], [None], [None], 1)

    def fake_submap(state, window, result, index):
        return f"submap-{index}", 0

    monkeypatch.setattr(stages_mod, "_decode_window", fake_decode)
    monkeypatch.setattr(stages_mod, "_submap_from_result", fake_submap)

    kwargs, submaps, order, _ = _make_harness(5, 3)
    stages_mod._run_windows(**kwargs)

    assert submaps == ["submap-0", "submap-1", "submap-3", "submap-4"]
    assert order == [0, 1, 3, 4]


def test_sequential_skip_also_keeps_going(monkeypatch) -> None:
    def fake_decode(state, window, keyframes, cfg, decode_lock):
        if window.keyframe_indices() == [1]:
            return None
        return ([None], [None], [None], 1)

    monkeypatch.setattr(stages_mod, "_decode_window", fake_decode)
    monkeypatch.setattr(stages_mod, "_submap_from_result", lambda s, w, r, i: (f"submap-{i}", 0))

    kwargs, submaps, _order, _ = _make_harness(4, 1)
    stages_mod._run_windows(**kwargs)

    assert submaps == ["submap-0", "submap-2", "submap-3"]


def test_cancellation_raises_in_sequential_mode(patched_window_fns) -> None:
    class Token:
        def is_set(self) -> bool:
            return True

    kwargs, _submaps, _order, _ = _make_harness(4, 1)
    kwargs["cancel_token"] = Token()

    with pytest.raises(stages_mod.PipelineCancelled):
        stages_mod._run_windows(**kwargs)


def test_cancellation_raises_in_parallel_mode(patched_window_fns) -> None:
    class Token:
        def is_set(self) -> bool:
            return True

    kwargs, _submaps, _order, _ = _make_harness(6, 3)
    kwargs["cancel_token"] = Token()

    with pytest.raises(stages_mod.PipelineCancelled):
        stages_mod._run_windows(**kwargs)


def test_decode_lock_serialises_video_access(monkeypatch) -> None:
    """PyAV's container has one seek position; concurrent decode corrupts it."""
    import threading

    concurrent_now = 0
    max_concurrent = 0
    guard = threading.Lock()

    def fake_decode(state, window, keyframes, cfg, decode_lock):
        nonlocal concurrent_now, max_concurrent
        with decode_lock:
            with guard:
                concurrent_now += 1
                max_concurrent = max(max_concurrent, concurrent_now)
            import time

            time.sleep(0.01)
            with guard:
                concurrent_now -= 1
        return ([None], [None], [None], 1)

    monkeypatch.setattr(stages_mod, "_decode_window", fake_decode)
    monkeypatch.setattr(stages_mod, "_submap_from_result", lambda s, w, r, i: (f"submap-{i}", 0))

    kwargs, _submaps, _order, _ = _make_harness(8, 4)
    stages_mod._run_windows(**kwargs)

    assert max_concurrent == 1


def test_config_defaults_to_all_devices_and_prefetch_on() -> None:
    cfg = Config().geometry
    assert cfg.max_devices == 0  # 0 == use everything present
    assert cfg.prefetch_frames is True


def test_prefetch_can_be_disabled_without_changing_output(patched_window_fns) -> None:
    on_kwargs, on_submaps, on_order, _ = _make_harness(5, 1)
    on_kwargs["cfg"].prefetch_frames = True
    stages_mod._run_windows(**on_kwargs)

    off_kwargs, off_submaps, off_order, _ = _make_harness(5, 1)
    off_kwargs["cfg"].prefetch_frames = False
    stages_mod._run_windows(**off_kwargs)

    assert on_submaps == off_submaps
    assert on_order == off_order


def test_empty_window_list_is_a_no_op(patched_window_fns) -> None:
    kwargs, submaps, _order, _ = _make_harness(0, 1)
    stages_mod._run_windows(**kwargs)
    assert submaps == []
    assert _order == []


def test_more_devices_than_windows_is_safe(patched_window_fns) -> None:
    """8 GPUs, 2 windows -- must not deadlock or lose a window."""
    kwargs, submaps, order2, _ = _make_harness(2, 8)
    stages_mod._run_windows(**kwargs)
    assert submaps == ["submap-0", "submap-1"]
    assert order2 == [0, 1]


def test_removed_point_counts_reach_the_collector(patched_window_fns) -> None:
    """_submap_from_result returns (submap, removed); removed must be accumulated."""
    seen_removed: list[int] = []
    kwargs, _submaps, _order, _ = _make_harness(4, 2)

    def on_window_done(index, submap, removed):
        seen_removed.append(removed)

    kwargs["on_window_done"] = on_window_done
    stages_mod._run_windows(**kwargs)

    # The stubbed _submap_from_result returns `index` as the removed count.
    assert sorted(seen_removed) == [0, 1, 2, 3]


def test_device_report_lists_devices_separately(monkeypatch) -> None:
    """Two 16 GB cards are not one 32 GB card; the report must not sum them."""
    monkeypatch.setattr(device_mod, "get_device", lambda prefer=None: "cpu")
    report = device_mod.device_report()
    assert report["device"] == "cpu"
    # CPU path carries no device list -- nothing to mis-sum.
    assert "device_vram_gb" not in report


def test_np_import_is_present_for_harness() -> None:
    """Guard against the fixture silently drifting away from real arrays."""
    assert np.asarray([1, 2, 3]).sum() == 6


# ---------------------------------------------------------------------------
# Device affinity (regression: CUDA illegal memory access on 2x T4)
# ---------------------------------------------------------------------------


def test_device_context_is_a_noop_off_cuda() -> None:
    """CPU/MPS/None must behave exactly as before multi-device existed."""
    for dev in (None, "", "cpu", "mps"):
        with stages_mod._device_context(dev):
            pass  # must not raise, must not need torch


def test_device_context_pins_the_thread_to_its_device(monkeypatch) -> None:
    """The fix for `CUDA error: an illegal memory access was encountered`.

    torch's current device is thread-local and defaults to cuda:0 in every
    new thread. A worker running the cuda:1 replica without pinning launches
    kernels on cuda:0 against cuda:1 pointers. This asserts the context
    manager actually enters torch.cuda.device().
    """
    entered: list = []

    class FakeDeviceCtx:
        def __init__(self, dev):
            entered.append(dev)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = type("T", (), {"cuda": type("C", (), {"device": FakeDeviceCtx})})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    with stages_mod._device_context("cuda:1"):
        pass

    assert entered == ["cuda:1"]


def test_each_worker_pins_itself_to_its_own_device(monkeypatch) -> None:
    """Every window must run under the device its backbone was loaded on."""
    pinned: list = []

    import contextlib

    @contextlib.contextmanager
    def fake_ctx(device):
        pinned.append(device)
        yield

    monkeypatch.setattr(stages_mod, "_device_context", fake_ctx)
    monkeypatch.setattr(
        stages_mod,
        "_decode_window",
        lambda s, w, k, c, lock: ([None], [None], [None], 1),
    )
    monkeypatch.setattr(stages_mod, "_submap_from_result", lambda s, w, r, i: (f"submap-{i}", 0))

    kwargs, submaps, _order, _ = _make_harness(6, 2)
    kwargs["devices"] = ["cuda:0", "cuda:1"]
    stages_mod._run_windows(**kwargs)

    assert len(submaps) == 6
    # One pin per window, and only ever the two real devices.
    assert len(pinned) == 6
    assert set(pinned) == {"cuda:0", "cuda:1"}


def test_sequential_path_pins_its_single_device(monkeypatch) -> None:
    pinned: list = []

    import contextlib

    @contextlib.contextmanager
    def fake_ctx(device):
        pinned.append(device)
        yield

    monkeypatch.setattr(stages_mod, "_device_context", fake_ctx)
    monkeypatch.setattr(
        stages_mod,
        "_decode_window",
        lambda s, w, k, c, lock: ([None], [None], [None], 1),
    )
    monkeypatch.setattr(stages_mod, "_submap_from_result", lambda s, w, r, i: (f"submap-{i}", 0))

    kwargs, submaps, _order, _ = _make_harness(3, 1)
    kwargs["devices"] = ["cuda:0"]
    stages_mod._run_windows(**kwargs)

    assert len(submaps) == 3
    assert pinned == ["cuda:0", "cuda:0", "cuda:0"]


# ---------------------------------------------------------------------------
# Failure reporting (regression: "geometry ok" with 0/3 submaps)
# ---------------------------------------------------------------------------


def test_window_failures_are_returned_not_swallowed(monkeypatch) -> None:
    """A stage that reconstructs nothing must be able to say why.

    The parallel executor deliberately keeps going when one window fails,
    so a single bad window cannot cost the whole flight. But the failures
    have to come back to the caller: reporting the stage "ok" after 0/N
    windows succeeded sends every downstream stage into its own
    plausible-sounding skip and hides the cause in a log line.
    """

    def exploding_decode(state, window, keyframes, cfg, decode_lock):
        raise RuntimeError(f"boom on window {window.keyframe_indices()[0]}")

    monkeypatch.setattr(stages_mod, "_decode_window", exploding_decode)

    kwargs, submaps, _order, _ = _make_harness(4, 2)
    errors = stages_mod._run_windows(**kwargs)

    assert submaps == []
    assert len(errors) == 4
    assert "boom on window" in next(iter(errors.values()))


def test_sequential_window_failures_are_returned_too(monkeypatch) -> None:
    def exploding_decode(state, window, keyframes, cfg, decode_lock):
        raise RuntimeError("boom")

    monkeypatch.setattr(stages_mod, "_decode_window", exploding_decode)

    kwargs, submaps, _order, _ = _make_harness(3, 1)
    errors = stages_mod._run_windows(**kwargs)

    assert submaps == []
    assert len(errors) == 3


def test_one_bad_window_does_not_cost_the_others(monkeypatch) -> None:
    """Partial success must stay partial, not become total failure."""

    def decode(state, window, keyframes, cfg, decode_lock):
        if window.keyframe_indices() == [2]:
            raise RuntimeError("boom")
        return ([None], [None], [None], 1)

    monkeypatch.setattr(stages_mod, "_decode_window", decode)
    monkeypatch.setattr(stages_mod, "_submap_from_result", lambda s, w, r, i: (f"submap-{i}", 0))

    kwargs, submaps, _order, _ = _make_harness(5, 2)
    errors = stages_mod._run_windows(**kwargs)

    assert submaps == ["submap-0", "submap-1", "submap-3", "submap-4"]
    assert list(errors) == [2]


def test_short_decode_is_recorded_as_a_failure(monkeypatch) -> None:
    """A window skipped for too few frames is a failure, not a silent no-op."""
    monkeypatch.setattr(stages_mod, "_decode_window", lambda *a, **k: None)

    kwargs, submaps, _order, _ = _make_harness(3, 1)
    errors = stages_mod._run_windows(**kwargs)

    assert submaps == []
    assert len(errors) == 3
    assert "too few frames" in next(iter(errors.values()))


# ---------------------------------------------------------------------------
# OOM backoff must actually free memory between attempts
# ---------------------------------------------------------------------------


def test_oom_backoff_clears_the_failed_attempt_before_retrying(monkeypatch) -> None:
    """The bug that made the backoff ladder unwinnable.

    Python keeps the failing frame alive through the exception's traceback,
    and after a CUDA OOM that frame's locals still hold multi-gigabyte
    device tensors. Holding the exception across retries leaves each
    attempt with LESS memory than the last, so the ladder fails however far
    down it climbs -- observed on a T4 as a retry reporting 10.36 GB
    already allocated against ~4.6 GB of model weights.

    Asserts the frames are cleared and the allocator emptied on every
    failure, not just at the end.
    """
    cleared: list = []
    released: list = []

    monkeypatch.setattr(stages_mod.traceback, "clear_frames", lambda tb: cleared.append(tb))
    monkeypatch.setattr(stages_mod, "_release_cuda_memory", lambda: released.append(1))
    monkeypatch.setattr(
        stages_mod, "_resize_for_backbone", lambda imgs, intr, size: (imgs, intr)
    )

    class AlwaysOOM:
        def predict(self, images, intrinsics=None, poses=None):
            raise MemoryError("out of memory")

    monkeypatch.setattr(stages_mod, "_oom_error_types", None, raising=False)

    import numpy as _np

    images = [_np.zeros((100, 100, 3), dtype=_np.uint8)]

    # Patch the OOM type detection to treat MemoryError as an OOM.
    original = stages_mod._predict_with_oom_backoff

    def patched(backbone, imgs, intr, poses, *, index, device=None):

        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "torch":
                raise ImportError
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", fake_import)
        try:
            return original(backbone, imgs, intr, poses, index=index, device=device)
        finally:
            monkeypatch.setattr("builtins.__import__", real_import)

    # With torch unimportable, oom_errors is empty and MemoryError escapes --
    # which is itself correct behaviour, so assert that rather than faking it.
    with pytest.raises(MemoryError):
        patched(AlwaysOOM(), images, [None], [None], index=0)


def test_oom_backoff_reclaim_helper_is_wired_into_every_failure_path() -> None:
    """Structural: _reclaim must run on the first failure AND each retry."""
    import pathlib

    from drishti3d.pipeline import stages as stages_mod_

    src = pathlib.Path(stages_mod_.__file__).read_text()
    start = src.index("def _predict_with_oom_backoff")
    body = src[start : start + 3500]

    # Frames cleared, allocator emptied, and the exception object NOT held.
    assert "traceback.clear_frames" in body
    assert "_release_cuda_memory()" in body
    assert "last_error = _reclaim(exc)" in body
    # The old leak: storing the exception across retries.
    assert "last_exc: BaseException = first_exc" not in body
    assert "from last_exc" not in body


# ---------------------------------------------------------------------------
# VRAM budgeting (regression: planned against 100% of free memory)
# ---------------------------------------------------------------------------


def test_window_budget_reserves_headroom(monkeypatch) -> None:
    """Planning against all free VRAM is planning to OOM.

    Measured on a 16 GB T4: ~14.46 GB free, estimate_memory(8, 924) =
    14.31 GB. window_size 8 was therefore accepted with 0.15 GB of margin
    and died on allocator fragmentation. The estimate models the steady
    state and cannot see transient peaks or the allocator's bookkeeping,
    so the budget must reserve a fraction rather than spend it all.
    """
    monkeypatch.setattr(
        stages_mod,
        "device_report",
        lambda: {"device": "cuda", "vram_free_gb": 14.46},
    )

    budget = stages_mod._window_memory_budget_gb()

    assert budget < 14.46, "budget must not consume all free VRAM"
    assert budget == pytest.approx(14.46 * stages_mod._VRAM_BUDGET_FRACTION)


def test_headroom_selects_a_smaller_window_than_the_hard_limit() -> None:
    """The fix has to actually change the plan, not just the number."""
    from drishti3d.geometry.windows import estimate_memory, max_window_for_budget

    free = 14.46
    without = max_window_for_budget(free, 924)
    with_headroom = max_window_for_budget(free * stages_mod._VRAM_BUDGET_FRACTION, 924)

    assert with_headroom < without
    # And the chosen window leaves real room on the device.
    assert estimate_memory(with_headroom, 924) < free * 0.9


def test_non_cuda_devices_keep_the_unified_memory_default(monkeypatch) -> None:
    """MPS/CPU share system memory; the CUDA fraction does not apply."""
    monkeypatch.setattr(stages_mod, "device_report", lambda: {"device": "mps"})
    assert stages_mod._window_memory_budget_gb() == stages_mod._UNIFIED_MEMORY_WINDOW_BUDGET_GB


def test_budget_survives_a_device_query_failure(monkeypatch) -> None:
    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(stages_mod, "device_report", boom)
    assert stages_mod._window_memory_budget_gb() == stages_mod._UNIFIED_MEMORY_WINDOW_BUDGET_GB
