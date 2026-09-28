"""Regression checks for build validation; no GPU or downloaded models needed."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load("smoke_test")


class BuildSmokeTests(unittest.TestCase):
    def test_nonzero_exit_stops_before_next_check(self):
        with patch.object(smoke.subprocess, "run", side_effect=subprocess.CalledProcessError(7, "app")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                smoke.main(["app"])
            self.assertEqual(run.call_count, 1)

    def test_timeout_is_a_failure(self):
        with patch.object(smoke.subprocess, "run", side_effect=subprocess.TimeoutExpired("app", 180)):
            with self.assertRaises(subprocess.TimeoutExpired):
                smoke.main(["app"])

    def test_waits_until_child_releases_files(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "finished"
            child = Path(directory) / "child.py"
            child.write_text("import time\nfrom pathlib import Path\ntime.sleep(0.1)\n"
                             + f"Path({str(marker)!r}).write_text('done')\n")
            with patch.object(smoke, "CHECKS", (("--help",),)):
                self.assertEqual(smoke.main([sys.executable, str(child)]), 0)
            self.assertEqual(marker.read_text(), "done")

    def test_native_operator_failure_is_not_swallowed(self):
        entry = load("entry")
        from types import SimpleNamespace

        def broken_nms(*args):
            raise RuntimeError("missing torchvision native operators")

        fake = SimpleNamespace(ops=SimpleNamespace(nms=broken_nms))
        with patch("importlib.import_module", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "native operators"):
                entry._selftest_imports(["torchvision"])


if __name__ == "__main__":
    unittest.main()
