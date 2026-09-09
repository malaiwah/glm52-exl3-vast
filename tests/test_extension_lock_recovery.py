#!/usr/bin/env python3
"""Regression tests for persistent PyTorch extension lock recovery."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "recover_torch_extension_lock.sh"


class ExtensionLockRecoveryTests(unittest.TestCase):
    def run_helper(self, root, **env):
        proc_env = os.environ.copy()
        proc_env.update(env)
        return subprocess.run(
            ["bash", str(HELPER), str(root)],
            text=True,
            capture_output=True,
            env=proc_env,
            timeout=15,
            check=False,
        )

    def test_old_ambiguous_filebaton_and_ninja_lock_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            ninja_lock = extension / ".ninja_lock"
            sentinel.touch()
            # Even a very old lock can belong to a foreign live namespace.
            old = time.time() - 7200
            os.utime(sentinel, (old, old))
            ninja_lock.write_text("keep")

            result = self.run_helper(temp)

            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertTrue(sentinel.exists())
            self.assertEqual(ninja_lock.read_text(), "keep")
            self.assertEqual(list(extension.glob("lock.stale-*")), [])

    def test_fresh_ownerless_sentinel_is_not_quarantined(self):
        # A sentinel with no owner visible in this PID namespace is NOT proof of
        # a stale lock: a compiler in another container sharing this persistent
        # /cache is invisible here and its live FileBaton looks ownerless. The
        # default minimum sentinel-age gate must leave a fresh one in place
        # rather than quarantine a live cross-container lock.
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            sentinel.touch()  # brand new: mtime is now

            result = self.run_helper(temp)

            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertTrue(sentinel.exists())
            self.assertEqual(list(extension.glob("lock.stale-*")), [])

    def test_disabling_legacy_age_gate_does_not_authorize_quarantine(self):
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            sentinel.touch()

            result = self.run_helper(temp, EXT_LOCK_MIN_AGE_S="0")

            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertTrue(sentinel.exists())
            self.assertEqual(list(extension.glob("lock.stale-*")), [])

    def test_disabled_recovery_leaves_sentinel_in_place(self):
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            sentinel.touch()

            result = self.run_helper(
                temp, RECOVER_STALE_EXTENSION_LOCKS="0"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(sentinel.exists())
            self.assertEqual(list(extension.glob("lock.stale-*")), [])

    @unittest.skipUnless(
        sys.platform.startswith("linux") and Path("/proc").is_dir(),
        "live compiler ownership is detected through Linux /proc",
    )
    def test_live_process_in_extension_directory_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            sentinel.touch()
            owner = subprocess.Popen(
                ["bash", "-c", "cd \"$1\" && sleep 10", "owner", str(extension)]
            )
            try:
                time.sleep(0.1)
                result = self.run_helper(
                    temp,
                    TORCH_EXTENSION_LOCK_WAIT_S="0",
                    TORCH_EXTENSION_PREFLIGHT_WAIT_S="1",
                )
                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertTrue(sentinel.exists())
                self.assertEqual(list(extension.glob("lock.stale-*")), [])
            finally:
                owner.terminate()
                owner.wait(timeout=5)



if __name__ == "__main__":
    unittest.main()
