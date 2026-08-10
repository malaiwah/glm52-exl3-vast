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
ENTRYPOINT = ROOT / "entrypoint.sh"


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

    def test_ownerless_filebaton_is_quarantined_but_ninja_lock_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            ninja_lock = extension / ".ninja_lock"
            sentinel.touch()
            # Age the sentinel well past the default minimum age so the
            # ownerless-quarantine path is exercised, not the fresh-lock guard.
            old = time.time() - 7200
            os.utime(sentinel, (old, old))
            ninja_lock.write_text("keep")

            result = self.run_helper(temp)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertEqual(ninja_lock.read_text(), "keep")
            quarantined = list(extension.glob("lock.stale-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].stat().st_size, 0)

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

    def test_old_ownerless_sentinel_recovers_when_age_gate_disabled(self):
        # EXT_LOCK_MIN_AGE_S=0 opts out of the cross-container age gate; a fresh
        # ownerless sentinel is then quarantined as before the gate existed.
        with tempfile.TemporaryDirectory() as temp:
            extension = Path(temp) / "sparkinfer_pcie_dma_ext"
            extension.mkdir()
            sentinel = extension / "lock"
            sentinel.touch()

            result = self.run_helper(temp, EXT_LOCK_MIN_AGE_S="0")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertEqual(len(list(extension.glob("lock.stale-*"))), 1)

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

    def test_entrypoint_runs_preflight_before_serve(self):
        source = ENTRYPOINT.read_text()
        preflight = source.index("if ! recover_sparkinfer_extension_lock; then")
        launch = source.index("serve_once &", preflight)
        self.assertLess(preflight, launch)
        self.assertIn(
            'bash "$SCRIPTS_DIR/recover_torch_extension_lock.sh" '
            '"$TORCH_EXTENSIONS_DIR"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
