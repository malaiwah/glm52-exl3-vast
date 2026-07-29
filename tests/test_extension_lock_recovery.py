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
            ninja_lock.write_text("keep")

            result = self.run_helper(temp)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertEqual(ninja_lock.read_text(), "keep")
            quarantined = list(extension.glob("lock.stale-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].stat().st_size, 0)

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
