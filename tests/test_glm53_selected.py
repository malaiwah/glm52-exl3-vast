#!/usr/bin/env python3
"""CPU-only selected-source installation and async compute-share regressions."""
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import apply_glm53_selected as installer

PAYLOADS = ROOT / "patches/glm53-selected"
ARCHIVE = ROOT / "maintenance/glm53-optimization-20260908/build"


class SelectedInstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "site-packages"
        self.mirror = self.directory / "opt"
        for payload, target, _, _ in installer.resolve_targets(self.root, self.mirror):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PAYLOADS / payload, target)

    def install(self, source=PAYLOADS, **kwargs):
        installer.install(source, root=self.root, mirror_root=self.mirror, **kwargs)

    def snapshot(self):
        return {str(path.relative_to(self.directory)): path.read_bytes()
                for path in self.directory.rglob("*") if path.is_file()}

    def test_shipped_sources_are_exact_selected_archive_and_reapplication_is_read_only(self):
        manifest = json.loads((ARCHIVE / "build-manifest.json").read_text())
        expected = {entry["path"]: (ARCHIVE / entry["source"]).read_bytes()
                    for entry in manifest["files"] if entry["path"].endswith(".py")}
        shipped = {str(path.relative_to(PAYLOADS)): path.read_bytes()
                   for path in PAYLOADS.rglob("*.py")}
        self.assertEqual(shipped, expected)
        before = self.snapshot()
        self.install(verify_only=True)
        self.install()
        self.assertEqual(self.snapshot(), before)

    def test_unknown_and_mixed_sources_fail_without_any_write(self):
        first = installer.OVERLAYS[0]
        added = next(row for row in installer.OVERLAYS if row[2] is None)
        for relative, mirror, data in ((first[1], False, b"unreviewed\n"),
                                       (first[1], True, b"unreviewed\n"),
                                       (added[1], False, None),
                                       (added[1], True, None)):
            with self.subTest(relative=relative, mirror=mirror):
                target = (self.mirror / Path(relative).parts[0] if mirror else self.root) / relative
                original = target.read_bytes()
                if data is None:
                    target.unlink()
                else:
                    target.write_bytes(data)
                before = self.snapshot()
                for verify_only in (False, True):
                    with self.assertRaises(RuntimeError):
                        self.install(verify_only=verify_only)
                    self.assertEqual(self.snapshot(), before)
                target.write_bytes(original)

    def test_optional_mirrors_are_independent_but_partial_mirrors_are_refused(self):
        shutil.rmtree(self.mirror / "b12x")
        self.install(verify_only=True)
        missing = self.mirror / "vllm/vllm/config/scheduler.py"
        missing.unlink()
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            self.install()
        self.assertEqual(self.snapshot(), before)
        shutil.rmtree(self.mirror / "vllm")
        self.install()

    def test_payload_and_ledger_corruption_fail_before_writes(self):
        copied = self.directory / "payloads"
        shutil.copytree(PAYLOADS, copied)
        for relative in (installer.OVERLAYS[-1][0], "provenance.json"):
            with self.subTest(relative=relative):
                path = copied / relative
                original = path.read_bytes()
                path.write_bytes(b"corrupt\n")
                before = self.snapshot()
                with self.assertRaises(RuntimeError):
                    self.install(copied)
                self.assertEqual(self.snapshot(), before)
                path.write_bytes(original)

    def test_broken_symlink_cannot_masquerade_as_absent_new_module(self):
        added = next(row for row in installer.OVERLAYS if row[2] is None)
        target = self.root / added[1]
        target.unlink()
        target.symlink_to(self.directory / "missing")
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            self.install()
        self.assertTrue(target.is_symlink())
        self.assertEqual(self.snapshot(), before)

    def test_whole_baseline_preflight_then_install_creates_new_module_in_both_trees(self):
        # Synthetic before bytes exercise the installer state machine without
        # shipping another megabyte of baseline GPU source as a test fixture.
        # Selected bytes and their after hashes remain the real production ones.
        rows = []
        for payload, relative, before, after in installer.OVERLAYS:
            baseline = ("# baseline " + relative + "\n").encode()
            rows.append((payload, relative,
                         hashlib.sha256(baseline).hexdigest() if before else None, after))
            for target in (self.root / relative,
                           self.mirror / Path(relative).parts[0] / relative):
                if before is None:
                    target.unlink()
                else:
                    target.write_bytes(baseline)
        source = self.directory / "synthetic-baseline-ledger"
        shutil.copytree(PAYLOADS, source)
        ledger = json.loads((source / "provenance.json").read_text())
        for entry, row in zip(ledger["files"], rows):
            entry["before_sha256"] = row[2]
        data = json.dumps(ledger).encode()
        (source / "provenance.json").write_bytes(data)
        with mock.patch.object(installer, "OVERLAYS", tuple(rows)), \
             mock.patch.object(installer, "PROVENANCE_SHA256", hashlib.sha256(data).hexdigest()):
            before = self.snapshot()
            self.install(source, verify_only=True)
            self.assertEqual(self.snapshot(), before)
            # A late unknown file must not leave earlier targets updated.
            last = self.mirror / "vllm" / rows[-1][1]
            original = last.read_bytes()
            last.write_bytes(b"unknown\n")
            corrupted = self.snapshot()
            with self.assertRaises(RuntimeError):
                self.install(source)
            self.assertEqual(self.snapshot(), corrupted)
            last.write_bytes(original)
            self.install(source)
            self.install(source, verify_only=True)
            self.install(source)
            for payload, target, _, _ in installer.resolve_targets(self.root, self.mirror):
                self.assertEqual(target.read_bytes(), (PAYLOADS / payload).read_bytes())


class ComputeShareTests(unittest.TestCase):
    def test_measured_service_share_with_unequal_cost_and_two_batches_in_flight(self):
        path = PAYLOADS / "vllm/v1/core/sched/compute_fairness.py"
        spec = importlib.util.spec_from_file_location("selected_compute_fairness", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            controller = module.PrefillComputeShareController(0.6)
        pending = deque()
        elapsed = {"decode": 0.002, "prefill": 0.017}
        totals = {"decode": 0.0, "prefill": 0.0}
        for _ in range(20000):
            service = controller.select(decode_runnable=True, prefill_runnable=True)
            controller.dispatch(service, contended=True)
            pending.append(service)
            if len(pending) == 2:
                completed = pending.popleft()
                controller.record(completed, elapsed[completed], contended=True)
                totals[completed] += elapsed[completed]
        for completed in pending:
            controller.record(completed, elapsed[completed], contended=True)
            totals[completed] += elapsed[completed]
        self.assertAlmostEqual(totals["prefill"] / sum(totals.values()), 0.6, delta=0.01)
        self.assertFalse(controller.has_pending_reservations)
        self.assertEqual(controller.select(decode_runnable=False, prefill_runnable=True), "prefill")
        self.assertIsNone(controller.select(decode_runnable=False, prefill_runnable=False))


if __name__ == "__main__":
    unittest.main()
