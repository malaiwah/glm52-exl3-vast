#!/usr/bin/env python3
"""CPU contract tests executing the exact shipped adapter methods, without CUDA."""
import ast
from dataclasses import dataclass
import enum
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import patch_scopedlmcache_retrieve as installer


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class FatalTimeout(TimeoutError):
    def __init__(self, message, *, session_id=""):
        super().__init__(message)
        self.session_id = session_id


class DeviceFuture:
    """Two independent completion gates, like MQ response plus CUDA IPC event."""
    def __init__(self):
        self.response_ready = False
        self.event_ready = False
        self.value = True
        self.query_error = None
        self.result_error = None
        self.result_calls = 0

    def query(self):
        if self.query_error:
            raise self.query_error
        return self.response_ready and self.event_ready

    def result(self):
        self.result_calls += 1
        if not self.query():
            raise AssertionError("blocking result called before device completion")
        if self.result_error:
            raise self.result_error
        return self.value


def load_adapter(clock):
    # AST extraction removes imports/decorators only. Execute whole production
    # classes and functions, not a copied approximation of the polling loop.
    path = Path(os.environ.get("LMCACHE_ADAPTER_TEST_SOURCE", installer.default_payload()))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "ExtraConfigDefault", "_resolve_extra_config", "ParallelStrategy",
        "_normalize_adapter_init_args", "LMCacheMPWorkerAdapter",
        "DEFAULT_MQ_TIMEOUT", "DEFAULT_HEARTBEAT_INTERVAL",
        "_EXTRA_CONFIG_KEY_PREFIX", "_SERVER_REAP_TIMEOUT_FLOOR_SECONDS",
    }
    body = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            body.append(node)
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + body
    ast.fix_missing_locations(tree)
    namespace = dict(
        enum=enum, math=math, os=os, threading=threading, uuid=uuid,
        dataclass=dataclass, time=clock, logger=logging.getLogger("cache-test"),
        _lmcache_nvtx_annotate=lambda fn: fn, LMCacheTimeoutError=FatalTimeout,
        MessageQueueClient=lambda *args: SimpleNamespace(close=lambda: None),
        get_lmcache_chunk_size=lambda *args, **kwargs: 256,
        RequestTelemetryFactory=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            on_request_store_finished=lambda **kwargs: None)),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class RetrieveDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ns = load_adapter(self.clock)
        strategy = self.ns["ParallelStrategy"](True, 4, 0, 4, 1, 1, dcp_size=4)
        self.adapter = self.ns["LMCacheMPWorkerAdapter"](
            "tcp://unused", None, "full-glm", 256, strategy,
            extra_config={"lmcache.mp.retrieve_timeout": 180},
        )
        self.adapter._ensure_heartbeat_started = lambda: None
        self.adapter._create_key = lambda *args, **kwargs: None
        self.adapter._block_ids_per_group = lambda op: [[7, 8]]
        self.future = DeviceFuture()
        self.adapter.transfer_ctx = SimpleNamespace(submit_retrieve=lambda *args, **kwargs: self.future)
        self.op = SimpleNamespace(token_ids=[1], start=0, end=256,
                                  flat_block_ids=[7, 8], skip_first_n_tokens=0)
        self.event = object()
        self.adapter.submit_retrieve_request("r", self.op, self.event)

    def assert_ownership_retained(self):
        self.assertEqual(self.adapter.get_block_ids_with_load_errors(), set())
        self.assertIs(self.adapter.retrieve_futures["r"][0], self.future)
        self.assertIs(self.adapter.retrieve_events["r"], self.event)

    def test_raw_response_and_device_event_share_submission_deadline(self):
        self.clock.now = 179.0
        self.assertEqual(self.adapter.get_finished(set()), (set(), set()))
        self.future.response_ready = True
        self.clock.now = 180.0
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assertEqual(self.future.result_calls, 0)
        self.assert_ownership_retained()
        # A caller that accidentally catches the fatal error must not ACK late DMA.
        self.future.event_ready = True
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        with self.assertRaises(FatalTimeout):
            self.adapter.submit_retrieve_request("new", self.op, object())
        self.assert_ownership_retained()

    def test_missing_raw_response_expires_without_ack(self):
        self.clock.now = 180.0
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assert_ownership_retained()

    def test_ready_device_wins_at_deadline_boundary(self):
        self.future.response_ready = self.future.event_ready = True
        self.clock.now = 180.0
        self.assertEqual(self.adapter.get_finished(set()), (set(), {"r"}))
        self.assertEqual(self.adapter.get_block_ids_with_load_errors(), set())
        self.assertEqual(self.adapter.get_finished(set()), (set(), set()))

    def test_deadline_includes_submission_time(self):
        def delayed_submit(*args, **kwargs):
            self.clock.now += 181
            return self.future
        self.adapter.transfer_ctx.submit_retrieve = delayed_submit
        self.adapter.submit_retrieve_request("slow-submit", self.op, self.event)
        # Remove the unrelated request so the slow submission alone is exercised.
        self.adapter.retrieve_futures.pop("r")
        self.adapter.retrieve_events.pop("r")
        self.adapter._retrieve_started_at.pop("r")
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())

    def test_completed_false_or_result_errors_recompute(self):
        for failure in (False, "result"):
            with self.subTest(failure=failure):
                self.setUp()
                self.future.response_ready = self.future.event_ready = True
                if failure is False:
                    self.future.value = False
                else:
                    self.future.result_error = RuntimeError("remote failure")
                self.assertEqual(self.adapter.get_finished(set()), (set(), {"r"}))
                self.assertEqual(self.adapter.get_block_ids_with_load_errors(), {7, 8})
                self.assertEqual(self.adapter.get_finished(set()), (set(), set()))
                self.assertEqual(self.adapter.get_block_ids_with_load_errors(), set())

    def test_query_error_before_completion_never_releases_target_blocks(self):
        # A raw reply can be ready while IPC import/event.query still fails.
        self.future.response_ready = True
        self.future.query_error = RuntimeError("CUDA IPC event import failed")
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assertEqual(self.future.result_calls, 0)
        self.assert_ownership_retained()
        self.future.query_error = None
        self.future.event_ready = True
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assert_ownership_retained()

    def test_transport_timeout_is_not_terminal_load_failure(self):
        self.future.query_error = TimeoutError("transport expired without cancellation")
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assert_ownership_retained()

    def test_health_loss_cannot_bypass_deadline_fence(self):
        self.adapter._health_event.clear()
        with self.assertRaises(FatalTimeout):
            self.adapter.get_finished(set())
        self.assert_ownership_retained()

    def test_unsubmitted_unhealthy_load_is_safe_to_recompute(self):
        self.adapter.retrieve_futures.clear()
        self.adapter.retrieve_events.clear()
        self.adapter._retrieve_started_at.clear()
        self.adapter._health_event.clear()
        self.adapter.submit_retrieve_request("not-sent", self.op, self.event)
        self.assertEqual(self.adapter.get_finished(set()), (set(), {"not-sent"}))
        self.assertEqual(self.adapter.get_block_ids_with_load_errors(), {7, 8})
        self.assertEqual(self.adapter.get_finished(set()), (set(), set()))

    def test_invalid_timeout_rejected_before_connecting(self):
        strategy = self.ns["ParallelStrategy"](True, 4, 0, 4, 1, 1, dcp_size=4)
        for timeout in (-1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.ns["LMCacheMPWorkerAdapter"](
                    "tcp://unused", None, "full-glm", 256, strategy,
                    extra_config={"lmcache.mp.retrieve_timeout": timeout},
                )


class InstallerTests(unittest.TestCase):
    def test_discovery_does_not_execute_package_initializers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = (
                "lmcache/integration/vllm/vllm_multi_process_adapter.py",
                "lmcache/integration/vllm/lmcache_mp_connector.py",
                "vllm/config/vllm.py",
            )
            for relative in sources:
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# installed source\n")
                for package in source.parents:
                    if package == root:
                        break
                    (package / "__init__.py").write_text(
                        "raise RuntimeError('package initialization is forbidden')\n"
                    )
            code = (
                "import sys; sys.path[:0] = sys.argv[1:3]; "
                "import patch_scopedlmcache_retrieve as p; "
                "print(p.default_target()); print(p.default_target(True)); "
                "print(p.package_source('vllm', 'config/vllm.py'))"
            )
            result = subprocess.run(
                [sys.executable, "-I", "-c", code, str(ROOT / "scripts"), directory],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(root / p) for p in sources])

    def test_unknown_source_leaves_target_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "adapter.py"
            original = b"raise RuntimeError('different installed source')\n"
            target.write_bytes(original)
            with self.assertRaises(RuntimeError):
                installer.patch(target)
            self.assertEqual(target.read_bytes(), original)

    def test_verify_and_repeat_preserve_reviewed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "adapter.py"
            target.write_bytes(installer.default_payload().read_bytes())
            target.chmod(0o640)
            self.assertEqual(installer.patch(target, verify_only=True), "verified")
            self.assertEqual(installer.patch(target), "verified")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main(verbosity=2)
