#!/usr/bin/env python3
"""CPU-only deterministic checks; no vLLM install, torch, model, or GPU needed.

Run: python3 validate_fairness.py
Loads production methods through AST to isolate scheduler logic from optional
GPU imports. Only cache/encoder/executor boundaries are faked; scheduler and
controller decisions, request advancement, KV recovery, and timers are real.
This does not validate kernels, LMCache transport, or HTTP responsiveness.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import math
import sys
import time
import unittest
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable, cast
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


def load_class(path, name, methods=None, namespace=None, bases=()):
    """Compile unchanged production method ASTs without importing GPU modules."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    node.decorator_list = []
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    env = dict(globals()) if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module), path, "exec"), env)
    return env[name]


spec = importlib.util.spec_from_file_location("fairness_controller", ROOT / "vllm/v1/core/sched/compute_fairness.py")
controller_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller_module
spec.loader.exec_module(controller_module)
Controller = controller_module.PrefillComputeShareController


class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    WAITING_FOR_REMOTE_KVS = 3
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 4
    WAITING_FOR_STREAMING_REQ = 5


class PauseState(IntEnum):
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulingPolicy(IntEnum):
    FCFS = 0
    PRIORITY = 1


class Queue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, request):
        self.appendleft(request)

    def prepend_requests(self, requests):
        self.extendleft(reversed(requests))


def create_request_queue(policy):
    return Queue()


class Blocks:
    def get_block_ids(self):
        return ([0],)


class Cache:
    empty_kv_cache_blocks = Blocks()
    usage = 0.0
    log_stats = False

    def __init__(self):
        self.local_hits = {}
        self.blocked = set()
        self.allocations = []
        self.cached = []

    def new_step_starts(self):
        pass

    def allocate_slots(self, request, count, **kwargs):
        self.allocations.append((request.request_id, count, kwargs))
        return None if request.request_id in self.blocked else Blocks()

    def get_computed_blocks(self, request):
        return Blocks(), self.local_hits.get(request.request_id, 0), 0

    def get_blocks(self, request_id):
        return Blocks()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_kv_cache_block_copies(self):
        return [], []

    def cache_blocks(self, request, count):
        self.cached.append((request.request_id, count))

    def free(self, request):
        pass


class Request:
    def __init__(self, request_id, prompt=32, decode=False):
        self.request_id = request_id
        self.num_prompt_tokens = prompt
        self.num_tokens = prompt + int(decode)
        self.num_computed_tokens = prompt if decode else 0
        self.num_output_placeholders = 0
        self.num_in_flight_tokens = 0
        self.is_prefill_chunk = not decode
        self.status = RequestStatus.RUNNING if decode else RequestStatus.WAITING
        self.next_decode_eligible_step = 0
        self.max_tokens = 100000
        self.spec_token_ids = []
        self.has_encoder_inputs = False
        self.use_structured_output = False
        self.lora_request = None
        self.prefill_stats = None
        self.skip_reading_prefix_cache = False
        self.num_preemptions = 0
        self.arrival_time = 0.0
        self.priority = 0
        self.prompt_token_ids = list(range(prompt))
        self._all_token_ids = list(range(self.num_tokens))
        self.mm_features = []
        self.sampling_params = NS(max_tokens=self.max_tokens)
        self.pooling_params = self.prompt_embeds = self.prompt_is_token_ids = None

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


NewRequestData = load_class(
    "vllm/v1/core/sched/output.py", "NewRequestData",
    {"from_request"}, bases=("NS",),
)


class SchedulerOutput(NS):
    def __init__(self, **kwargs):
        super().__init__(
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            **kwargs,
        )

    def resolve_num_spec_tokens_to_schedule(self, default):
        return default if self.num_spec_tokens_to_schedule is None else self.num_spec_tokens_to_schedule


ComputeServiceClass = str
HybridKVCacheCoordinator = type("HybridKVCacheCoordinator", (), {})
logger = Mock()
record_function_or_nullcontext = lambda *args: nullcontext()
Scheduler = load_class(
    "vllm/v1/core/sched/scheduler.py", "Scheduler",
    {"schedule", "record_compute_time", "_update_after_schedule", "_is_blocked_waiting_status", "_select_waiting_queue_for_scheduling", "_try_promote_blocked_waiting_request", "_update_waiting_for_remote_kv"},
)


def scheduler(*, enabled=True, budget=16, threshold=0):
    s = Scheduler()
    s.scheduler_config = NS(long_prefill_token_threshold=threshold, enable_chunked_prefill=True, prefill_compute_share=0.5 if enabled else None)
    s.compute_share_controller = Controller(0.5) if enabled else None
    s._decode_compute_seconds = s._prefill_compute_seconds = 0.0
    s.current_step = 0
    s.max_num_scheduled_tokens = budget
    s.max_num_running_reqs = 8
    s.max_model_len = 1000000
    s.num_sampled_tokens_per_step = 1
    s._pause_state = PauseState.UNPAUSED
    s.prefill_capacity_bound = False
    s.running = []
    s.waiting = Queue()
    s.skipped_waiting = Queue()
    s.requests = {}
    s.kv_cache_manager = Cache()
    s.kv_cache_config = NS(kv_cache_groups=[None])
    s.max_num_encoder_input_tokens = 0
    s.num_waiting_for_streaming_input = 0
    s.num_lookahead_tokens = s.num_spec_tokens = 0
    s.dynamic_sd_lookup = s.acceptance_length_controller = None
    s.policy = SchedulingPolicy.FCFS
    s.lora_config = s.connector = s.ec_connector = None
    s.connector_prefix_cache_stats = None
    s.scheduler_reserve_full_isl = True
    s.need_mamba_block_aligned_split = s.has_mamba_layers = False
    s.needs_kv_cache_zeroing = s.defer_block_free = False
    s.use_eagle = s.use_v2_model_runner = False
    s.is_encoder_decoder = False
    s.log_stats = False
    s._inflight_prefills = set()
    s.finished_recving_kv_req_ids = set()
    s.failed_recving_kv_req_ids = set()
    s.finished_req_ids = set()
    s.reset_preempted_req_ids = set()
    s.prev_step_scheduled_req_ids = set()
    s.enable_return_routed_experts = False
    s.encoder_cache_manager = NS(get_freed_mm_hashes=lambda: [])
    s._get_new_block_ids_to_zero = lambda: None
    s._make_cached_request_data = lambda *args: NS()
    s._inflight_prefill_reserved_blocks = lambda: 0
    s._build_kv_connector_meta = lambda *args: NS()
    s._preempt_request = lambda r, timestamp: setattr(r, "status", RequestStatus.PREEMPTED)
    return s


def add(s, r):
    s.requests[r.request_id] = r
    (s.running if r.status == RequestStatus.RUNNING else s.waiting).append(r)
    return r


def complete(s, output, elapsed=0.01):
    if output.compute_service_class is not None:
        s.record_compute_time(output.compute_service_class, elapsed, contended=output.compute_contention)
    for rid in output.num_scheduled_tokens:
        r = s.requests[rid]
        if not r.is_prefill_chunk:
            r.num_tokens += 1


class SchedulerChecks(unittest.TestCase):
    def test_legacy_mixed_batch_and_cadence_escape(self):
        for throttle, capacity, expected in [(False, False, {"d", "p"}), (True, False, {"d"}), (True, True, {"d", "p"})]:
            s = scheduler(enabled=False)
            add(s, Request("d", decode=True))
            add(s, Request("p"))
            s.prefill_capacity_bound = capacity
            out = s.schedule(throttle)
            self.assertEqual(set(out.num_scheduled_tokens), expected)
            self.assertIsNone(out.compute_service_class)

    def test_decode_tie_then_prefill_first_with_decode_remainder(self):
        s = scheduler()
        add(s, Request("d", decode=True))
        add(s, Request("p", prompt=4))
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"d": 1})
        complete(s, out)
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"p": 4, "d": 1})
        self.assertEqual(out.compute_service_class, "prefill")

    def test_selected_prefill_uses_budget_and_partial_fcfs(self):
        s = scheduler(threshold=8)
        add(s, Request("d", decode=True))
        add(s, Request("p0"))
        add(s, Request("p1"))
        complete(s, s.schedule())
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"p0": 8, "p1": 8})
        complete(s, out)
        complete(s, s.schedule())
        self.assertEqual(s.schedule().num_scheduled_tokens, {"p0": 8, "p1": 8})

    def test_decode_async_guard_falls_back_to_running_prefill(self):
        s = scheduler()
        d = add(s, Request("d", decode=True))
        p = Request("p")
        p.status = RequestStatus.RUNNING
        add(s, p)
        d.num_output_placeholders = 1
        d.max_tokens = 0
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"p": 16})
        self.assertEqual(out.compute_service_class, "prefill")

    def test_alignment_blocked_prefill_reclaims_capacity_for_decode(self):
        s = scheduler()
        add(s, Request("d", decode=True))
        add(s, Request("p"))
        complete(s, s.schedule())
        s.need_mamba_block_aligned_split = True
        s._mamba_block_aligned_split = lambda r, n, *args: 0 if r.request_id == "p" else n
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"d": 1})
        self.assertEqual(out.compute_service_class, "decode")

    def test_full_apc_hit_is_decode(self):
        s = scheduler()
        add(s, Request("d", decode=True))
        add(s, Request("cached", prompt=33))
        s.kv_cache_manager.local_hits["cached"] = 32
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"d": 1, "cached": 1})
        self.assertEqual(out.compute_service_class, "decode")

    def test_async_restore_and_failed_restore_recompute(self):
        s = scheduler()
        p = add(s, Request("restore", prompt=64))
        s.connector = NS(get_num_new_matched_tokens=lambda *args: (32, True), update_state_after_alloc=lambda *args: None)
        out = s.schedule()
        self.assertEqual(out.total_num_scheduled_tokens, 0)
        self.assertIsNone(out.compute_service_class)
        self.assertEqual(p.status, RequestStatus.WAITING_FOR_REMOTE_KVS)
        self.assertIn(p, s._inflight_prefills)
        d = add(s, Request("d", decode=True))
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"d": 1})
        self.assertFalse(out.compute_contention)
        complete(s, out)
        # Invalid-block handling already rewinds computed tokens before promotion.
        p.num_computed_tokens = 16
        s.failed_recving_kv_req_ids.add("restore")
        s.finished_recving_kv_req_ids.add("restore")
        complete(s, s.schedule())
        out = s.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"restore": 16})
        self.assertEqual(out.compute_service_class, "prefill")
        self.assertNotIn("restore", s.failed_recving_kv_req_ids)
        self.assertIn(("restore", 16), s.kv_cache_manager.cached)

    def test_pause_all_never_dispatches_credit(self):
        s = scheduler()
        add(s, Request("d", decode=True))
        add(s, Request("p"))
        s._pause_state = PauseState.PAUSED_ALL
        out = s.schedule()
        self.assertEqual(out.total_num_scheduled_tokens, 0)
        self.assertFalse(s.compute_share_controller.has_pending_reservations)


class ControllerChecks(unittest.TestCase):
    def test_measured_unequal_cost_convergence_with_two_inflight(self):
        for share in (0.2, 0.5, 0.8):
            c = Controller(share)
            pending = deque()
            total = {"decode": 0.0, "prefill": 0.0}
            for _ in range(10000):
                while len(pending) < 2:
                    cls = c.select(decode_runnable=True, prefill_runnable=True)
                    c.dispatch(cls, contended=True)
                    pending.append(cls)
                cls = pending.popleft()
                elapsed = 0.7 if cls == "prefill" else 0.01
                c.record(cls, elapsed, contended=True)
                total[cls] += elapsed
            self.assertAlmostEqual(total["prefill"] / sum(total.values()), share, delta=0.01)

    def test_zero_elapsed_releases_estimated_reservation(self):
        c = Controller(0.5, last_decode_seconds=0.1)
        c.select(decode_runnable=True, prefill_runnable=True)
        c.dispatch("decode", contended=True)
        c.record("decode", 0.0, contended=True)
        self.assertFalse(c.has_pending_reservations)
        self.assertEqual(c.decode_virtual_runtime, 0.0)

    def test_uncontended_reset_and_changed_cost(self):
        c = Controller(0.5)
        counts = []
        for cost in (1.0, 0.1):
            count = 0
            for _ in range(1000):
                cls = c.select(decode_runnable=True, prefill_runnable=True)
                c.dispatch(cls, contended=True)
                c.record(cls, cost if cls == "prefill" else 0.01, contended=True)
                count += cls == "prefill"
            counts.append(count)
        self.assertGreater(counts[1], counts[0] * 5)
        self.assertEqual(c.select(decode_runnable=True, prefill_runnable=False), "decode")
        self.assertEqual(c.prefill_virtual_runtime, 0.0)
        self.assertFalse(c.contention_active)


_ModelExecutionTiming = load_class("vllm/v1/engine/core.py", "_ModelExecutionTiming")
EngineCore = load_class("vllm/v1/engine/core.py", "EngineCore", {"_execute_model", "_record_compute_time", "_should_throttle_prefills"})


class TimingChecks(unittest.TestCase):
    def test_callback_captures_completion_and_untagged_is_free(self):
        e = EngineCore()
        f = Future()
        e.model_executor = NS(execute_model=lambda *args, **kwargs: f)
        _, timer = e._execute_model(NS(compute_service_class=None))
        self.assertIsNone(timer)
        with patch.object(time, "perf_counter", side_effect=[10.0, 10.25]):
            returned, timer = e._execute_model(NS(compute_service_class="prefill"))
            self.assertIs(returned, f)
            self.assertIsNone(timer.completed_at)
            final = Future()
            timer.bind(final)
            f.set_result(None)
            self.assertIsNone(timer.completed_at)
            final.set_result(NS())
        self.assertEqual(timer.elapsed_seconds, 0.25)

    def test_fifo_feedback_subtracts_overlap_and_rejects_incomplete(self):
        e = EngineCore()
        e.scheduler = Mock()
        e._last_model_completion_time = None
        decode = NS(compute_service_class="decode", compute_contention=True)
        prefill = NS(compute_service_class="prefill", compute_contention=True)
        e._record_compute_time(decode, NS(started_at=10.0, completed_at=10.1))
        e._record_compute_time(prefill, NS(started_at=10.01, completed_at=10.3))
        calls = e.scheduler.record_compute_time.call_args_list
        self.assertEqual(calls[0].args[0], "decode")
        self.assertAlmostEqual(calls[0].args[1], 0.1)
        self.assertEqual(calls[1].args[0], "prefill")
        self.assertAlmostEqual(calls[1].args[1], 0.2)
        with self.assertRaises(RuntimeError):
            e._record_compute_time(prefill, NS(started_at=10.0, completed_at=None))

    def test_original_cadence_contract(self):
        e = EngineCore()
        e.vllm_config = NS(scheduler_config=NS(prefill_schedule_interval=3))
        e.scheduler = NS(current_step=0)
        values = []
        for step in range(6):
            e.scheduler.current_step = step
            values.append(e._should_throttle_prefills())
        self.assertEqual(values, [False, True, True, False, True, True])
        e.scheduler.current_step = True
        with self.assertRaises(RuntimeError):
            e._should_throttle_prefills()


class ConfigurationChecks(unittest.TestCase):
    def config(self, **options):
        cls = load_class(
            "vllm/config/scheduler.py", "SchedulerConfig",
            {"__post_init__", "verify_max_model_len"},
        )
        config = cls()
        values = dict(
            fairness_engine=None, prefill_compute_share=None,
            prefill_schedule_interval=1, enable_chunked_prefill=True,
            scheduler_cls=None, max_num_batched_tokens=16, max_num_seqs=4,
            max_num_partial_prefills=2, max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
        )
        values.update(options)
        config.__dict__.update(values)
        config.__post_init__(128, False)
        return config

    def test_explicit_selector_and_cadence_exclusion(self):
        for options in (
            {"prefill_compute_share": 0.5},
            {"fairness_engine": "compute_share"},
            {"fairness_engine": "compute_share", "prefill_compute_share": 0.5, "prefill_schedule_interval": 2},
        ):
            with self.assertRaises(ValueError):
                self.config(**options)

    def test_old_partial_prefill_semantics_remain_available(self):
        for mode in (None, "compute_share"):
            config = self.config(
                fairness_engine=mode,
                prefill_compute_share=0.5 if mode else None,
            )
            # The existing long-prompt threshold still derives from model len;
            # compute_share must not reinterpret partial-prefill knobs as micro-slicing.
            self.assertEqual(config.long_prefill_token_threshold, 5)
            self.assertEqual(config.max_num_partial_prefills, 2)


class QueueTimingChecks(unittest.TestCase):
    def test_deferred_grammar_preserves_exact_completion_credit(self):
        env = dict(globals(), ModelRunnerOutput=NS)
        core = load_class(
            "vllm/v1/engine/core.py", "EngineCore",
            {"_execute_model", "_record_compute_time", "step_with_batch_queue"},
            namespace=env,
        )
        e = core()
        decode = NS(
            compute_service_class="decode", compute_contention=True,
            total_num_scheduled_tokens=1, pending_structured_output_tokens=False,
        )
        prefill = NS(
            compute_service_class="prefill", compute_contention=True,
            total_num_scheduled_tokens=16, pending_structured_output_tokens=True,
        )
        pending = deque([decode, prefill])
        e.scheduler = NS(
            has_requests=lambda: bool(pending),
            schedule=lambda _: pending.popleft(),
            get_grammar_bitmask=lambda _: None,
            update_from_output=lambda *args: {},
            record_compute_time=Mock(),
        )
        e._last_model_completion_time = None
        e.batch_queue_size = 2
        e.batch_queue = deque(maxlen=2)
        e.is_ec_consumer = True
        e.is_pooling_model = False
        e.requires_host_draft_token_ids = False
        e.check_for_draft_tokens = False
        e._should_throttle_prefills = lambda: False
        e.capture_iteration_details = lambda _: nullcontext()
        e.log_error_detail = lambda _: nullcontext()
        e._process_aborts_queue = lambda: None
        e._attach_iteration_details = lambda *args: None
        decode_result, prefill_result = Future(), Future()
        results = deque([decode_result, prefill_result])
        clock = [0.0]

        def execute(output, **kwargs):
            if output is prefill:
                clock[0] = 0.1
                decode_result.set_result(NS())
            future = Future()
            future.set_result(None)
            return future

        e.model_executor = NS(
            execute_model=execute,
            sample_tokens=lambda *args, **kwargs: results.popleft(),
        )
        with patch.object(time, "perf_counter", side_effect=lambda: clock[0]):
            self.assertEqual(e.step_with_batch_queue(), (None, True))
            clock[0] = 0.05
            e.step_with_batch_queue()
            clock[0] = 0.4
            prefill_result.set_result(NS())
            e.step_with_batch_queue()
        calls = e.scheduler.record_compute_time.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["decode", "prefill"])
        self.assertAlmostEqual(calls[0].args[1], 0.1)
        self.assertAlmostEqual(calls[1].args[1], 0.3)


class V2SchedulerChecks(SchedulerChecks):
    """Run scheduler behavior cases through the production V2 output branch."""

    def setUp(self):
        original_factory = scheduler

        def v2_factory(**kwargs):
            s = original_factory(**kwargs)
            s.use_v2_model_runner = True
            return s

        replacement = patch.dict(globals(), {"scheduler": v2_factory})
        replacement.start()
        self.addCleanup(replacement.stop)


class V2AsyncChecks(unittest.TestCase):
    def test_skipped_decode_retains_mtp_width_and_request_eligibility(self):
        async_cls = load_class(
            "validation_sources/async_scheduler.py", "AsyncScheduler",
            {"_update_after_schedule"}, bases=("Scheduler",),
        )
        s = scheduler()
        s.__class__ = async_cls
        s.use_v2_model_runner = True
        s.pp_size = 1
        s.num_spec_tokens = s.num_lookahead_tokens = 3
        decode = add(s, Request("decode", decode=True))
        add(s, Request("prefill", prompt=64))
        first = s.schedule()
        self.assertEqual(first.num_scheduled_tokens, {"decode": 1})
        s.record_compute_time("decode", 0.01, contended=True)
        second = s.schedule()
        self.assertEqual(second.num_scheduled_tokens, {"prefill": 16})
        s.record_compute_time("prefill", 0.7, contended=True)
        third = s.schedule()
        # A prefill-only step must not lose an excluded decoder's persistent
        # speculative width, nor schedule it twice in one engine decision.
        self.assertEqual(third.num_scheduled_tokens, {"decode": 4})
        self.assertEqual(len(third.scheduled_spec_decode_tokens["decode"]), 3)
        self.assertEqual(decode.next_decode_eligible_step, s.current_step + 1)

    def test_v2_consumer_reconstructs_resumed_generated_prefix(self):
        s = scheduler()
        s.use_v2_model_runner = True
        r = Request("resumed", prompt=32)
        r.status = RequestStatus.PREEMPTED
        r.num_preemptions = 1
        r.num_tokens = 40
        r._all_token_ids = list(range(40))
        add(s, r)
        out = s.schedule()
        runner_cls = load_class(
            "validation_sources/v2_contracts.py", "GPUModelRunner",
            {"add_requests"},
        )
        worker = runner_cls()
        restored = {}

        def restore(**kwargs):
            restored[kwargs["req_id"]] = list(kwargs["all_token_ids"])

        worker.req_states = NS(
            add_request=restore, req_id_to_index={"resumed": 0},
            apply_staged_writes=lambda: None,
        )
        worker._remove_request = lambda _: None
        worker.verification_capacity_manager = worker.encoder_cache = None
        worker.model_state = NS(add_request=lambda *args: None, apply_staged_writes=lambda: None)
        worker.block_tables = NS(append_block_ids=lambda *args, **kwargs: None)
        worker.lora_state = NS(add_request=lambda *args: None)
        worker.is_last_pp_rank = False
        worker.sampler = None
        worker.add_requests(out)
        self.assertEqual(restored["resumed"], list(range(40)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
