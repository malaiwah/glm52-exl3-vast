#!/usr/bin/env python3
"""CPU regressions executing the shipped Gilded-derived runtime code.

The exact unchanged events/lexer fixtures keep these tests self-contained.
GLM53_REFRESH_BASE_ROOT optionally selects an extracted Gilded tree for those
modules; it defaults to the read-only extraction at /tmp/lil-gg-installed-before
when that is present, which also cross-checks the pinned before bytes.
GLM53_REFRESH_TEST_ROOT optionally selects all eight runtime sources instead of
payloads, so the same regressions can reproduce against the unpatched base.
No vLLM package initialization, Torch import, CUDA or model weights are needed.
"""
import ast
from contextlib import nullcontext
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
PAYLOADS = REPO / "patches/glm53-refresh"
EXTRACTED_BASE = Path(os.environ.get("GLM53_REFRESH_BASE_ROOT",
                                     "/tmp/lil-gg-installed-before"))
SPEC = importlib.util.spec_from_file_location(
    "apply_glm53_refresh", REPO / "scripts/apply_glm53_refresh.py")
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)
LEDGER = json.loads((PAYLOADS / "provenance.json").read_text())


def base_bytes(entry):
    """The pinned Gilded before bytes for one ledger entry.

    Derived by reversing the recorded hunks in the payload, which is exactly
    what the installer's provenance check proves. When the read-only extraction
    from the live Gilded container is available the derived bytes are compared
    against it, so a silently re-pinned payload cannot pass.
    """
    text = (PAYLOADS / entry["payload"]).read_text()
    for replacement in reversed(entry["replacements"]):
        if text.count(replacement["after"]) != 1:
            raise RuntimeError(f"non-unique refresh anchor: {entry['payload']}")
        text = text.replace(replacement["after"], replacement["before"], 1)
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["before_sha256"]:
        raise RuntimeError(f"payload does not reproduce pinned base: {entry['path']}")
    extracted = EXTRACTED_BASE / entry["path"]
    if extracted.is_file():
        actual = hashlib.sha256(extracted.read_bytes()).hexdigest()
        if actual != entry["before_sha256"]:
            raise RuntimeError(
                f"extracted base disagrees with pinned before state: {entry['path']}")
    return data


def source_path(name):
    root = os.environ.get("GLM53_REFRESH_TEST_ROOT")
    if root:
        relative = next(row[1] for row in INSTALLER.OVERLAYS if row[0] == name)
        return Path(root) / relative
    return PAYLOADS / name


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_methods(filename, class_name, method_names, namespace):
    """Execute complete production methods; remove only unrelated imports/init."""
    path = source_path(filename)
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    body = [node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names]
    for node in body:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0)] + body,
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_glm(namespace):
    path = source_path("glm47_moe.py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    tree.body.insert(0, ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0))
    namespace.update(re=re, json=json, functools=functools, TYPE_CHECKING=TYPE_CHECKING)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


class Tokenizer:
    def __init__(self, tokens, special_ids):
        self.tokens = tokens
        self.all_special_ids = special_ids
        self.all_special_tokens = [tokens[index] for index in special_ids]

    def decode(self, ids):
        return "".join(self.tokens[index] for index in ids)

    def get_vocab(self):
        return {text: index for index, text in self.tokens.items()}


class ScannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("glm53_test_scanner", source_path("token_id_scanner.py"))

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("glm53_test_scanner", None)

    def scanner(self):
        m = self.module
        tokens = {1: "prefix", 2: "<tool_call>", 3: "</tool_call>",
                  4: "record<arg_key>v</arg_key><arg_value></tool_call> and <tool_call></arg_value>",
                  5: "<|observation|>", 6: "<|user|>", 7: "after"}
        return m.TokenIDScanner(
            {2: "TOOL_START", 3: "TOOL_END", 5: m.DROP_TERMINAL, 6: m.DROP_TERMINAL},
            Tokenizer(tokens, [2, 3, 5, 6]))

    def test_reconstruction_matches_suffix_not_earlier_literal(self):
        scanner = self.scanner()
        m = self.module
        items = scanner.scan("prefix</tool_call> held prefix</tool_call>", [1, 3])
        self.assertTrue(all(isinstance(item, m.TextChunk) for item in items[:-1]))
        self.assertEqual("".join(item.text for item in items[:-1]),
                         "prefix</tool_call> held prefix")
        self.assertEqual(items[-1], m.PreLexedTerminal("TOOL_END", 3, "</tool_call>"))

    def test_stripped_stop_preserves_structural_anchors_and_multiple_drop_order(self):
        scanner = self.scanner()
        m = self.module
        text = scanner.tokenizer.decode([1, 2, 4, 3])
        self.assertEqual(scanner.scan(text, [1, 2, 4, 3, 5, 6]), [
            m.TextChunk("prefix"), m.PreLexedTerminal("TOOL_START", 2, "<tool_call>"),
            m.TextChunk(scanner.tokenizer.tokens[4]),
            m.PreLexedTerminal("TOOL_END", 3, "</tool_call>")])
        self.assertEqual(scanner.flush_pending(), [
            m.PreLexedTerminal(m.DROP_TERMINAL, 5, "<|observation|>"),
            m.PreLexedTerminal(m.DROP_TERMINAL, 6, "<|user|>")])
        self.assertEqual(scanner.flush_pending(), [])

    def test_holdback_and_deferred_stop_resolve_on_next_delta(self):
        scanner = self.scanner()
        m = self.module
        items = scanner.scan("held prefix", [1, 5])
        self.assertTrue(all(isinstance(item, m.TextChunk) for item in items))
        self.assertEqual("".join(item.text for item in items), "held prefix")
        self.assertEqual(scanner.scan("<|observation|>after", [7]), [
            m.PreLexedTerminal(m.DROP_TERMINAL, 5, "<|observation|>"),
            m.TextChunk("after")])
        self.assertEqual(scanner.flush_pending(), [])

    def test_context_decode_mismatch_still_uses_real_delta_text(self):
        scanner = self.scanner()
        m = self.module
        self.assertEqual(scanner.scan("different</tool_call>", [1, 3]), [
            m.TextChunk("different"), m.PreLexedTerminal("TOOL_END", 3, "</tool_call>")])


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.convert = load_glm({})["_glm47_arg_converter"]

    def test_literal_end_tags_and_unicode_remain_argument_data(self):
        value = '雪 "quoted" \\ </arg_value> then </tool_call> as data <tool_call>'
        raw = f"<arg_key>a</arg_key><arg_value>{value}</arg_value>\n<arg_key>b</arg_key><arg_value>2</arg_value>"
        self.assertEqual(json.loads(self.convert(raw, False)), {"a": value, "b": "2"})

    def test_unconfirmed_end_tag_is_held_until_later_end_keeps_deltas_monotonic(self):
        raw = "<arg_key>a</arg_key><arg_value>x</arg_value>"
        self.assertEqual(json.loads(self.convert(raw, True)), {"a": "x"})
        self.assertEqual(json.loads(self.convert(raw + "y", True)), {"a": "x"})
        self.assertEqual(json.loads(self.convert(raw + "y</arg_value>", True)),
                         {"a": "x</arg_value>y"})

    def test_missing_end_preserves_data_but_last_unconfirmed_end_discards_malformed_tail(self):
        raw = "<arg_key>a</arg_key><arg_value>x</tool_call>after"
        self.assertEqual(json.loads(self.convert(raw, False)), {"a": "x</tool_call>after"})
        self.assertEqual(json.loads(self.convert(raw + "</arg_value> stray", False)),
                         {"a": "x</tool_call>after"})


class StreamingParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = REPO / "tests/fixtures/glm53-refresh"
        extracted = EXTRACTED_BASE / "vllm/parser/engine"
        helpers = extracted if extracted.is_dir() else fixtures
        provenance = json.loads((fixtures / "provenance.json").read_text())
        for entry in provenance["files"]:
            actual = hashlib.sha256((helpers / entry["fixture"]).read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                raise RuntimeError(f"unreviewed parser dependency: {entry['fixture']}")
        cls.modules = mock.patch.dict(sys.modules)
        cls.modules.start()
        # This exact lexer uses compile/escape only to retain .pattern;
        # all matching and chunk holdback runs its real literal/prefix logic.
        # stdlib re supplies that metadata without adding a CI binary dependency.
        sys.modules["regex"] = re
        for name in ("vllm", "vllm.parser", "vllm.parser.engine"):
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        prefix = "vllm.parser.engine."
        try:
            cls.events = load_module(prefix + "events", helpers / "events.py")
            load_module(prefix + "incremental_lexer", helpers / "incremental_lexer.py")
            config = load_module(prefix + "parser_engine_config", source_path("parser_engine_config.py"))
            load_module(prefix + "token_id_scanner", source_path("token_id_scanner.py"))
            cls.engine = load_module(prefix + "streaming_parser_engine", source_path("streaming_parser_engine.py"))
            cls.glm = load_glm(dict(EventType=cls.events.EventType,
                                   ParserEngineConfig=config.ParserEngineConfig,
                                   ParserState=config.ParserState, Transition=config.Transition))
        except BaseException:
            cls.modules.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()

    def parse(self, chunks, tokenizer=None, ids=None):
        engine = self.engine.StreamingParserEngine(self.glm["glm47_moe_config"](), tokenizer)
        events = []
        for index, chunk in enumerate(chunks):
            events.extend(engine.feed(chunk, ids[index] if ids else []))
        events.extend(engine.finish())
        E = self.events.EventType
        args = "".join(e.value for e in events if e.type == E.ARG_VALUE_CHUNK)
        return (
            "".join(e.value for e in events if e.type == E.TOOL_NAME).strip(),
            json.loads(self.glm["_glm47_arg_converter"](args, False)),
            "".join(e.value for e in events if e.type == E.TEXT_CHUNK),
            sum(e.type == E.TOOL_CALL_END for e in events),
        )

    def test_literal_markers_survive_whole_tag_and_character_chunks(self):
        value = '雪 "quoted" \\ </arg_value> then </tool_call> as data <tool_call>'
        output = f"</think><tool_call>record<arg_key>a</arg_key><arg_value>{value}</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>"
        tag_chunks = [part for part in re.split(r"(<[^>]+>)", output) if part]
        for chunks in ([output], tag_chunks, list(output)):
            with self.subTest(chunk_count=len(chunks)):
                self.assertEqual(self.parse(chunks), ("record", {"a": value, "b": "2"}, "", 1))

    def test_stripped_stop_with_literal_tool_ids_in_argument(self):
        tokens = {1: "</think>", 2: "<tool_call>", 3: "record", 4: "<arg_key>",
                  5: "a", 6: "</arg_key>", 7: "<arg_value>", 8: "</tool_call>",
                  9: " and ", 10: "</arg_value>", 11: "<|observation|>"}
        tokenizer = Tokenizer(tokens, [1, 2, 4, 6, 7, 8, 10, 11])
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10, 8]
        output = tokenizer.decode(sequence)
        self.assertEqual(self.parse([output], tokenizer, [sequence + [11]]),
                         ("record", {"a": "</tool_call> and <tool_call>"}, "", 1))

    def test_empty_call_and_documented_malformed_termination(self):
        self.assertEqual(self.parse(["</think><tool_call>record</tool_call>"]),
                         ("record", {}, "", 1))
        start = "</think><tool_call>record<arg_key>a</arg_key><arg_value>x"
        self.assertEqual(self.parse(list(start + "</tool_call>after")),
                         ("record", {"a": "x</tool_call>after"}, "", 1))
        self.assertEqual(self.parse(list(start + "</arg_value> stray </tool_call>after")),
                         ("record", {"a": "x"}, "", 1))


class Queue(list):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.pop(0)

    def prepend_request(self, request):
        self.insert(0, request)

    def prepend_requests(self, requests):
        self[:0] = requests


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.throttle = load_methods("engine_core.py", "EngineCore",
                                     {"_should_throttle_prefills"}, {})["_should_throttle_prefills"]
        self.dp_throttle = load_methods("engine_core.py", "DPEngineCoreProc",
                                        {"_should_throttle_prefills"}, {})["_should_throttle_prefills"]
        self.pause = SimpleNamespace(PAUSED_ALL=2, UNPAUSED=0)
        self.status = SimpleNamespace(WAITING=0, PREEMPTED=1, RUNNING=2, WAITING_FOR_REMOTE_KVS=3)
        self.schedule = load_methods("scheduler.py", "Scheduler", {"schedule"}, dict(
            time=time, PauseState=self.pause, RequestStatus=self.status,
            record_function_or_nullcontext=lambda *_: nullcontext(),
            create_request_queue=lambda *_: Queue(), SchedulerOutput=SimpleNamespace,
            NewRequestData=SimpleNamespace(from_request=lambda req, *_: SimpleNamespace(req_id=req.request_id)),
        ))["schedule"]

    def engine(self, scheduler, interval=4):
        return SimpleNamespace(scheduler=scheduler, vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(prefill_schedule_interval=interval)))

    def request(self, name, *, prefill, computed=0, eligible=0):
        return SimpleNamespace(request_id=name, is_prefill_chunk=prefill,
            num_output_placeholders=0, num_computed_tokens=computed,
            num_prompt_tokens=8, max_tokens=32, next_decode_eligible_step=eligible,
            num_tokens_with_spec=8 if prefill else 9, num_tokens=8 if prefill else 9,
            has_encoder_inputs=False, spec_token_ids=[], num_stale_output_tokens=0,
            status=self.status.WAITING, drop_stale_output=False,
            shared_prefix_boundary=0, mm_features=None, block_hashes=[],
            num_preemptions=0, skip_reading_prefix_cache=False,
            prefill_stats=None)

    def scheduler(self, running=(), waiting=(), *, capacity_bound=False):
        blocks = SimpleNamespace(get_block_ids=lambda: ([1],))
        cache = SimpleNamespace(new_step_starts=lambda: None,
            allocate_slots=lambda *_, **__: blocks, get_blocks=lambda *_: blocks,
            record_prefix_cache_stats=lambda *_: None, log_stats=False,
            get_computed_blocks=lambda _: (blocks, 0, 0),
            empty_kv_cache_blocks=blocks, get_num_common_prefix_blocks=lambda *_: [0],
            take_kv_cache_block_copies=lambda: ([], []))
        scheduler = SimpleNamespace(current_step=0, max_num_scheduled_tokens=50,
            _pause_state=0, max_num_encoder_input_tokens=0, kv_cache_manager=cache,
            prefill_capacity_bound=capacity_bound, running=list(running),
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0, enable_chunked_prefill=True),
            max_model_len=1024, num_sampled_tokens_per_step=1,
            need_mamba_block_aligned_split=False, num_lookahead_tokens=0,
            lora_config=None, waiting=Queue(waiting), skipped_waiting=Queue(),
            policy=0, num_waiting_for_streaming_input=0, max_num_running_reqs=16,
            connector=None, ec_connector=None, num_spec_tokens=0,
            dynamic_sd_lookup=None, is_encoder_decoder=False,
            scheduler_reserve_full_isl=False, log_stats=False,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
            use_v2_model_runner=False, prev_step_scheduled_req_ids=set(),
            acceptance_length_controller=None, reset_preempted_req_ids=set(),
            finished_req_ids=set(), defer_block_free=False,
            encoder_cache_manager=SimpleNamespace(get_freed_mm_hashes=lambda: [], get_manager_metadata=lambda: None),
            _get_new_block_ids_to_zero=lambda: [],
            _make_cached_request_data=lambda *args: SimpleNamespace(req_ids=[r.request_id for r in args[0]]),
            _update_after_schedule=lambda _: None,
            _is_blocked_waiting_status=lambda _: False)
        scheduler._select_waiting_queue_for_scheduling = lambda: scheduler.waiting
        return scheduler

    def test_non_dp_cadence_first_decision_and_invalid_counter(self):
        engine = self.engine(SimpleNamespace(current_step=0))
        observed = []
        for completed in range(5):
            engine.scheduler.current_step = completed
            observed.append(self.throttle(engine))
        self.assertEqual(observed, [False, True, True, True, False])
        for counter in (None, -1, True, 1.5):
            with self.subTest(counter=counter):
                engine.scheduler.current_step = counter
                with self.assertRaises(RuntimeError):
                    self.throttle(engine)
        engine.vllm_config.scheduler_config.prefill_schedule_interval = 1
        self.assertFalse(self.throttle(engine))

    def test_dp_uses_synchronized_step_and_releases_after_idle(self):
        engine = SimpleNamespace(prefill_schedule_interval=4, step_counter=0,
                                 scheduler=SimpleNamespace(current_step=77))
        self.assertFalse(self.dp_throttle(engine))
        engine.step_counter = 3
        self.assertTrue(self.dp_throttle(engine))
        engine.step_counter = 0
        self.assertFalse(self.dp_throttle(engine))

    def test_eligible_decode_defers_chunked_prefill_until_release(self):
        scheduler = self.scheduler([self.request("decode", prefill=False, computed=8),
                                    self.request("prefill", prefill=True)])
        scheduler.current_step = 1
        output = self.schedule(scheduler, self.throttle(self.engine(scheduler)))
        self.assertEqual(output.num_scheduled_tokens, {"decode": 1})
        scheduler.current_step = 4
        output = self.schedule(scheduler, self.throttle(self.engine(scheduler)))
        self.assertEqual(output.num_scheduled_tokens, {"decode": 1, "prefill": 8})

    def test_no_decode_or_temporarily_ineligible_decode_makes_prefill_progress(self):
        for running in ([], [self.request("decode", prefill=False, computed=8, eligible=2)]):
            with self.subTest(has_decode=bool(running)):
                scheduler = self.scheduler(running, [self.request("prefill", prefill=True)])
                output = self.schedule(scheduler, True)
                self.assertEqual(output.num_scheduled_tokens, {"prefill": 8})

    def test_capacity_bound_queue_overrides_throttle(self):
        scheduler = self.scheduler([self.request("decode", prefill=False, computed=8),
                                    self.request("prefill", prefill=True)], capacity_bound=True)
        self.assertEqual(self.schedule(scheduler, True).num_scheduled_tokens,
                         {"decode": 1, "prefill": 8})

    def test_remote_resume_only_defers_when_local_prefill_is_needed(self):
        for computed, expected in ((4, {"decode": 1}), (7, {"decode": 1, "remote": 1})):
            with self.subTest(computed=computed):
                scheduler = self.scheduler(
                    [self.request("decode", prefill=False, computed=8)],
                    [self.request("remote", prefill=True, computed=computed)])
                self.assertEqual(self.schedule(scheduler, True).num_scheduled_tokens, expected)

    def test_empty_scheduler_still_advances_completed_decision_counter(self):
        scheduler = self.scheduler()
        self.assertEqual(self.schedule(scheduler, True).num_scheduled_tokens, {})
        self.assertEqual(scheduler.current_step, 1)
        self.assertTrue(self.throttle(self.engine(scheduler)))


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "site-packages"
        self.mirror = Path(self.temp.name) / "opt-vllm"
        self.originals = {}
        for entry in LEDGER["files"]:
            data = base_bytes(entry)
            for target in (self.root / entry["path"], self.mirror / entry["path"]):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                self.originals[target] = data

    def installed(self):
        return {self.root / e["path"]: (PAYLOADS / e["payload"]).read_bytes()
                for e in LEDGER["files"]}

    def snapshot(self):
        return {path: path.read_bytes() for path in self.originals}

    def install(self, source=PAYLOADS, **kwargs):
        kwargs.setdefault("root", self.root)
        kwargs.setdefault("mirror_root", self.mirror)
        INSTALLER.install(source, **kwargs)

    def test_pinned_roots_and_relative_targets_compose_both_trees(self):
        self.assertEqual(str(INSTALLER.DEFAULT_ROOT),
                         "/opt/venv/lib/python3.12/site-packages")
        self.assertEqual(str(INSTALLER.DEFAULT_MIRROR_ROOT), "/opt/vllm")
        # Enumerating the manifest must not require the image to be present.
        targets = [str(target) for _payload, target, _b, _a
                   in INSTALLER.resolve_targets()]
        self.assertIn("/opt/venv/lib/python3.12/site-packages/vllm/parser/glm47_moe.py",
                      targets)
        self.assertIn("/opt/vllm/vllm/parser/glm47_moe.py", targets)
        self.assertEqual(len(targets), 2 * len(INSTALLER.OVERLAYS))

    def test_verify_then_install_and_idempotent_reapplication(self):
        self.install(verify_only=True)
        self.assertEqual(self.snapshot(), self.originals)
        self.install()
        expected = self.installed()
        self.assertEqual({p: p.read_bytes() for p in expected}, expected)
        # The debug source tree receives the identical payload.
        for entry in LEDGER["files"]:
            self.assertEqual((self.mirror / entry["path"]).read_bytes(),
                             (PAYLOADS / entry["payload"]).read_bytes())
        self.install(verify_only=True)
        self.install()
        self.assertEqual({p: p.read_bytes() for p in expected}, expected)

    def test_absent_mirror_installs_runtime_root_only(self):
        shutil.rmtree(self.mirror)
        self.install()
        expected = self.installed()
        self.assertEqual({p: p.read_bytes() for p in expected}, expected)
        self.install(verify_only=True)

    def test_partial_mirror_is_refused_without_any_write(self):
        (self.mirror / LEDGER["files"][0]["path"]).unlink()
        remaining = {path: data for path, data in self.originals.items()
                     if path.exists()}
        with self.assertRaises(RuntimeError):
            self.install()
        self.assertEqual({p: p.read_bytes() for p in remaining}, remaining)

    def test_mixed_and_unknown_states_reject_without_partial_mutation(self):
        first = LEDGER["files"][0]
        payload = (PAYLOADS / first["payload"]).read_bytes()
        for label, path, invalid in (
                ("mixed root", self.root / first["path"], payload),
                ("mixed mirror", self.mirror / first["path"], payload),
                ("unknown root", self.root / first["path"], b"unreviewed source\n"),
                ("corrupt byte", self.mirror / first["path"],
                 self.originals[self.mirror / first["path"]][:-1] + b"#")):
            with self.subTest(state=label):
                original = path.read_bytes()
                path.write_bytes(invalid)
                before = self.snapshot()
                with self.assertRaises(RuntimeError):
                    self.install()
                self.assertEqual(self.snapshot(), before)
                path.write_bytes(original)

    def test_corrupt_payload_rejects_before_any_target_write(self):
        copies = Path(self.temp.name) / "payloads"
        shutil.copytree(PAYLOADS, copies)
        first = LEDGER["files"][0]
        (copies / first["payload"]).write_text("corrupt\n")
        with self.assertRaises(RuntimeError):
            self.install(copies)
        self.assertEqual(self.snapshot(), self.originals)


if __name__ == "__main__":
    unittest.main()
