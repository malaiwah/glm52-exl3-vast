import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import soul_config as sc  # noqa: E402
import soul_controller as controller  # noqa: E402
import soul_launcher as launcher  # noqa: E402
import secure_erase  # noqa: E402
import config_cli  # noqa: E402
import landing  # noqa: E402


class SoulConfigTests(unittest.TestCase):
    def env(self, root, **values):
        result = {
            "GLM_STATE_DIR": str(root / "state"),
            "GLM_RUNTIME_DIR": str(root / "runtime"),
        }
        result.update({key: str(value) for key, value in values.items()})
        return result

    def test_precedence_and_startup_ceiling(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = self.env(root, SOUL_AUTONOMY_LEVEL=1,
                           SOUL_AUTONOMY_MAX_LEVEL=2,
                           SOUL_JOURNAL_INTERVAL_S=7200)
            effective, sources, notes = sc.resolve(
                env, {"autonomyLevel": 3, "journalIntervalS": 1800})
            self.assertEqual(effective["requestedLevel"], 3)
            self.assertEqual(effective["autonomyLevel"], 2)
            self.assertEqual(effective["journalIntervalS"], 1800)
            self.assertEqual(sources["autonomyLevel"], "override")
            self.assertTrue(any("capped" in note for note in notes))

    def test_invalid_level_fails_closed_and_interval_falls_through(self):
        env = {
            "SOUL_AUTONOMY_LEVEL": "not-a-level",
            "SOUL_AUTONOMY_MAX_LEVEL": "3",
            "SOUL_HEARTBEAT_INTERVAL_S": "120",
        }
        effective, _sources, _notes = sc.resolve(
            env, {"autonomyLevel": "9", "heartbeatIntervalS": "bad"})
        self.assertEqual(effective["autonomyLevel"], 0)
        self.assertEqual(effective["heartbeatIntervalS"], 120)

    def test_atomic_override_writes_audit_and_reconcile(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = self.env(root, SOUL_AUTONOMY_MAX_LEVEL=2)
            with mock.patch("os.kill"):
                effective = sc.write_override({
                    "autonomyLevel": 3,
                    "heartbeatIntervalS": 60,
                    "timezone": "UTC",
                }, env)
            self.assertEqual(effective["autonomyLevel"], 2)
            self.assertTrue(sc.paths(env)["reconcile"].exists())
            config = json.loads(sc.paths(env)["config"].read_text())
            self.assertEqual(config["values"]["autonomyLevel"], 3)
            journal = sc.read_journal(env=env)
            self.assertEqual(journal[0]["kind"], "audit")

    def test_redaction_covers_headers_urls_and_common_tokens(self):
        raw = (
            "Authorization: Bearer abcdef api_key=verysecret "
            "VLLM_API_KEY=plainsecret SOUL_PROBE_API_KEY='probe-secret' "
            "OPEN_BUTTON_TOKEN=button-secret "
            '\"SERVICE_TOKEN\": \"json-secret\" '
            "https://local.test/x?token=signed-value hf_abcdefghijklmno"
        )
        clean = sc.redact(raw)
        self.assertNotIn("abcdef", clean)
        self.assertNotIn("verysecret", clean)
        self.assertNotIn("plainsecret", clean)
        self.assertNotIn("probe-secret", clean)
        self.assertNotIn("button-secret", clean)
        self.assertNotIn("json-secret", clean)
        self.assertNotIn("signed-value", clean)
        self.assertNotIn("hf_abcdefghijklmno", clean)
        self.assertEqual(
            sc.redact({"VLLM_API_KEY": "dictionary-secret", "safe": "visible"}),
            {"safe": "visible"},
        )

    def test_journal_pagination_is_newest_first(self):
        with tempfile.TemporaryDirectory() as raw:
            env = self.env(Path(raw))
            path = sc.ensure_tree(env)["journal"]
            for number in range(4):
                sc.append_jsonl(path, {"id": f"id-{number}", "timestamp": sc.utcnow()})
            self.assertEqual([row["id"] for row in sc.read_journal(2, env=env)],
                             ["id-3", "id-2"])
            self.assertEqual([row["id"] for row in sc.read_journal(
                2, before="id-2", env=env)], ["id-1", "id-0"])


class SoulControllerTests(unittest.TestCase):
    def test_every_analysis_gets_bounded_cross_incident_journal_context(self):
        entries = [{
            "timestamp": f"2026-07-29T00:00:0{number}Z",
            "kind": "incident",
            "severity": "warning",
            "headline": f"Repeated failure {number}",
            "summary": "x" * 1200,
            "incidentId": f"incident-{number}",
            "analysisStatus": "complete",
            "evidenceRefs": ["/must/not/be/replayed"],
        } for number in range(10)]
        with mock.patch.object(controller.sc, "read_journal",
                               return_value=entries[:8]) as read:
            result = controller._with_journal_context({"snapshot": {"ok": False}})
        read.assert_called_once_with(limit=controller.JOURNAL_CONTEXT_LIMIT)
        previous = result["previousJournalEntries"]
        self.assertEqual(len(previous), 8)
        self.assertEqual(set(previous[0]), {
            "timestamp", "kind", "severity", "headline", "summary",
            "incidentId", "analysisStatus",
        })
        self.assertEqual(len(previous[0]["summary"]),
                         controller.JOURNAL_CONTEXT_SUMMARY_CHARS)
        self.assertNotIn("evidenceRefs", previous[0])

    def test_explicit_daily_history_is_not_duplicated(self):
        supplied = [{"headline": "daily"}]
        with mock.patch.object(controller.sc, "read_journal") as read:
            result = controller._with_journal_context({"recentJournal": supplied})
        read.assert_not_called()
        self.assertEqual(result["recentJournal"], supplied)

    def test_incident_requires_two_failures_and_posts_one_recovery(self):
        with tempfile.TemporaryDirectory() as raw:
            book = controller.IncidentBook(Path(raw) / "incidents.json")
            finding = {"reachability:health": {
                "failed": True, "severity": "warning", "summary": "down", "requires": 2}}
            self.assertEqual(book.reconcile(finding, "s1"), [])
            opened = book.reconcile(finding, "s2")
            self.assertEqual([event["event"] for event in opened], ["opened"])
            self.assertEqual(book.reconcile(finding, "s3"), [])
            recovered = book.reconcile({}, "s4")
            self.assertEqual([event["event"] for event in recovered], ["recovered"])
            self.assertEqual(book.reconcile({}, "s5"), [])

    def test_immediate_fatal_incident(self):
        with tempfile.TemporaryDirectory() as raw:
            book = controller.IncidentBook(Path(raw) / "incidents.json")
            events = book.reconcile({"gpu:xid": {
                "failed": True, "severity": "critical", "summary": "XID", "requires": 1}},
                "s1")
            self.assertEqual(events[0]["severity"], "critical")

    def test_failed_inference_probes_are_immediate_incidents(self):
        findings = controller._severity({
            "filesystem": {"mounts": []},
            "endpoint": {"health": {"ok": True}, "models": {"ok": True}},
            "network": {},
            "logs": {},
            "gpu": {},
            "canary": {"ok": False, "response": "not SOUL-CANARY-OK"},
            "longContextProbe": {"ok": False, "verdict": {"ok": False}},
        })
        self.assertEqual(findings["inference:canary"]["requires"], 1)
        self.assertEqual(findings["inference:canary"]["severity"], "critical")
        self.assertEqual(findings["inference:long-context"]["requires"], 1)

    def test_canary_requires_an_exact_normalized_response(self):
        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({
                    "choices": [{"message": {"content": self.content}}],
                }).encode()

        requests = []

        def open_response(request, **_kwargs):
            requests.append(json.loads(request.data))
            return Response("not SOUL-CANARY-OK")

        with mock.patch.object(controller.urllib.request, "urlopen",
                               side_effect=open_response):
            self.assertFalse(controller._cheap_canary(8000, "", "local")["ok"])
        self.assertEqual(
            requests[0]["chat_template_kwargs"], {"enable_thinking": False})
        with mock.patch.object(controller.urllib.request, "urlopen",
                               return_value=Response("\nSOUL-CANARY-OK \n")):
            self.assertTrue(controller._cheap_canary(8000, "", "local")["ok"])
        with mock.patch.object(controller.urllib.request, "urlopen",
                               return_value=Response(None)):
            result = controller._cheap_canary(8000, "", "local")
        self.assertFalse(result["ok"])
        self.assertEqual(result["response"], "")

    def test_nanobot_document_disables_dream_and_level_one_shell(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            doc = controller._nanobot_document(
                root / "workspace", "local-model", 8000, 1, "UTC",
                api_key="sk-in-memory",
            )
            self.assertFalse(doc["agents"]["defaults"]["dream"]["enabled"])
            self.assertFalse(doc["tools"]["exec"]["enable"])
            self.assertEqual(doc["providers"]["custom"]["apiKey"], "sk-in-memory")
            self.assertEqual(
                doc["providers"]["custom"]["apiBase"],
                "http://127.0.0.1:8000/v1")
            self.assertEqual(
                doc["modelPresets"]["soul-local"]["maxTokens"], 2400)
            self.assertFalse((root / "nanobot-config.json").exists())
            level_two = controller._nanobot_document(
                root / "workspace", "local-model", 8000, 2, "UTC")
            self.assertTrue(level_two["tools"]["exec"]["enable"])

    def test_api_key_handoff_waits_for_hardening_then_closes_descriptors(self):
        key_read, key_write = os.pipe()
        ready_read, ready_write = os.pipe()
        result = []
        errors = []

        def consume():
            try:
                result.append(controller._read_api_key_fd())
            except Exception as exc:  # surfaced below in the test process
                errors.append(exc)

        env = {
            "SOUL_API_KEY_FD": str(key_read),
            "SOUL_KEY_READY_FD": str(ready_write),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            thread = __import__("threading").Thread(target=consume)
            thread.start()
            self.assertEqual(os.read(ready_read, 1), b"R")
            self.assertTrue(thread.is_alive())
            payload = b"k" * controller.MAX_API_KEY_BYTES
            launcher._write_all(key_write, len(payload).to_bytes(4, "big"))
            launcher._write_all(key_write, payload)
            os.close(key_write)
            os.close(ready_read)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertEqual(result, [payload.decode()])
            self.assertNotIn("SOUL_API_KEY_FD", os.environ)
            self.assertNotIn("SOUL_KEY_READY_FD", os.environ)
        for fd in (key_read, ready_write):
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_api_key_handoff_rejects_truncated_length_frame(self):
        key_read, key_write = os.pipe()
        ready_read, ready_write = os.pipe()
        errors = []

        def consume():
            try:
                controller._read_api_key_fd()
            except Exception as exc:  # surfaced below in the test process
                errors.append(exc)

        env = {
            "SOUL_API_KEY_FD": str(key_read),
            "SOUL_KEY_READY_FD": str(ready_write),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            thread = __import__("threading").Thread(target=consume)
            thread.start()
            self.assertEqual(os.read(ready_read, 1), b"R")
            os.write(key_write, (10).to_bytes(4, "big") + b"partial")
            os.close(key_write)
            os.close(ready_read)
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertIn("incomplete", str(errors[0]))

    def test_controller_never_persists_nanobot_provider_key(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            boot = root / "boot.json"
            boot.write_text(json.dumps({
                "phase": "serving", "served_model": "local-model",
                "endpoint": "",
            }))
            env = {
                "GLM_STATE_DIR": str(root / "state"),
                "GLM_RUNTIME_DIR": str(root / "runtime"),
                "SOUL_AUTONOMY_LEVEL": "2",
                "SOUL_AUTONOMY_MAX_LEVEL": "2",
            }
            fake_bot = SimpleNamespace()
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch.object(controller, "_disable_process_dumping"), \
                 mock.patch.object(
                     controller, "_nanobot_from_document",
                     return_value=fake_bot) as build, \
                 mock.patch.object(controller, "_restrict_nanobot_tools"), \
                 mock.patch.object(controller, "_redact_shell_tool_results"), \
                 mock.patch.object(controller, "_pin_soul_prompt"):
                instance = controller.SoulController(
                    boot, 8000, api_key="controller-only-secret",
                    start_probe_worker=False)
                result = asyncio.run(instance.get_bot(
                    "local-model", {"autonomyLevel": 2, "timezone": "UTC"}))
            self.assertIs(result, fake_bot)
            document = build.call_args.args[0]
            self.assertEqual(
                document["providers"]["custom"]["apiKey"],
                "controller-only-secret")
            self.assertFalse(
                (instance.paths["runtime"] / "nanobot-config.json").exists())
            self.assertNotIn("controller-only-secret",
                             json.dumps(sc.read_json(
                                 instance.paths["status"], {})))

    def test_direct_tls_uses_the_certificate_hostname_on_the_local_port(self):
        status = {
            "endpoint": "https://model-123.example.test:45678",
            "cert": "/workspace/.lego/certificates/model-123.example.test.crt",
            "keyfile": "/workspace/.lego/certificates/model-123.example.test.key",
        }
        self.assertEqual(
            controller._local_api_base(status, 8000),
            "https://model-123.example.test:8000")
        self.assertEqual(
            controller._local_api_base(
                {"endpoint": "https://pod-8000.proxy.runpod.net"}, 8000),
            "http://127.0.0.1:8000")
        runpod_status = {**status, "provider": "runpod"}
        self.assertEqual(
            controller._local_api_base(runpod_status, 8000),
            "http://127.0.0.1:8000")
        self.assertEqual(
            controller._tls_probe_target(
                runpod_status, "model-123.example.test", 45678, 8000),
            ("127.0.0.1", 8443, "model-123.example.test"),
        )
        self.assertEqual(
            controller._tls_probe_target(
                status, "model-123.example.test", 45678, 8000),
            ("model-123.example.test", 8000, "model-123.example.test"),
        )
        with tempfile.TemporaryDirectory() as raw:
            doc = controller._nanobot_document(
                Path(raw) / "workspace", "local-model", 8000, 1,
                "UTC", "https://model-123.example.test:8000")
            self.assertEqual(
                doc["providers"]["custom"]["apiBase"],
                "https://model-123.example.test:8000/v1")

    def test_nanobot_registry_is_shell_only_at_levels_two_and_three(self):
        class Registry:
            def __init__(self):
                self.tool_names = ["exec", "write_stdin", "spawn", "message", "my"]

            def unregister(self, name):
                self.tool_names.remove(name)

        bot = SimpleNamespace(_loop=SimpleNamespace(tools=Registry()))
        controller._restrict_nanobot_tools(bot, 2)
        self.assertEqual(bot._loop.tools.tool_names, ["exec", "write_stdin"])
        bot = SimpleNamespace(_loop=SimpleNamespace(tools=Registry()))
        controller._restrict_nanobot_tools(bot, 1)
        self.assertEqual(bot._loop.tools.tool_names, [])

    def test_deep_probe_keeps_api_key_inside_nondumpable_controller(self):
        with mock.patch("verify_serving.main", return_value=0) as run, \
                mock.patch("subprocess.Popen") as spawn, \
                mock.patch.object(controller, "_read_json",
                                  return_value={"ok": True}):
            result = controller._long_context_probe(
                8000, "deep-secret", "model", 524288, 32768,
                Path("/tmp/result"))
        self.assertTrue(result["ok"])
        arguments = run.call_args.args[0]
        self.assertEqual(
            arguments[arguments.index("--api-key") + 1], "deep-secret")
        spawn.assert_not_called()
        self.assertNotIn("SOUL_PROBE_API_KEY", os.environ)
        self.assertEqual(controller.LONG_PROBE_COOLDOWN_S, 6 * 3600)

    def test_deep_probe_worker_hardens_before_receiving_key_free_requests(self):
        class Connection:
            def __init__(self):
                self.requests = iter([
                    (8000, "model", 524288, 32768, "/tmp/result", ""),
                    None,
                ])
                self.sent = []
                self.closed = False

            def recv(self):
                return next(self.requests)

            def send(self, value):
                self.sent.append(value)

            def close(self):
                self.closed = True

        connection = Connection()
        with mock.patch.object(controller, "_disable_process_dumping") as harden, \
                mock.patch.object(controller, "_long_context_probe",
                                  return_value={"ok": True}) as probe:
            controller._deep_probe_worker(connection, "worker-only-secret")
        harden.assert_called_once_with()
        probe.assert_called_once_with(
            8000, "worker-only-secret", "model", 524288, 32768,
            Path("/tmp/result"), "")
        self.assertEqual(connection.sent, [{"ok": True}])
        self.assertTrue(connection.closed)

    def test_deep_probe_worker_timeout_is_hard_and_key_free(self):
        class Connection:
            def __init__(self):
                self.sent = []
                self.closed = False

            def send(self, value):
                self.sent.append(value)

            def poll(self, _timeout):
                return False

            def close(self):
                self.closed = True

        class Process:
            def __init__(self):
                self.alive = True
                self.terminated = False

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                del timeout

            def terminate(self):
                self.terminated = True
                self.alive = False

            def kill(self):
                self.alive = False

        instance = object.__new__(controller.SoulController)
        instance.port = 8000
        instance.api_key = "worker-only-secret"
        connection = Connection()
        process = Process()
        instance.deep_probe_connection = connection
        instance.deep_probe_process = process
        result = instance.run_deep_probe(
            "model", 524288, 32768, Path("/tmp/result"),
            "http://127.0.0.1:8000")
        self.assertIsNone(instance.deep_probe_connection)
        self.assertIsNone(instance.deep_probe_process)
        self.assertEqual(connection.sent[0], (
            8000, "model", 524288, 32768, "/tmp/result",
            "http://127.0.0.1:8000",
        ))
        self.assertNotIn("worker-only-secret", repr(connection.sent))
        self.assertTrue(connection.closed)
        self.assertTrue(process.terminated)
        self.assertFalse(result["ok"])
        self.assertIn(str(controller.RUN_TIMEOUT_S), result["error"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux fork")
    def test_real_deep_probe_worker_round_trip_and_shutdown(self):
        instance = object.__new__(controller.SoulController)
        instance.port = 8000
        instance.api_key = "worker-only-secret"
        instance.deep_probe_connection = None
        instance.deep_probe_process = None
        try:
            with mock.patch.object(
                    controller, "_long_context_probe",
                    return_value={"ok": True, "returncode": 0}):
                instance._start_deep_probe_worker()
                result = instance.run_deep_probe(
                    "model", 524288, 32768, Path("/tmp/result"),
                    "http://127.0.0.1:8000")
            self.assertEqual(result, {"ok": True, "returncode": 0})
            self.assertTrue(instance.deep_probe_process.is_alive())
        finally:
            instance._stop_deep_probe_worker()
        self.assertIsNone(instance.deep_probe_process)
        self.assertIsNone(instance.deep_probe_connection)

    def test_controller_requests_supervisor_restart_after_worker_loss(self):
        class DeadProcess:
            @staticmethod
            def is_alive():
                return False

            @staticmethod
            def join(timeout=None):
                del timeout

        instance = object.__new__(controller.SoulController)
        instance.stop = False
        instance.bot = None
        instance.bot_key = None
        instance.deep_probe_connection = None
        instance.deep_probe_process = DeadProcess()
        with mock.patch.object(
                controller.sc, "resolve", return_value=({}, {}, [])), \
             mock.patch.object(controller.sc, "audit"), \
             mock.patch.object(controller, "_atomic_status") as status, \
             mock.patch.object(controller.signal, "signal"):
            returncode = asyncio.run(instance.run())
        self.assertEqual(returncode, controller.CONTROLLER_RESTART_EXIT)
        self.assertEqual(status.call_args.kwargs["state"], "Error")
        self.assertIn("restart requested", status.call_args.kwargs["lastError"])

    def test_soul_marker_suppresses_duplicate_direct_analysis(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state"
            failure = state / "failures" / "20260727T000000Z"
            failure.mkdir(parents=True)
            env = {"GLM_STATE_DIR": str(state),
                   "GLM_RUNTIME_DIR": str(root / "runtime")}
            with mock.patch.dict(os.environ, env, clear=False), \
                    mock.patch("builtins.print"):
                self.assertEqual(config_cli.cmd_pending_analysis(None), 0)
                marker = state / "soul" / "incidents" / (
                    "rollback-" + failure.name + ".json")
                marker.parent.mkdir(parents=True)
                marker.write_text("{}")
                self.assertEqual(config_cli.cmd_pending_analysis(None), 3)

    def test_structured_output_repairs_trailing_comma_and_validates(self):
        record = controller._journal_record(
            '{"headline":"OK","summary":"healthy","observations":[],'
            '"suggestions":[],"confidence":0.9,}',
            kind="hourly", severity="info", incident_id=None, evidence=[],
            level=1, model="local", tools=[])
        self.assertEqual(record["headline"], "OK")
        self.assertEqual(record["analysisStatus"], "complete")
        self.assertEqual(record["schemaVersion"], 1)

    def test_malformed_output_is_escaped_unstructured_data(self):
        record = controller._journal_record(
            "<script>alert(1)</script> token=secret",
            kind="incident", severity="warning", incident_id="i", evidence=[],
            level=2, model="local", tools=["exec"])
        self.assertEqual(record["analysisStatus"], "unstructured")
        self.assertIn("<script>", record["summary"])
        self.assertNotIn("token=secret", record["summary"])

    def test_retention_removes_old_evidence_before_journal(self):
        with tempfile.TemporaryDirectory() as raw:
            env = {
                "GLM_STATE_DIR": str(Path(raw) / "state"),
                "GLM_RUNTIME_DIR": str(Path(raw) / "run"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                paths = sc.ensure_tree()
                old = paths["evidence"] / "old.json"
                old.write_text("x" * 100)
                stale = time.time() - 10 * 86400
                os.utime(old, (stale, stale))
                sc.append_jsonl(paths["journal"], {
                    "id": "journal", "timestamp": sc.utcnow(), "kind": "hourly"})
                config = sc.resolve(env, {
                    "evidenceRetentionDays": 7, "journalRetentionDays": 90,
                    "maxStateMB": 16,
                })[0]
                controller.enforce_retention(config)
                self.assertFalse(old.exists())
                self.assertTrue(paths["journal"].exists())

    def test_retention_bounds_active_log_and_compacts_closed_incidents(self):
        with tempfile.TemporaryDirectory() as raw:
            env = {
                "GLM_STATE_DIR": str(Path(raw) / "state"),
                "GLM_RUNTIME_DIR": str(Path(raw) / "run"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                paths = sc.ensure_tree()
                log = paths["logs"] / "controller.log"
                log.write_text("x" * (2 * 1024 * 1024) + "\nnewest-tail\n")
                incident_path = paths["incidents"] / "index.json"
                incident_path.write_text(json.dumps({
                    "checks": {"old": {"failures": 0, "incidentId": None}},
                    "incidents": {
                        "closed": {
                            "id": "closed", "state": "recovered",
                            "summary": "y" * (2 * 1024 * 1024),
                        },
                    },
                }))
                book = controller.IncidentBook(incident_path)
                controller.enforce_retention({
                    "evidenceRetentionDays": 7,
                    "journalRetentionDays": 90,
                    "maxStateMB": 1,
                }, book)
                total = sum(path.stat().st_size for path in paths["root"].glob("**/*")
                            if path.is_file())
                self.assertLessEqual(total, 1024 * 1024)
                self.assertIn("newest-tail", log.read_text())
                self.assertEqual(book.data, {"checks": {}, "incidents": {}})


class SoulSecureEraseTests(unittest.TestCase):
    def test_secure_erase_selects_entire_soul_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state"
            soul_file = state / "soul" / "workspace" / "sessions" / "journal.session"
            soul_file.parent.mkdir(parents=True)
            soul_file.write_text("private session")
            model = root / "model"
            model.mkdir()
            env = {
                "GLM_STATE_DIR": str(state),
                "GLM_RUNTIME_DIR": str(root / "runtime"),
                "MODEL_DIR": str(model),
                "ERASE_WORKSPACE": str(root / "workspace"),
                "ERASE_HOME": str(root / "home"),
                "ERASE_TMP": str(root / "tmp"),
                "ERASE_VARLOG": str(root / "varlog"),
                "ERASE_ETC_SSH": str(root / "etc-ssh"),
                "ERASE_EXTRA_ROOTS": "",
                "ERASE_CONFINE_TO": str(root),
            }
            for name in ("runtime", "workspace", "home", "tmp", "varlog", "etc-ssh"):
                (root / name).mkdir(exist_ok=True)
            with mock.patch.dict(os.environ, env, clear=False), \
                    mock.patch.object(secure_erase.gc, "effective_env",
                                      side_effect=lambda passed=None: dict(env)), \
                    mock.patch.object(secure_erase.gc, "state_dir",
                                      return_value=str(state)), \
                    mock.patch.object(secure_erase.gc, "runtime_dir",
                                      return_value=str(root / "runtime")):
                plan = secure_erase.plan(secure_erase.Paths(env))
            targets = {item["path"]: item for item in plan["targets"]}
            self.assertIn(str(soul_file), targets)
            self.assertEqual(targets[str(soul_file)]["group"], "soul-state")


class SoulWiringTests(unittest.TestCase):
    def test_image_and_pid1_keep_soul_isolated_from_vllm(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        entrypoint = (ROOT / "entrypoint.sh").read_text()
        landing = (ROOT / "landing.py").read_text()
        controller_source = (SCRIPTS / "soul_controller.py").read_text()
        launcher_source = (SCRIPTS / "soul_launcher.py").read_text()
        start_soul = entrypoint.split("start_soul() {", 1)[1].split(
            "stop_soul() {", 1)[0]
        self.assertIn("/opt/nanobot-venv/bin/pip", dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("diff -u /tmp/vllm-packages.before", dockerfile)
        self.assertIn("EXPOSE 22 8000 8443 1111\n", dockerfile)
        self.assertIn("soul_launcher.py", start_soul)
        self.assertIn("SOUL_ROOT_API_KEY=", start_soul)
        self.assertNotIn("SOUL_API_KEY_FD=", start_soul)
        self.assertNotIn('VLLM_API_KEY="${VLLM_API_KEY:-}"', start_soul)
        self.assertIn("SOUL_KEY_READY_FD", launcher_source)
        self.assertIn("select.select(", launcher_source)
        self.assertIn("_kill_uid(uid, signal.SIGKILL)", launcher_source)
        self.assertIn("pass_fds=(key_read, ready_write)", launcher_source)
        self.assertIn("start_new_session=True", launcher_source)
        self.assertNotIn("nanobot-config.json", controller_source)
        self.assertIn("_disable_process_dumping()", controller_source)
        self.assertIn("if [ \"$(soul_level)\" -gt 0 ]", entrypoint)
        self.assertIn("stop_soul_supervisor\n      kill_server_tree", entrypoint)
        self.assertIn("SOUL: queued rollback evidence", entrypoint)
        self.assertIn("analyze_failure.py", entrypoint)
        self.assertIn("TCPServer.server_bind(self)", landing)

    def test_root_launcher_waits_for_hardening_and_accepts_max_key(self):
        created = []
        secret = "k" * launcher.MAX_API_KEY_BYTES

        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.command = command
                self.env = kwargs["env"]
                self.pid = 424242
                self.returncode = None
                self.received = b""
                key_fd = os.dup(kwargs["pass_fds"][0])
                ready_fd = os.dup(kwargs["pass_fds"][1])

                def controller_child():
                    os.write(ready_fd, b"R")
                    os.close(ready_fd)
                    chunks = []
                    while True:
                        chunk = os.read(key_fd, 4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    os.close(key_fd)
                    self.received = b"".join(chunks)
                    self.returncode = 0

                self.thread = __import__("threading").Thread(
                    target=controller_child)
                self.thread.start()
                created.append(self)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.thread.join(timeout)
                if self.thread.is_alive():
                    raise subprocess.TimeoutExpired(self.command, timeout)
                return self.returncode

        with mock.patch.dict(
                os.environ,
                {"SOUL_ROOT_API_KEY": secret,
                 "HOME": "/state/soul/workspace",
                 "PATH": "/usr/bin:/bin"}, clear=True), \
             mock.patch.object(launcher.os, "geteuid", return_value=0), \
             mock.patch.object(
                 launcher.pwd, "getpwnam",
                 return_value=SimpleNamespace(pw_uid=45678)), \
             mock.patch.object(launcher, "_kill_uid") as kill_uid, \
             mock.patch.object(
                 launcher.subprocess, "Popen", side_effect=FakeProcess), \
             mock.patch.object(launcher.signal, "signal"):
            rc = launcher.main([
                "--user", "soul", "--", "python", "controller.py"])

        self.assertEqual(rc, 0)
        self.assertEqual(
            created[0].received,
            len(secret.encode()).to_bytes(4, "big") + secret.encode())
        self.assertNotIn("SOUL_ROOT_API_KEY", created[0].env)
        self.assertNotIn(secret, repr(created[0].env))
        self.assertIn("SOUL_API_KEY_FD", created[0].env)
        self.assertIn("SOUL_KEY_READY_FD", created[0].env)
        self.assertEqual(created[0].command[:11], [
            "nice", "-n", "10", "ionice", "-c", "3", "runuser",
            "-u", "soul", "--preserve-environment", "--",
        ])
        self.assertEqual(created[0].env["HOME"], "/state/soul/workspace")
        self.assertGreaterEqual(kill_uid.call_count, 2)

    def test_root_launcher_cancels_post_spawn_signal_before_key_handoff(self):
        created = []

        class InterruptedProcess:
            def __init__(self, _command, **kwargs):
                self.env = kwargs["env"]
                self.pid = 424242
                self.returncode = None
                created.append(self)
                launcher._STOP_SIGNAL = launcher.signal.SIGTERM

            def poll(self):
                return self.returncode

        with mock.patch.dict(
                os.environ,
                {"SOUL_ROOT_API_KEY": "must-not-be-written",
                 "PATH": "/usr/bin:/bin"}, clear=True), \
             mock.patch.object(launcher.os, "geteuid", return_value=0), \
             mock.patch.object(
                 launcher.pwd, "getpwnam",
                 return_value=SimpleNamespace(pw_uid=45678)), \
             mock.patch.object(launcher, "_kill_uid"), \
             mock.patch.object(launcher, "_signal_child") as stop_child, \
             mock.patch.object(launcher, "_write_all") as write_key, \
             mock.patch.object(
                 launcher.subprocess, "Popen",
                 side_effect=InterruptedProcess), \
             mock.patch.object(launcher.signal, "signal"):
            rc = launcher.main([
                "--user", "soul", "--", "python", "controller.py"])

        self.assertEqual(rc, 128 + launcher.signal.SIGTERM)
        write_key.assert_not_called()
        stop_child.assert_any_call(launcher.signal.SIGTERM)
        self.assertNotIn("SOUL_ROOT_API_KEY", created[0].env)


    def test_uid_reap_fails_closed_while_any_active_process_remains(self):
        with mock.patch.object(
                launcher, "_uid_pids",
                side_effect=[[101], [102], [103], [104]]), \
             mock.patch.object(launcher.os, "kill"), \
             mock.patch.object(launcher.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "could not reap"):
                launcher._kill_uid(45678, launcher.signal.SIGKILL)

class SoulLandingTests(unittest.TestCase):
    def start_landing(self, root, token=""):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        state = root / "state"
        runtime = root / "runtime"
        state.mkdir(exist_ok=True)
        runtime.mkdir(exist_ok=True)
        status = root / "boot.json"
        status.write_text(json.dumps({
            "phase": "serving", "endpoint": "", "served_model": "model",
            "tls": "not configured", "model_dir": str(root / "model"),
        }))
        env = dict(os.environ)
        env.update({
            "GLM_STATE_DIR": str(state),
            "GLM_RUNTIME_DIR": str(runtime),
            "GLM_SCRIPTS_DIR": str(SCRIPTS),
            "STATUS_FILE": str(status),
            "LANDING_PORT": str(port),
            "LANDING_ALLOW_INSECURE": "1",
            "SOUL_AUTONOMY_LEVEL": "1",
            "SOUL_AUTONOMY_MAX_LEVEL": "2",
            "OPEN_BUTTON_TOKEN": token,
        })
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "landing.py")], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        base = f"http://127.0.0.1:{port}"
        ready_url = base + ("/?token=" + urllib.parse.quote(token) if token else "/")
        for _ in range(50):
            try:
                urllib.request.urlopen(ready_url, timeout=.2).read()
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.05)
        else:
            error = proc.stderr.read().decode() if proc.poll() is not None else "timeout"
            proc.terminate()
            raise AssertionError("landing page did not start: " + error)
        return proc, base, env

    def stop_landing(self, proc):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stderr is not None:
            proc.stderr.close()

    def test_no_token_shows_sanitized_card_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = {
                "GLM_STATE_DIR": str(root / "state"),
                "GLM_RUNTIME_DIR": str(root / "runtime"),
            }
            sc.ensure_tree(env)
            sc.atomic_json(sc.paths(env)["status"], {
                "state": "Observing",
                "latestHeadline": "<script>alert(1)</script> token=secret",
            })
            proc, base, _ = self.start_landing(root)
            try:
                page = urllib.request.urlopen(base + "/", timeout=2).read().decode()
                self.assertIn("Appliance SOUL", page)
                self.assertNotIn("<script>alert(1)</script>", page)
                self.assertNotIn("token=secret", page)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/soul", timeout=2)
                self.assertEqual(caught.exception.code, 403)
            finally:
                self.stop_landing(proc)

    def test_soul_resolution_failure_returns_degraded_status(self):
        with mock.patch.object(
                landing.soul, "resolve",
                side_effect=ValueError("damaged config")):
            doc = landing.soul_status()
        self.assertEqual(doc["state"], "Disabled")
        self.assertFalse(doc["controllerHealthy"])
        self.assertEqual(doc["level"], 0)
        self.assertIn("could not be resolved", doc["configNotes"][0])

    def test_profile_form_switch_drops_untouched_old_variant_defaults(self):
        old, _sources, _notes = landing.gc.resolve(
            state_values={"MODEL_VARIANT": "exl3-tr3-3.42bpw"},
            env_values={"GLM_GPU_COUNT": "4"},
        )
        submitted = {
            knob["key"]: old[knob["key"]]
            for knob in landing.gc.KNOBS
            if knob.get("editable", True)
        }
        submitted["MODEL_VARIANT"] = "exl3-tr3-glm53-3.42bpw"
        submitted["MAX_NUM_SEQS"] = 4
        filtered = landing._profile_switch_form_values(submitted, old)
        with mock.patch.object(
                landing.gc, "load_startup_env",
                return_value={"GLM_GPU_COUNT": "4"}):
            minimal = landing.gc.minimize(filtered)
        switched, _sources, _notes = landing.gc.resolve(
            state_values=minimal, env_values={"GLM_GPU_COUNT": "4"})
        self.assertNotIn("MAX_MODEL_LEN", minimal)
        self.assertNotIn("KV_CACHE_MEMORY_BYTES", minimal)
        self.assertNotIn("SERVED_MODEL_NAME", minimal)
        self.assertEqual(minimal["MAX_NUM_SEQS"], 4)
        self.assertEqual(switched["MAX_MODEL_LEN"], 393216)
        self.assertEqual(switched["KV_CACHE_MEMORY_BYTES"], 3415867392)
        self.assertEqual(switched["SERVED_MODEL_NAME"], "GLM-5.3")

    def test_profile_apply_rejects_a_stale_rendered_baseline(self):
        rendered, _sources, _notes = landing.gc.resolve(
            state_values={"MODEL_VARIANT": "exl3-tr3-3.42bpw"},
            env_values={"GLM_GPU_COUNT": "4"},
        )
        changed = dict(rendered)
        changed["MAX_NUM_SEQS"] = 4
        revision = landing._config_revision(rendered)
        with mock.patch.object(
                landing.gc, "resolve", return_value=(changed, {}, [])), \
             mock.patch.object(
                 landing.gc, "state_lock", return_value=mock.MagicMock()), \
             mock.patch.object(landing.gc, "write_json_atomic") as write_state:
            ok, findings, message = landing.apply_values(
                {"MODEL_VARIANT": "exl3-tr3-glm53-3.42bpw"},
                form_submission=True,
                form_revision=revision,
            )
        self.assertFalse(ok)
        self.assertEqual(findings[0]["id"], "stale-form")
        self.assertIn("stale", message)
        write_state.assert_not_called()


    def test_correctness_reports_structured_output_failure_before_long_context(self):
        verdict = {
            "health": True,
            "short_prompt": {"ok": True},
            "structured_output": {"ok": False, "detail": "grammar rejected"},
            "stochastic_sampling": {"attempted": False, "ok": False},
            "long_context": {"attempted": True, "ok": True},
        }
        with mock.patch.object(landing, "verify_last", return_value=verdict):
            headline, ok, detail = landing.correctness()
        self.assertFalse(ok)
        self.assertEqual(headline, "STRUCTURED-OUTPUT PROBE FAILED")
        self.assertEqual(detail, "grammar rejected")

    def test_correctness_reports_stochastic_failure_before_long_context(self):
        verdict = {
            "health": True,
            "short_prompt": {"ok": True},
            "structured_output": {"ok": True},
            "stochastic_sampling": {
                "attempted": True,
                "ok": False,
                "detail": "decode stopped at 19 tokens",
            },
            "long_context": {"attempted": True, "ok": True},
        }
        with mock.patch.object(landing, "verify_last", return_value=verdict):
            headline, ok, detail = landing.correctness()
        self.assertFalse(ok)
        self.assertEqual(headline, "TEMPERATURE-1 SAMPLING PROBE FAILED")
        self.assertEqual(detail, "decode stopped at 19 tokens")

    def test_omp_reasoning_capability_follows_effective_serve_argv(self):
        qwen, _sources, _notes = landing.gc.resolve(
            state_values={"MODEL_FAMILY": "qwen36"},
            env_values={"GLM_GPU_COUNT": "1"},
        )
        custom, _sources, _notes = landing.gc.resolve(
            state_values={"MODEL_FAMILY": "custom"},
            env_values={"GLM_GPU_COUNT": "1", "MODEL_ID": "org/model"},
        )
        with mock.patch.object(
                landing.gc, "resolve", return_value=(qwen, {}, [])):
            self.assertEqual(landing.deployment()["reasoning"], "true")
        with mock.patch.object(
                landing.gc, "resolve", return_value=(custom, {}, [])):
            self.assertEqual(landing.deployment()["reasoning"], "false")

    def test_token_gates_journal_and_ceiling_is_enforced_without_vllm_restart(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            proc, base, env = self.start_landing(root, token="capability")
            try:
                soul_root = sc.ensure_tree(env)
                sc.append_jsonl(soul_root["journal"], {
                    "schemaVersion": 1, "id": "xss", "timestamp": sc.utcnow(),
                    "kind": "incident", "severity": "warning",
                    "headline": "<img src=x onerror=alert(1)>",
                    "summary": "<script>alert(1)</script>", "observations": [],
                    "suggestions": [], "evidenceRefs": [],
                })
                page = urllib.request.urlopen(
                    base + "/soul?token=capability", timeout=2).read().decode()
                self.assertNotIn("<script>alert(1)</script>", page)
                self.assertIn("&lt;script&gt;", page)
                self.assertIn("1 Observe:", page)
                self.assertIn("2 Investigate:", page)
                self.assertIn("3 Verify:", page)
                data = urllib.parse.urlencode({
                    "token": "capability", "autonomyLevel": "3",
                    "heartbeatIntervalS": "60", "journalIntervalS": "600",
                    "timezone": "UTC",
                }).encode()
                request = urllib.request.Request(
                    base + "/soul/config?token=capability", data=data, method="POST")
                urllib.request.urlopen(request, timeout=2).read()
                effective = sc.resolve(env)[0]
                self.assertEqual(effective["requestedLevel"], 3)
                self.assertEqual(effective["autonomyLevel"], 2)
                self.assertFalse((Path(env["GLM_RUNTIME_DIR"]) / "restart-request").exists())
                status = json.loads(urllib.request.urlopen(
                    base + "/soul/status?token=capability", timeout=2).read())
                self.assertEqual(status["maximumLevel"], 2)
            finally:
                self.stop_landing(proc)

    def test_bare_runpod_proxy_url_is_safe_read_only_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            proc, base, _env = self.start_landing(root, token="capability")
            try:
                status_path = root / "boot.json"
                status = json.loads(status_path.read_text())
                status.update({
                    "endpoint": "https://pod-8000.proxy.runpod.net",
                    "api_key": "sk-must-not-leak",
                    "provider": "runpod",
                })
                status_path.write_text(json.dumps(status))
                page = urllib.request.urlopen(base + "/", timeout=2).read().decode()
                self.assertIn("Read-only status", page)
                self.assertIn("The appliance is reachable", page)
                self.assertNotIn("sk-must-not-leak", page)
                self.assertNotIn('href="/config?token=', page)
                self.assertNotIn('href="/chat?token=', page)
                self.assertNotIn('href="/terminate?token=', page)
                self.assertNotIn("METRIC_NAMES", page)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(
                        base + "/?token=incorrect", timeout=2)
                self.assertEqual(caught.exception.code, 403)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/config", timeout=2)
                self.assertEqual(caught.exception.code, 403)
                full = urllib.request.urlopen(
                    base + "/?token=capability", timeout=2).read().decode()
                self.assertIn("sk-must-not-leak", full)
            finally:
                self.stop_landing(proc)

    def test_home_has_one_configuration_link_and_no_stray_correctness_detail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            proc, base, env = self.start_landing(root, token="capability")
            try:
                status = json.loads((root / "boot.json").read_text())
                status.update({
                    "endpoint": "https://model.example.test:12345",
                    "api_key": "not-rendered-over-http",
                })
                (root / "boot.json").write_text(json.dumps(status))
                state = Path(env["GLM_STATE_DIR"])
                (state / "verify-last.json").write_text(json.dumps({
                    "health": True,
                    "short_prompt": {"ok": True},
                    "long_context": {
                        "attempted": True, "ok": True, "found": 3,
                        "total": 3, "tokens": 32773,
                    },
                    "reason": "STRAY-CORRECTNESS-DETAIL",
                }))
                page = urllib.request.urlopen(
                    base + "/?token=capability", timeout=2).read().decode()
                self.assertEqual(
                    page.count('href="/config?token=capability"'), 1)
                self.assertIn("Change model, draft, context, concurrency", page)
                self.assertNotIn("<b>Configure &rarr;</b>", page)
                self.assertNotIn("STRAY-CORRECTNESS-DETAIL", page)
            finally:
                self.stop_landing(proc)


class RedactShellToolResultsTests(unittest.TestCase):
    """Test _redact_shell_tool_results without requiring the nanobot package.

    The function lazily imports ToolResult inside its body, so we mock that
    import path and build a minimal fake bot with fake tools to verify the
    redaction wrapping is applied correctly."""

    def test_shell_tool_output_is_redacted(self):
        """A secret in a tool's output is scrubbed before reaching the model."""
        import asyncio

        async def fake_execute(*args, **kwargs):
            return "api_key=sk-secret123456789 token=bearer_xyz"

        class FakeTool:
            def __init__(self):
                self.execute = fake_execute

        class FakeRegistry:
            def __init__(self):
                self._tools = {
                    "exec": FakeTool(),
                    "list_exec_sessions": FakeTool(),
                    "write_stdin": FakeTool(),
                }

            def get(self, name):
                return self._tools.get(name)

        class FakeBot:
            def __init__(self):
                self._loop = SimpleNamespace(tools=FakeRegistry())

        # Mock the nanobot import that happens inside _redact_shell_tool_results
        class FakeToolResult:
            def __init__(self, content, is_error=False):
                self.content = content
                self.is_error = is_error

            def __str__(self):
                return str(self.content)

        with mock.patch.dict(sys.modules, {
            "nanobot": mock.MagicMock(),
            "nanobot.agent": mock.MagicMock(),
            "nanobot.agent.tools": mock.MagicMock(),
            "nanobot.agent.tools.base": mock.MagicMock(ToolResult=FakeToolResult),
        }):
            bot = FakeBot()
            controller._redact_shell_tool_results(bot)

            # All three shell tools should now have wrapped execute methods
            for name in ("exec", "list_exec_sessions", "write_stdin"):
                tool = bot._loop.tools.get(name)
                self.assertIsNot(tool.execute, fake_execute,
                                 f"{name} execute was not wrapped")
                result = asyncio.run(tool.execute())
                self.assertNotIn("sk-secret123456789", result,
                                  f"secret leaked through {name} redaction")
                self.assertNotIn("bearer_xyz", result,
                                  f"token leaked through {name} redaction")

    def test_toolresult_preserves_error_flag(self):
        """When the original returns a ToolResult, the redacted result keeps is_error."""
        import asyncio

        class FakeToolResult:
            def __init__(self, content, is_error=False):
                self.content = content
                self.is_error = is_error

            def __str__(self):
                return str(self.content)

        async def fake_execute(*args, **kwargs):
            return FakeToolResult("api_key=sk-leak password=hunter2",
                                  is_error=True)

        class FakeTool:
            def __init__(self):
                self.execute = fake_execute

        class FakeRegistry:
            def __init__(self):
                self._tools = {"exec": FakeTool()}

            def get(self, name):
                return self._tools.get(name)

        class FakeBot:
            def __init__(self):
                self._loop = SimpleNamespace(tools=FakeRegistry())

        with mock.patch.dict(sys.modules, {
            "nanobot": mock.MagicMock(),
            "nanobot.agent": mock.MagicMock(),
            "nanobot.agent.tools": mock.MagicMock(),
            "nanobot.agent.tools.base": mock.MagicMock(ToolResult=FakeToolResult),
        }):
            bot = FakeBot()
            controller._redact_shell_tool_results(bot)
            tool = bot._loop.tools.get("exec")
            result = asyncio.run(tool.execute())
            self.assertIsInstance(result, FakeToolResult)
            self.assertTrue(result.is_error)
            self.assertNotIn("sk-leak", str(result))
            self.assertNotIn("hunter2", str(result))

    def test_missing_tools_are_skipped(self):
        """Tools that don't exist (None) are silently skipped, not an error."""

        class FakeRegistry:
            def __init__(self):
                self._tools = {}  # no tools registered

            def get(self, name):
                return self._tools.get(name)

        class FakeBot:
            def __init__(self):
                self._loop = SimpleNamespace(tools=FakeRegistry())

        with mock.patch.dict(sys.modules, {
            "nanobot": mock.MagicMock(),
            "nanobot.agent": mock.MagicMock(),
            "nanobot.agent.tools": mock.MagicMock(),
            "nanobot.agent.tools.base": mock.MagicMock(),
        }):
            bot = FakeBot()
            # Should not raise
            controller._redact_shell_tool_results(bot)


@unittest.skipUnless(importlib.util.find_spec("nanobot"),
                     "Nanobot is installed only in the image's isolated venv")
class SoulSdkIntegrationTests(unittest.TestCase):
    def test_level_one_uses_real_sdk_against_fake_openai_endpoint(self):
        class FakeOpenAI(BaseHTTPRequestHandler):
            calls = []

            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/health":
                    body, ctype = b"{}", "application/json"
                elif self.path == "/v1/models":
                    body, ctype = json.dumps({
                        "object": "list",
                        "data": [{"id": "fake-model", "object": "model"}],
                    }).encode(), "application/json"
                elif self.path == "/metrics":
                    body, ctype = b"vllm:num_requests_running 0\n", "text/plain"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                self.__class__.calls.append(request)
                content = json.dumps({
                    "headline": "All checks nominal",
                    "summary": "The fake appliance is serving.",
                    "observations": ["health and model discovery passed"],
                    "suggestions": [],
                    "confidence": 0.99,
                })
                body = json.dumps({
                    "id": "chatcmpl-fake", "object": "chat.completion",
                    "created": int(time.time()), "model": "fake-model",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": content,
                    }, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                              "total_tokens": 130},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state"
            runtime = root / "runtime"
            boot = root / "boot.json"
            boot.write_text(json.dumps({
                "phase": "serving", "served_model": "fake-model",
                "model_family": "fake", "auth": "key",
                "port": server.server_port, "endpoint": "",
            }))
            env = {
                "GLM_STATE_DIR": str(state), "GLM_RUNTIME_DIR": str(runtime),
                "SOUL_AUTONOMY_LEVEL": "1", "SOUL_AUTONOMY_MAX_LEVEL": "1",
                "SOUL_HEARTBEAT_INTERVAL_S": "30",
                "SOUL_JOURNAL_INTERVAL_S": "300", "SOUL_TIMEZONE": "UTC",
                "SOUL_PROMPT": str(ROOT / "soul" / "SOUL.md"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                instance = controller.SoulController(
                    boot, server.server_port, api_key="fake-key",
                    start_probe_worker=False)
                (instance.paths["workspace"] / "AGENTS.md").write_text(
                    "INJECTED-WORKSPACE-INSTRUCTION")
                asyncio.run(instance.one_cycle())
                asyncio.run(instance.close_bot())
                entries = sc.read_journal(env=env)
            persisted = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in instance.paths["root"].rglob("*")
                if path.is_file()
            )
            self.assertNotIn("fake-key", persisted)
            self.assertFalse(
                (instance.paths["runtime"] / "nanobot-config.json").exists())
            self.assertEqual(entries[0]["headline"], "All checks nominal")
            self.assertEqual(entries[0]["toolsUsed"], [])
            self.assertTrue(FakeOpenAI.calls)
            self.assertNotIn("tools", FakeOpenAI.calls[0])
            self.assertNotIn("INJECTED-WORKSPACE-INSTRUCTION",
                             json.dumps(FakeOpenAI.calls[0]))
            self.assertIn("Appliance SOUL", json.dumps(FakeOpenAI.calls[0]))
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
