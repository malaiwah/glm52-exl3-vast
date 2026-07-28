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
sys.path.insert(0, str(SCRIPTS))

import soul_config as sc  # noqa: E402
import soul_controller as controller  # noqa: E402
import secure_erase  # noqa: E402
import config_cli  # noqa: E402


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

        with mock.patch.object(controller.urllib.request, "urlopen",
                               return_value=Response("not SOUL-CANARY-OK")):
            self.assertFalse(controller._cheap_canary(8000, "", "local")["ok"])
        with mock.patch.object(controller.urllib.request, "urlopen",
                               return_value=Response("\nSOUL-CANARY-OK \n")):
            self.assertTrue(controller._cheap_canary(8000, "", "local")["ok"])

    def test_nanobot_config_disables_dream_and_level_one_shell(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "config.json"
            controller._nanobot_config(path, root / "workspace", "local-model",
                                       8000, 1, "UTC")
            doc = json.loads(path.read_text())
            self.assertFalse(doc["agents"]["defaults"]["dream"]["enabled"])
            self.assertFalse(doc["tools"]["exec"]["enable"])
            self.assertEqual(doc["providers"]["custom"]["apiKey"], "${VLLM_API_KEY}")
            self.assertNotIn("actual-secret", path.read_text())
            controller._nanobot_config(path, root / "workspace", "local-model",
                                       8000, 2, "UTC")
            self.assertTrue(json.loads(path.read_text())["tools"]["exec"]["enable"])

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

    def test_deep_probe_keeps_api_key_out_of_process_arguments(self):
        observed = {}

        class FakeProcess:
            returncode = 0

            def communicate(self, timeout=None):
                observed["timeout"] = timeout
                return "", None

            def kill(self):
                pass

        def fake_popen(command, **kwargs):
            observed["command"] = command
            observed["env"] = kwargs["env"]
            return FakeProcess()

        with mock.patch("subprocess.Popen", side_effect=fake_popen), \
                mock.patch.object(controller, "_read_json",
                                  return_value={"ok": True}):
            result = controller._long_context_probe(
                8000, "deep-secret", "model", 524288, 32768, Path("/tmp/result"))
        self.assertTrue(result["ok"])
        self.assertNotIn("deep-secret", observed["command"])
        self.assertEqual(observed["env"]["SOUL_PROBE_API_KEY"], "deep-secret")
        self.assertEqual(observed["timeout"], 600)
        self.assertEqual(controller.LONG_PROBE_COOLDOWN_S, 6 * 3600)

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
        self.assertIn("/opt/nanobot-venv/bin/pip", dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("diff -u /tmp/vllm-packages.before", dockerfile)
        self.assertIn("EXPOSE 22 8000 8443 1111\n", dockerfile)
        self.assertIn("runuser -u soul -- env -i", entrypoint)
        self.assertIn("if [ \"$(soul_level)\" -gt 0 ]", entrypoint)
        self.assertIn("stop_soul_supervisor\n      kill_server_tree", entrypoint)
        self.assertIn("SOUL: queued rollback evidence", entrypoint)
        self.assertIn("analyze_failure.py", entrypoint)


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
                "VLLM_API_KEY": "fake-key",
                "SOUL_PROMPT": str(ROOT / "soul" / "SOUL.md"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                instance = controller.SoulController(boot, server.server_port)
                (instance.paths["workspace"] / "AGENTS.md").write_text(
                    "INJECTED-WORKSPACE-INSTRUCTION")
                asyncio.run(instance.one_cycle())
                asyncio.run(instance.close_bot())
                entries = sc.read_journal(env=env)
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
