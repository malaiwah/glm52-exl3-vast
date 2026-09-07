"""HTTP capability boundaries and supervisor/dashboard configuration transactions."""
import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config_cli
import glm_config as gc
import landing


class ConfigFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {
            "GLM_STATE_DIR": str(self.root / "state"),
            "GLM_RUNTIME_DIR": str(self.root / "runtime"),
            "GLM_GPU_COUNT": "4",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        gc.snapshot_startup_env({"GLM_GPU_COUNT": "4"})
        self.args = SimpleNamespace(log="", reason="failed probe")

    def state(self, **values):
        gc.write_json_atomic(gc.p_state(), {"values": values})

    def begin(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(config_cli.main(["env", "--begin-attempt"]), 0)
        exports = {}
        for line in output.getvalue().splitlines():
            if line.startswith("export "):
                key, value = line[7:].split("=", 1)
                exports[key] = shlex.split(value)[0] if shlex.split(value) else ""
        os.environ["GLM_CONFIG_ATTEMPT"] = exports["GLM_CONFIG_ATTEMPT"]
        return exports


class ConfigTransactionTests(ConfigFixture, unittest.TestCase):
    def test_older_boot_cannot_rollback_or_mark_new_apply_good(self):
        self.state(SERVED_MODEL_NAME="boot-A")
        self.begin()
        ok, findings, _ = landing.apply_values({"SERVED_MODEL_NAME": "pending-B"})
        self.assertTrue(ok, findings)
        before = Path(gc.p_state()).read_bytes()
        apply_state = Path(gc.p_apply_state()).read_bytes()
        self.assertEqual(config_cli.cmd_should_rollback(self.args), 3)
        self.assertEqual(config_cli.cmd_validate(SimpleNamespace(quiet=True)), 3)
        self.assertEqual(config_cli.cmd_rollback(self.args), 3)
        self.assertEqual(config_cli.cmd_mark_good(self.args), 3)
        self.assertEqual(Path(gc.p_state()).read_bytes(), before)
        self.assertEqual(Path(gc.p_apply_state()).read_bytes(), apply_state)
        self.assertFalse(Path(gc.p_known_good()).exists())
        self.assertFalse(Path(gc.p_failures()).exists())
        self.assertTrue(Path(gc.p_restart_flag()).exists())

    def test_new_attempt_of_same_state_invalidates_old_completion(self):
        self.state(SERVED_MODEL_NAME="same-config")
        first = self.begin()["GLM_CONFIG_ATTEMPT"]
        second = self.begin()["GLM_CONFIG_ATTEMPT"]
        os.environ["GLM_CONFIG_ATTEMPT"] = first
        self.assertEqual(config_cli.cmd_mark_good(self.args), 3)
        self.assertFalse(Path(gc.p_known_good()).exists())
        os.environ["GLM_CONFIG_ATTEMPT"] = second
        self.assertEqual(config_cli.cmd_mark_good(self.args), 0)
        self.assertEqual(gc.read_json(gc.p_known_good())["effective"]["SERVED_MODEL_NAME"],
                         "same-config")

    def test_mark_good_records_exact_configuration_exported_to_engine(self):
        gc.snapshot_startup_env({"GLM_GPU_COUNT": "4", "SERVED_MODEL_NAME": "launched"})
        exported = self.begin()
        # Completion must not re-resolve even if an external operator replaces
        # the startup snapshot while this attempt is running.
        gc.snapshot_startup_env({"GLM_GPU_COUNT": "4", "SERVED_MODEL_NAME": "not-launched"})
        self.assertEqual(config_cli.cmd_mark_good(self.args), 0)
        good = gc.read_json(gc.p_known_good())
        self.assertEqual(good["effective"]["SERVED_MODEL_NAME"], exported["SERVED_MODEL_NAME"])
        self.assertEqual(good["sources"], json.loads(exported["GLM_CONFIG_SOURCES"]))

    def test_missing_attempt_fails_closed_but_manual_rollback_remains_available(self):
        self.state(SERVED_MODEL_NAME="unbootable")
        os.environ["GLM_CONFIG_ATTEMPT"] = "missing-attempt"
        self.assertEqual(config_cli.cmd_rollback(self.args), 3)
        self.assertEqual(config_cli.cmd_validate(SimpleNamespace(quiet=True)), 3)
        self.assertTrue(Path(gc.p_state()).exists())
        del os.environ["GLM_CONFIG_ATTEMPT"]
        self.assertEqual(config_cli.cmd_rollback(self.args), 0)
        self.assertFalse(Path(gc.p_state()).exists())
        self.assertEqual(gc.apply_state()["mode"], "rolled-back")

    def test_reset_rejects_invalid_template_without_destroying_working_override(self):
        gc.snapshot_startup_env({"GLM_GPU_COUNT": "4", "CUDAGRAPH_CAPTURE_SIZES": "1,2"})
        self.state(CUDAGRAPH_CAPTURE_SIZES="4,8,16,32,64,128")
        before = Path(gc.p_state()).read_bytes()
        revision = landing._config_revision(gc.resolve()[0])
        ok, findings, _ = landing.apply_values({}, reset=True, form_revision=revision)
        self.assertFalse(ok)
        self.assertIn("capture-sizes-floor", {finding["id"] for finding in findings})
        self.assertEqual(Path(gc.p_state()).read_bytes(), before)
        self.assertFalse(Path(gc.p_restart_flag()).exists())

    def test_reset_validates_then_removes_state_and_requests_restart(self):
        self.state(SERVED_MODEL_NAME="override")
        revision = landing._config_revision(gc.resolve()[0])
        ok, findings, _ = landing.apply_values({}, reset=True, form_revision=revision)
        self.assertTrue(ok, findings)
        self.assertFalse(Path(gc.p_state()).exists())
        self.assertTrue(Path(gc.p_restart_flag()).exists())
        self.assertEqual(gc.apply_state()["mode"], "trial")


class LandingHTTPTests(ConfigFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in {
            "TOKEN": "dashboard-capability", "ALLOW_INSECURE": False,
            "TRUST_PROXY_HTTPS": False, "soul": None,
        }.items():
            self.stack.enter_context(mock.patch.object(landing, name, value))
        self.status = {"endpoint": "https://api.example", "api_key": "private-api-key",
                       "phase": "serving", "served_model": "model"}
        self.stack.enter_context(mock.patch.object(landing, "status", side_effect=lambda: self.status))
        self.stack.enter_context(mock.patch.object(landing, "engine_state", return_value="serving"))
        self.stack.enter_context(mock.patch.object(landing, "weights_state", return_value="ready"))
        self.stack.enter_context(mock.patch.object(landing, "ssl_ctx", return_value=None))
        self.server = landing.DualProtocolServer(("127.0.0.1", 0), landing.Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)

    def request(self, path, form=None, headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            body = None if form is None else urlencode(form)
            all_headers = dict(headers or {})
            if form is not None:
                all_headers["Content-Type"] = "application/x-www-form-urlencoded"
            conn.request("GET" if form is None else "POST", path, body, all_headers)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            conn.close()

    def test_tls_redirect_preserves_duplicate_blank_and_encoded_query_parameters(self):
        self.status["https_hostport"] = "dashboard.example:1111"
        query = "token=dashboard-capability&before=a%2Bb%26c&tag=one&tag=two&empty="
        with mock.patch.object(landing, "ssl_ctx", return_value=object()):
            status, headers, _ = self.request("/soul?" + query)
        self.assertEqual(status, 302)
        self.assertEqual(headers["Location"], "https://dashboard.example:1111/soul?" + query)

    def test_proxy_opt_out_and_token_gate_both_protect_api_key(self):
        path = "/chat?token=dashboard-capability"
        status, _, body = self.request(path, headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(status, 403)
        self.assertNotIn("private-api-key", body)
        with mock.patch.object(landing, "TRUST_PROXY_HTTPS", True):
            status, _, body = self.request("/chat", headers={"X-Forwarded-Proto": "https"})
            self.assertEqual(status, 403)
            self.assertNotIn("private-api-key", body)
            status, headers, body = self.request(path, headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(status, 200)
        self.assertIn("private-api-key", body)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")

    def test_optional_redactor_failure_does_not_expose_boot_log_credentials(self):
        Path(gc.runtime_dir(), "boot-notes.log").write_text("secret-provider-credential\n")
        for path in ("/", "/?token=dashboard-capability"):
            status, _, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertNotIn("secret-provider-credential", body)
        with mock.patch.object(landing, "ALLOW_INSECURE", True):
            status, _, body = self.request("/?token=dashboard-capability")
        self.assertEqual(status, 200)
        self.assertIn("secret-provider-credential", body)

    def test_import_and_reset_reject_stale_page_before_writing(self):
        self.state(SERVED_MODEL_NAME="rendered")
        revision = landing._config_revision(gc.resolve()[0])
        self.state(SERVED_MODEL_NAME="newer-tab")
        before = Path(gc.p_state()).read_bytes()
        for path in ("/config/import", "/config/reset"):
            status, _, _ = self.request(path, {
                "token": "dashboard-capability", "config_revision": revision,
                "doc": json.dumps({"SERVED_MODEL_NAME": "older-import"}),
            })
            self.assertEqual(status, 200)
            self.assertEqual(Path(gc.p_state()).read_bytes(), before)
            self.assertFalse(Path(gc.p_restart_flag()).exists())

    def test_mutations_require_capability_even_with_trusted_proxy_header(self):
        self.state(SERVED_MODEL_NAME="protected")
        before = Path(gc.p_state()).read_bytes()
        status, _, _ = self.request("/config/reset", {
            "config_revision": landing._config_revision(gc.resolve()[0]),
        }, headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(status, 403)
        self.assertEqual(Path(gc.p_state()).read_bytes(), before)
        self.assertFalse(Path(gc.p_restart_flag()).exists())


if __name__ == "__main__":
    unittest.main()
