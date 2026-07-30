#!/usr/bin/env python3
"""Tests for the terminate control, the provider layer, the session erase and
the two termination switches.

    python3 tests/test_termination.py

NOTHING HERE CAN DESTROY ANYTHING. Three layers of protection, and the tests
assert on all three:

  1. provider.HttpTransport refuses any DELETE/POST/PUT/PATCH unless it was
     constructed with allow_destructive=True (test_transport_refuses_unarmed).
  2. After that test runs, this file REPLACES provider.HttpTransport with a
     class that raises on construction, so a test that forgets to inject a stub
     fails loudly instead of reaching the network.
  3. Every provider call takes its transport as an argument; the stubs record
     the request and return canned statuses.

The filesystem tests run entirely inside a temporary directory: MODEL_DIR,
GLM_STATE_DIR, GLM_RUNTIME_DIR and the ERASE_HOME/ERASE_TMP/ERASE_WORKSPACE/
ERASE_VARLOG roots are all redirected there, so /root and /tmp are never
touched.
"""
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import glm_config as gc          # noqa: E402
import provider                  # noqa: E402
import secure_erase              # noqa: E402
import terminate_worker          # noqa: E402

FAILURES = []
PASSED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append((name, detail))
        print(f"  FAIL {name} {detail}")
        raise AssertionError(detail)


def section(title):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
# 0. the arming property, tested with the REAL transport class (no network is
#    reached: NotArmed is raised before any socket is opened)
# --------------------------------------------------------------------------

def test_transport_refuses_unarmed():
    section("transport arming")
    t = provider.HttpTransport()          # not armed
    raised = ""
    try:
        t("DELETE", "https://example.invalid/api/v0/instances/1/")
    except provider.NotArmed as e:
        raised = str(e)
    check("an unarmed transport refuses DELETE", "not armed" in raised, raised)
    check("nothing was sent", t.sent == [], str(t.sent))

    t2 = provider.HttpTransport(allow_destructive=True, dry_run=True)
    status, body = t2("DELETE", "https://example.invalid/x")
    check("a dry-run transport returns 0 and sends nothing",
          status == 0 and json.loads(body).get("dry_run") is True, body)

    # A GraphQL read is a POST too, so intent — not method — decides.
    raised = ""
    t3 = provider.HttpTransport()
    try:
        t3("POST", "https://example.invalid/graphql", destructive=True)
    except provider.NotArmed as e:
        raised = str(e)
    check("a POST declared destructive is refused when unarmed", "not armed" in raised)
    check("the default for POST is the strict reading",
          "POST" in provider.DESTRUCTIVE_METHODS)
    check("a credential key in a URL is redacted before it can be logged",
          provider.redact("https://api.runpod.io/graphql?api_key=sk-secret")
          == "https://api.runpod.io/graphql?api_key=REDACTED",
          provider.redact("https://api.runpod.io/graphql?api_key=sk-secret"))


class StubTransport:
    """Records requests, returns canned responses. The only transport the rest
    of this file is allowed to use."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers=None, body=None, destructive=None):
        self.calls.append({"method": method, "url": url,
                           "auth": (headers or {}).get("Authorization", ""),
                           "destructive": destructive,
                           "body": (body or b"").decode("utf-8", "replace")})
        return self.responses.pop(0) if self.responses else (500, "no stub response")


class _NoRealTransport(provider.HttpTransport):
    def __init__(self, *a, **kw):
        raise AssertionError(
            "a test tried to construct a REAL provider transport; inject a stub")


# --------------------------------------------------------------------------
# 1. provider detection
# --------------------------------------------------------------------------

def test_detection():
    section("provider detection")
    check("vast.ai from CONTAINER_ID + CONTAINER_API_KEY",
          provider.detect({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"}) == provider.VAST)
    check("vast.ai from the port variables alone",
          provider.detect({"VAST_TCP_PORT_8000": "40001"}) == provider.VAST)
    check("RunPod from RUNPOD_POD_ID",
          provider.detect({"RUNPOD_POD_ID": "abc"}) == provider.RUNPOD)
    check("JarvisLabs from JARVISLABS_MACHINE_ID",
          provider.detect({"JARVISLABS_MACHINE_ID": "460920"})
          == provider.JARVISLABS)
    check("bare environment is unknown", provider.detect({}) == provider.UNKNOWN)
    check("manual override wins over the environment",
          provider.detect({"TERMINATE_PROVIDER": "runpod", "CONTAINER_ID": "1",
                           "CONTAINER_API_KEY": "k"}) == provider.RUNPOD)
    check("a nonsense override degrades to unknown, it does not crash",
          provider.detect({"TERMINATE_PROVIDER": "aws", "RUNPOD_POD_ID": "x"})
          == provider.UNKNOWN)
    ev = provider.detection_evidence({"CONTAINER_ID": "42", "CONTAINER_API_KEY": "secret"})
    check("detection evidence never echoes a key",
          ev["CONTAINER_ID"] == "42" and "secret" not in json.dumps(ev), json.dumps(ev))
    bad_vast = provider.get({"CONTAINER_ID": "../42", "CONTAINER_API_KEY": "k"})
    check("a malformed Vast instance id is not actionable",
          bad_vast.ready()[0] is False and "invalid format" in bad_vast.ready()[1])
    bad_runpod = provider.get({
        "RUNPOD_POD_ID": 'pod") } mutation { podTerminate',
        "RUNPOD_API_KEY": "k",
    })
    check("GraphQL punctuation is refused in a Runpod pod id",
          bad_runpod.ready()[0] is False and "invalid format" in bad_runpod.ready()[1])
    bad_jarvis = provider.get({
        "JARVISLABS_MACHINE_ID": "../460920",
        "JARVISLABS_REGION": "IN1",
        "JARVISLABS_API_KEY": "k",
    })
    check("a malformed JarvisLabs machine id is not actionable",
          bad_jarvis.ready()[0] is False
          and "invalid format" in bad_jarvis.ready()[1])
    no_calls = StubTransport([])
    bad_runpod.terminate(no_calls)
    check("an invalid provider id cannot reach the transport", no_calls.calls == [])


def test_unknown_provider_degrades():
    section("unknown provider degrades clearly")
    p = provider.get({})
    ok, reason = p.ready()
    check("not ready", not ok)
    check("says termination is not supported here",
          "could not identify the cloud provider" in reason, reason)
    check("points at the dashboard", "dashboard" in reason, reason)
    check("names the override", "TERMINATE_PROVIDER" in reason, reason)
    t = StubTransport([])
    res = p.terminate(t)
    check("terminate() is a no-op that reports failure", res["ok"] is False)
    check("and it made no request at all", t.calls == [], str(t.calls))


# --------------------------------------------------------------------------
# 2. provider calls (stubbed)
# --------------------------------------------------------------------------

VAST_ENV = {"CONTAINER_ID": "9876543", "CONTAINER_API_KEY": "vast-key"}
RUNPOD_ENV = {"RUNPOD_POD_ID": "pod-abc123", "RUNPOD_API_KEY": "pod-scoped-key"}
JARVIS_ENV = {
    "JARVISLABS_MACHINE_ID": "460920",
    "JARVISLABS_REGION": "IN1",
    "JARVISLABS_API_KEY": "jarvis-account-key",
}


def test_vast_terminate():
    section("vast.ai terminate (stubbed)")
    p = provider.get(VAST_ENV)
    t = StubTransport([(200, '{"success": true, "msg": "Instance destroyed successfully"}')])
    res = p.terminate(t)
    check("reports success", res["ok"] is True, json.dumps(res))
    check("DELETE to the documented URL",
          t.calls[0]["method"] == "DELETE"
          and t.calls[0]["url"] == "https://console.vast.ai/api/v0/instances/9876543/",
          str(t.calls))
    check("bearer auth from CONTAINER_API_KEY",
          t.calls[0]["auth"] == "Bearer vast-key", str(t.calls))
    check("tells the user to confirm on the dashboard",
          "dashboard" in res["detail"], res["detail"])

    t = StubTransport([(200, '{"success": false, "msg": "nope"}')])
    res = p.terminate(t)
    check("a 200 with success:false is NOT treated as success", res["ok"] is False,
          json.dumps(res))


def test_runpod_terminate():
    section("RunPod terminate (stubbed)")
    p = provider.get(RUNPOD_ENV)
    t = StubTransport([(204, "")])
    res = p.terminate(t, runner=lambda *a, **k: (1, "runner must not be called"))
    check("204 is success", res["ok"] is True, json.dumps(res))
    check("DELETE to rest.runpod.io/v1/pods/<id>",
          t.calls[0]["url"] == "https://rest.runpod.io/v1/pods/pod-abc123", str(t.calls))

    check("the REST call is declared destructive",
          t.calls[0]["destructive"] is True, str(t.calls))

    # REST refuses -> GraphQL podTerminate (the API runpodctl speaks)
    real_which = provider.shutil.which
    provider.shutil.which = lambda name: None
    try:
        t = StubTransport([(403, '{"error":"forbidden"}'),
                           (200, '{"data":{"podTerminate":"pod-abc123"}}')])
        res = p.terminate(t)
    finally:
        provider.shutil.which = real_which
    check("a 403 from REST falls back to GraphQL", res["ok"] is True, json.dumps(res)[:200])
    check("GraphQL goes to api.runpod.io/graphql",
          t.calls[1]["url"] == "https://api.runpod.io/graphql", str(t.calls[1]["url"]))
    check("with the verified podTerminate mutation and podId input",
          'podTerminate(input: { podId: \\"pod-abc123\\" })' in t.calls[1]["body"]
          or 'podTerminate(input: { podId: \"pod-abc123\" })' in t.calls[1]["body"],
          t.calls[1]["body"])
    check("as a Bearer header, not a key in the URL",
          t.calls[1]["auth"].startswith("Bearer ") and "api_key" not in t.calls[1]["url"])
    check("and declared destructive", t.calls[1]["destructive"] is True)

    # GraphQL 200 carrying an errors block is NOT success
    provider.shutil.which = lambda name: None
    try:
        t = StubTransport([(403, "no"), (200, '{"errors":[{"message":"denied"}]}')])
        res = p.terminate(t)
    finally:
        provider.shutil.which = real_which
    check("a GraphQL 200 with an errors block is not treated as success",
          res["ok"] is False, json.dumps(res)[:200])

    # the CLI fallback, only if the binary happens to exist
    calls = []

    runner_envs = []

    def runner(argv, timeout=120, env=None):
        calls.append(argv)
        runner_envs.append(env)
        return 0, "pod removed"
    provider.shutil.which = lambda name: "/usr/local/bin/runpodctl"
    try:
        t = StubTransport([(403, "no"), (403, "no")])
        res = p.terminate(t, runner=runner)
    finally:
        provider.shutil.which = real_which
    check("if both APIs refuse and runpodctl exists, it is the last resort",
          res["ok"] is True and calls
          and calls[0][1:] == ["pod", "delete", "pod-abc123"], json.dumps(res)[:200])
    check("and it uses delete (terminate), never stop",
          "stop" not in calls[0], str(calls[0]))
    check("the CLI receives the key in its environment, not argv",
          runner_envs[0]["RUNPOD_API_KEY"] == "pod-scoped-key"
          and "pod-scoped-key" not in " ".join(calls[0]), str(calls[0]))

    calls.clear()
    runner_envs.clear()

    def legacy_runner(argv, timeout=120, env=None):
        calls.append(argv)
        runner_envs.append(env)
        if argv[1:3] == ["pod", "delete"]:
            return 1, 'unknown command "pod"'
        return 0, "pod removed"
    provider.shutil.which = lambda name: "/usr/local/bin/runpodctl"
    try:
        t = StubTransport([(403, "no"), (403, "no")])
        res = p.terminate(t, runner=legacy_runner)
    finally:
        provider.shutil.which = real_which
    check("an older runpodctl falls back from pod delete to remove pod",
          res["ok"] is True and len(calls) == 2
          and calls[1][1:] == ["remove", "pod", "pod-abc123"],
          json.dumps(res)[:300])
    check("the legacy fallback also receives its key without exposing it",
          runner_envs[1]["RUNPOD_API_KEY"] == "pod-scoped-key"
          and all("pod-scoped-key" not in " ".join(call) for call in calls),
          str(calls))

    provider.shutil.which = lambda name: None
    try:
        t = StubTransport([(403, "no"), (403, "no")])
        res = p.terminate(t, runner=runner)
    finally:
        provider.shutil.which = real_which
    check("otherwise it fails", res["ok"] is False)
    check("and says the instance is still billing",
          "STILL BILLING" in res["detail"], res["detail"])
    check("and names the fix for the pod-scoped key",
          "RUNPOD_TERMINATE_API_KEY" in res["detail"], res["detail"])

    p2 = provider.get(dict(RUNPOD_ENV, RUNPOD_TERMINATE_API_KEY="account-key"))
    t = StubTransport([(204, "")])
    p2.terminate(t)
    check("an account key supplied by the user takes precedence",
          t.calls[0]["auth"] == "Bearer account-key", str(t.calls))


def test_runpod_probe_and_volume():
    section("RunPod credential probe and network-volume warning")
    p = provider.get(RUNPOD_ENV)

    # MEASURED on a live pod: this is what the pod-scoped key gets back when it
    # reads its own pod.
    t = StubTransport([(200, '{"data":{"pod":{"id":"pod-abc123","name":"glm-recon2",'
                             '"desiredStatus":"RUNNING"}}}')])
    state, detail = p.probe(t)
    check("a pod-scoped key that can read its own pod probes OK", state == "ok", detail)
    check("the probe asks about THIS POD, not the account",
          "pod(input:" in t.calls[0]["body"] and "myself" not in t.calls[0]["body"],
          t.calls[0]["body"])
    check("the probe is a READ, declared non-destructive",
          t.calls[0]["destructive"] is False, str(t.calls))
    check("the probe body carries no mutation", "mutation" not in t.calls[0]["body"],
          t.calls[0]["body"])

    # REGRESSION: the old pre-check asked `myself`, which a pod-scoped key is
    # NOT allowed to answer — measured Unauthorized for a key that could
    # nonetheless terminate the pod. Probing with `myself` condemned a working
    # credential and sent the user off for an account key they did not need.
    check("a pod-scoped Unauthorized on `myself` can no longer be reached "
          "when a pod id is known",
          "myself" not in t.calls[0]["body"], t.calls[0]["body"])
    t = StubTransport([(200, '{"errors":[{"message":"Unauthorized","path":["myself"],'
                             '"extensions":{"code":"RUNPOD"}}],"data":{"myself":null}}')])
    state, detail = p.probe(t)
    check("and if that shape ever comes back, it is reported as rejected",
          state == "rejected" and "REFUSED" in detail, detail)

    # a valid key pointed at a pod it cannot see
    state, detail = p.probe(StubTransport([(200, '{"data":{"pod":null}}')]))
    check("200 with data.pod null is caught as a bad pod id, not a pass",
          state == "rejected" and "pod id may be wrong" in detail, detail)

    # no pod id at all -> the account query is the only thing left to ask
    p_noid = provider.get({"TERMINATE_PROVIDER": "runpod", "RUNPOD_API_KEY": "k"})
    t = StubTransport([(200, '{"data":{"myself":{"id":"user_1"}}}')])
    state, _ = p_noid.probe(t)
    check("with no pod id it falls back to the account query",
          state == "ok" and "myself" in t.calls[0]["body"], t.calls[0]["body"])
    check("the account fallback asks only for the id, not email or balance",
          "email" not in t.calls[0]["body"] and "clientBalance" not in t.calls[0]["body"],
          t.calls[0]["body"])

    state, detail = p.probe(StubTransport([(401, "unauthorized")]))
    check("a rejected key is reported before the user types anything",
          state == "rejected" and "REFUSED" in detail, detail)
    check("and it names the fix", "RUNPOD_TERMINATE_API_KEY" in detail, detail)

    check("vast.ai deliberately has no probe",
          provider.get(VAST_ENV).probe(StubTransport([]))[0] == "unknown")

    # The injected pod key is real but deletion permission varies by deployment.
    src = provider.get(RUNPOD_ENV).key_source()
    check("the injected pod key describes deployment-dependent delete permission",
          "delete permission varies" in src and "verified to terminate" not in src,
          src)
    check("and no warning tells the user to go get an account key",
          not any("RUNPOD_TERMINATE_API_KEY" in w
                  for w in provider.get(RUNPOD_ENV).warnings()),
          str(provider.get(RUNPOD_ENV).warnings()))

    # MEASURED success shape of the mutation (live pod, 2026-07-26): the
    # podTerminate mutation returns Void, so a real success is
    # {"data":{"podTerminate":null}} — null IS success; what matters is that
    # the `podTerminate` key is present with no top-level `errors`.
    t = StubTransport([(403, "no"), (200, '{"data":{"podTerminate":null}}')])
    real_which = provider.shutil.which
    provider.shutil.which = lambda name: None
    try:
        res = p.terminate(t)
    finally:
        provider.shutil.which = real_which
    check("{'data':{'podTerminate':null}} is the measured SUCCESS shape",
          res["ok"] is True, json.dumps(res)[:200])
    provider.shutil.which = lambda name: None
    try:
        res_missing = provider.get(RUNPOD_ENV).terminate(
            StubTransport([(403, "no"), (200, '{"data":{}}')]))
    finally:
        provider.shutil.which = real_which
    check("a response missing the podTerminate key is NOT success",
          not res_missing["ok"], json.dumps(res_missing)[:200])
    check("and the failure detail does not falsely claim acceptance",
          "accepted" not in res_missing["detail"].lower(), res_missing["detail"])

    p = provider.get(dict(RUNPOD_ENV, RUNPOD_VOLUME_ID="vol-xyz"))
    warns = " ".join(p.warnings())
    check("an attached network volume is called out", "NETWORK VOLUME" in warns, warns)
    check("and it says the session data SURVIVES the terminate",
          "SURVIVES THIS TERMINATION" in warns, warns)
    check("and points at the erase box", "erase box" in warns, warns)
    check("and warns it keeps billing", "billing" in warns, warns)
    check("no volume, no volume warning",
          "NETWORK VOLUME" not in " ".join(provider.get(RUNPOD_ENV).warnings()))
    check("stop-vs-terminate is explicit in the billing copy",
          "not a stop" in provider.RunPod.billing
          and "KEEPS CHARGING" in provider.RunPod.billing, provider.RunPod.billing)


def test_jarvislabs_terminate_and_probe():
    section("JarvisLabs terminate and probe (stubbed)")
    p = provider.get(JARVIS_ENV)
    check("JarvisLabs provider is ready with id, region, and account key",
          p.ready() == (True, "ready"), str(p.ready()))
    check("IN1 resolves to the current Chennai backend",
          p.base_url() == "https://backendc.jarvislabs.net", p.base_url())

    t = StubTransport([(
        200,
        '{"success":true,"instance":{"machine_id":460920,"status":"Running"}}',
    )])
    state, detail = p.probe(t)
    check("read-only probe recognizes this VM",
          state == "ok" and "460920" in detail, f"{state}: {detail}")
    check("probe uses GET and is explicitly non-destructive",
          t.calls[0]["method"] == "GET"
          and t.calls[0]["url"]
          == "https://backendn.jarvislabs.net/users/fetch/460920"
          and t.calls[0]["destructive"] is False,
          str(t.calls))

    t = StubTransport([(200, '{"success":true}')])
    res = p.terminate(t)
    check("JarvisLabs reports destroy accepted", res["ok"] is True, json.dumps(res))
    check("destroy POSTs to the VM's regional SDK endpoint",
          t.calls[0]["method"] == "POST"
          and t.calls[0]["url"]
          == "https://backendc.jarvislabs.net/templates/vm/destroy?machine_id=460920",
          str(t.calls))
    check("destroy is armed and uses bearer auth",
          t.calls[0]["destructive"] is True
          and t.calls[0]["auth"] == "Bearer jarvis-account-key",
          str(t.calls))

    rejected = provider.get(dict(JARVIS_ENV, JARVISLABS_REGION="moon"))
    check("unknown JarvisLabs region fails closed",
          rejected.ready()[0] is False and "IN1, IN2, or EU1" in rejected.ready()[1],
          str(rejected.ready()))
    missing_key = provider.get({
        "JARVISLABS_MACHINE_ID": "460920",
        "JARVISLABS_REGION": "IN1",
    })
    check("JarvisLabs self-termination requires an explicitly supplied key",
          missing_key.ready()[0] is False and "credential" in missing_key.ready()[1],
          str(missing_key.ready()))


def test_pid1_env(tmp):
    section("environment is read from PID 1, not the calling shell")
    # RunPod injects into PID 1 only; an SSH login shell sees nothing.
    saved = gc._PID1_ENV
    try:
        gc._PID1_ENV = {"RUNPOD_POD_ID": "pod-from-pid1", "RUNPOD_API_KEY": "k1"}
        for var in ("RUNPOD_POD_ID", "RUNPOD_API_KEY", "CONTAINER_ID",
                    "CONTAINER_API_KEY"):
            os.environ.pop(var, None)
        check("a shell with an empty environment still detects RunPod",
              provider.detect() == provider.RUNPOD, provider.detect())
        check("and resolves the pod id from PID 1",
              provider.get().instance_id() == "pod-from-pid1")
        os.environ["RUNPOD_POD_ID"] = "pod-from-shell"
        check("an explicit value in this process still wins",
              provider.get().instance_id() == "pod-from-shell")
        os.environ.pop("RUNPOD_POD_ID")
        check("an explicitly passed env dict is never merged with PID 1",
              provider.detect({}) == provider.UNKNOWN)
        gc._PID1_ENV = {"TERMINATE_ENABLED": "1"}
        check("the switches are derived from PID 1's environment too",
              gc.switches_from_env()["enabled"] is True)
        pid1_runtime = os.path.join(tmp, "pid1-worker-run")
        pid1_state = os.path.join(tmp, "pid1-worker-state")
        gc._PID1_ENV = {
            "RUNPOD_POD_ID": "pod-from-pid1",
            "RUNPOD_API_KEY": "k1",
            "TERMINATE_ENABLED": "1",
            "GLM_RUNTIME_DIR": pid1_runtime,
            "GLM_STATE_DIR": pid1_state,
        }
        gc.init_switches(gc._PID1_ENV)
        transport = StubTransport([(204, "")])
        doc = terminate_worker.run(
            {"confirm": "pod-from-pid1", "erase": False},
            transport=transport, stopper=lambda _pr: (True, "stub"), env=None)
        check("the detached worker also falls back to PID 1",
              doc["ok"] is True and len(transport.calls) == 1,
              json.dumps(doc)[:300])
    finally:
        gc._PID1_ENV = saved


def test_failure_reporting():
    section("failure reporting")
    p = provider.get(VAST_ENV)
    for status, must in ((401, "STILL BILLING"), (403, "STILL BILLING"),
                         (404, "CHECK THE DASHBOARD"), (429, "STILL BILLING"),
                         (500, "MAY STILL BE RUNNING")):
        res = p.terminate(StubTransport([(status, "body")]))
        check(f"HTTP {status} reports failure and billing status",
              res["ok"] is False and must in res["detail"], res["detail"])
    res = p.terminate(StubTransport([(None, "connection refused")]))
    check("a network failure is reported, not swallowed",
          res["ok"] is False and "STILL RUNNING" in res["detail"], res["detail"])


# --------------------------------------------------------------------------
# 3. the switches: kill switch, anti-kill switch, ratchet
# --------------------------------------------------------------------------

def test_switch_defaults(tmp):
    section("switch defaults and the state-file ban")
    os.environ["GLM_RUNTIME_DIR"] = os.path.join(tmp, "run")
    os.environ["GLM_STATE_DIR"] = os.path.join(tmp, "state")
    os.makedirs(os.environ["GLM_RUNTIME_DIR"], exist_ok=True)
    os.makedirs(os.environ["GLM_STATE_DIR"], exist_ok=True)

    gc.init_switches({})
    allowed, why = gc.termination_allowed({})
    check("termination is DISABLED by default", allowed is False)
    check("and says how to enable it (startup env only)",
          "TERMINATE_ENABLED=1" in why and "not from this page" in why, why)

    gc.init_switches({"TERMINATE_ENABLED": "1"})
    allowed, _ = gc.termination_allowed({})
    check("TERMINATE_ENABLED=1 in the startup env enables it", allowed is True)

    gc.init_switches({"TERMINATE_ENABLED": "1", "TERMINATE_LOCKED": "1"})
    allowed, why = gc.termination_allowed({})
    check("the anti-kill switch beats the kill switch", allowed is False)
    check("and says it cannot be unlocked from the UI",
          "cannot be unlocked from this page" in why, why)
    check("and says what would be required instead",
          "container is restarted" in why and "dashboard" in why, why)


def test_state_file_cannot_set_switches(tmp):
    section("a state file may not set either switch")
    for key in ("TERMINATE_ENABLED", "TERMINATE_LOCKED"):
        gc.write_json_atomic(gc.p_state(), {"values": {key: True, "MTP_TOKENS": 5}})
        raised = ""
        try:
            gc.load_state_file()
        except gc.ConfigError as e:
            raised = str(e)
        check(f"state file setting {key} is REJECTED", "state file rejected" in raised,
              raised)
        check(f"and the error names {key}", key in raised, raised)

        effective, sources, notes = gc.resolve()
        baseline, baseline_sources, _ = gc.resolve(
            state_values={}, env_values=gc.load_startup_env())
        check("the whole file is ignored, not just the key",
              effective["MTP_TOKENS"] == baseline["MTP_TOKENS"]
              and sources["MTP_TOKENS"] == baseline_sources["MTP_TOKENS"]
              and sources["MTP_TOKENS"] != "file",
              f"{effective['MTP_TOKENS']} / {sources['MTP_TOKENS']}")
        check("and the rejection is surfaced as a note",
              any("state file rejected" in n for n in notes), str(notes))
    os.remove(gc.p_state())


def test_ratchet(tmp):
    section("the ratchet only tightens")
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    check("start: enabled, unlocked",
          gc.read_switches()["enabled"] and not gc.read_switches()["locked"])

    gc.tighten(enabled=True, locked=False, reason="no-op")
    st = gc.read_switches()
    check("asking for the same state changes nothing",
          st["enabled"] and not st["locked"], json.dumps(st))

    gc.tighten(enabled=False, reason="disable")
    check("enabled -> disabled is allowed", gc.read_switches()["enabled"] is False)
    gc.tighten(enabled=True, reason="try to re-enable")
    check("disabled -> enabled is REFUSED by construction",
          gc.read_switches()["enabled"] is False, json.dumps(gc.read_switches()))

    gc.tighten(locked=True, reason="lock")
    check("locking works", gc.read_switches()["locked"] is True)
    gc.tighten(locked=False, reason="try to unlock")
    check("locked -> unlocked is REFUSED by construction",
          gc.read_switches()["locked"] is True, json.dumps(gc.read_switches()))

    check("there is no loosening API at all",
          not any(n in dir(gc) for n in ("loosen", "unlock", "set_switches")),
          str([n for n in dir(gc) if "lock" in n or "loosen" in n]))

    # a container restart re-derives from the environment: that is the only way back
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    st = gc.read_switches()
    check("a container restart with a fresh env clears the runtime lock",
          st["enabled"] and not st["locked"], json.dumps(st))
    check("the switch state lives in the runtime dir, not the volume",
          gc.p_switches().startswith(gc.runtime_dir()), gc.p_switches())


# --------------------------------------------------------------------------
# 4. the worker: gates, erase-off-by-default, no destructive call when refused
# --------------------------------------------------------------------------

def worker_env(tmp, **extra):
    env = dict(VAST_ENV)
    env.update({"GLM_RUNTIME_DIR": os.path.join(tmp, "run"),
                "GLM_STATE_DIR": os.path.join(tmp, "state")})
    env.update(extra)
    return env


def test_worker_gates(tmp):
    section("worker gates")
    env = worker_env(tmp, TERMINATE_ENABLED="1")

    # locked: refuse even with a valid confirmation
    gc.init_switches({"TERMINATE_ENABLED": "1", "TERMINATE_LOCKED": "1"})
    t = StubTransport([(200, '{"success": true}')])
    doc = terminate_worker.run({"confirm": "9876543", "erase": False}, transport=t,
                               stopper=lambda pr: (True, "stub"), env=env)
    check("a LOCKED instance refuses a correct confirmation",
          doc["ok"] is False and doc.get("refused") == "switch", json.dumps(doc)[:200])
    check("and made no provider call", t.calls == [], str(t.calls))

    # disabled
    gc.init_switches({})
    t = StubTransport([(200, '{"success": true}')])
    doc = terminate_worker.run({"confirm": "9876543"}, transport=t,
                               stopper=lambda pr: (True, "stub"), env=env)
    check("a DISABLED instance refuses", doc["ok"] is False
          and doc.get("refused") == "switch")
    check("and made no provider call", t.calls == [], str(t.calls))

    # enabled, but the wrong confirmation
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    t = StubTransport([(200, '{"success": true}')])
    doc = terminate_worker.run({"confirm": "yes"}, transport=t,
                               stopper=lambda pr: (True, "stub"), env=env)
    check("a wrong typed confirmation refuses",
          doc["ok"] is False and doc.get("refused") == "confirmation", json.dumps(doc)[:200])
    check("and made no provider call", t.calls == [], str(t.calls))

    # enabled, no credential
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    t = StubTransport([(200, "{}")])
    doc = terminate_worker.run({"confirm": "TERMINATE"}, transport=t,
                               stopper=lambda pr: (True, "stub"),
                               env={"GLM_RUNTIME_DIR": os.path.join(tmp, "run"),
                                    "GLM_STATE_DIR": os.path.join(tmp, "state")})
    check("an unidentified provider refuses",
          doc["ok"] is False and doc.get("refused") == "provider", json.dumps(doc)[:200])
    check("and made no provider call", t.calls == [], str(t.calls))


def test_worker_happy_path(tmp):
    section("worker happy path (stubbed transport, erase off)")
    env = worker_env(tmp, TERMINATE_ENABLED="1")
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    stopped = []
    t = StubTransport([(200, '{"success": true}')])
    doc = terminate_worker.run({"confirm": "9876543", "erase": False}, transport=t,
                               stopper=lambda pr: (stopped.append(1), (True, "stub"))[1],
                               env=env)
    check("succeeds", doc["ok"] is True, json.dumps(doc)[:300])
    check("stopped the engine before terminating", stopped == [1])
    phases = [s["phase"] for s in doc["steps"]]
    check("erase was SKIPPED because it is off by default",
          "erase-skipped" in phases and "erasing" not in phases, str(phases))
    check("exactly one provider call", len(t.calls) == 1, str(t.calls))
    check("progress is written for the page to poll",
          os.path.isfile(terminate_worker.p_progress()))

    check("the terminate flag is set so the supervisor stands down",
          os.path.isfile(terminate_worker.p_request_stop()) is False,
          "default_stopper was stubbed, so no flag is expected here")


def test_worker_dry_run(tmp):
    section("dry run")
    env = worker_env(tmp, TERMINATE_ENABLED="1", TERMINATE_DRY_RUN="1")
    gc.init_switches({"TERMINATE_ENABLED": "1"})
    stopper_calls = []
    def _stub_stopper(pr):
        stopper_calls.append(1)
        return (True, "stub")
    doc = terminate_worker.run(
        {"confirm": "9876543", "erase": False},
        transport=StubTransport([(0, '{"dry_run": true}')]),
        stopper=_stub_stopper, env=env)
    check("dry run is recorded in the progress doc", doc["dry_run"] is True)
    check("and reported as success without claiming destruction",
          doc["ok"] is True and "DRY RUN" in doc["detail"], doc["detail"])
    check("dry run does NOT call the real stopper", stopper_calls == [],
          f"stopper was called {len(stopper_calls)} time(s)")
    check("dry run does not leave a request-stop flag",
          not os.path.isfile(terminate_worker.p_request_stop()),
          "the request-stop flag should not exist in a dry run")


# --------------------------------------------------------------------------
# 5. session erase, in a sandbox
# --------------------------------------------------------------------------

def build_fake_instance(root, with_manifest=True):
    """A miniature of the real layout: public checkpoint + our state + secrets."""
    md = os.path.join(root, "workspace", "GLM-5.2-EXL3-TR3-3.0bpw")
    home = os.path.join(root, "root")
    for d in (md, home, os.path.join(root, "workspace", ".lego", "certificates"),
              os.path.join(home, ".ssh"), os.path.join(home, ".cache", "huggingface"),
              os.path.join(root, "tmp"), os.path.join(root, "varlog"),
              os.path.join(root, "state", "logs"),
              os.path.join(root, "state", "failures", "20260726T000000Z"),
              os.path.join(root, "run")):
        os.makedirs(d, exist_ok=True)

    def w(path, data="x", size=0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data.encode() if not size else os.urandom(size))
        return path

    public = [w(os.path.join(md, "config.json"), '{"a":1}'),
              w(os.path.join(md, "model-layer-000.safetensors"), size=1024),
              w(os.path.join(md, "tokenizer.json"), "{}")]
    if with_manifest:
        for name in ("config.json", "model-layer-000.safetensors", "tokenizer.json"):
            w(os.path.join(md, ".cache", "huggingface", "download", name + ".metadata"),
              "hash")
    secrets = [
        w(os.path.join(root, "workspace", ".vllm-api-key"), "sk-secret"),
        w(os.path.join(root, "workspace", ".lego", "certificates", "x.key"), "PRIVATE"),
        w(os.path.join(home, ".ssh", "authorized_keys"), "ssh-ed25519 AAA"),
        w(os.path.join(home, ".bash_history"), "curl -H 'Authorization: Bearer sk-'"),
        w(os.path.join(home, ".cache", "huggingface", "token"), "hf_xxx"),
        w(os.path.join(home, ".runpod", "config.toml"), 'apiKey = "rp-secret"'),
        # the HF token where RunPod's defaults actually put it: on the network
        # volume, which termination does NOT clear
        w(os.path.join(root, "runpod-volume", ".cache", "huggingface", "token"), "hf_v"),
        w(os.path.join(root, "runpod-volume", "vllm.log"), "prompt text on the volume"),
        w(os.path.join(root, "state", "config.json"), "{}"),
        w(os.path.join(root, "state", "logs", "serve-current.log"), "prompt text"),
        w(os.path.join(root, "state", "failures", "20260726T000000Z", "error.log"), "boom"),
        w(os.path.join(root, "varlog", "syslog"), "logs"),
        w(os.path.join(root, "tmp", "scratch.txt"), "temp"),
        w(os.path.join(root, "glm-boot-status.json"), '{"api_key":"sk-secret"}'),
    ]
    user = [w(os.path.join(md, "my-adapter.safetensors"), size=2048),
            w(os.path.join(md, "notes.txt"), "my private notes")]
    derived = [w(os.path.join(md, ".vllm-cache", "kernel.so"), size=512),
               w(os.path.join(md, "config.json.orig"), "{}"),
               w(os.path.join(md, ".download-complete"), "")]
    return md, home, public, secrets, user, derived


def erase_env(root):
    md = os.path.join(root, "workspace", "GLM-5.2-EXL3-TR3-3.0bpw")
    return {"MODEL_DIR": md,
            # RunPod's PID-1 default: the whole HF cache on the network volume
            "HF_HOME": os.path.join(root, "runpod-volume", ".cache", "huggingface"),
            "ERASE_EXTRA_ROOTS": os.path.join(root, "runpod-volume"),
            "ERASE_ETC_SSH": os.path.join(root, "etc-ssh"),
            # hard fence: nothing outside the sandbox may even be planned
            "ERASE_CONFINE_TO": root,
            "ERASE_WORKSPACE": os.path.join(root, "workspace"),
            "ERASE_HOME": os.path.join(root, "root"),
            "ERASE_TMP": os.path.join(root, "tmp"),
            "ERASE_VARLOG": os.path.join(root, "varlog"),
            "STATUS_FILE": os.path.join(root, "glm-boot-status.json"),
            "GLM_STATE_DIR": os.path.join(root, "state"),
            "GLM_RUNTIME_DIR": os.path.join(root, "run")}


def test_erase_plan(tmp):
    section("erase plan: keeps the public weights, takes the session")
    root = os.path.join(tmp, "inst1")
    md, home, public, secrets, user, derived = build_fake_instance(root)
    os.makedirs(os.path.join(root, "etc-ssh"), exist_ok=True)
    open(os.path.join(root, "etc-ssh", "ssh_host_ed25519_key"), "w").write("HOSTKEY")
    env = erase_env(root)
    os.environ.update(env)
    p = secure_erase.Paths(env)
    doc = secure_erase.plan(p)
    chosen = {t["path"] for t in doc["targets"]}

    # THE test: nothing outside the sandbox may ever be planned. This module
    # walks absolute system paths that are only correct inside the container.
    stray = [c for c in chosen if not os.path.realpath(c).startswith(
        os.path.realpath(root) + "/")]
    check("no planned path escapes the sandbox root", not stray, str(stray[:5]))
    check("sshd host keys are taken",
          os.path.join(root, "etc-ssh", "ssh_host_ed25519_key") in chosen)

    check("manifest was found and used", doc["manifest_used"] is True)
    for f in public:
        check(f"PUBLIC kept: {os.path.basename(f)}", f not in chosen)
    for f in secrets:
        check(f"session data taken: {os.path.relpath(f, root)}", f in chosen)
    for f in user:
        check(f"user file taken: {os.path.basename(f)}", f in chosen)
    for f in derived:
        check(f"derived artifact kept: {os.path.relpath(f, md)}", f not in chosen)
    check("groups cover credentials/tls/config-state/logs/history/user-files",
          {"credentials", "tls", "config-state", "logs", "history", "user-files"}
          <= set(doc["groups"]), str(doc["groups"]))
    shutil.rmtree(root, ignore_errors=True)


def test_erase_without_manifest(tmp):
    section("erase plan without a download manifest")
    root = os.path.join(tmp, "inst2")
    md, home, public, secrets, user, derived = build_fake_instance(root, with_manifest=False)
    big = os.path.join(md, "big-unknown.safetensors")
    with open(big, "wb") as f:
        f.write(b"\0" * (secure_erase.LARGE_FILE_MB * (1 << 20) + 16))
    env = erase_env(root)
    os.environ.update(env)
    doc = secure_erase.plan(secure_erase.Paths(env))
    chosen = {t["path"] for t in doc["targets"]}
    check("manifest not used", doc["manifest_used"] is False)
    check("small unknown files are still erased",
          os.path.join(md, "notes.txt") in chosen)
    check("large indistinguishable files are NOT erased silently",
          big not in chosen and any(u["path"] == big for u in doc["unknown_large"]),
          json.dumps(doc["unknown_large"])[:200])
    shutil.rmtree(root, ignore_errors=True)


def test_erase_execution(tmp):
    section("erase execution")
    root = os.path.join(tmp, "inst3")
    md, home, public, secrets, user, derived = build_fake_instance(root)
    env = erase_env(root)
    os.environ.update(env)
    doc = secure_erase.plan(secure_erase.Paths(env))
    keyfile = os.path.join(root, "workspace", ".vllm-api-key")

    dry = secure_erase.erase(doc, dry_run=True)
    check("dry run erases nothing", os.path.isfile(keyfile) and dry["dry_run"])

    res = secure_erase.erase(doc)
    check("files are gone", not os.path.isfile(keyfile))
    check("public weights survive",
          os.path.isfile(os.path.join(md, "model-layer-000.safetensors")))
    check("counted what it did", res["erased"] >= len(secrets), json.dumps(res)[:200])
    check("no failures", not res["failed"], json.dumps(res["failed"])[:200])
    check("guarantees say weights are deliberately kept",
          any("public model weights" in s for s in secure_erase.GUARANTEES["does_not"]))
    check("guarantees are blunt about SSD wear levelling",
          any("wear levelling" in s for s in secure_erase.GUARANTEES["does_not"]))
    check("the runpodctl config (world-readable API key) is gone",
          not os.path.isfile(os.path.join(home, ".runpod", "config.toml")))
    shutil.rmtree(root, ignore_errors=True)


def test_no_secret_written_world_readable(tmp):
    section("nothing this code writes leaks a credential by mode")
    os.environ["GLM_STATE_DIR"] = os.path.join(tmp, "modes", "state")
    os.environ["GLM_RUNTIME_DIR"] = os.path.join(tmp, "modes", "run")
    gc.write_json_atomic(gc.p_state(), {"values": {"MTP_TOKENS": 5}}, mode=0o600)
    mode = oct(os.stat(gc.p_state()).st_mode & 0o777)
    check("the config state file is 0o600", mode == "0o600", mode)
    gc.init_switches({})
    swmode = oct(os.stat(gc.p_switches()).st_mode & 0o777)
    check("the switch file is 0o644 and carries no secret",
          swmode == "0o644" and "key" not in json.dumps(gc.read_switches()).lower(),
          swmode)
    p = provider.get({"RUNPOD_POD_ID": "x", "RUNPOD_API_KEY": "super-secret"})
    blob = json.dumps(p.describe())
    check("describe() never contains the key itself", "super-secret" not in blob, blob[:200])


def test_erase_keeps_progress_file(tmp):
    section("erase keeps what the worker still needs")
    root = os.path.join(tmp, "inst4")
    build_fake_instance(root)
    env = erase_env(root)
    os.environ.update(env)
    progress = os.path.join(env["GLM_RUNTIME_DIR"], "terminate.json")
    os.makedirs(env["GLM_RUNTIME_DIR"], exist_ok=True)
    with open(progress, "w") as f:
        f.write("{}")
    with open(os.path.join(env["GLM_RUNTIME_DIR"], "config.env"), "w") as f:
        f.write("export X=1")
    doc = secure_erase.plan(secure_erase.Paths(env), keep=(progress,))
    chosen = {t["path"] for t in doc["targets"]}
    check("the progress file is preserved", progress not in chosen)
    check("but other runtime scratch is erased",
          os.path.join(env["GLM_RUNTIME_DIR"], "config.env") in chosen, str(chosen))
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
# 6. the landing page: auth, confirmation, locked refusal
# --------------------------------------------------------------------------

def _free_port():
    """Bind to port 0, read the OS-assigned port, release it for the landing page."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def http(method, path, data=None, port=18333):
    url = f"http://127.0.0.1:{port}{path}"
    body = data.encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def http_headers(path, port=18333):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            response.read()
            return dict(response.headers.items())
    except Exception:
        return {}


def test_landing(tmp):
    section("landing page: auth, confirmation, locked refusal")
    root = os.path.join(tmp, "inst5")
    build_fake_instance(root)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C"), "LC_ALL": os.environ.get("LC_ALL", "C")}
    env.update(erase_env(root))
    env.update(VAST_ENV)
    port = _free_port()
    env.update({"OPEN_BUTTON_TOKEN": "tok", "LANDING_ALLOW_INSECURE": "1",
                "LANDING_PORT": str(port), "GLM_SCRIPTS_DIR": os.path.join(REPO, "scripts"),
                "TERMINATE_ENABLED": "1"})
    # the page reads switches from the runtime dir; seed it as a container start would
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "config_cli.py"),
                    "snapshot-env"], env=env, capture_output=True)
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "landing.py")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if http("GET", "/?token=tok", port=port)[0] == 200:
                break
            time.sleep(0.25)

        headers = {k.lower(): v for k, v in http_headers("/?token=tok", port=port).items()}
        check("credential-bearing landing responses are not cached",
              headers.get("cache-control") == "no-store", str(headers))
        check("landing capability tokens are not sent as referrers",
              headers.get("referrer-policy") == "no-referrer", str(headers))
        check("the landing page cannot be framed",
              headers.get("x-frame-options") == "DENY", str(headers))

        code, _ = http("GET", "/terminate", port=port)
        check("GET /terminate with no token is refused", code == 403, str(code))
        code, _ = http("GET", "/terminate?token=wrong", port=port)
        check("GET /terminate with a wrong token is refused", code == 403, str(code))
        code, body = http("POST", "/terminate", "confirm=9876543&understand=1", port=port)
        check("POST /terminate with no token is refused", code == 403, str(code))
        time.sleep(0.5)  # let the handler thread clean up after the 403

        code, body = http("GET", "/terminate?token=tok", port=port)
        check("the confirmation page renders", code == 200, str(code))
        check("it shows the instance id to type", "9876543" in body)
        check("it says irreversible", "irreversible" in body.lower())
        check("it shows what is destroyed", "Destroyed:" in body)
        check("erase checkbox is present and NOT pre-checked",
              'name=erase value=1' in body and 'name=erase value=1 checked' not in body)
        check("it states the weights are not erased",
              "NOT erased" in body or "not erased" in body)
        check("it states the SSD/overlay/network-volume limits",
              "wear levelling" in body and "network volume" in body)

        code, body = http("POST", "/terminate?token=tok",
                          "token=tok&confirm=9876543", port=port)
        check("POST without the understand checkbox does nothing",
              code == 200 and "did not tick the box" in body, str(code))
        code, body = http("POST", "/terminate?token=tok",
                          "token=tok&understand=1&confirm=WRONG", port=port)
        check("POST with a wrong typed confirmation does nothing",
              code == 200 and "did not match" in body, str(code))
        check("no worker was started",
              not os.path.isfile(os.path.join(root, "run", "terminate.json")))

        # now LOCK it through the UI ratchet, and prove a perfect request fails
        code, body = http("POST", "/terminate/lock?token=tok",
                          "token=tok&understand=1&action=lock", port=port)
        check("the lock button locks", code == 200 and "now LOCKED" in body, body[:200])
        code, body = http("POST", "/terminate?token=tok",
                          "token=tok&understand=1&confirm=9876543", port=port)
        check("a LOCKED instance refuses a perfect request",
              code == 200 and "LOCKED" in body, body[:300])
        check("still no worker started",
              not os.path.isfile(os.path.join(root, "run", "terminate.json")))
        code, body = http("GET", "/terminate/status?token=tok", port=port)
        st = json.loads(body)
        check("status endpoint reports the lock",
              st["switches"]["locked"] is True and st["allowed"] is False, body[:200])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    shutil.rmtree(root, ignore_errors=True)


def test_landing_unknown_provider(tmp):
    section("landing page on an unidentified provider")
    root = os.path.join(tmp, "inst6")
    build_fake_instance(root)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C"), "LC_ALL": os.environ.get("LC_ALL", "C")}
    env.update(erase_env(root))
    port = _free_port()
    env.update({"OPEN_BUTTON_TOKEN": "tok", "LANDING_ALLOW_INSECURE": "1",
                "LANDING_PORT": str(port), "GLM_SCRIPTS_DIR": os.path.join(REPO, "scripts"),
                "TERMINATE_ENABLED": "1", "TERMINATE_PROVIDER": ""})
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "config_cli.py"),
                    "snapshot-env"], env=env, capture_output=True)
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "landing.py")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if http("GET", "/?token=tok", port=port)[0] == 200:
                break
            time.sleep(0.25)
        code, body = http("GET", "/terminate?token=tok", port=port)
        check("page renders", code == 200, str(code))
        check("says termination is not supported here",
              "not supported here" in body, body[:300])
        check("tells the user to use the dashboard", "dashboard" in body)
        check("offers the manual override", "TERMINATE_PROVIDER" in body)
        check("shows no destroy button",
              "Destroy this instance" not in body, "a destroy button was rendered")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------

def _run(test, *args):
    """Run one test_*(); an AssertionError is already recorded by check(), so
    swallow it here so the direct-invocation path continues and exits 1 via
    main()'s FAILURES check. pytest still sees the raised AssertionError."""
    try:
        test(*args)
    except AssertionError:
        pass


def main():
    _run(test_transport_refuses_unarmed)
    # From here on, constructing a real transport is a test failure.
    provider.HttpTransport = _NoRealTransport

    tmp = tempfile.mkdtemp(prefix="glm-term-test-")
    saved_env = dict(os.environ)
    try:
        _run(test_detection)
        _run(test_unknown_provider_degrades)
        _run(test_vast_terminate)
        _run(test_runpod_terminate)
        _run(test_runpod_probe_and_volume)
        _run(test_jarvislabs_terminate_and_probe)
        _run(test_pid1_env, tmp)
        _run(test_failure_reporting)
        _run(test_switch_defaults, tmp)
        _run(test_state_file_cannot_set_switches, tmp)
        _run(test_ratchet, tmp)
        _run(test_worker_gates, tmp)
        _run(test_worker_happy_path, tmp)
        _run(test_worker_dry_run, tmp)
        _run(test_erase_plan, tmp)
        _run(test_erase_without_manifest, tmp)
        _run(test_erase_execution, tmp)
        _run(test_erase_keeps_progress_file, tmp)
        _run(test_no_secret_written_world_readable, tmp)
        _run(test_landing, tmp)
        _run(test_landing_unknown_provider, tmp)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILURES)} failed")
    for name, detail in FAILURES:
        print(f"  FAILED: {name} {detail}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
