#!/usr/bin/env python3
"""Cloud-provider abstraction: identify the instance we are running on, and
destroy it on request.

THIS MODULE ISSUES A REQUEST THAT DESTROYS A PAID INSTANCE AND EVERY BYTE ON
ITS DISKS. Three safety properties, in order of importance:

  1. The network transport REFUSES to send a destructive request unless it was
     constructed with allow_destructive=True. Nothing constructs it that way
     except terminate_worker.py, after a typed confirmation.
  2. Every provider call takes its transport as an argument, so tests inject a
     stub and no real endpoint is ever contacted. There is no code path in the
     test suite that can reach the network.
  3. TERMINATE_DRY_RUN=1 performs the whole flow and prints the exact request
     it WOULD have sent, without sending it.

Provider facts, with sources (see docs/termination-and-erase.md for the full
matrix and for what is unverified):

vast.ai
  - CONTAINER_ID / CONTAINER_API_KEY are injected into every instance; the
    per-instance key "can only start, stop, or destroy that specific instance".
    https://docs.vast.ai/documentation/instances/templates/docker-environment
  - DELETE /api/v0/instances/{id}/ with `Authorization: Bearer <key>`;
    200 -> {"success": true, "msg": "Instance destroyed successfully"}.
    https://docs.vast.ai/api-reference/instances/destroy-instance
  - The entrypoint already uses these same credentials for the dashboard label,
    so we know they work from inside the container.

RunPod
  - RUNPOD_POD_ID and a "Pod-scoped API key" in RUNPOD_API_KEY are documented as
    set automatically. https://docs.runpod.io/pods/templates/environment-variables
  - REST: DELETE https://rest.runpod.io/v1/pods/{podId} with a Bearer token; 204
    on success. https://docs.runpod.io/api-reference/pods/DELETE/pods/podId
  - GraphQL (what runpodctl itself is a thin client over): POST
    https://api.runpod.io/graphql with `Authorization: Bearer <key>` and
    `mutation { podTerminate(input: { podId: "<id>" }) }`. Mutation name and
    input shape from RunPod's own Python SDK
    (runpod/api/mutations/pods.py, generate_pod_terminate_mutation) and the
    schema at https://graphql-spec.runpod.io/ (podTerminate(input:
    PodTerminateInput!) -> Void); endpoint and Bearer header from
    runpod/api/graphql.py.
  - Current `runpodctl pod delete <id>` and legacy
    `runpodctl remove pod <id>` both terminate a pod. They are DIFFERENT from
    `stop`, which leaves storage billing. This module always terminates.
  - RunPod currently preinstalls runpodctl in Pods, but deployed versions vary.
    It is therefore an opportunistic final fallback rather than the primary
    API path.
  - MEASURED on live pods: the auto-injected pod-scoped RUNPOD_API_KEY
    terminated its own pod through GraphQL on 2026-07-26, but REST and GraphQL
    both refused it on a different Secure deployment on 2026-07-29. The
    appliance therefore retains the CLI ladder and supports an explicit
    RUNPOD_TERMINATE_API_KEY for deployments whose scoped key cannot delete.
  - VERIFIED: RunPod injects its variables into PID 1 ONLY. A helper started
    from a new session — an SSH login shell, say — sees none of them, so the
    environment is read through glm_config.effective_env(), which layers this
    process's environment over /proc/1/environ.
  - VERIFIED on 2026-07-29: a RunPod Secure Pod carried an older runpodctl whose
    termination spelling was `remove pod`, not the current `pod delete`.

JarvisLabs
  - The current `jarvislabs` 0.2.x SDK/CLI manages instances through regional
    HTTPS backends. `jl destroy <machine_id>` resolves the instance region and
    POSTs `templates/vm/destroy?machine_id=<id>` for a VM.
    https://docs.jarvislabs.ai/cli
  - JarvisLabs VMs do not inject an instance-scoped API credential into nested
    Docker containers. The launcher must pass JARVISLABS_MACHINE_ID and
    JARVISLABS_REGION; self-termination additionally requires an account key in
    JARVISLABS_TERMINATE_API_KEY (or JARVISLABS_API_KEY/JL_API_KEY).
  - Pause stops compute billing but retains chargeable storage. Destroy removes
    the VM and its disk. https://docs.jarvislabs.ai/getting_started
"""
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc  # noqa: E402

VAST = "vastai"
RUNPOD = "runpod"
JARVISLABS = "jarvislabs"
UNKNOWN = "unknown"

DESTRUCTIVE_METHODS = ("DELETE", "POST", "PUT", "PATCH")


class NotArmed(RuntimeError):
    """Raised when a destructive request is attempted through a transport that
    was not explicitly armed. This is the last line of defence, and it is the
    reason a stray import or a test cannot destroy anything."""


def redact(url: str) -> str:
    """Never let a credential reach a log, a progress file or the landing page.

    RunPod's own documentation shows the GraphQL key passed as `?api_key=...`
    in the URL. This module always sends it as a Bearer header instead (which
    is what RunPod's SDK does), but URLs get recorded and displayed, so strip
    anything key-shaped on the way out regardless."""
    return re.sub(r"(api_key|apiKey|token)=[^&]*", r"\1=REDACTED", url)


class HttpTransport:
    """The only thing in this repo that can actually talk to a provider API."""

    def __init__(self, allow_destructive=False, dry_run=False, timeout=30):
        self.allow_destructive = bool(allow_destructive)
        self.dry_run = bool(dry_run)
        self.timeout = timeout
        self.sent = []

    def __call__(self, method, url, headers=None, body=None, destructive=None):
        # `destructive` is the CALLER's declaration of intent, because method
        # alone cannot tell a GraphQL read from a GraphQL mutation — both are
        # POSTs. It defaults to the pessimistic reading of the method, so a
        # caller that says nothing gets the strict behaviour.
        if destructive is None:
            destructive = method.upper() in DESTRUCTIVE_METHODS
        record = {"method": method, "url": redact(url),
                  "headers": sorted((headers or {}).keys()),
                  "destructive": bool(destructive)}
        if destructive and not self.allow_destructive:
            raise NotArmed(
                f"refusing to send {method} {redact(url)}: this transport is not "
                "armed. Destructive calls are only made by terminate_worker.py "
                "after a typed confirmation.")
        if self.dry_run:
            self.sent.append(dict(record, dry_run=True))
            return 0, json.dumps({"dry_run": True, **record})
        req = urllib.request.Request(url, data=body, method=method,
                                     headers=headers or {})
        ctx = ssl.create_default_context() if url.startswith("https") else None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as r:
                text = r.read().decode("utf-8", "replace")
                self.sent.append(dict(record, status=r.status))
                return r.status, text
        except urllib.error.HTTPError as e:          # noqa: F821 (urllib.error via urllib.request)
            text = ""
            try:
                text = e.read().decode("utf-8", "replace")
            except Exception:
                pass
            self.sent.append(dict(record, status=e.code))
            return e.code, text
        except Exception as e:
            self.sent.append(dict(record, status=None, error=str(e)))
            return None, str(e)


def default_runner(argv, timeout=120, env=None):
    """Subprocess runner for CLI fallbacks (runpodctl). Injected in tests."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return None, str(e)


# --------------------------------------------------------------------------

class Provider:
    name = UNKNOWN
    label = "unknown provider"
    supported = False
    # what a terminate destroys, for the confirmation screen
    destroys = []
    survives = []
    billing = ""
    docs = ""

    def __init__(self, env=None):
        # effective_env: this process's environment over PID 1's. RunPod injects
        # only into PID 1, so os.environ alone is wrong for anything invoked
        # from a separate session.
        self.env = gc.effective_env(env)

    # -- identity ---------------------------------------------------------
    def instance_id(self):
        return ""

    def valid_instance_id(self):
        return True

    def api_key(self):
        return ""

    def key_source(self):
        return ""

    def ready(self):
        """-> (ok, reason). Never raises; the UI shows `reason` verbatim."""
        if not self.supported:
            return False, (
                "This image could not identify the cloud provider it is running on, "
                "so it cannot terminate the instance for you. Terminate it from your "
                "provider's dashboard instead — that is the only way to stop billing. "
                "If you know the provider, set "
                "TERMINATE_PROVIDER=vastai|runpod|jarvislabs.")
        if not self.instance_id():
            return False, f"{self.label}: the instance id is not in the environment"
        if not self.valid_instance_id():
            return False, f"{self.label}: the instance id has an invalid format"
        if not self.api_key():
            return False, f"{self.label}: no API credential is available"
        return True, "ready"

    def probe(self, transport):
        """Cheap read-only credential check, so the UI can say "this key will
        not work" BEFORE the user types an instance id. -> (state, detail) where
        state is "ok" | "rejected" | "unknown"."""
        return "unknown", "this provider offers no read-only credential check"

    def warnings(self):
        """Provider-specific things the confirmation screen must not bury."""
        return []

    def describe(self):
        ok, reason = self.ready()
        return {"provider": self.name, "label": self.label, "supported": self.supported,
                "instance_id": self.instance_id(), "ready": ok, "reason": reason,
                "key_source": self.key_source(), "destroys": self.destroys,
                "survives": self.survives, "billing": self.billing, "docs": self.docs,
                "warnings": self.warnings()}

    # -- action -----------------------------------------------------------
    def terminate(self, transport, runner=None):
        return {"ok": False, "status": None, "provider": self.name,
                "detail": self.ready()[1], "attempts": []}


class VastAI(Provider):
    name = VAST
    label = "vast.ai"
    supported = True
    destroys = ["the instance and every disk attached to it",
                "the downloaded model weights",
                "the persisted API key, TLS certificate and config state"]
    survives = ["nothing on this instance; vast.ai instances have no external volume "
                "in this template"]
    billing = ("Destroying is what stops billing. Vast.ai also has a separate 'stop', "
               "which keeps charging for storage — this control does a DESTROY.")
    docs = "https://docs.vast.ai/api-reference/instances/destroy-instance"

    def instance_id(self):
        return (self.env.get("CONTAINER_ID") or "").strip()

    def valid_instance_id(self):
        return bool(re.fullmatch(r"[0-9]+", self.instance_id()))

    def api_key(self):
        return (self.env.get("VAST_API_KEY") or self.env.get("CONTAINER_API_KEY") or "").strip()

    def key_source(self):
        if self.env.get("VAST_API_KEY"):
            return "VAST_API_KEY (account key supplied by the user)"
        if self.env.get("CONTAINER_API_KEY"):
            return "CONTAINER_API_KEY (per-instance key injected by vast.ai)"
        return ""

    def probe(self, transport):
        # Deliberately not probed. CONTAINER_API_KEY is documented as able to
        # "only start, stop, or destroy that specific instance", so a read call
        # may well be refused for a key that CAN destroy — a probe here would
        # produce false alarms, which is worse than no probe.
        return "unknown", ("vast.ai's per-instance key is scoped to start/stop/"
                           "destroy only, so there is no read call that would "
                           "prove it works")

    def terminate(self, transport, runner=None):
        ok, reason = self.ready()
        if not ok:
            return {"ok": False, "status": None, "provider": self.name,
                    "detail": reason, "attempts": []}
        iid = self.instance_id()
        url = f"https://console.vast.ai/api/v0/instances/{iid}/"
        status, text = transport("DELETE", url,
                                 {"Authorization": "Bearer " + self.api_key(),
                                  "Accept": "application/json"},
                                 destructive=True)
        attempt = {"method": "DELETE", "url": redact(url), "status": status,
                   "body": (text or "")[:400]}
        ok = status == 200
        if ok:
            try:
                ok = bool(json.loads(text).get("success", True))
            except (ValueError, AttributeError):
                ok = True                      # 200 with a non-JSON body: take it
        elif status == 0:
            ok = True                          # dry run
        if status == 200 and not ok:
            # HTTP 200 with success=false must not read as the accepted-
            # terminate wording _explain gives every 200.
            detail = ("Vast returned HTTP 200 but success=false for the "
                      "destroy request. THE INSTANCE MAY STILL BE RUNNING "
                      "AND BILLING — check the dashboard.")
        else:
            detail = _explain(status, text, self)
        return {"ok": ok, "status": status, "provider": self.name,
                "detail": detail, "attempts": [attempt]}


class RunPod(Provider):
    name = RUNPOD
    label = "RunPod"
    supported = True
    destroys = ["the Pod, its container disk and its volume disk (/workspace)",
                "the downloaded model weights if they are on the volume disk",
                "the persisted API key, TLS certificate and config state"]
    survives = ["a NETWORK volume, if one is attached — RunPod keeps network volumes "
                "after a Pod is deleted, and it keeps billing for them"]
    billing = ("This does a TERMINATE (delete), not a stop. RunPod's 'stop' keeps the "
               "volume disk and KEEPS CHARGING for its storage; terminate deletes the "
               "container disk and the volume disk and stops those charges. RunPod "
               "documents terminate as permanently deleting all data not on a network "
               "volume.")
    docs = "https://docs.runpod.io/pods/manage-pods"

    def instance_id(self):
        return (self.env.get("RUNPOD_POD_ID") or "").strip()

    def valid_instance_id(self):
        # Provider-generated Pod ids are short URL-safe identifiers. Reject
        # quotes, slashes and GraphQL punctuation before the id reaches either
        # a URL path or the measured inline GraphQL operation.
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                                 self.instance_id()))

    def api_key(self):
        return (self.env.get("RUNPOD_TERMINATE_API_KEY")
                or self.env.get("RUNPOD_API_KEY") or "").strip()

    def key_source(self):
        if self.env.get("RUNPOD_TERMINATE_API_KEY"):
            return "RUNPOD_TERMINATE_API_KEY (account key supplied by the user)"
        if self.env.get("RUNPOD_API_KEY"):
            return ("RUNPOD_API_KEY (pod-scoped key injected by RunPod; delete "
                    "permission varies by deployment)")
        return ""

    GRAPHQL_URL = "https://api.runpod.io/graphql"
    # A live pod accepted this with its injected scoped key on 2026-07-26.
    # Another Secure pod refused that key on 2026-07-29, so it is one rung in
    # the ladder rather than a universal permission claim.
    TERMINATE_MUTATION = 'mutation { podTerminate(input: { podId: "%s" }) }'
    # The pre-check must ask a question the POD-SCOPED key is allowed to answer.
    # MEASURED: `query { myself { id } }` returns
    #   {"errors":[{"message":"Unauthorized","path":["myself"]}],"data":{"myself":null}}
    # for a key that can nonetheless terminate its own pod. Probing with `myself`
    # therefore condemns a perfectly good credential and sends the user off to
    # fetch an account key they do not need. Reading the pod itself is both
    # permitted and strictly more useful: it also proves the podId we resolved is
    # the right one.
    POD_PROBE_QUERY = 'query { pod(input: {podId: "%s"}) { id desiredStatus } }'
    # Only used when no pod id is known — an account key with no pod to point at.
    ACCOUNT_PROBE_QUERY = "query { myself { id } }"

    def network_volume_id(self):
        return (self.env.get("RUNPOD_VOLUME_ID") or "").strip()

    def warnings(self):
        out = []
        if self.network_volume_id():
            out.append(
                "A NETWORK VOLUME (%s) is attached. RunPod keeps network volumes when "
                "a Pod is deleted — 'Terminating permanently deletes all data not "
                "stored in a network volume' — and keeps billing for them. Everything "
                "this template writes lives under /workspace, which is where a network "
                "volume mounts, so YOUR SESSION DATA SURVIVES THIS TERMINATION unless "
                "you tick the erase box. Delete the volume itself from the dashboard "
                "when you no longer want to pay for it."
                % self.network_volume_id())
        return out

    def probe(self, transport):
        """A read, not a mutation. Declared destructive=False so it works
        through an UNARMED transport: the landing page checks the credential
        without ever holding a loaded gun.

        Asks about THIS POD, not about the account, because the pod-scoped key
        is scoped — see POD_PROBE_QUERY. A pass here means the key can see the
        pod it is about to terminate, which is the authorisation that matters."""
        if not self.api_key():
            return "rejected", "no API key is available in this pod"
        iid = self.instance_id()
        if iid and not self.valid_instance_id():
            return "rejected", (
                f"RunPod: the pod id in the environment ({iid}) has an invalid "
                "format and is not safe to interpolate into a query. Set "
                "RUNPOD_POD_ID to the pod's short id.")
        query = self.POD_PROBE_QUERY % iid if iid else self.ACCOUNT_PROBE_QUERY
        assert "mutation" not in query, "the probe must never mutate"
        status, text = transport("POST", self.GRAPHQL_URL,
                                 {"Authorization": "Bearer " + self.api_key(),
                                  "Content-Type": "application/json"},
                                 json.dumps({"query": query}).encode(),
                                 destructive=False)
        text = text or ""
        if status == 200 and '"errors"' not in text:
            if iid:
                # 200 + data.pod == null means the key is fine but the id is not
                # a pod it can see: worth catching before the user types it out.
                try:
                    if (json.loads(text).get("data") or {}).get("pod") is None:
                        return "rejected", (
                            f"RunPod accepted the key but returned nothing for pod "
                            f"{iid}. The pod id may be wrong, or this key belongs to "
                            "another account. Terminating would fail.")
                except ValueError:
                    pass
                return "ok", (f"the API key can read this pod ({iid}); deletion "
                              "permission will be checked by the terminate call")
            return "ok", "the API key was accepted by RunPod"
        if status in (401, 403) or '"Unauthorized"' in text:
            return "rejected", (
                "RunPod REFUSED this API key for this pod (HTTP %s). Terminating will "
                "fail. Supply an account key as RUNPOD_TERMINATE_API_KEY, or use the "
                "dashboard." % status)
        return "unknown", f"the credential check was inconclusive (HTTP {status})"

    def terminate(self, transport, runner=None):
        ok, reason = self.ready()
        if not ok:
            return {"ok": False, "status": None, "provider": self.name,
                    "detail": reason, "attempts": []}
        iid = self.instance_id()
        key = self.api_key()
        attempts = []

        # 1. the documented REST endpoint
        url = f"https://rest.runpod.io/v1/pods/{iid}"
        status, text = transport("DELETE", url,
                                 {"Authorization": "Bearer " + key,
                                  "Accept": "application/json"},
                                 destructive=True)
        attempts.append({"method": "DELETE", "url": redact(url), "status": status,
                         "body": (text or "")[:400]})
        if status in (200, 204) or status == 0:
            return {"ok": True, "status": status, "provider": self.name,
                    "detail": _explain(status, text, self), "attempts": attempts}

        # 2. GraphQL — the API runpodctl itself speaks. Worth trying separately
        #    because REST and GraphQL are different services and a key rejected
        #    by one is not necessarily rejected by the other.
        body = json.dumps({"query": self.TERMINATE_MUTATION % iid}).encode()
        gstatus, gtext = transport("POST", self.GRAPHQL_URL,
                                   {"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"},
                                   body, destructive=True)
        attempts.append({"method": "POST", "url": redact(self.GRAPHQL_URL) + " (podTerminate)",
                         "status": gstatus, "body": (gtext or "")[:400]})
        if gstatus == 0:
            return {"ok": True, "status": gstatus, "provider": self.name,
                    "detail": _explain(gstatus, gtext, self), "attempts": attempts}
        if gstatus == 200:
            # Parse the body rather than substring-match "errors": a 200 without
            # the literal token is not proof of success. RunPod's podTerminate
            # mutation returns Void — MEASURED on a live pod (2026-07-26):
            # {"data":{"podTerminate":null}} IS the success shape. Success
            # therefore requires a parsed object with no top-level `errors` key
            # and a `data` object that actually HAS the `podTerminate` key
            # (key presence, not truthiness — null is the documented success
            # value). A body that fails to parse, or lacks the key entirely, is
            # a failure, not a silent success.
            try:
                payload = json.loads(gtext or "")
            except ValueError:
                payload = None
            data = payload.get("data") if isinstance(payload, dict) else None
            if (isinstance(payload, dict) and "errors" not in payload
                    and isinstance(data, dict) and "podTerminate" in data):
                return {"ok": True, "status": gstatus, "provider": self.name,
                        "detail": ("Terminate accepted by RunPod (GraphQL podTerminate). "
                                   "Confirm in the dashboard that the Pod is gone and "
                                   "billing has stopped."),
                        "attempts": attempts}
            # HTTP 200 is not proof of GraphQL-level success (GraphQL returns
            # 200 for content-level errors too). Mark this attempt so the final
            # failure report below doesn't fall through to _explain's blanket
            # "200 = accepted" — that would tell the user termination succeeded
            # when the GraphQL response actually reported it did not.
            attempts[-1]["graphql_failed"] = True

        # 3. RunPod currently places runpodctl in Pods. Its command grammar has
        #    changed over time, so try the documented current form and then the
        #    retained legacy alias. Secure erase may already have removed
        #    ~/.runpod/config.toml; pass the key only in the subprocess
        #    environment, never argv or the attempt log.
        runner = runner or default_runner
        cli = shutil.which("runpodctl")
        if cli:
            cli_env = dict(os.environ)
            cli_env["RUNPOD_API_KEY"] = key
            for cli_args in (["pod", "delete", iid], ["remove", "pod", iid]):
                rc, out = runner([cli, *cli_args], env=cli_env)
                rendered = " ".join([cli, *cli_args])
                attempts.append({"method": "runpodctl", "url": redact(rendered),
                                 "status": rc, "body": (out or "")[:400]})
                if rc == 0:
                    return {"ok": True, "status": rc, "provider": self.name,
                            "detail": ("Pod deleted via runpodctl (%s; delete = "
                                       "terminate, so volume-disk billing stops too)."
                                       % " ".join(cli_args[:2])),
                            "attempts": attempts}
        last = attempts[-1]
        if last.get("graphql_failed"):
            detail = ("RunPod's GraphQL termination call returned HTTP 200 but did "
                      "not report success (%s). THE INSTANCE MAY STILL BE RUNNING "
                      "AND BILLING — check the dashboard."
                      % ((last["body"] or "no body")[:200]))
        else:
            detail = _explain(last["status"], last["body"], self)
        return {"ok": False, "status": last["status"], "provider": self.name,
                "detail": detail, "attempts": attempts}


class JarvisLabs(Provider):
    name = JARVISLABS
    label = "JarvisLabs"
    supported = True
    destroys = ["the VM and its complete persistent disk (/home included)",
                "the downloaded model weights",
                "the persisted API key, TLS certificate and config state"]
    survives = ["a separately attached JarvisLabs File Storage filesystem, if one "
                "is attached; delete that filesystem separately when it is no "
                "longer needed"]
    billing = ("This does a DESTROY, not a pause. JarvisLabs pauses release compute "
               "but retain persistent storage billing; destroy permanently removes "
               "the VM disk and stops its instance storage charge.")
    docs = "https://docs.jarvislabs.ai/getting_started"

    REGION_BASE_URLS = {
        "IN1": "https://backendc.jarvislabs.net",
        "INDIA-CHENNAI-01": "https://backendc.jarvislabs.net",
        "IN2": "https://backendn.jarvislabs.net",
        "INDIA-NOIDA-01": "https://backendn.jarvislabs.net",
        "EU1": "https://backendeu.jarvislabs.net",
        "EUROPE-01": "https://backendeu.jarvislabs.net",
    }
    FETCH_BASE_URL = "https://backendn.jarvislabs.net"

    def instance_id(self):
        return (self.env.get("JARVISLABS_MACHINE_ID") or "").strip()

    def valid_instance_id(self):
        return bool(re.fullmatch(r"[0-9]+", self.instance_id()))

    def api_key(self):
        return (self.env.get("JARVISLABS_TERMINATE_API_KEY")
                or self.env.get("JARVISLABS_API_KEY")
                or self.env.get("JL_API_KEY") or "").strip()

    def key_source(self):
        if self.env.get("JARVISLABS_TERMINATE_API_KEY"):
            return "JARVISLABS_TERMINATE_API_KEY (account key supplied by the user)"
        if self.env.get("JARVISLABS_API_KEY"):
            return "JARVISLABS_API_KEY (account key supplied by the user)"
        if self.env.get("JL_API_KEY"):
            return "JL_API_KEY (JarvisLabs CLI/SDK account key supplied by the user)"
        return ""

    def region(self):
        return (self.env.get("JARVISLABS_REGION") or "").strip().upper()

    def base_url(self):
        return self.REGION_BASE_URLS.get(self.region(), "")

    def ready(self):
        ok, reason = super().ready()
        if not ok:
            return ok, reason
        if not self.base_url():
            return False, (
                "JarvisLabs: JARVISLABS_REGION must be one of IN1, IN2, or EU1 "
                "so the destroy request reaches the VM's regional backend")
        return True, "ready"

    def warnings(self):
        return [
            "JarvisLabs does not inject a VM-scoped deletion credential. "
            "Self-termination uses the account key explicitly supplied to this "
            "appliance; leave termination disabled and use the dashboard if you "
            "do not want that credential on the rented host."
        ]

    def probe(self, transport):
        if not self.api_key():
            return "rejected", "no JarvisLabs API key is available in this VM"
        if not self.instance_id() or not self.valid_instance_id():
            return "rejected", "the JarvisLabs machine id is missing or invalid"
        url = f"{self.FETCH_BASE_URL}/users/fetch/{self.instance_id()}"
        status, text = transport(
            "GET", url,
            {"Authorization": "Bearer " + self.api_key(),
             "Accept": "application/json"},
            destructive=False)
        text = text or ""
        if status == 200:
            try:
                payload = json.loads(text)
            except ValueError:
                payload = None
            instance = payload.get("instance") if isinstance(payload, dict) else None
            if isinstance(instance, dict):
                returned = str(instance.get("machine_id", ""))
                if returned == self.instance_id():
                    return "ok", (
                        f"the API key can read this JarvisLabs VM "
                        f"({self.instance_id()})")
            return "rejected", (
                "JarvisLabs accepted the credential but did not return this VM; "
                "the machine id, account, or region may be wrong")
        if status in (401, 403):
            return "rejected", (
                "JarvisLabs REFUSED this API key (HTTP %s). Terminating will "
                "fail; supply a current account key or use the dashboard."
                % status)
        if status == 404:
            return "rejected", (
                f"JarvisLabs returned no VM for machine id {self.instance_id()}")
        return "unknown", f"the credential check was inconclusive (HTTP {status})"

    def terminate(self, transport, runner=None):
        ok, reason = self.ready()
        if not ok:
            return {"ok": False, "status": None, "provider": self.name,
                    "detail": reason, "attempts": []}
        iid = self.instance_id()
        url = f"{self.base_url()}/templates/vm/destroy?machine_id={iid}"
        status, text = transport(
            "POST", url,
            {"Authorization": "Bearer " + self.api_key(),
             "Accept": "application/json"},
            destructive=True)
        attempt = {"method": "POST", "url": redact(url), "status": status,
                   "body": (text or "")[:400]}
        accepted = status in (200, 204) or status == 0
        if accepted and status != 0 and text:
            try:
                payload = json.loads(text)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                declared = payload.get("success")
                if declared is False or str(declared).lower() == "false":
                    accepted = False
        return {
            "ok": accepted,
            "status": status,
            "provider": self.name,
            "detail": _explain(status, text, self) if accepted else (
                "JarvisLabs did not accept the destroy request. THE VM MAY STILL "
                "BE RUNNING AND BILLING — check the dashboard."),
            "attempts": [attempt],
        }


class UnknownProvider(Provider):
    pass


PROVIDERS = {
    VAST: VastAI,
    RUNPOD: RunPod,
    JARVISLABS: JarvisLabs,
    UNKNOWN: UnknownProvider,
}


def _explain(status, text, provider):
    """Turn a status code into something a user can act on. 'It failed' without
    'you are still being billed' is the one message we must never send."""
    if status == 0:
        return ("DRY RUN — the request above was NOT sent. The instance is still "
                "running and still billing.")
    if status in (200, 204):
        return ("Terminate accepted by the provider. The instance should disappear "
                "from your dashboard within a minute; confirm there that billing "
                "has stopped.")
    if status in (401, 403):
        extra = ""
        if provider.name == RUNPOD:
            extra = (" RunPod's injected pod-scoped key has deleted its own pod on "
                     "some deployments and has been refused on others. Supply an "
                     "account API key as RUNPOD_TERMINATE_API_KEY (Settings -> API "
                     "Keys), or terminate from the dashboard.")
        return ("The provider REFUSED the credential (HTTP %s). THE INSTANCE IS STILL "
                "RUNNING AND STILL BILLING — terminate it from the dashboard.%s"
                % (status, extra))
    if status == 404:
        return ("The provider does not know this instance id (HTTP 404). It may "
                "already be gone — CHECK THE DASHBOARD; do not assume billing stopped.")
    if status == 429:
        return ("Rate limited by the provider (HTTP 429). THE INSTANCE IS STILL "
                "RUNNING AND STILL BILLING — retry, or use the dashboard.")
    if status is None:
        return ("The request never got a reply (%s). THE INSTANCE IS PROBABLY STILL "
                "RUNNING AND STILL BILLING — check the dashboard." % (text or "network error"))
    return ("Unexpected response HTTP %s: %s. THE INSTANCE MAY STILL BE RUNNING AND "
            "BILLING — check the dashboard." % (status, (text or "")[:200]))


def detect(env=None):
    """Which provider are we on? Manual override wins, then unambiguous
    provider-injected identifiers, then unknown.

    There is no cloud metadata service to fall back on — VERIFIED unreachable
    (169.254.169.254) from a RunPod pod — so identity comes entirely from the
    variables the provider injected into PID 1."""
    env = gc.effective_env(env)
    override = (env.get("TERMINATE_PROVIDER") or "").strip().lower()
    if override:
        return override if override in PROVIDERS else UNKNOWN
    if env.get("JARVISLABS_MACHINE_ID"):
        return JARVISLABS
    if env.get("RUNPOD_POD_ID"):
        return RUNPOD
    if env.get("CONTAINER_ID") and env.get("CONTAINER_API_KEY"):
        return VAST
    # A vast instance with the label-update credentials missing still looks like
    # vast if the port variables are there.
    if any(k.startswith("VAST_TCP_PORT_") for k in env):
        return VAST
    return UNKNOWN


def get(env=None):
    env = gc.effective_env(env)
    return PROVIDERS.get(detect(env), UnknownProvider)(env)


def detection_evidence(env=None):
    """What the detector saw — shown on the confirmation screen so a user can
    tell a misdetection from a missing credential."""
    env = gc.effective_env(env)
    keys = ("TERMINATE_PROVIDER", "JARVISLABS_MACHINE_ID",
            "JARVISLABS_REGION", "JARVISLABS_API_KEY",
            "JARVISLABS_TERMINATE_API_KEY", "JL_API_KEY",
            "RUNPOD_POD_ID", "RUNPOD_API_KEY",
            "RUNPOD_TERMINATE_API_KEY", "CONTAINER_ID", "CONTAINER_API_KEY",
            "VAST_API_KEY", "PUBLIC_IPADDR", "RUNPOD_DC_ID", "RUNPOD_GPU_NAME",
            "RUNPOD_POD_HOSTNAME", "RUNPOD_VOLUME_ID")
    out = {}
    for k in keys:
        v = env.get(k)
        if not v:
            continue
        out[k] = v if not k.endswith(("API_KEY",)) and k != "JL_API_KEY" \
            else f"set ({len(v)} chars)"
    ports = sorted(k for k in env if k.startswith(("VAST_TCP_PORT_", "RUNPOD_TCP_PORT_")))
    if ports:
        out["port_vars"] = ", ".join(ports[:6])
    return out


if __name__ == "__main__":                      # read-only introspection
    p = get()
    print(json.dumps({"detected": detect(), "describe": p.describe(),
                      "evidence": detection_evidence()}, indent=1))
