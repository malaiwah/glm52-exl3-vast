#!/usr/bin/env python3
"""Stop the engine, erase the session, destroy the instance — in that order.

Run detached by the landing page so the flow survives the HTTP request that
started it. Progress is written to $GLM_RUNTIME_DIR/terminate.json, which the
page polls; that file is deliberately excluded from the erase and is the last
thing standing when the provider pulls the plug.

SAFETY. The destructive provider call happens in exactly one place, through a
transport that must be constructed with allow_destructive=True. Every gate is
re-checked HERE and not merely in the UI, because "the UI checked it" is not a
property of the system:

  1. the anti-kill switch / kill switch (glm_config.termination_allowed)
  2. the typed confirmation, re-validated against the instance id
  3. the provider being identified and credentialed

TERMINATE_DRY_RUN=1 (or --dry-run) runs everything except the destroy request
and says so loudly. That is how this was exercised during development; no real
endpoint was ever called.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc      # noqa: E402
import provider as prov      # noqa: E402
import secure_erase          # noqa: E402

STOP_TIMEOUT_S = 180


def p_progress():
    return os.path.join(gc.runtime_dir(), "terminate.json")


def p_request_stop():
    return os.path.join(gc.runtime_dir(), "terminate-in-progress")


def p_engine_stopped():
    return os.path.join(gc.runtime_dir(), "engine-stopped")
def _clear_request_stop():
    """Remove the terminate-in-progress flag so the supervisor may restart the
    engine on a failed terminate. Safe to call when the flag is absent."""
    try:
        os.remove(p_request_stop())
    except OSError:
        pass


def read_progress():
    return gc.read_json(p_progress())


class Progress:
    def __init__(self, dry_run=False, sink=None):
        self.doc = {"started": gc.utcnow_iso(), "phase": "starting", "steps": [],
                    "done": False, "ok": None, "detail": "", "dry_run": bool(dry_run)}
        self.sink = sink
        self.flush()

    def step(self, phase, detail="", **kw):
        self.doc["phase"] = phase
        self.doc["steps"].append({"ts": gc.utcnow_iso(), "phase": phase,
                                  "detail": detail})
        self.doc.update(kw)
        self.flush()

    def finish(self, ok, detail, **kw):
        self.doc.update({"done": True, "ok": bool(ok), "detail": detail,
                         "finished": gc.utcnow_iso()})
        self.doc.update(kw)
        self.doc["phase"] = "terminated" if ok else "failed"
        self.flush()
        return self.doc

    def flush(self):
        if self.sink is not None:
            self.sink(self.doc)
            return
        gc.write_json_atomic(p_progress(), self.doc, mode=0o644)


def default_stopper(progress, timeout=STOP_TIMEOUT_S):
    """Ask the supervisor to stop vLLM and stay stopped.

    The supervisor owns the engine's lifecycle; racing it with our own kill
    would just have it restart the engine underneath the erase. It watches for
    this flag, kills the tree, and touches engine-stopped."""
    os.makedirs(gc.runtime_dir(), exist_ok=True)
    with open(p_request_stop(), "w") as f:
        f.write("terminate\n")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(p_engine_stopped()):
            return True, "engine stopped by the supervisor"
        time.sleep(2)
    return False, (f"the supervisor did not confirm the engine stopped within "
                   f"{timeout}s; continuing anyway")


def run(opts, transport=None, runner=None, provider_obj=None, stopper=None,
        progress=None, env=None):
    """-> the final progress document. Never raises."""
    # A worker launched by a separately restarted landing process may not have
    # the provider variables RunPod injected into PID 1. Preserve explicit test
    # environments exactly, but otherwise use the repository-wide PID 1
    # fallback contract instead of freezing this process's reduced os.environ.
    runtime_env = gc.effective_env(env)
    dry_run = bool(opts.get("dry_run")) or gc.env_flag(
        runtime_env, "TERMINATE_DRY_RUN", False)
    pr = progress or Progress(dry_run=dry_run)
    p = provider_obj or prov.get(runtime_env)

    # ---- gate 1: the switches -------------------------------------------
    allowed, why = gc.termination_allowed(runtime_env)
    if not allowed:
        return pr.finish(False, why, refused="switch")

    # ---- gate 2: the typed confirmation ---------------------------------
    expected = p.instance_id() or "TERMINATE"
    if (opts.get("confirm") or "").strip() != expected:
        return pr.finish(False, "the typed confirmation did not match the instance id; "
                                "nothing was done", refused="confirmation")

    # ---- gate 3: provider is usable -------------------------------------
    ok, reason = p.ready()
    if not ok:
        return pr.finish(False, reason, refused="provider")

    pr.step("stopping-engine", "stopping vLLM so VRAM can be cleared and no new "
                               "requests are logged")
    if dry_run:
        stopped, detail = True, "dry run: engine stop skipped"
    else:
        stopper = stopper or default_stopper
        stopped, detail = stopper(pr)
    pr.step("engine-stopped", detail, engine_stopped=stopped)

    # ---- erase -----------------------------------------------------------
    if opts.get("erase"):
        pr.step("planning-erase", "listing session data")
        plan_doc = secure_erase.plan(keep=(p_progress(), p_request_stop(),
                                           p_engine_stopped()))
        pr.step("erasing",
                f"{plan_doc['count']} files, {plan_doc['total_bytes'] / 2**20:.1f} MiB",
                erase_plan={k: plan_doc[k] for k in
                            ("count", "total_bytes", "groups", "manifest_used")},
                unknown_large=plan_doc["unknown_large"])
        result = secure_erase.erase(plan_doc, dry_run=dry_run)
        pr.step("erased", f"{result['erased']} file(s), "
                          f"{result['bytes'] / 2**20:.1f} MiB overwritten",
                erase_result=result)
        if opts.get("ram"):
            pr.step("erasing-ram", "dropping caches and overwriting free RAM")
            pr.step("erased-ram", "", ram_result=secure_erase.erase_ram(dry_run=dry_run))
        if opts.get("vram"):
            pr.step("erasing-vram", "zeroing device memory")
            pr.step("erased-vram", "", vram_result=secure_erase.erase_vram(dry_run=dry_run))
    else:
        pr.step("erase-skipped", "secure erase was not requested (it is off by default)")

    # ---- the destructive call -------------------------------------------
    pr.step("terminating", f"asking {p.label} to destroy instance {p.instance_id()}")
    # Re-check the kill switch: a lock can be thrown mid-flight through the
    # landing-page ratchet. If termination is no longer allowed, clean up the
    # request-stop flag so the supervisor may restart the engine, then bail.
    allowed, why = gc.termination_allowed(runtime_env)
    if not allowed:
        _clear_request_stop()
        return pr.finish(False, f"termination was refused mid-flight: {why}. "
                                "Serving was stopped; the operator may need to "
                                "restart the engine via the provider dashboard.",
                        refused="switch")
    transport = transport or prov.HttpTransport(allow_destructive=True, dry_run=dry_run)
    try:
        res = p.terminate(transport, runner=runner)
    except prov.NotArmed as e:
        _clear_request_stop()
        return pr.finish(False, f"internal safety refusal: {e}. Serving was "
                                "stopped; the operator may need to restart the "
                                "engine via the provider dashboard.",
                        refused="not-armed")
    except Exception as e:
        _clear_request_stop()
        return pr.finish(False, f"the terminate call raised {type(e).__name__}: {e}. "
                                "Serving was stopped; THE INSTANCE MAY STILL BE "
                                "RUNNING AND BILLING — check your provider "
                                "dashboard, and restart the engine if needed.")
    if not res.get("ok"):
        _clear_request_stop()
        return pr.finish(False, res.get("detail", "") + " Serving was stopped; the "
                                "operator may need to restart the engine via the "
                                "provider dashboard.", terminate=res)
    return pr.finish(bool(res.get("ok")), res.get("detail", ""), terminate=res)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--erase", action="store_true")
    ap.add_argument("--ram", action="store_true")
    ap.add_argument("--vram", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    doc = run({"confirm": args.confirm, "erase": args.erase, "ram": args.ram,
               "vram": args.vram, "dry_run": args.dry_run})
    print(json.dumps(doc, indent=1))
    return 0 if doc.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
