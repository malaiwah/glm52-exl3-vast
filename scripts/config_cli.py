#!/usr/bin/env python3
"""Shell-facing side of the self-service config (entrypoint.sh drives this).

    config_cli.py snapshot-env          freeze the startup env layer (once, at boot)
    config_cli.py env                   print `export K=V` for the resolved config
    config_cli.py show                  human-readable table of value + source
    config_cli.py validate [--quiet]    exit 2 if the resolved config has errors
    config_cli.py mark-good --log F     record the running config as known-good
    config_cli.py rollback --log F --reason R
                                        preserve the failed config+log, restore the
                                        last known-good, ask for a restart
    config_cli.py should-rollback       exit 0 if a rollback would change anything
    config_cli.py switches              print the termination switches as JSON
    config_cli.py pending-analysis      print a failure dir that has no analysis yet
    config_cli.py request-restart       set the restart flag
    config_cli.py clear-restart         clear the restart flag

Exit codes: 0 ok, 2 validation errors, 3 nothing to do.
"""
import argparse
import json
import os
import shlex
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc  # noqa: E402

TAIL_BYTES = 512 * 1024
TAIL_LINES = 4000
KEEP_FAILURES = 10


def _tail(path, lines=TAIL_LINES, nbytes=TAIL_BYTES):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
                f.readline()          # drop the partial first line
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    parts = data.splitlines()
    return "\n".join(parts[-lines:])


def _now():
    return gc.utcnow_stamp()


def _resolved():
    effective, sources, notes = gc.resolve()
    return effective, sources, notes


# --------------------------------------------------------------------------

def cmd_snapshot_env(_args):
    """Once per container start: freeze the env layer AND derive the
    termination switches from it. Both are per-container by design — the
    switches live in the runtime dir so that loosening them requires a restart
    with a different environment, i.e. provider-level access."""
    layer = gc.snapshot_startup_env()
    os.makedirs(gc.state_dir(), exist_ok=True)
    os.makedirs(gc.runtime_dir(), exist_ok=True)
    os.makedirs(gc.p_logs(), exist_ok=True)
    os.makedirs(gc.p_failures(), exist_ok=True)
    print(f"startup env layer: {len(layer)} knob(s) set from the environment"
          + (": " + ", ".join(sorted(layer)) if layer else ""))
    sw = gc.init_switches()
    print(">>> termination: kill switch %s, anti-kill switch %s" % (
        "ON (TERMINATE_ENABLED=1)" if sw["enabled"] else "off (default)",
        "LOCKED (TERMINATE_LOCKED=1)" if sw["locked"] else "not locked"))
    return 0


def cmd_env(_args):
    effective, sources, notes = _resolved()
    derived = gc.derive(effective)
    for note in notes:
        print(f"!!! config: {note}", file=sys.stderr)
    lines = []
    for knob in gc.KNOBS:
        key = knob["key"]
        lines.append("export %s=%s" % (key, shlex.quote(gc.to_text(knob, effective[key]))))
    for key in ("MODEL_REPO", "MODEL_DIRNAME", "QUANTIZATION", "MTP78_MODE",
                "DRAFT_MODEL", "DRAFT_QUANTIZATION"):
        lines.append("export %s=%s" % (key, shlex.quote(str(derived.get(key, "")))))
    # a compact source map so the boot log can explain where a value came from
    lines.append("export GLM_CONFIG_SOURCES=%s" % shlex.quote(json.dumps(sources)))
    print("\n".join(lines))
    return 0


def cmd_show(_args):
    effective, sources, notes = _resolved()
    findings = gc.validate(effective)
    width = max(len(k["key"]) for k in gc.KNOBS)
    print(">>> effective configuration (default < env < state file):")
    for knob in gc.KNOBS:
        key = knob["key"]
        print("      %-*s = %-28s [%s]" % (width, key,
                                           gc.to_text(knob, effective[key]), sources[key]))
    for note in notes:
        print(f"!!! config: {note}")
    for f in findings:
        print(f"{'!!!' if f['level'] == 'error' else ' * '} {f['level']}: {f['message']}")
    return 0


def cmd_validate(args):
    effective, _sources, _notes = _resolved()
    findings = gc.validate(effective)
    errs = gc.errors(findings)
    if not args.quiet:
        for f in findings:
            print(f"{f['level']}: [{f['id']}] {f['message']}")
    return 2 if errs else 0


def _state_values():
    try:
        return gc.load_state_file()
    except Exception:
        return {}


def cmd_mark_good(args):
    effective, sources, _notes = _resolved()
    verify = gc.read_json(gc.p_verify_last())
    doc = {"ts": _now(), "values": _state_values(), "effective": effective,
           "sources": sources, "verify": verify}
    gc.write_json_atomic(gc.p_known_good(), doc, mode=0o644)
    if args.log and os.path.exists(args.log):
        os.makedirs(gc.p_logs(), exist_ok=True)
        with open(os.path.join(gc.p_logs(), "last-good.log"), "w") as f:
            f.write(_tail(args.log))
    gc.set_apply_state("steady", since=doc["ts"], detail="verified")
    print(">>> config marked known-good")
    return 0


def _prune_failures():
    root = gc.p_failures()
    try:
        dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    except OSError:
        return
    for d in dirs[:-KEEP_FAILURES]:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)


def cmd_rollback(args):
    """Preserve the failed config + its log, restore the last known-good."""
    effective, _sources, _notes = _resolved()
    failed_values = _state_values()
    good = gc.read_json(gc.p_known_good())
    log_text = _tail(args.log) if args.log else ""

    ts = _now()
    fdir = os.path.join(gc.p_failures(), ts)
    os.makedirs(fdir, exist_ok=True)
    gc.write_json_atomic(os.path.join(fdir, "config.json"),
                         {"values": failed_values, "effective": effective}, mode=0o644)
    with open(os.path.join(fdir, "error.log"), "w") as f:
        f.write(log_text)
    with open(os.path.join(fdir, "diff.txt"), "w") as f:
        f.write(gc.diff_text(good.get("effective", {}), effective))
    gc.write_json_atomic(os.path.join(fdir, "meta.json"), {
        "ts": ts,
        "reason": args.reason,
        "signatures": gc.match_signatures(log_text),
        "had_known_good": bool(good),
        "analysis": None,
    }, mode=0o644)

    if not good:
        # Nothing to fall back to. Clearing the state file at least drops the
        # user's edits and returns to template env + built-in defaults; if the
        # state file was already empty, the failure is in env/defaults and only
        # the crash-loop budget applies.
        if failed_values:
            try:
                os.remove(gc.p_state())
            except OSError:
                pass
            gc.set_apply_state("rolled-back", since=ts, failure=ts, detail=args.reason,
                               restored="built-in defaults + template env")
            print(">>> no known-good config yet: dropped the state file, "
                  "falling back to env + defaults")
            open(gc.p_restart_flag(), "w").write("rollback\n")
            _prune_failures()
            return 0
        gc.set_apply_state("failed", since=ts, failure=ts, detail=args.reason,
                           restored="")
        print("!!! no known-good config and nothing to drop — the failure is in the "
              "template env or the built-in defaults; not restarting in a loop")
        _prune_failures()
        return 3

    gc.write_json_atomic(gc.p_state(), {"values": good.get("values", {}),
                                        "restored_from": good.get("ts")}, mode=0o600)
    gc.set_apply_state("rolled-back", since=ts, failure=ts, detail=args.reason,
                       restored=good.get("ts"))
    open(gc.p_restart_flag(), "w").write("rollback\n")
    _prune_failures()
    print(f">>> rolled back to the known-good config from {good.get('ts')} "
          f"(failed config preserved in {fdir})")
    return 0


def cmd_should_rollback(_args):
    """Exit 0 when rolling back would actually change something.

    Rollback is only meaningful when the running configuration DIFFERS from the
    last known-good one. Restarting the same config in a loop is a crash loop,
    not a rollback, and the supervisor's restart budget is the right tool for
    that. With no known-good config at all, dropping a non-empty state file is
    still a real move (it returns the instance to template env + defaults)."""
    effective, _sources, _notes = _resolved()
    good = gc.read_json(gc.p_known_good())
    if not good:
        return 0 if _state_values() else 1
    return 0 if gc.diff(good.get("effective", {}), effective) else 1


def cmd_switches(_args):
    allowed, reason = gc.termination_allowed()
    print(json.dumps({"switches": gc.read_switches(), "allowed": allowed,
                      "reason": reason}, indent=1))
    return 0


def cmd_pending_analysis(_args):
    root = gc.p_failures()
    try:
        dirs = sorted(os.listdir(root))
    except OSError:
        return 3
    for d in reversed(dirs):
        fdir = os.path.join(root, d)
        if os.path.isdir(fdir) and not os.path.exists(os.path.join(fdir, "analysis.md")):
            print(fdir)
            return 0
    return 3


def cmd_request_restart(_args):
    os.makedirs(gc.runtime_dir(), exist_ok=True)
    with open(gc.p_restart_flag(), "w") as f:
        f.write("cli\n")
    return 0


def cmd_clear_restart(_args):
    try:
        os.remove(gc.p_restart_flag())
    except OSError:
        pass
    return 0


COMMANDS = {
    "snapshot-env": cmd_snapshot_env,
    "env": cmd_env,
    "show": cmd_show,
    "validate": cmd_validate,
    "mark-good": cmd_mark_good,
    "rollback": cmd_rollback,
    "should-rollback": cmd_should_rollback,
    "switches": cmd_switches,
    "pending-analysis": cmd_pending_analysis,
    "request-restart": cmd_request_restart,
    "clear-restart": cmd_clear_restart,
}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--log", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
