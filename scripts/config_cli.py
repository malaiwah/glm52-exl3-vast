#!/usr/bin/env python3
"""Shell-facing side of the self-service config (entrypoint.sh drives this).

    config_cli.py snapshot-env          freeze the startup env layer (once, at boot)
    config_cli.py env                   print `export K=V` for the resolved config
    config_cli.py env --begin-attempt   freeze this boot's config; export its token
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

Exit codes: 0 ok, 1 no useful rollback, 2 validation errors,
3 superseded attempt or nothing to do.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import uuid

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


def _context():
    """Validation context the shell can supply: how many GPUs this host has,
    and which keys the state file actually persisted."""
    ctx = {}
    # A fresh SSH shell does not inherit the boot exports, so fall back to
    # PID 1's environment and then to the frozen startup snapshot — the same
    # sources the rest of the config layer reads — before giving up and
    # silently skipping the gpu-count validations.
    for source in (os.environ, gc.effective_env(), gc.load_startup_env()):
        try:
            ctx["gpu_count"] = int(str(source.get("GLM_GPU_COUNT", "")).strip())
            break
        except (TypeError, ValueError):
            continue
    try:
        ctx["state_keys"] = list(gc.load_state_file())
    except Exception:
        ctx["state_keys"] = []
    return ctx


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


def cmd_env(args):
    with gc.state_lock():
        effective, sources, notes = _resolved()
        derived = gc.derive(effective)
        attempt = None
        if getattr(args, "begin_attempt", False):
            attempt = {
                "token": uuid.uuid4().hex,
                "state_identity": _state_identity(),
                "effective": effective,
                "sources": sources,
                "values": _state_values(),
            }
            gc.write_json_atomic(_attempt_path(), attempt, mode=0o600)
    for note in notes:
        print(f"!!! config: {note}", file=sys.stderr)
    lines = []
    for knob in gc.KNOBS:
        key = knob["key"]
        lines.append("export %s=%s" % (key, shlex.quote(gc.to_text(knob, effective[key]))))
    # NB: QUANTIZATION is deliberately NOT exported. --quantization is part of
    # FAMILY_SERVE_ARGS now (it comes from the variant), and exporting it as well
    # would be a second source of truth for one flag — the kind of thing the knob
    # wiring audit exists to catch.
    for key in ("MODEL_REPO", "MODEL_REVISION", "MODEL_DIRNAME", "MTP78_MODE",
                "DRAFT_MODEL", "DRAFT_QUANTIZATION", "FAMILY_ENV_BLOCK", "SPEC_METHOD",
                "NATIVE_MTP_FORMAT", "MTP_GRAFT_COMPATIBLE"):
        lines.append("export %s=%s" % (key, shlex.quote(str(derived.get(key, "")))))
    # A bash ARRAY, not a string: these values are JSON with spaces and braces,
    # and word-splitting them would corrupt the serve line. Arrays cannot be
    # exported, which is fine — config.env is sourced into the same shell.
    lines.append("FAMILY_SERVE_ARGS=(%s)" % " ".join(
        shlex.quote(a) for a in derived.get("FAMILY_SERVE_ARGS", [])))
    lines.append("PROFILE_RUNTIME_ENV=(%s)" % " ".join(
        shlex.quote(a) for a in derived.get("PROFILE_RUNTIME_ENV", [])))
    # a compact source map so the boot log can explain where a value came from
    lines.append("export GLM_CONFIG_SOURCES=%s" % shlex.quote(json.dumps(sources)))
    if attempt is not None:
        lines.append("export GLM_CONFIG_ATTEMPT=%s" % shlex.quote(attempt["token"]))
    print("\n".join(lines))
    return 0


def cmd_show(_args):
    effective, sources, notes = _resolved()
    findings = gc.validate(effective, _context())
    width = max(len(k["key"]) for k in gc.KNOBS)
    fam = gc.family(effective.get("MODEL_FAMILY"))
    print(">>> model family: %s%s" % (fam["label"],
                                      "" if fam.get("tested") else "  [UNTESTED PRESET]"))
    print(">>> effective configuration "
          "(default < family < variant < env < state file):")
    for knob in gc.KNOBS:
        key = knob["key"]
        if sources[key] == "n/a":
            print("      %-*s = %-28s [not applicable to this family]" % (
                width, key, "-"))
            continue
        val = gc.to_text(knob, effective[key])
        if key == "DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS" and val == "-1":
            val = "disabled"
        print("      %-*s = %-28s [%s]" % (width, key, val, sources[key]))
    for note in notes:
        print(f"!!! config: {note}")
    for f in findings:
        print(f"{'!!!' if f['level'] == 'error' else ' * '} {f['level']}: {f['message']}")
    return 0


def cmd_validate(args):
    with gc.state_lock():
        attempt = _current_attempt()
        if attempt is False:
            return 3
        effective = attempt["effective"] if attempt else _resolved()[0]
        context = _context()
        if attempt:
            context["state_keys"] = list(attempt["values"])
        findings = gc.validate(effective, context)
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


def _attempt_path():
    return os.path.join(gc.runtime_dir(), "config-attempt.json")


def _state_identity():
    """Include raw metadata such as written_at, not just resolved knob values."""
    try:
        with open(gc.p_state(), "rb") as state:
            return hashlib.sha256(state.read()).hexdigest()
    except FileNotFoundError:
        return None


def _current_attempt():
    """None is an explicit manual CLI operation; False is a superseded boot.

    Call only under state_lock. Comparing the token prevents an earlier boot
    from completing after a later attempt of the same configuration; comparing
    raw state catches dashboard writes before PID 1 begins the next attempt.
    """
    if "GLM_CONFIG_ATTEMPT" not in os.environ:
        return None
    token = os.environ["GLM_CONFIG_ATTEMPT"]
    attempt = gc.read_json(_attempt_path())
    if (not token or not isinstance(attempt, dict)
            or attempt.get("token") != token
            or "state_identity" not in attempt
            or not isinstance(attempt.get("effective"), dict)
            or not isinstance(attempt.get("sources"), dict)
            or not isinstance(attempt.get("values"), dict)
            or attempt["state_identity"] != _state_identity()):
        print(">>> config attempt superseded; retry the current configuration",
              file=sys.stderr)
        return False
    return attempt


def cmd_mark_good(args):
    # Serialize against a concurrent landing-page apply/reset (see gc.state_lock).
    with gc.state_lock():
        attempt = _current_attempt()
        if attempt is False:
            return 3
        if attempt is None:
            effective, sources, _notes = _resolved()
            values = _state_values()
        else:
            effective, sources = attempt["effective"], attempt["sources"]
            values = attempt["values"]
        verify = gc.read_json(gc.p_verify_last())
        doc = {"ts": _now(), "values": values, "effective": effective,
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
    """Preserve the failed config + its log, restore the last known-good.

    Under the shared state lock, reject completions from superseded boots
    before preserving or restoring anything. Locking alone only serializes a
    newer dashboard apply and an older boot failure; the attempt check stops
    the latter from overwriting the former after it acquires the lock."""
    with gc.state_lock():
        attempt = _current_attempt()
        if attempt is False:
            return 3
        return _cmd_rollback_locked(args, attempt)


def _known_good_replay(good):
    """Return state values that reproduce a stored effective config now.

    Known-good snapshots may have been recorded entirely from the old startup
    environment, leaving their historical ``values`` diff empty. Re-minimizing
    the full effective snapshot against the current startup environment makes
    rollback portable across container replacements. Refuse rather than claim
    success if a non-editable or detected layer prevents an exact replay.
    """
    target = good.get("effective") if isinstance(good, dict) else None
    if not isinstance(target, dict):
        return None, "known-good record has no effective configuration"
    current_env = gc.load_startup_env()
    replay = gc.minimize(target)
    restored, _sources, _notes = gc.resolve(
        state_values=replay, env_values=current_env)
    family_name = str(target.get("MODEL_FAMILY") or "glm52")
    comparable_keys = {"MODEL_FAMILY", "MODEL_VARIANT"}
    comparable_keys.update(
        knob["key"] for knob in gc.KNOBS
        if gc.applies_to(knob, family_name)
    )
    target_contract = {
        key: value for key, value in target.items()
        if key in comparable_keys
    }
    restored_contract = {
        key: value for key, value in restored.items()
        if key in target_contract
    }
    differences = gc.diff(target_contract, restored_contract)
    if differences:
        keys = ", ".join(key for key, _old, _new in differences)
        return None, (
            "known-good configuration is not replayable under this startup "
            f"environment ({keys})"
        )

    context = _context()
    context["state_keys"] = list(replay)
    try:
        context["gpu_count"] = int(
            str(current_env.get("GLM_GPU_COUNT", "")).strip())
    except (TypeError, ValueError):
        pass
    replay_errors = gc.errors(gc.validate(restored, context))
    if replay_errors:
        finding_ids = ", ".join(finding["id"] for finding in replay_errors)
        return None, (
            "known-good configuration is invalid on the current host "
            f"({finding_ids})"
        )
    return replay, ""


def _cmd_rollback_locked(args, attempt=None):
    if attempt is None:
        effective, _sources, _notes = _resolved()
        failed_values = _state_values()
    else:
        effective, failed_values = attempt["effective"], attempt["values"]
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
        f.write(gc.diff_text(good.get("effective", {}) if good else {}, effective))
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
            except FileNotFoundError:
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
        _prune_failures()
        print("!!! rollback: no known-good config and no state override to drop",
              file=sys.stderr)
        return 3

    replay, error = _known_good_replay(good)
    if error:
        print(f"!!! rollback refused: {error}", file=sys.stderr)
        gc.set_apply_state("failed", since=ts, failure=ts, detail=args.reason,
                           restored="")
        _prune_failures()
        return 3
    if not gc.diff(effective, good["effective"]):
        print("!!! rollback: known-good is identical to failed config; "
              "not restarting into the same failure", file=sys.stderr)
        gc.set_apply_state("failed", since=ts, failure=ts, detail=args.reason,
                           restored="")
        _prune_failures()
        return 3

    gc.write_json_atomic(gc.p_state(), {
        "values": replay,
        "written_at": gc.utcnow_iso(),
    }, mode=0o600)
    gc.set_apply_state("rolled-back", since=ts, failure=ts, detail=args.reason,
                       restored=good.get("ts"))
    open(gc.p_restart_flag(), "w").write("rollback\n")
    _prune_failures()
    print(f">>> rolled back to the known-good config from {good.get('ts')} "
          f"(failed config preserved in {fdir})")
    return 0


def cmd_should_rollback(_args):
    """Exit 0 for a useful rollback, 1 for none, 3 for a superseded boot."""
    with gc.state_lock():
        attempt = _current_attempt()
        if attempt is False:
            return 3
        if attempt is None:
            effective, _sources, _notes = _resolved()
            values = _state_values()
        else:
            effective, values = attempt["effective"], attempt["values"]
        good = gc.read_json(gc.p_known_good())
        if not good:
            return 0 if values else 1
        _replay, error = _known_good_replay(good)
        if error:
            return 1
        return 0 if gc.diff(effective, good["effective"]) else 1


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
        soul_marker = os.path.join(gc.state_dir(), "soul", "incidents",
                                   "rollback-" + d + ".json")
        if (os.path.isdir(fdir)
                and not os.path.exists(os.path.join(fdir, "analysis.md"))
                and not os.path.exists(soul_marker)):
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


def cmd_mark_unverified(args):
    with gc.state_lock():
        if _current_attempt() is False:
            return 3
        gc.set_apply_state("degraded", detail=args.reason or "verification failed")
    return 0


COMMANDS = {
    "snapshot-env": cmd_snapshot_env,
    "env": cmd_env,
    "show": cmd_show,
    "validate": cmd_validate,
    "mark-good": cmd_mark_good,
    "mark-unverified": cmd_mark_unverified,
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
    ap.add_argument("--begin-attempt", action="store_true",
                    help="env: freeze this boot's configuration for guarded completion")
    args = ap.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
