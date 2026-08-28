#!/usr/bin/env python3
"""Configuration and durable-state helpers for the embedded appliance SOUL.

This module intentionally uses only the Python standard library so the landing
page and PID 1 can resolve SOUL settings without importing Nanobot's venv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

SCHEMA_VERSION = 1

FIELDS = {
    "autonomyLevel": ("SOUL_AUTONOMY_LEVEL", 0, 0, 3),
    "heartbeatIntervalS": ("SOUL_HEARTBEAT_INTERVAL_S", 300, 30, 86400),
    "journalIntervalS": ("SOUL_JOURNAL_INTERVAL_S", 3600, 300, 604800),
    "journalRetentionDays": ("SOUL_JOURNAL_RETENTION_DAYS", 90, 1, 3650),
    "evidenceRetentionDays": ("SOUL_EVIDENCE_RETENTION_DAYS", 7, 1, 365),
    "maxStateMB": ("SOUL_MAX_STATE_MB", 256, 16, 4096),
}
TIMEZONE_ENV = "SOUL_TIMEZONE"
TIMEZONE_DEFAULT = "UTC"

SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(authorization\s*:\s*)(?:bearer|basic)\s+\S+"), r"\1[REDACTED]"),
    (re.compile(
        r"(?i)(?<![A-Za-z0-9])"
        r"(api[-_]?key|secret[-_]?key|private[-_]?key|token|secret|password|pass|pwd|authorization)"
        r"(\s*[=:]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"),
     r"\1\2[REDACTED]"),
    (re.compile(
        r"""(?ix)(["'](?:[A-Za-z][A-Za-z0-9_-]*[_-])?
        (?:api[-_]?key|token|secret|password|private[-_]?key|authorization)
        ["']\s*:\s*)(?:"[^"]*"|'[^']*'|[^\s,;}]+)"""),
     r'\1"[REDACTED]"'),
    (re.compile(r"(?i)(https?://[^\s?#]+[?&](?:token|access_token|key|api_key|"
                r"signature|sig|x-amz-signature)=)[^&#\s]+"),
     r"\1[REDACTED]"),
    (re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(?:hf_|sk-)[A-Za-z0-9_-]{12,}\b"), "[REDACTED TOKEN]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
                r"[A-Za-z0-9_-]{8,}\b"), "[REDACTED JWT]"),
)


def _secret_key(name: Any) -> bool:
    normalized = re.sub(r"[-\s]+", "_", str(name).strip().lower())
    parts = normalized.split("_")
    return (
        any(part in {"token", "secret", "password", "pass", "pwd", "authorization"} for part in parts)
        or normalized.endswith(("api_key", "apikey", "private_key", "secret_access_key"))
    )


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def state_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    return Path(e.get("GLM_STATE_DIR", "/workspace/.glm-config")) / "soul"


def runtime_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    return Path(e.get("GLM_RUNTIME_DIR", "/tmp/glm-runtime")) / "soul"


def paths(env: Optional[Mapping[str, str]] = None) -> Dict[str, Path]:
    root = state_dir(env)
    run = runtime_dir(env)
    return {
        "root": root,
        "config": root / "config.json",
        "status": root / "status.json",
        "journal": root / "journal.jsonl",
        "incidents": root / "incidents",
        "evidence": root / "evidence",
        "snapshots": root / "snapshots",
        "workspace": root / "workspace",
        "logs": root / "logs",
        "runtime": run,
        "reconcile": run / "reconcile",
    }


def ensure_tree(env: Optional[Mapping[str, str]] = None) -> Dict[str, Path]:
    p = paths(env)
    for key in ("root", "incidents", "evidence", "snapshots", "workspace", "logs", "runtime"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p


def _parse_int(value: Any, low: int, high: int) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if low <= parsed <= high else None


def _timezone(value: Any) -> Optional[str]:
    value = str(value or "").strip()
    if not value or len(value) > 64 or not re.fullmatch(r"[A-Za-z0-9_+\-/]+", value):
        return None
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(value)
    except Exception:
        return None
    return value


def load_override(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    try:
        doc = json.loads(paths(env)["config"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    values = doc.get("values", doc) if isinstance(doc, dict) else {}
    return values if isinstance(values, dict) else {}


def resolve(env: Optional[Mapping[str, str]] = None,
            override: Optional[Mapping[str, Any]] = None
            ) -> Tuple[Dict[str, Any], Dict[str, str], list[str]]:
    """Resolve built-ins < startup environment < persistent override.

    Invalid autonomy levels fail closed to 0. Invalid interval-like settings
    fall through to their lower layer. The maximum is startup-only.
    """
    e = dict(os.environ if env is None else env)
    supplied = load_override(e) if override is None else dict(override)
    result: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    notes: list[str] = []
    for name, (env_name, default, low, high) in FIELDS.items():
        value, source = default, "default"
        if env_name in e:
            parsed = _parse_int(e.get(env_name), low, high)
            if parsed is not None:
                value, source = parsed, "environment"
            elif name == "autonomyLevel":
                value, source = 0, "environment-invalid"
                notes.append(f"{env_name} is invalid; autonomy failed closed to 0")
            else:
                notes.append(f"{env_name} is invalid; using {value} from {source}")
        if name in supplied:
            parsed = _parse_int(supplied.get(name), low, high)
            if parsed is not None:
                value, source = parsed, "override"
            elif name == "autonomyLevel":
                value, source = 0, "override-invalid"
                notes.append("stored autonomyLevel is invalid; autonomy failed closed to 0")
            else:
                notes.append(f"stored {name} is invalid; using {value} from {source}")
        result[name], sources[name] = value, source

    timezone, tz_source = TIMEZONE_DEFAULT, "default"
    if TIMEZONE_ENV in e:
        candidate = _timezone(e.get(TIMEZONE_ENV))
        if candidate:
            timezone, tz_source = candidate, "environment"
        else:
            notes.append(f"{TIMEZONE_ENV} is invalid; using {timezone}")
    if "timezone" in supplied:
        candidate = _timezone(supplied.get("timezone"))
        if candidate:
            timezone, tz_source = candidate, "override"
        else:
            notes.append(f"stored timezone is invalid; using {timezone}")
    result["timezone"], sources["timezone"] = timezone, tz_source

    maximum = _parse_int(e.get("SOUL_AUTONOMY_MAX_LEVEL", 3), 0, 3)
    if maximum is None:
        maximum = 0
        notes.append("SOUL_AUTONOMY_MAX_LEVEL is invalid; ceiling failed closed to 0")
    result["maximumLevel"] = maximum
    requested = result["autonomyLevel"]
    result["requestedLevel"] = requested
    result["autonomyLevel"] = min(requested, maximum)
    if requested > maximum:
        notes.append(f"requested autonomy {requested} is capped at startup ceiling {maximum}")
    return result, sources, notes


def atomic_json(path: Path, document: Mapping[str, Any], mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # The landing page runs as root while the controller runs as `soul`.
        # Inherit the directory/file group across atomic replacements so either
        # side can continue reading the shared contract after a UI write.
        try:
            gid = path.stat().st_gid if path.exists() else path.parent.stat().st_gid
            os.chown(tmp, -1, gid)
        except (OSError, PermissionError):
            pass
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): redact(v) for k, v in value.items() if not _secret_key(k)}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def append_jsonl(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(redact(dict(document)), ensure_ascii=False, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        try:
            os.fchown(fd, -1, path.parent.stat().st_gid)
            os.fchmod(fd, 0o640)
        except (OSError, PermissionError):
            pass
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def audit(headline: str, summary: str, config: Mapping[str, Any],
          env: Optional[Mapping[str, str]] = None) -> None:
    append_jsonl(paths(env)["journal"], {
        "schemaVersion": SCHEMA_VERSION,
        "id": "audit-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "timestamp": utcnow(),
        "kind": "audit",
        "severity": "info",
        "headline": headline[:160],
        "summary": summary[:2000],
        "observations": [],
        "suggestions": [],
        "confidence": 1.0,
        "incidentId": None,
        "evidenceRefs": [],
        "autonomyLevel": int(config.get("autonomyLevel", 0)),
        "model": "",
        "toolsUsed": [],
        "analysisStatus": "deterministic",
    })


def write_override(values: Mapping[str, Any],
                   env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    e = dict(os.environ if env is None else env)
    before, _, _ = resolve(e)
    cleaned: Dict[str, Any] = {}
    for name, (_env_name, _default, low, high) in FIELDS.items():
        if name not in values:
            continue
        parsed = _parse_int(values[name], low, high)
        if parsed is None:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
        cleaned[name] = parsed
    if "timezone" in values:
        parsed_tz = _timezone(values["timezone"])
        if parsed_tz is None:
            raise ValueError("timezone must be a valid IANA timezone")
        cleaned["timezone"] = parsed_tz
    p = ensure_tree(e)
    atomic_json(p["config"], {
        "schemaVersion": SCHEMA_VERSION,
        "writtenAt": utcnow(),
        "values": cleaned,
    })
    after, _, _ = resolve(e, cleaned)
    changed = [name for name in (*FIELDS, "timezone")
               if before.get(name) != after.get(name)]
    audit("SOUL configuration changed",
          "Effective values changed: " + (", ".join(changed) if changed else "none"),
          after, e)
    p["reconcile"].touch()
    try:
        os.kill(1, signal.SIGUSR1)
    except (OSError, PermissionError):
        pass
    return after


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_journal(limit: int = 20, before: str = "",
                 env: Optional[Mapping[str, str]] = None) -> list[Dict[str, Any]]:
    limit = max(1, min(int(limit), 50))
    try:
        lines = paths(env)["journal"].read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out, accepting = [], not bool(before)
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if not accepting:
            if item.get("id") == before:
                accepting = True
            continue
        if isinstance(item, dict):
            out.append(redact(item))
        if len(out) >= limit:
            break
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("level")
    sub.add_parser("show")
    args = parser.parse_args(argv)
    config, sources, notes = resolve()
    if args.command == "level":
        print(config["autonomyLevel"])
    else:
        print(json.dumps({"config": config, "sources": sources, "notes": notes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
