# Embedded appliance SOUL

SOUL is an optional, appliance-owned Nanobot 0.3.0 runtime. It runs inside the
turnkey container, against the local OpenAI-compatible vLLM endpoint, and adds
no process container, public port, gateway, WebUI, or first-run setup.

It is disabled by default:

| level | behaviour |
|---|---|
| `0` — Off | No controller process or proactive LLM calls. The existing direct post-rollback failure explanation remains active. |
| `1` — Observe | Deterministic snapshots, incident interpretation, and an hourly journal using supplied evidence only. The Nanobot shell tool is disabled. |
| `2` — Investigate | Everything in level 1, plus bounded, read-only shell investigation when the supplied evidence is insufficient. |
| `3` — Verify | Everything in level 2, plus an idle-only exact-response canary and a conditional, cooldown-bounded long-context probe. It never restarts, rolls back, changes configuration, installs packages, deletes data, or remediates. |

Startup verification and rollback are always authoritative. SOUL only observes,
explains, and suggests.

## Isolation and lifecycle

Nanobot and every locked dependency are installed in `/opt/nanobot-venv`.
Nothing is installed into vLLM's Python environment. The image build verifies
both the Nanobot version and that vLLM's package set did not change.

The controller runs as the `soul` Unix user. That user owns only
`$GLM_STATE_DIR/soul`; selected appliance configuration, rollback evidence, and
engine logs are group-readable but not group-writable. SOUL records rollback
analysis under its own incident tree. Nanobot receives normal diagnostic shell
access at levels 2 and 3, with its upstream destructive-command guards. The
SDK's messaging, goals, subagent, CLI app, file, web, image, and
self-configuration tools are removed. Level 1 receives no tools; levels 2 and 3
receive only the upstream shell tools, without an appliance command allowlist.
`/opt/soul/SOUL.md` is immutable and pinned directly into each SDK prompt.
Dream and background compaction are disabled, so memory consolidation cannot
rewrite the governing prompt.

PID 1 reconciles the effective level every five seconds. Level zero stops the
controller; levels 1–3 start it without touching vLLM. Controller crashes use
their own capped 5, 15, 60, and 300 second backoff. A controller failure never
changes endpoint status or consumes the vLLM restart budget. The controller
survives vLLM restarts, waits for verified `serving` state before LLM work, and
is stopped before secure erase or container exit.

## Configuration

SOUL has a separate precedence chain:

```text
built-in defaults < startup environment < $GLM_STATE_DIR/soul/config.json
effective level = min(requested level, SOUL_AUTONOMY_MAX_LEVEL)
```

| environment variable | default |
|---|---:|
| `SOUL_AUTONOMY_LEVEL` | `0` |
| `SOUL_AUTONOMY_MAX_LEVEL` | `3` |
| `SOUL_HEARTBEAT_INTERVAL_S` | `300` |
| `SOUL_JOURNAL_INTERVAL_S` | `3600` |
| `SOUL_JOURNAL_RETENTION_DAYS` | `90` |
| `SOUL_EVIDENCE_RETENTION_DAYS` | `7` |
| `SOUL_MAX_STATE_MB` | `256` |
| `SOUL_TIMEZONE` | `UTC` |

The maximum level is startup-only. The token-gated landing page can request a
different level or interval, but it cannot exceed the ceiling. Invalid levels
fail closed to zero; invalid intervals fall back to the lower layer. Every
change appends a deterministic audit record and never requests a vLLM restart.

## Monitoring and incidents

Every heartbeat records a bounded, redacted snapshot containing:

- boot phase, verification, active and known-good configuration;
- local `/health`, `/v1/models`, selected Prometheus metrics, latency, and errors;
- disk bytes/inodes, cgroup memory/pressure, CPU load, and host temperatures;
- NVIDIA utilization, memory, temperature/slowdown threshold, power, ECC,
  XID, and driver output;
- DNS, local TCP, TLS chain/SAN, and certificate expiry;
- bounded engine logs and known fatal failure signatures.

Ordinary endpoint reachability needs two consecutive failures. Disk warnings
open below 10% free and become critical below 5%. Certificates warn below 30
days and become critical below 7. GPU temperatures warn within 10°C of the
reported slowdown threshold and become critical at it. New XIDs, uncorrectable
ECC, fatal log signatures, and failed canaries open immediately. Incidents are
fingerprinted, repeated updates are coalesced for 15 minutes, and recovery is a
first-class journal event.

Level 3 runs a cheap exact-response canary only when the verified endpoint has
no active requests, at most hourly. Probe evidence is separate and never feeds
the rollback state machine.

## Durable state and privacy

```text
$GLM_STATE_DIR/soul/
├── config.json
├── status.json
├── journal.jsonl
├── incidents/
├── evidence/
├── snapshots/
├── workspace/
└── logs/
```

The controller owns journal writes. Nanobot returns a structured candidate;
the controller repairs trailing commas once, validates fields and lengths, and
falls back to redacted `unstructured` text. Stored content is never rendered as
HTML. Credentials, authorization headers, signed URLs, private keys, and common
token patterns are redacted before persistence and before LLM submission.
Agent turns are ephemeral; after each run the controller persists only a short,
redacted continuity note under the stable journal, incident, or daily session
key. That gives updates and recovery for one incident direct continuity. A
recovered check receives a new incident id if it later fails again, so every
analysis also receives a bounded digest of the eight newest redacted journal
entries. Daily synthesis receives up to 50. This lets Nanobot identify a
recurring symptom across incident sessions without replaying raw snapshots,
commands, or tool output into its session files.

Journal records use append-only JSONL. Status and indexes use atomic JSON
replacement, avoiding SQLite locking assumptions on provider network volumes.
Evidence is retained for seven days and hourly journal entries for 90 days.
The size cap removes the oldest evidence before journal history. Secure erase
selects the entire persistent and runtime SOUL tree, including sessions,
prompts, command evidence, and logs.

## Landing-page interface

The home page always shows a sanitized state and headline. A full journal,
structured observations/suggestions, escaped evidence, and settings are
available only with `OPEN_BUTTON_TOKEN`:

- `GET /soul`
- `GET /soul/status`
- `GET /soul/journal?limit=1..50&before=<entry-id>`
- `POST /soul/config`

When no token is configured, `/soul` and the JSON/config endpoints return 403.
Paths, commands, log excerpts, settings, and evidence are not exposed.

## Updating the dependency lock

`requirements-soul.lock` is a universal, hash-locked resolution generated from
`requirements-soul.in`:

```bash
uv pip compile requirements-soul.in --generate-hashes \
  --python-version 3.11 --universal -o requirements-soul.lock
```

Review Nanobot release and security changes before regenerating or changing the
pinned `nanobot-ai==0.3.0`.
