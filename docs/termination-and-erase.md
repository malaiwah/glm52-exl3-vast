# Terminating the instance, and erasing the session first

A rented box is someone else's hardware, and it goes to someone else next. This
adds two things to the landing page: a way to **destroy the instance from
inside it** (so billing stops the moment you are done, and so an erase can run
first), and an optional **session erase** that removes the evidence you were
there before the disk is recycled.

Both are off unless you asked for them at launch. Read §3 before enabling
anything.

| file | role |
|---|---|
| `scripts/provider.py` | which cloud are we on, and the destroy call for each. The armed-transport safety property lives here. |
| `scripts/secure_erase.py` | what counts as session data, and how it is destroyed. |
| `scripts/terminate_worker.py` | the sequence: stop engine → erase → destroy. Re-checks every gate. |
| `landing.py` | `/terminate` confirmation UI, `/terminate/lock` ratchet, `/terminate/status`. |
| `entrypoint.sh` | derives the switches at boot; stands the supervisor down when a termination starts. |
| `tests/test_termination.py` | 172 assertions, all provider calls stubbed. |

---

## 1. Provider matrix

Everything below is from the providers' own documentation, cited. Anything I
could not confirm is marked **UNVERIFIED** and the code treats it as a
possibility, not a fact.

### vast.ai — confirmed

| | |
|---|---|
| identifies the instance | `CONTAINER_ID` |
| credential | `CONTAINER_API_KEY`, injected into every instance |
| what that key may do | "The instance_api_key is a restricted key injected into the container as CONTAINER_API_KEY, it can only start, stop, or destroy that specific instance." |
| call | `DELETE https://console.vast.ai/api/v0/instances/{id}/`, header `Authorization: Bearer <key>` |
| success | `200` with `{"success": true, "msg": "Instance destroyed successfully"}` |
| errors | `400 invalid_args`, `404 not_found`, `429 rate_limit_exceeded` |
| stop vs destroy | vast has both. **This control destroys.** A stopped instance keeps charging for storage. |

Sources: [Docker execution environment / injected env vars](https://docs.vast.ai/documentation/instances/templates/docker-environment),
[destroy instance API](https://docs.vast.ai/api-reference/instances/destroy-instance),
[vastai destroy instance CLI](https://docs.vast.ai/cli/reference/destroy-instance).

Confidence is high for a second reason: `entrypoint.sh` has been using these
same two variables against `console.vast.ai/api/v0/instances/{id}/` (a `PUT` to
set the dashboard label) since long before this feature, and that works.

### RunPod — verified on live pods

Two pods were created, exercised and destroyed on 2026-07-26
(`runpod/base:0.6.2-cuda12.4.1`, SECURE cloud). Everything in this table is
measured from inside a running pod unless marked otherwise.

| | | how known |
|---|---|---|
| identifies the pod | `RUNPOD_POD_ID` | **verified present in PID 1** |
| credential | `RUNPOD_API_KEY`, injected automatically, pod-scoped | **verified present in PID 1** |
| **self-termination with the injected key** | `mutation { podTerminate(input:{podId:"<own id>"}) }` → `{"data":{"podTerminate":null}}`; pod gone from `runpodctl pod list` within 20 s | **VERIFIED — it works** |
| success shape | `null`. The mutation returns `Void`, so `data.podTerminate: null` *is* success | **verified** |
| what the pod key may NOT do | account-level reads: `query { myself { id } }` → `{"errors":[{"message":"Unauthorized","path":["myself"]}],"data":{"myself":null}}` | **verified** |
| what the pod key MAY read | its own pod: `query { pod(input:{podId:"<own id>"}) { id name desiredStatus } }` → the pod | **verified** |
| auth mechanism | one API key, `Authorization: Bearer <key>` to `https://api.runpod.io/graphql` | verified (runpodctl config + RunPod's SDK) |
| where the CLI stores it | `~/.runpod/config.toml`, **world-readable (0644)** | verified on a real install |
| REST terminate | `DELETE https://rest.runpod.io/v1/pods/{podId}`, Bearer, `204` | documented API reference |
| CLI terminate / stop | `runpodctl pod delete <id>` vs `runpodctl pod stop <id>` — different verbs, different billing | verified from the CLI; **runpodctl is NOT in the base image** |
| metadata service | 169.254.169.254 **unreachable** — there is none to fall back on | **verified** |

Sources: [environment variables](https://docs.runpod.io/pods/templates/environment-variables),
[DELETE /v1/pods/{podId}](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId),
[manage pods — stop vs terminate](https://docs.runpod.io/pods/manage-pods),
[GraphQL manage-pods](https://docs.runpod.io/sdks/graphql/manage-pods),
[GraphQL schema](https://graphql-spec.runpod.io/),
[runpod-python mutations](https://github.com/runpod/runpod-python/blob/main/runpod/api/mutations/pods.py),
[runpod-python graphql client](https://github.com/runpod/runpod-python/blob/main/runpod/api/graphql.py).

**The pod-scoped key is scoped, not weak — and the pre-check has to respect
that.** An earlier revision of this feature probed the credential with
`query { myself { id } }`. That is exactly the query a pod-scoped key is *not*
allowed to answer, so the pre-check condemned a credential that could in fact
terminate the pod, and told the user to go and fetch an account key they did not
need. The probe now reads **the pod itself**
(`pod(input:{podId:"<id>"})`), which the key is permitted to do and which
additionally proves the pod id we resolved is the right one. `myself` is only
used as a fallback when no pod id is known. A `200` carrying
`{"data":{"pod":null}}` is treated as a *failure* — valid key, wrong pod.

**Environment variables come from PID 1, not from your shell.** RunPod injects
into the container's main process only. A helper started from a new session —
an SSH login shell being the obvious one — sees *none* of them, which reads
exactly like "no provider detected" while the box is plainly a RunPod pod. All
provider identity, the switches and the erase paths therefore go through
`glm_config.effective_env()`: this process's environment layered over
`/proc/1/environ`, with an explicitly-passed dict never merged. The entrypoint
itself is PID 1 and never needed this; everything invocable from another session
did.

Confirmed present in PID 1: `RUNPOD_API_KEY`, `RUNPOD_POD_ID`,
`RUNPOD_POD_HOSTNAME`, `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_DC_ID`,
`RUNPOD_GPU_COUNT`, `RUNPOD_GPU_NAME`, `RUNPOD_CPU_COUNT`, `RUNPOD_MEM_GB`, and
`PUBLIC_KEY` (the account's SSH public keys — session evidence, see §4).

**Why no `runpodctl` binary is bundled.** Auth is a plain API key and the
operation is one HTTP POST; a 13.8 MB static Go binary to make that call would
be dead weight in an image that already POSTs to vast's API with `urllib`. It is
also **not present in RunPod's own base image**, so depending on it would have
been wrong twice. The code still uses it opportunistically as the last of three
attempts if the image happens to carry it.

**Terminate, never stop.** `pod stop` and `pod delete` are different verbs with
different billing: a stopped pod keeps paying for volume storage. Someone who
clicks "terminate" means "stop paying". The code only ever deletes, and a test
asserts the CLI fallback never passes `stop`.

**The ladder.** REST `DELETE` → GraphQL `podTerminate` → `runpodctl pod delete`
if present. REST and GraphQL are different services, so a key refused by one is
not necessarily refused by the other, and trying both costs one extra
round-trip on a path that runs once. A GraphQL `200` carrying an `errors` block
is **not** treated as success.

**`RUNPOD_TERMINATE_API_KEY` is still supported**, for the cases the pod key
cannot cover: a key that is missing or has been altered, or a deployment that
wants to terminate a *different* pod. It is no longer presented as the expected
path, and nothing warns the user to go and get one unless a call actually fails.

**Network volumes: the flagged case is the DEFAULT case.** A RunPod network
volume mounts at `/workspace` (replacing the volume disk), "must be attached
during Pod creation and cannot be detached later", is retained independently,
and RunPod's own wording is that terminating "permanently deletes all data not
stored in a network volume" — while the volume keeps billing. On top of that,
PID 1's environment points the **entire Hugging Face cache** at the volume by
default: `HF_HOME=/runpod-volume/.cache/huggingface/`, plus
`HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE` and friends. So on a stock RunPod
pod, **your HF token and anything cached through it land on storage that
termination does not clear and that goes on charging you.** This is not an edge
case to warn about; it is what happens if you change nothing. The erase resolves
every HF cache root from PID 1's environment and takes the `token` /
`stored_tokens` files from each (the cached model *blobs* are left alone for the
same reason the weights are — public bytes), and it sweeps `/runpod-volume` for
logs. When `RUNPOD_VOLUME_ID` is set the confirmation page says all of this in a
red banner, including that the volume itself must be deleted from the dashboard
to stop its charges.

### Unknown provider

Detection order: `TERMINATE_PROVIDER` override → `RUNPOD_POD_ID` →
`CONTAINER_ID`+`CONTAINER_API_KEY` → any `VAST_TCP_PORT_*` → unknown. An
unrecognised environment does not fail obscurely: `/terminate` renders a page
that says *termination is not supported here, terminate from your provider's
dashboard, and here is what the detector actually saw* (a dump of the
provider-identifying variables, with key values replaced by
`set (N chars)`). No destroy button is rendered at all.

---

## 2. Confirmation flow

1. **Auth.** `/terminate` and every POST under it require the same
   `OPEN_BUTTON_TOKEN` as `/chat` and `/config`. With no token configured the
   control does not exist. There is no unauthenticated path, and the tests
   assert 403 for no-token and wrong-token on both GET and POST.
2. **Switches.** The kill switch must be on and the anti-kill switch off (§3).
3. **Information, before anything is typed.** Provider, instance id, which
   credential is being used and where it came from, an explicit list of what is
   destroyed and what survives (network volumes get their own line on RunPod),
   the billing consequence, and — if you tick erase — what the erase does and
   does not guarantee.
4. **Typed confirmation.** You must type the **instance id** exactly
   (`9876543`, `pod-abc123`); if no id is known the phrase is `TERMINATE`. A
   misclick cannot destroy anything: there is no state in which a single click
   is sufficient.
5. **Explicit acknowledgement.** A separate "I understand this destroys the
   instance and everything on its disks, permanently" checkbox.
6. **Hand-off.** Only then does the page spawn `terminate_worker.py` detached,
   so the flow outlives the HTTP request. The worker **re-checks all three
   gates** — switches, typed confirmation against the instance id, provider
   readiness — because "the UI already checked" is not a property of a system.
7. **Progress and outcome.** The page shows each phase
   (`stopping-engine` → `erasing` → `terminating` → `terminated`/`failed`), the
   provider calls with their HTTP statuses, and the final message. `GET
   /terminate/status` returns the same as JSON.

**Failure is reported as failure.** Every non-success path says whether the
instance is still running and still billing, in those words, and points at the
dashboard. A 404 says "it may already be gone — check the dashboard; do not
assume billing stopped". A timeout says the request never got a reply. The one
thing this must never do is leave you thinking the box is gone when it isn't.

**`TERMINATE_DRY_RUN=1`** runs the entire flow — gates, erase, provider
selection, the constructed request — and does not send it. The UI says
`DRY RUN — the destroy request is being prepared but NOT sent`. This is how the
flow was exercised during development.

### Why the destroy call cannot fire by accident

`provider.HttpTransport` raises `NotArmed` on any `DELETE`/`POST`/`PUT`/`PATCH`
unless constructed with `allow_destructive=True`. Exactly one call site does
that: `terminate_worker.run()`, after the gates. Every provider method takes its
transport as an argument, so the test suite injects stubs; after the
arming test, the suite **replaces `HttpTransport` with a class that raises on
construction**, so a test that forgets to inject a stub fails instead of
reaching the network. No test has ever contacted a provider API.

---

## 3. The two switches

Set **only** as startup environment variables:

| variable | default | meaning |
|---|---|---|
| `TERMINATE_ENABLED` | **0** | kill switch — is the terminate control available at all? |
| `TERMINATE_LOCKED` | **0** | anti-kill switch — hard lock; termination is refused regardless of `TERMINATE_ENABLED` |

### Why `TERMINATE_ENABLED` defaults to OFF

This ships as a public template, so the question is what happens to someone who
does not read this file — in both directions.

*If it defaulted ON*: the destroy button sits behind the Open-button token,
which travels **in the URL query string**. URLs leak — browser history, shared
screenshots, a pasted link in a support channel, referrer headers. That token
already exposes the API key and the config editor, so this is not a new class of
exposure, but it is a categorically worse consequence: a leaked config editor
costs a restart, a leaked destroy button costs the job that has been running for
six hours and the 332 GB download that precedes restarting it. There is also a
plain human risk: a page whose main affordances are "Configure" and "Terminate"
invites a mis-click on a phone.

*If it defaults OFF*: the user who wanted it discovers a disabled control that
says exactly which variable to set at launch — and, crucially, **they lose
nothing in the meantime**, because every provider dashboard can already
terminate the instance in one click. The in-container control is a
*convenience*, plus the only way to run an erase first. The "forgotten instance
still billing" failure mode — the strongest argument for defaulting ON — is not
actually made worse by defaulting OFF, because the dashboard path is universal
and is the path that provider documentation, billing emails and the console all
point at.

So the asymmetry is: default-ON risks irreversible loss for people who did not
opt in; default-OFF risks a moment of inconvenience for people who did not read
the docs, with a fully functional alternative one tab away. Default OFF.

### Why `TERMINATE_LOCKED` defaults to OFF

Locked-by-default would be redundant (the kill switch already denies by
default) and it would create a worse failure: a user who sets
`TERMINATE_ENABLED=1`, sees the control still refuse, and cannot work out why —
with the fix requiring a relaunch. The lock's value is that it can be
*acquired* at runtime and never released, so it is a deliberate act, not a
resting state. Set `TERMINATE_LOCKED=1` at launch when you are handing someone
landing-page access to an instance you do not want them to destroy.

### The invariants, and why they are structural

**(a) The state file can never set either one.** `FORBIDDEN_STATE_KEYS` is
checked in `load_state_file()`, which **rejects the entire file** and names the
offending key:

> state file rejected: 'TERMINATE_LOCKED' is a startup-environment control and
> can never be set from the state file or the landing page.

Not "ignored" — rejected. A file that contains one of these keys is evidence of
an escalation attempt, and the rest of its contents have not earned any trust
either. `resolve()` then falls back to env + defaults and surfaces the rejection
as a note on `/config` and in the boot log. The apply/import path in the landing
page runs the same check before writing anything, so the error is shown to
whoever tried.

**(b) Runtime changes ratchet one way.** `glm_config.tighten()` is the only
mutator, and it computes

```python
enabled := enabled AND requested_enabled
locked  := locked  OR  requested_locked
```

Loosening is not "disallowed by a check" — it is **unrepresentable**. There is
no `loosen()`, no `set_switches()`, no `unlock()`; the test suite asserts that
no such symbol exists. The landing page's two buttons ("Disable the terminate
control", "Lock termination (hard)") can only call `tighten()`.

**(c) The ratchet state is per-container.** It lives in
`$GLM_RUNTIME_DIR/terminate-switches.json` — `/tmp`, not the volume — and the
entrypoint re-derives it from the startup environment on every container start.
So the *only* way back to a looser state is a container restart with a different
environment, which requires editing the template or the pod configuration:
provider-dashboard authority, which is precisely what a landing-page token
holder may not have.

**(c-UI)** `/terminate` always shows both switches with their current values and
an explanation. When locked it says, in the page body, that it cannot be
unlocked from the UI and that clearing it requires restarting the container with
`TERMINATE_LOCKED` unset — a greyed-out button with no explanation would not be
enough. The form controls are disabled *in addition to* the server-side refusal;
the tests drive a perfectly-formed POST at a locked instance and assert that it
is refused and that no worker starts.

---

## 4. Session erase

**Off by default.** It is a checkbox on the confirmation screen, unchecked, with
sub-options for RAM and VRAM.

### What it does NOT do: the model weights

The ~332 GB checkpoint is **public**. It comes from Hugging Face, anyone can
download the identical bytes, and finding it on a recycled disk tells the next
tenant nothing except that someone ran a popular open-weights model. Overwriting
it would take the better part of an hour on local NVMe and much longer on
network storage — which is what made an earlier "erase everything" design
impractical and forced tiers on it.

Dropping the weights is not an omission, it is the point: what remains is small,
fast, and can be done *completely*. The erase now finishes in seconds to a
couple of minutes and covers every file that says a particular person was here.

### What it destroys

| group | contents |
|---|---|
| credentials | the persisted vLLM API key (`.vllm-api-key`), the boot status file (which carries the key in plaintext for the landing page), the Hugging Face token in **every** cache root resolved from PID 1's environment — on RunPod that defaults to `/runpod-volume/.cache/huggingface`, i.e. storage that termination does NOT clear, provider CLI credentials — including `~/.runpod/config.toml`, which `runpodctl` writes **world-readable (0644)** with the account API key in it (verified on a real install) — plus `~/.vast_api_key` and `~/.config/*`, everything under `~/.ssh` (authorized_keys, known_hosts, any private keys) |
| tls | all of `/workspace/.lego` — the Let's Encrypt **account key**, the certificate, and the **private key** for your domain |
| config-state | `config.json`, `known-good.json`, `apply-state.json`, `verify-last.json`, `checkpoint-baseline.json`, the switch state, and every preserved failure directory |
| logs | `logs/serve-current.log`, `logs/last-good.log`, every `failures/*/error.log`, every `failures/*/analysis.md` (the model's written description of your configuration), `*.log`/`*.jsonl` at the volume root, `/var/log` |
| runtime | resolved config, the startup-env snapshot, verification verdicts, `/tmp` scratch |
| history | `.bash_history`, `.python_history`, `.zsh_history`, `.viminfo`, `.lesshst`, IPython history, Jupyter runtime |
| user-files | anything under the model dir that is not part of the public checkpoint |

Logs matter more than they look: this template's engine logs are where prompt
text lands when request or trace logging is on, and the rollback machinery
deliberately *preserves* failed boot logs plus a model-written analysis of your
configuration. Those are the most descriptive artefacts on the box.

### Telling your files from the public checkpoint

`snapshot_download(local_dir=...)` "will create a `.cache/huggingface/` folder at
the root of `local_dir` to store some metadata related to the downloaded files"
([HF download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)).
Each downloaded file leaves a sidecar ending in `.metadata`, mirroring the repo
layout. `public_manifest()` reads that folder and treats every path it names as
public — deliberately accepting *any* nesting that ends in `.metadata`, because
the exact layout has moved between hub versions.

Anything under the model dir that is **not** in the manifest and **not** a
derived artefact of this template (`.vllm-cache`, `.mtp78-overlay`,
`.mtp78-draft`, `.vision`, `*.orig`, `*.text-only`, the marker files) is, by
elimination, something you added — an adapter, a dataset, notes — and is erased.

**When the manifest is missing** (a hub version whose layout we guessed wrong,
or weights placed there by hand), the code does not silently skip: files under
64 MiB that it cannot account for are erased anyway (a stray `notes.txt` is far
more likely to be yours than part of a safetensors repo), and every larger
unidentifiable file is **listed in the report and on the page** as *not erased,
could not be distinguished from the public checkpoint*. You are told, rather
than quietly missed. **UNVERIFIED:** the sidecar naming was not exercised
against a real 332 GB download — no network here — so the fallback path is the
one to expect if the layout differs.

### RAM and VRAM

- **RAM** (opt-in): `sync` + `/proc/sys/vm/drop_caches` (often read-only in a
  container — reported honestly when it is), then overwrite free memory in
  256 MiB blocks up to 60% of `MemAvailable`, capped at 32 GiB. Bounded because
  an unbounded allocation on an oversubscribed host gets you OOM-killed *before*
  the destroy call goes out.
- **VRAM** (opt-in): after the engine has stopped, allocate and `zero_()` device
  memory on every visible GPU until allocation fails, then release. Best-effort
  by construction: it needs torch, it needs the engine gone, and the driver may
  have already scrubbed pages. **UNTESTED** — there is no GPU in the build
  environment and the development host was serving production traffic.

### What it does and does not guarantee

**Does:** overwrite every planned file with one pass of random bytes, `fsync` it
to the device, truncate, rename the inode and unlink; then drop caches and
overwrite free RAM; then zero VRAM. Defeats undelete tools and casual forensics.

**Does not:**

- **SSD wear levelling and over-provisioning.** A logical overwrite may land on
  a different physical page; the old one stays in the flash translation layer's
  pool until it is garbage-collected. A container cannot issue TRIM or a
  sanitize command to the device.
- **Copy-on-write and overlay filesystems.** Docker's overlay2, btrfs and ZFS
  write the new bytes elsewhere and unlink the old extent. The old extent is
  still on the media. This applies to *every* container filesystem, which is to
  say: to this one.
- **Network-backed storage.** RunPod network volumes and NFS are overwritten on
  a server we do not control, with its own replication and snapshots.
- **Provider snapshots and backups** taken before the erase.
- **The instance console log.** Every line this container printed to stdout —
  including prompts, if request logging was on — is in your provider's dashboard,
  off the box, unreachable from in here.

One pass, not three: multi-pass overwriting is theatre on any storage made after
the 1990s, and it costs time we may not have before the instance disappears.

The honest summary, now that the 300 GB problem is gone: **the data is removed
from the filesystem, reliably and completely, and recovery through the operating
system will fail. This is not media sanitisation.** If your threat model
includes an adversary with the flash in hand, do not put the secret on rented
hardware.

---

## 5. Sequencing

```
landing page                worker                       supervisor (entrypoint)
  POST /terminate  ──────►  gate: switches
                            gate: typed confirmation
                            gate: provider ready
                            touch terminate-in-progress ──►  sees the flag
                                                             kills vLLM
                                                             touch engine-stopped
                            wait (≤180 s) ◄──────────────────  stands down, stays PID 1
                            erase files
                            erase RAM / VRAM
                            DELETE to the provider  (armed transport, once)
                            write the outcome
  poll /terminate/status ◄─ progress JSON
```

Why the supervisor is asked rather than raced: it owns the engine's lifecycle,
and killing vLLM behind its back just gets the engine restarted underneath the
erase — rewriting the very logs being destroyed and re-allocating the VRAM being
zeroed. Why PID 1 stays alive: exiting the entrypoint kills the container, which
on some providers simply restarts it, mid-erase. If the flag is ever cleared,
the supervisor resumes normally.

The progress file, the stop flag and the stopped marker are excluded from the
erase — they are the last things standing when the provider pulls the plug.

---

## 6. Template compatibility: vast.ai vs RunPod

Beyond termination, these differ in ways that affect this image.

| | vast.ai | RunPod |
|---|---|---|
| **persistent path** | the instance's own disk; this template uses `/workspace` and nothing survives a destroy | **volume disk** mounted at `/workspace` by default; survives stop, destroyed on terminate. A **network volume** also mounts at `/workspace`, replaces the volume disk, and survives the pod entirely |
| **SSH** | vast runs sshd; the entrypoint only repairs key permissions | nothing runs sshd unless the image does — this image now starts one when `PUBLIC_KEY` is present (`SSHD=0` to opt out) |
| **port exposure** | `-p 8000:8000 -p 1111:1111` in Docker options; external ports arrive as `VAST_TCP_PORT_8000` etc. and are usually *not* the same numbers | **ports must be requested AT POD CREATION** (`--ports "22/tcp,1111/http,8000/http"`); HTTP ports are proxied at `https://[POD_ID]-[PORT].proxy.runpod.net` (max 10, **100-second Cloudflare timeout**); raw TCP needs a public IP and symmetrical ports above 70000, surfaced as `RUNPOD_TCP_PORT_*` |
| **HF cache location** | `~/.cache/huggingface` on the instance disk; dies with the instance | PID 1 sets `HF_HOME=/runpod-volume/.cache/huggingface/` — **the network volume by default**, which survives termination and keeps billing |
| **landing page on :1111** | works; `OPEN_BUTTON_PORT=1111` makes the dashboard's Open button hit it | reachable via the HTTP proxy at `https://<pod>-1111.proxy.runpod.net`; there is no Open button, so the URL must be constructed by hand, and `OPEN_BUTTON_TOKEN` is not injected — **it must be set manually or the config/terminate UI stays disabled** |
| **streaming / long requests** | direct TCP; a 20-minute generation is fine | the HTTP proxy's **100 s timeout kills long non-streaming requests** (524). Use the TCP-mapped port for the API, not the proxy, or keep responses streaming |
| **env vars** | template env + injected `CONTAINER_ID`, `CONTAINER_API_KEY`, `PUBLIC_IPADDR`, `VAST_TCP_PORT_*`, `DATA_DIRECTORY`, `SSH_PUBLIC_KEY` | template/pod env + injected `RUNPOD_POD_ID`, `RUNPOD_API_KEY`, `RUNPOD_DC_ID`, `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_VOLUME_ID`, `PUBLIC_KEY` |
| **GPU selection** | host filters in the template (4× RTX PRO 6000, disk, bandwidth) | GPU type and count chosen at deploy time; no equivalent of vast's host-level filters for disk speed or network |
| **billing while idle** | stopped instances keep paying for storage | stopped pods keep paying for volume storage; **network volumes bill whether or not a pod exists** |

### RunPod: expose the ports at creation, or the UI does not exist

**VERIFIED the hard way.** A pod created without `--ports` came up with
`ports: null`: the container ran, but nothing was reachable — not the landing
page, not the API, not even SSH. There is no way to add a port to a running
pod; the pod has to be destroyed and recreated.

That is materially worse than on vast.ai, where the Docker options are part of
the template and a missed port is visible immediately in the instance's port
list. On RunPod the pod looks healthy, the logs look healthy, and the entire
self-service surface — config editor, terminate control, status — is simply
unreachable, with the only fix being to recreate the pod and lose the 332 GB
download.

So a RunPod deployment of this image **must** request, at creation time:

```
--ports "22/tcp,1111/http,8000/http"
```

`1111/http` is the landing page (config editor + terminate control), `8000/http`
is the OpenAI-compatible endpoint, `22/tcp` is SSH. Remember the proxy's
100-second timeout applies to the `/http` ports: keep long generations
streaming, or reach the API over a TCP-mapped port instead.

### RunPod: this image now starts sshd, because nothing else will

**MEASURED.** This image does not run `sshd`, and RunPod does not run one for
you — their SSH works because *their* base images start one. With this image as
PID 1 the pod reports RUNNING, `runpodctl ssh info` resolves happily, and every
connection is refused. RunPod injects `PUBLIC_KEY`, but something in the
container has to consume it.

Combined with the ports trap above, that produced the worst possible failure
mode: a pod that is billing, looks healthy, and is a **black box** — no shell,
no landing page, only the console log.

**Decision: start sshd when the provider handed us keys** (`SSHD=auto`, the
default), rather than documenting "RunPod users get no SSH". The argument:

* *Against* starting it: more surface, more moving parts, and one more thing
  that can fail at boot.
* *For*: the access it grants is the access the provider was **already handing
  out** — key-only auth using key material RunPod itself injected, reachable
  only on a port the user had to expose deliberately at pod creation. It grants
  nothing new to anyone; it just stops silently withholding it from the owner.
* And the decisive asymmetry: without SSH, the *only* remaining way into a
  misconfigured pod is the landing page — which may be exactly what is
  misconfigured. Two independent ways in is the difference between "fix it" and
  "destroy it and re-download 332 GB".

Behaviour: `auto` starts sshd only when `PUBLIC_KEY` is set (so vast.ai, where
sshd already runs, is untouched) and only when nothing is already listening on
:22. It installs the injected keys into `/root/.ssh/authorized_keys`
(append-only, deduplicated, 0600), generates host keys, and starts the daemon.
`SSHD=1` forces it, `SSHD=0` disables it entirely for anyone who wants the
smaller surface.

If the image has no `sshd` binary, it says so loudly and names the consequence —
"there is NO SHELL ACCESS on providers that expect the container to run one;
the landing page on :1111 is the only way in" — rather than failing silently.
`SSHD_INSTALL=1` will `apt-get` it at boot for anyone who wants that instead of
a rebuild.

The generated host keys are added to the session erase: they are what
fingerprints this box to a client.

### What in the current entrypoint assumes vast.ai

Flagged rather than changed — none of it breaks a RunPod boot, but a RunPod
template needs to know:

1. **`VAST_TCP_PORT_8000` / `VAST_TCP_PORT_1111`** are used to build the
   endpoint URL printed in the logs, the landing page and the dashboard label.
   On RunPod these are unset, so the URL falls back to
   `http://$PUBLIC_IPADDR:8000` — and `PUBLIC_IPADDR` is also vast-only
   (RunPod's is `RUNPOD_PUBLIC_IP`). **The endpoint URL shown on a RunPod pod
   will be wrong or empty.**
2. **The dashboard-label update** (`PUT console.vast.ai/api/v0/instances/...`)
   is vast-specific; it is already conditional on `CONTAINER_API_KEY` +
   `CONTAINER_ID`, so it silently does nothing on RunPod. There is no RunPod
   equivalent.
3. **`OPEN_BUTTON_TOKEN`** is a vast concept. On RunPod nothing sets it, so the
   config editor and the terminate control are simply absent until the user sets
   it in the pod environment.
4. **`/workspace` as the volume root** happens to be correct on both, but for
   different reasons — on RunPod it is the volume-disk mount point, and if a
   *network* volume is attached it replaces that mount, which means the 332 GB
   download lands on shared, billed-separately storage that survives the pod.
5. **`--ulimit memlock` / `--ipc=host`** are passed as vast Docker options.
   RunPod's UI does not expose arbitrary Docker flags, so DRAM KV offload runs
   under whatever memlock the pod has — the warn-and-proceed default already
   covers this, but it is untested there.
6. **SSH key repair** targets `/root/.ssh/authorized_keys`, which vast injects.
   RunPod injects `PUBLIC_KEY` and expects the image to install it and to run
   sshd — **now handled** (see above); on vast nothing changes, because
   `PUBLIC_KEY` is unset there and sshd is already listening.

Making the image fully RunPod-native is a separate piece of work; this document
is the list it would start from.

---

## 7. Untested and unverified

**Verified on live pods (2026-07-26).** Two RunPod pods were created, exercised
and destroyed: the injected pod-scoped key terminates its own pod through
GraphQL `podTerminate` (`{"data":{"podTerminate":null}}`, pod gone within 20 s);
`RUNPOD_POD_ID` / `RUNPOD_API_KEY` and the rest are present in PID 1 and absent
from a fresh SSH session; `myself` is Unauthorized for the pod key while
`pod(input:{podId})` is allowed; ports must be requested at creation or nothing
is reachable; `HF_HOME` defaults onto the network volume; `runpodctl` is not in
the base image; there is no metadata service.

**Still untested (cannot be tested without destroying a paid instance from this
codebase):**

- **This code has never issued a terminate call.** The vast.ai and RunPod paths
  are exercised only through stubbed transports. The RunPod GraphQL request this
  code builds is byte-identical in shape to the one that was verified by hand,
  but it has not itself been sent.
- The vast.ai destroy call. High confidence — the entrypoint already uses the
  same two credentials against the same host for the dashboard label — but
  unexercised.
- The `runpodctl` fallback's behaviour *after* REST and GraphQL have both been
  refused; it is also moot on the stock image, which has no runpodctl.
- The full worker sequence against a live supervisor: the
  `terminate-in-progress` → `engine-stopped` handshake is implemented on both
  sides, passes `bash -n` and shellcheck, and is driven in tests with a stubbed
  stopper — but no container has run it end to end.
- VRAM zeroing (needs torch and GPUs) and the RAM overwrite under a cgroup
  limit. `drop_caches` is expected to be unavailable in most containers and is
  reported rather than assumed.
- The erase against a real 332 GB checkpoint directory; only synthetic layouts
  were used.

**Still unverified from documentation:**

- The exact `.metadata` sidecar naming under `local_dir/.cache/huggingface/`
  (§4). The folder's existence is documented; the per-file naming is handled
  defensively and degrades to the size rule, which is also what a user gets on a
  hub version whose layout we guessed wrong.
- RunPod's REST `DELETE /v1/pods/{podId}` — documented, never called. GraphQL is
  the path that was actually proven, and it is tried second; if the REST call
  turns out to be wrong for pod-scoped keys, the ladder still lands on the
  verified one.
- Whether a vast.ai instance ever lacks `CONTAINER_API_KEY` (the docs say it is
  always injected).
