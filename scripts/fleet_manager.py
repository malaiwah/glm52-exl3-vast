#!/usr/bin/env python3
"""Fleet manager for the JarvisLabs GLM-5.3 appliance fleet.

Runs on the router CPU VM beside litellm and closes the loop:

  watch   jl list          which replica slots exist / were reaped
  probe   each replica     authenticated /v1/models through the proxy
  scrape  /metrics         vLLM queue depth, unauthenticated
  decide  min/max replicas scale up under queueing, down when idle
  act     jl create/destroy  autonomous replicas via the startup script
  wire    litellm          regenerate config.yaml + restart the proxy

Everything is stdlib; the only external dependency is the jl CLI being
authenticated on this VM. State lives beside the config; logs to stdout
(journalctl captures them).
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

CONFIG_PATH = os.environ.get("FLEET_MANAGER_CONFIG",
                             os.path.expanduser("~/fleet-manager/fleet.json"))
NUMERIC_SLOT_RE = re.compile(r"^glm53-serve-\d+$")
METRIC_RE = {
    "waiting": re.compile(rb'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)', re.M),
    "running": re.compile(rb'^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)', re.M),
    "cache":   re.compile(rb'^vllm:gpu_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)', re.M),
}


def log(msg):
    print(time.strftime("[%Y-%m-%dT%H:%M:%SZ]", time.gmtime()), msg, flush=True)


def jl(*args):
    out = subprocess.run(["jl", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"jl {args[0]} failed: {out.stderr.strip()[:200]}")
    return out.stdout


def jl_json(*args):
    return json.loads(jl(*args, "--json"))


def http_code(url, key=None, timeout=15):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


class Replica:
    def __init__(self, info):
        self.info = info
        self.name = info.get("name", "")
        self.mid = info.get("machine_id")
        self.status = info.get("status", "")
        self.is_spot = info.get("is_spot")
        self.healthy = False
        self.waiting = None
        self.running = None
        self.cache = None
    def probe(self):
        """Find the replica's vLLM proxy URL. /metrics is the discriminator:
        it answers 200 without authentication on the appliance's engine port.
        The 'url' field in jl get is the Jupyter lab URL, which answers 200 to
        everything and must never be used as an API base."""
        detail = jl_json("get", str(self.mid))
        for base in detail.get("endpoints") or []:
            code, body = http_code(f"{base}/metrics")
            # 200 alone is not proof: the Jupyter lab proxy answers 200 to
            # everything, and port 1111 (landing page) is exposed through
            # the same proxy family. Only vLLM's own /metrics will do.
            if code == 200 and b"vllm:" in body:
                self.api_url = base
                self.healthy = True
                return
        # VM replicas have no HTTPS proxy and no endpoints list: the engine
        # port is exposed directly on the public IP. Plain HTTP is
        # acceptable only on the provider's private network; a public
        # plaintext probe is still fine here (/metrics leaks nothing), but
        # routing production traffic to it is a docs-level decision.
        ip = detail.get("public_ip")
        if ip:
            code, body = http_code(f"http://{ip}:8000/metrics")
            if code == 200 and b"vllm:" in body:
                self.api_url = f"http://{ip}:8000"
                self.healthy = True
                return
        self.healthy = False

    def scrape(self):
        """vLLM /metrics is unauthenticated on the appliance."""
        if not self.healthy:
            return
        code, body = http_code(f"{self.api_url}/metrics")
        if code != 200:
            return
        for name, rx in METRIC_RE.items():
            m = rx.search(body)
            if m:
                setattr(self, name, float(m.group(1)))


class Fleet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fleet_key = cfg["fleet_key"]
        self.last_scale = 0
        self.wired = set()
        self.unhealthy_since = {}
        self.idle_since = None
        self.retiring = {}
        self.first_seen = {}
    def slots(self):
        """Only numeric slots are manager-owned. Manually created replicas
        (glm53-serve-a etc.) keep their own appliance-generated API key,
        which the fleet key cannot authenticate against — adopting them
        would wire a deployment that 401s every request."""
        rows = jl_json("list")
        rows = rows if isinstance(rows, list) else rows.get("instances", rows.get("data", []))
        return [r for r in rows if NUMERIC_SLOT_RE.match(r.get("name", ""))]

    def ensure_min(self, replicas):
        alive = [r for r in replicas if r.status in ("Running", "Pending", "Provisioning")]
        missing = self.cfg["min_replicas"] - len(alive)
        if missing > 0:
            self.create_slot()

    def create_slot(self):
        current = self.slots()
        # max_replicas bounds the whole fleet, not just autoscale.
        if len(current) >= self.cfg["max_replicas"]:
            log(f"create skipped: at max_replicas={self.cfg['max_replicas']}")
            return
        # Serialize COLD BOOTS, not just creations: two slots booting at
        # once would write the one shared quantization cache concurrently.
        # Any alive slot younger than boot_timeout is treated as booting;
        # since a just-created slot is Pending and young, this also
        # collapses the ensure_min/reap double-create within one cycle.
        for row in current:
            if row.get("status") in ("Running", "Pending", "Provisioning"):
                if self.age_of(Replica(row)) < self.cfg.get("boot_timeout_seconds", 2400):
                    log(f"create deferred: {row.get('name')} still booting")
                    return
        used = {r["name"] for r in current}
        n = 0
        while f"glm53-serve-{n}" in used:
            n += 1
        name = f"glm53-serve-{n}"
        # Maintain an on-demand backbone: on_demand_min replicas are kept
        # non-spot so capacity pressure can never pause/destroy the whole
        # fleet; everything beyond that stays spot for price. The backbone
        # can be a GPU VM (on_demand_kind=vm) at the same hourly price as
        # an on-demand container — a VM takes the driver/module tuning
        # containers cannot (P2P atomics measured off on containers).
        alive = [r for r in current if r.get("status") in ("Running", "Pending", "Provisioning")]
        on_demand_alive = sum(1 for r in alive if not r.get("is_spot"))
        args = ["--gpu", self.cfg["gpu"], "--num-gpus", str(self.cfg["num_gpus"]),
                "--region", self.cfg["region"], "--fs-id", str(self.cfg["fs_id"]),
                "--name", name, "--yes"]
        kind = "spot"
        if on_demand_alive < self.cfg.get("on_demand_min", 0):
            if self.cfg.get("on_demand_kind") == "vm":
                kind = "on-demand-vm"
                # VMs are SSH-only: no startup scripts, no HTTPS proxy, no
                # --http-ports. Ports are exposed directly on the public
                # IP; the boot is driven over ssh (see docs/jarvislabs-fleet.md).
                args += ["--vm"]
            else:
                kind = "on-demand"
                args += ["--script-id", str(self.cfg["script_id"]),
                         "--http-ports", "8000,1111"]
        else:
            args.insert(0, "--spot")
            args += ["--script-id", str(self.cfg["script_id"]),
                     "--http-ports", "8000,1111"]
        log(f"creating {kind} replica slot {name}"
            + (f" (script {self.cfg['script_id']})" if "--script-id" in args else ""))
        jl("create", *args)
        self.last_scale = time.time()
    def destroy(self, replica):
        log(f"destroying {replica.name} ({replica.mid})")
        jl("destroy", str(replica.mid), "--yes")

    def reap(self, replicas):
        """Replace slots whose instance died (spot reclamation, crash).

        A fresh instance is NOT unhealthy just because it is not serving
        yet: a cold boot takes 15-25 minutes. Reaping is only allowed once
        the instance is older than boot_timeout, or its status is gone
        (destroyed/reaped spot) — otherwise the manager destroys every
        booting replica after the grace window and thrashes forever.
        """
        grace = self.cfg.get("unhealthy_grace_seconds", 300)
        boot_timeout = self.cfg.get("boot_timeout_seconds", 2400)
        for r in replicas:
            key = r.name
            if r.healthy:
                self.unhealthy_since.pop(key, None)
                continue
            age = self.age_of(r)
            if r.status == "Paused":
                # The provider pauses spot instances under capacity
                # pressure instead of destroying them: billing stops and
                # the instance state is kept. Resuming continues the warm
                # boot — far cheaper than a recreate. If resume fails
                # (still no capacity) the paused slot costs nothing and
                # ensure_min creates a replacement meanwhile.
                log(f"slot {r.name} is paused; attempting resume")
                # Resume preserves the slot's billing type: jl defaults a
                # bare resume to on-demand (measured: a paused spot came
                # back on-demand at 2x price with a NEW machine id). The
                # mix logic reads is_spot fresh from jl list each cycle, so
                # the id change is harmless, but the price flip is not.
                args = ["resume", str(r.mid)]
                if r.is_spot:
                    args.append("--spot")
                args.append("--yes")
                try:
                    jl(*args)
                except RuntimeError as exc:
                    log(f"resume failed for {r.name}: {str(exc)[:120]}")
                return  # one action per cycle; the next pass re-checks
            if r.status in ("Running", "Pending", "Provisioning") and age < boot_timeout:
                continue  # alive and young: still booting, leave it alone
            first = self.unhealthy_since.setdefault(key, time.time())
            if r.status not in ("Running",) or time.time() - first > grace:
                log(f"slot {r.name} is {r.status or 'unreachable'} "
                    f"(age {int(age)}s); recreating")
                if r.status == "Running":
                    self.destroy(r)  # hung past boot timeout: kill + replace
                self.create_slot()
                self.unhealthy_since.pop(key, None)
                return  # one action per cycle; the next pass re-checks

    def age_of(self, replica):
        """Seconds since the instance was created, from jl's runtime field
        ("2 hours 13 minutes"), falling back to first-seen tracking."""
        runtime = (replica.info.get("runtime") or "").strip()
        total = 0
        for value, unit in re.findall(r"(\d+)\s*(day|hour|minute|second)", runtime):
            total += int(value) * {"day": 86400, "hour": 3600,
                                   "minute": 60, "second": 1}[unit]
        if total:
            return total
        first = self.first_seen.setdefault(replica.mid, time.time())
        return time.time() - first

    def autoscale(self, replicas):
        healthy = [r for r in replicas if r.healthy]
        self.finish_retirements(replicas)
        if not healthy:
            return
        cooldown = self.cfg.get("cooldown_seconds", 300)
        if time.time() - self.last_scale < cooldown:
            return
        queued = sum(r.waiting or 0 for r in healthy)
        busy = sum(r.running or 0 for r in healthy)
        cap = self.cfg.get("capacity_per_replica", 12) * len(healthy)
        if queued > self.cfg.get("scale_up_waiting", 2) or busy >= cap:
            if len(replicas) < self.cfg["max_replicas"]:
                log(f"scale up: queued={queued} busy={busy}/{cap}")
                self.create_slot()
            return
        # Scale down on LIGHT load, not just total silence: while the fleet
        # is under the concurrency threshold (default: fewer than 4 requests
        # in flight across all replicas) for the idle window, it shrinks
        # toward the minimum — keeping one warm replica through light
        # traffic instead of oscillating on and off zero.
        concurrency = sum((r.running or 0) + (r.waiting or 0) for r in healthy)
        threshold = self.cfg.get("scale_down_concurrency", 4)
        idle_for = self.cfg.get("scale_down_idle_seconds", 900)
        now = time.time()
        if concurrency < threshold:
            self.idle_since = self.idle_since or now
            if now - self.idle_since >= idle_for and len(replicas) > self.cfg["min_replicas"]:
                # Retire the least-loaded replica. Rewire FIRST so litellm
                # stops routing new requests (and new sessions) to it, let
                # the drain window pass, and only then destroy it —
                # destroying first would kill every in-flight request and
                # every session pinned to it by deployment_affinity.
                candidates = [r for r in healthy if r.name not in self.retiring]
                if not candidates:
                    return
                # Retire the least-loaded replica, preferring spot over the
                # on-demand backbone (which is kept for stability, not load).
                victim = min(candidates,
                             key=lambda r: ((0 if r.is_spot else 1),
                                            (r.running or 0) + (r.waiting or 0)))
                log(f"scale down: concurrency {concurrency} < {threshold} for "
                    f"{int(now - self.idle_since)}s; draining {victim.name}")
                self.retiring[victim.name] = now
                self.wire_litellm([r for r in replicas if r.name != victim.name])
                self.idle_since = None
        else:
            self.idle_since = None

    def finish_retirements(self, replicas):
        """After the drain window, destroy drained replicas."""
        drain = self.cfg.get("scale_down_drain_seconds", 120)
        for name, started in list(self.retiring.items()):
            if time.time() - started < drain:
                continue
            replica = next((r for r in replicas if r.name == name), None)
            if replica:
                log(f"retirement drain elapsed; destroying {name}")
                self.destroy(replica)
            self.retiring.pop(name)
            self.last_scale = time.time()

    def wire_litellm(self, replicas):
        """Reconfigure litellm when the set of healthy replicas changes."""
        healthy = sorted((r for r in replicas
                          if r.healthy and r.name not in self.retiring),
                         key=lambda r: r.name)
        ids = {r.name for r in healthy}
        # Never rewrite the config to an empty model_list: if every replica
        # is briefly unhealthy, the last good config keeps serving while
        # reap rebuilds the fleet.
        if not ids:
            return
        if self.cfg.get("router_api"):
            # DB-backed router: deployments are managed through the admin
            # API. Adding/removing them is HOT — no restart, no dropped
            # connections, and existing session pins survive scale-up.
            if ids == self.wired:
                return
            self.wire_via_api(healthy, ids)
            return
        if ids == self.wired and os.path.exists(os.path.expanduser("~/router/config.yaml")):
            return
        cfg = ["model_list:"]
        for r in healthy:
            cfg += [
                "  - model_name: GLM-5.3",
                "    litellm_params:",
                "      model: openai/GLM-5.3",
                f"      api_base: {r.api_url}/v1",
                f"      api_key: {self.fleet_key}",
            ]
        cfg += [
            "router_settings:",
            "  routing_strategy: simple-shuffle",
            "  model_group_affinity_config:",
            "    GLM-5.3:",
            "      - deployment_affinity",
            "      - session_affinity",
            "  deployment_affinity_ttl_seconds: 3600",
            "general_settings:",
            "  master_key: os.environ/LITELLM_MASTER_KEY",
            "litellm_settings:",
            "  drop_params: true",
        ]
        path = os.path.expanduser("~/router/config.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".new"
        with open(tmp, "w") as f:
            f.write("\n".join(cfg) + "\n")
        os.replace(tmp, path)
        rc = subprocess.run(["sudo", "systemctl", "restart", "litellm-router"],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            # The unit is transient (--collect): once litellm exits,
            # restart reports the unit as gone and cannot bring it back.
            # Leave self.wired unchanged so the next cycle retries, and say
            # loudly that the router needs a human.
            log(f"litellm restart FAILED rc={rc.returncode}: "
                f"{(rc.stderr or '').strip()[:200]}")
            return
        self.wired = ids
        log(f"litellm rewired: {sorted(ids)}")

    def router_api(self, method, path, payload=None):
        req = urllib.request.Request(
            f"{self.cfg['router_url']}{path}",
            data=json.dumps(payload).encode() if payload else None,
            headers={"Authorization": f"Bearer {self.cfg['master_key']}",
                     "Content-Type": "application/json"},
            method=method)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
        return json.loads(body) if body else {}

    def wire_via_api(self, healthy, ids):
        """Hot-manage deployments on a DB-backed router (Postgres+Redis):
        /model/new and /model/delete apply live via litellm's config-sync;
        no process restart, no dropped connections, pins survive."""
        info = self.router_api("GET", "/v1/model/info")
        current = {}  # api_base -> deployment id
        for d in info.get("data", []):
            params = d.get("litellm_params", {})
            current[params.get("api_base")] = d.get("model_info", {}).get("id")
        desired = {f"{r.api_url}/v1": r for r in healthy}
        for base, dep_id in current.items():
            if base not in desired and dep_id:
                self.router_api("POST", "/model/delete", {"id": dep_id})
                log(f"router: removed stale deployment {base}")
        for base, r in desired.items():
            if base not in current:
                self.router_api("POST", "/model/new", {
                    "model_name": "GLM-5.3",
                    "litellm_params": {
                        "model": "openai/GLM-5.3",
                        "api_base": base,
                        "api_key": self.fleet_key,
                    }})
                log(f"router: added deployment {r.name} at {base}")
        self.wired = ids
        log(f"litellm hot-wired: {sorted(ids)}")

    def cycle(self):
        replicas = [Replica(r) for r in self.slots()]
        for r in replicas:
            if r.status == "Running":
                r.probe()
                r.scrape()
        self.ensure_min(replicas)
        self.reap(replicas)
        self.autoscale(replicas)
        # refresh after possible creates so litellm wiring sees reality
        replicas = [Replica(r) for r in self.slots()]
        for r in replicas:
            if r.status == "Running":
                r.probe()
        self.wire_litellm(replicas)
        states = ", ".join(f"{r.name}:{r.status}:{'ok' if r.healthy else '-'}"
                           f"(run={r.running} wait={r.waiting})" for r in replicas)
        log(f"fleet: {states or 'empty'}")


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    fleet = Fleet(cfg)
    log(f"manager up: min={cfg['min_replicas']} max={cfg['max_replicas']} "
        f"poll={cfg.get('poll_seconds', 30)}s")
    while True:
        try:
            fleet.cycle()
        except Exception as exc:
            log(f"cycle error: {exc}")
        time.sleep(cfg.get("poll_seconds", 30))


if __name__ == "__main__":
    main()
