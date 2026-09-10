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
SLOT_RE = re.compile(r"^glm53-serve-")
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
        self.api_url = ""
        self.healthy = False
        self.waiting = None
        self.running = None
        self.cache = None
        self.idle_since = None

    def probe(self, key):
        """Discover the verified proxy URL from jl's endpoint list."""
        detail = jl_json("get", str(self.mid))
        candidates = list(detail.get("endpoints") or [])
        url = detail.get("url") or ""
        if url and url not in candidates:
            candidates.append(url)
        for base in candidates:
            code, _ = http_code(f"{base}/v1/models", key)
            if code == 200:
                self.api_url = base
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
        self.last_change = 0
        self.last_scale = 0
        self.wired = set()
        self.unhealthy_since = {}

    def slots(self):
        rows = jl_json("list")
        rows = rows if isinstance(rows, list) else rows.get("instances", rows.get("data", []))
        return [r for r in rows if SLOT_RE.match(r.get("name", ""))]

    def ensure_min(self, replicas):
        alive = [r for r in replicas if r.status in ("Running", "Pending", "Provisioning")]
        for _ in range(self.cfg["min_replicas"] - len(alive)):
            self.create_slot()

    def create_slot(self):
        used = {r.name for r in self.slots()}
        n = 0
        while f"glm53-serve-{n}" in used:
            n += 1
        name = f"glm53-serve-{n}"
        log(f"creating replica slot {name} (script {self.cfg['script_id']})")
        jl("create", "--gpu", self.cfg["gpu"], "--num-gpus", str(self.cfg["num_gpus"]),
           "--spot", "--region", self.cfg["region"], "--fs-id", str(self.cfg["fs_id"]),
           "--script-id", str(self.cfg["script_id"]), "--http-ports", "8000,1111",
           "--name", name, "--yes")
        self.last_scale = time.time()

    def destroy(self, replica):
        log(f"destroying {replica.name} ({replica.mid})")
        jl("destroy", str(replica.mid), "--yes")

    def reap(self, replicas):
        """Replace slots whose instance died (spot reclamation, crash)."""
        grace = self.cfg.get("unhealthy_grace_seconds", 300)
        for r in replicas:
            key = r.name
            if r.healthy:
                self.unhealthy_since.pop(key, None)
                continue
            first = self.unhealthy_since.setdefault(key, time.time())
            if r.status not in ("Running",) or time.time() - first > grace:
                log(f"slot {r.name} is {r.status or 'unreachable'}; recreating")
                if r.status == "Running":
                    self.destroy(r)  # hung: kill before replacing
                self.create_slot()
                self.unhealthy_since.pop(key, None)
                return  # one action per cycle; the next pass re-checks

    def autoscale(self, replicas):
        healthy = [r for r in replicas if r.healthy]
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
        # scale down: every replica fully idle (no running, no waiting) and
        # the fleet above minimum, for idle_seconds.
        idle_for = self.cfg.get("scale_down_idle_seconds", 900)
        now = time.time()
        victim = None
        for r in healthy:
            if (r.running or 0) > 0 or (r.waiting or 0) > 0:
                r.idle_since = None
            else:
                r.idle_since = r.idle_since or now
                if now - r.idle_since >= idle_for:
                    victim = victim or r
        if victim and len(replicas) > self.cfg["min_replicas"]:
            log(f"scale down: {victim.name} idle for {int(now - victim.idle_since)}s")
            self.destroy(victim)
            self.last_scale = time.time()

    def wire_litellm(self, replicas):
        """Reconfigure litellm when the set of healthy replicas changes."""
        healthy = sorted((r for r in replicas if r.healthy), key=lambda r: r.name)
        ids = {r.name for r in healthy}
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
        tmp = path + ".new"
        with open(tmp, "w") as f:
            f.write("\n".join(cfg) + "\n")
        os.replace(tmp, path)
        subprocess.run(["sudo", "systemctl", "restart", "litellm-router"],
                       capture_output=True)
        self.wired = ids
        log(f"litellm rewired: {sorted(ids)}")

    def cycle(self):
        replicas = [Replica(r) for r in self.slots()]
        for r in replicas:
            if r.status == "Running":
                r.probe(self.fleet_key)
                r.scrape()
        self.ensure_min(replicas)
        self.reap(replicas)
        self.autoscale(replicas)
        # refresh after possible creates so litellm wiring sees reality
        replicas = [Replica(r) for r in self.slots()]
        for r in replicas:
            if r.status == "Running":
                r.probe(self.fleet_key)
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
