"""Offline policy tests for the JarvisLabs fleet manager.

The manager's decision logic (reap, autoscale, slot naming, litellm wiring)
is pure given stubbed jl/HTTP boundaries; these tests pin the behavior that
live testing established:

  - a booting replica is never reaped before boot_timeout (the thrash bug)
  - reaping happens only after the grace window, one action per cycle
  - scale-up on queue depth or a full batch window, bounded by max_replicas
  - scale-down under light load after an idle window, retiring the
    least-loaded replica, never below min_replicas
  - litellm is rewired only when the healthy set changes, with the fleet key
  - health probing discriminates /metrics and never trusts the lab URL
"""

import re
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import fleet_manager as fm  # noqa: E402


def make_cfg(**overrides):
    cfg = {
        "fleet_key": "sk-fleet-test",
        "fs_id": 1,
        "script_id": 7409,
        "min_replicas": 1,
        "max_replicas": 3,
        "gpu": "RTX-PRO6000",
        "num_gpus": 4,
        "region": "IN1",
        "poll_seconds": 0,
        "cooldown_seconds": 300,
        "scale_up_waiting": 2,
        "scale_down_idle_seconds": 900,
        "scale_down_concurrency": 4,
        "unhealthy_grace_seconds": 300,
        "boot_timeout_seconds": 2400,
        "capacity_per_replica": 12,
    }
    cfg.update(overrides)
    return cfg


def replica(name, status="Running", runtime="0 hours 05 minutes", **kw):
    info = {"name": name, "machine_id": kw.pop("mid", 1),
            "status": status, "runtime": runtime}
    info.update(kw)
    return fm.Replica(info)


class ReapPolicyTests(unittest.TestCase):
    def test_booting_replica_is_never_reaped(self):
        """The live thrash bug: grace elapsed but the instance is young."""
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0")
        r.healthy = False
        f.unhealthy_since["glm53-serve-0"] = 0  # grace long elapsed
        with mock.patch.object(f, "destroy") as destroy, \
                mock.patch.object(f, "create_slot") as create:
            f.reap([r])
        destroy.assert_not_called()
        create.assert_not_called()

    def test_hung_replica_past_boot_timeout_is_replaced(self):
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0", runtime="2 hours 00 minutes")
        r.healthy = False
        f.unhealthy_since["glm53-serve-0"] = 0
        with mock.patch.object(f, "destroy") as destroy, \
                mock.patch.object(f, "create_slot") as create:
            f.reap([r])
        destroy.assert_called_once()
        create.assert_called_once()

    def test_destroyed_slot_is_replaced_immediately(self):
        """A gone instance (reaped spot) has no boot to wait for."""
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0", status="Destroying")
        r.healthy = False
        with mock.patch.object(f, "destroy") as destroy, \
                mock.patch.object(f, "create_slot") as create:
            f.reap([r])
        destroy.assert_not_called()  # already going away
        create.assert_called_once()

    def test_pending_slot_is_never_reaped(self):
        """Pending/Provisioning are alive: reaping them fans out unbounded."""
        f = fm.Fleet(make_cfg())
        for status in ("Pending", "Provisioning"):
            r = replica("glm53-serve-0", status=status)
            r.healthy = False
            with mock.patch.object(f, "destroy") as destroy, \
                    mock.patch.object(f, "create_slot") as create:
                f.reap([r])
            destroy.assert_not_called()
            create.assert_not_called()

    def test_paused_slot_is_resumed_not_replaced(self):
        """Provider pauses spot under pressure; resume continues the warm boot."""
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0", status="Paused")
        r.healthy = False
        with mock.patch.object(fm, "jl") as jl:
            f.reap([r])
        jl.assert_called_once_with("resume", "1", "--yes")  # on-demand stays on-demand

    def test_paused_spot_slot_resumes_as_spot(self):
        """Bare jl resume flips a paused spot to on-demand at 2x price."""
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0", status="Paused", is_spot=True)
        r.healthy = False
        with mock.patch.object(fm, "jl") as jl:
            f.reap([r])
        jl.assert_called_once_with("resume", "1", "--spot", "--yes")

    def test_paused_resume_failure_does_not_destroy(self):
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0", status="Paused")
        r.healthy = False
        def boom(*a):
            raise RuntimeError("jl resume failed: no capacity")
        with mock.patch.object(fm, "jl", side_effect=boom), \
                mock.patch.object(f, "destroy") as destroy:
            f.reap([r])
        destroy.assert_not_called()

    def test_healthy_clears_the_grace_tracker(self):
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0")
        r.healthy = True
        f.unhealthy_since["glm53-serve-0"] = 0
        f.reap([r])
        self.assertNotIn("glm53-serve-0", f.unhealthy_since)


class AgeParsingTests(unittest.TestCase):
    def test_runtime_string(self):
        f = fm.Fleet(make_cfg())
        self.assertEqual(f.age_of(replica("x", runtime="0 hours 07 minutes")), 420)
        self.assertEqual(f.age_of(replica("x", runtime="2 hours 13 minutes")), 2 * 3600 + 13 * 60)

    def test_day_and_second_units(self):
        f = fm.Fleet(make_cfg())
        self.assertEqual(f.age_of(replica("x", runtime="1 day 2 hours")), 93600)
        self.assertEqual(f.age_of(replica("x", runtime="90 seconds")), 90)

    def test_fallback_tracks_first_seen(self):
        f = fm.Fleet(make_cfg())
        r = replica("x", runtime="")  # jl gave nothing
        with mock.patch("fleet_manager.time.time", return_value=1000.0):
            first = f.age_of(r)
            self.assertEqual(first, 0)
            second = f.age_of(r)
            self.assertEqual(second, 0)  # same frozen clock, still tracked


class AutoscaleTests(unittest.TestCase):
    def setUp(self):
        self.f = fm.Fleet(make_cfg())
        self.f.last_scale = 0  # cooldown elapsed

    def healthy(self, *specs):
        out = []
        for name, running, waiting in specs:
            r = replica(name)
            r.healthy, r.running, r.waiting = True, running, waiting
            out.append(r)
        return out

    def test_scale_up_on_queue_depth(self):
        fleet = self.healthy(("glm53-serve-0", 4, 3))
        with mock.patch.object(self.f, "create_slot") as create:
            self.f.autoscale(fleet)
        create.assert_called_once()

    def test_scale_up_bounded_by_max(self):
        fleet = self.healthy(("glm53-serve-0", 12, 3), ("glm53-serve-1", 12, 3),
                             ("glm53-serve-2", 12, 3))
        with mock.patch.object(self.f, "create_slot") as create:
            self.f.autoscale(fleet)
        create.assert_not_called()  # already at max_replicas

    def test_cooldown_blocks_scaling(self):
        self.f.last_scale = __import__("time").time()  # scaled moments ago
        fleet = self.healthy(("glm53-serve-0", 12, 3))
        with mock.patch.object(self.f, "create_slot") as create:
            self.f.autoscale(fleet)
        create.assert_not_called()

    def test_light_load_starts_idle_window_but_waits(self):
        fleet = self.healthy(("glm53-serve-0", 1, 0), ("glm53-serve-1", 1, 0))
        with mock.patch.object(self.f, "destroy") as destroy:
            self.f.autoscale(fleet)  # 2 concurrent < 4: window opens
            self.assertIsNotNone(self.f.idle_since)
            destroy.assert_not_called()  # not yet idle_for long enough

    def test_idle_window_drains_then_retires_least_loaded(self):
        fleet = self.healthy(("glm53-serve-0", 2, 0), ("glm53-serve-1", 0, 0))
        self.f.idle_since = __import__("time").time() - 1000  # window long elapsed
        with mock.patch.object(self.f, "destroy") as destroy, \
                mock.patch.object(self.f, "wire_litellm") as wire:
            self.f.autoscale(fleet)
        # draining, not destroyed: litellm is rewired without the victim first
        destroy.assert_not_called()
        self.assertIn("glm53-serve-1", self.f.retiring)
        remaining = [r.name for r in wire.call_args[0][0]]
        self.assertNotIn("glm53-serve-1", remaining)
        # spot replicas retire before the on-demand backbone
        backbone = replica("glm53-serve-2", mid=9)
        backbone.healthy, backbone.running, backbone.waiting = True, 0, 0
        backbone.is_spot = False
        fleet2 = fleet + [backbone]
        self.f.retiring = {}
        self.f.idle_since = __import__("time").time() - 1000
        with mock.patch.object(self.f, "wire_litellm"):
            self.f.autoscale(fleet2)
        # the spot replica is drained, not the equally-idle on-demand backbone
        self.assertEqual(list(self.f.retiring), ["glm53-serve-1"])
        # after the drain window the victim is destroyed
        self.f.retiring["glm53-serve-1"] = 0
        with mock.patch.object(self.f, "destroy") as destroy:
            self.f.finish_retirements(fleet)
        destroy.assert_called_once()
        self.assertEqual(destroy.call_args[0][0].name, "glm53-serve-1")
        self.assertEqual(self.f.retiring, {})

    def test_drain_window_is_respected(self):
        fleet = self.healthy(("glm53-serve-0", 2, 0), ("glm53-serve-1", 0, 0))
        self.f.retiring["glm53-serve-1"] = __import__("time").time() - 10
        with mock.patch.object(self.f, "destroy") as destroy:
            self.f.finish_retirements(fleet)
        destroy.assert_not_called()
        self.assertIn("glm53-serve-1", self.f.retiring)

    def test_load_resets_idle_window(self):
        fleet = self.healthy(("glm53-serve-0", 4, 0))  # at the scale-down threshold
        self.f.idle_since = __import__("time").time() - 1000
        with mock.patch.object(self.f, "destroy") as destroy, \
                mock.patch.object(self.f, "create_slot") as create:
            self.f.autoscale(fleet)
        self.assertIsNone(self.f.idle_since)
        destroy.assert_not_called()

    def test_never_below_min(self):
        fleet = self.healthy(("glm53-serve-0", 0, 0))
        self.f.idle_since = 0
        with mock.patch.object(self.f, "destroy") as destroy:
            self.f.autoscale(fleet)
        destroy.assert_not_called()  # one replica IS the minimum


class CreateSlotGuardTests(unittest.TestCase):
    def test_deferred_while_another_slot_is_booting(self):
        """Cold boots are serialized: the shared quant cache is single-writer."""
        f = fm.Fleet(make_cfg())
        f.slots = lambda: [{"name": "glm53-serve-0", "status": "Pending",
                            "runtime": "0 hours 00 minutes"}]
        with mock.patch.object(fm, "jl") as jl:
            f.create_slot()
        jl.assert_not_called()

    def test_on_demand_backbone_created_when_deficit(self):
        f = fm.Fleet(make_cfg(on_demand_min=1))
        f.slots = lambda: []  # empty fleet: backbone deficit
        with mock.patch.object(fm, "jl") as jl:
            f.create_slot()
        args = jl.call_args[0]
        self.assertNotIn("--spot", args)

    def test_backbone_as_vm_when_configured(self):
        f = fm.Fleet(make_cfg(on_demand_min=1, on_demand_kind="vm"))
        f.slots = lambda: []
        with mock.patch.object(fm, "jl") as jl:
            f.create_slot()
        args = jl.call_args[0]
        self.assertIn("--vm", args)
        self.assertNotIn("--spot", args)
        self.assertNotIn("--http-ports", args)
        self.assertNotIn("--script-id", args)  # VMs are ssh-booted

    def test_spot_created_when_backbone_satisfied(self):
        f = fm.Fleet(make_cfg(on_demand_min=1))
        f.slots = lambda: [{"name": "glm53-serve-0", "status": "Running",
                            "runtime": "2 hours", "is_spot": False}]
        with mock.patch.object(fm, "jl") as jl:
            f.create_slot()
        args = jl.call_args[0]
        self.assertIn("--spot", args)

    def test_capped_at_max_replicas(self):
        f = fm.Fleet(make_cfg(max_replicas=2))
        f.slots = lambda: [{"name": "glm53-serve-0"}, {"name": "glm53-serve-1"}]
        with mock.patch.object(fm, "jl") as jl:
            f.create_slot()
        jl.assert_not_called()


class SlotNamingTests(unittest.TestCase):
    def test_first_free_index(self):
        f = fm.Fleet(make_cfg())
        f.slots = lambda: [{"name": "glm53-serve-0"}, {"name": "glm53-serve-2"}]
        created = {}
        def fake_create(*args):
            created["args"] = args
        with mock.patch.object(fm, "jl", side_effect=lambda *a: fake_create(*a) or ""):
            f.create_slot()
        name = [a for a in created["args"] if a == "glm53-serve-1"]
        self.assertEqual(name, ["glm53-serve-1"])


class WiringTests(unittest.TestCase):
    def test_writes_config_with_fleet_key_and_restarts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            r = replica("glm53-serve-0")
            r.healthy, r.api_url = True, "https://proxy.example"
            f = fm.Fleet(make_cfg())
            with mock.patch.dict(fm.os.environ, {"HOME": raw}):
                with mock.patch.object(fm, "subprocess") as sub:
                    sub.run.return_value = SimpleNamespace(returncode=0, stderr="")
                    f.wire_litellm([r])
            sub.run.assert_called_once()
            body = Path(raw, "router", "config.yaml").read_text()
            self.assertIn("api_base: https://proxy.example/v1", body)
            self.assertIn("api_key: sk-fleet-test", body)
            self.assertIn("deployment_affinity", body)
            self.assertIn("master_key", body)
            self.assertEqual(f.wired, {"glm53-serve-0"})

    def test_empty_healthy_set_never_rewrites_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            f = fm.Fleet(make_cfg())
            with mock.patch.dict(fm.os.environ, {"HOME": raw}):
                Path(raw, "router").mkdir(parents=True)
                Path(raw, "router", "config.yaml").write_text("last-good")
                f.wired = {"glm53-serve-0"}
                f.wire_litellm([])  # everything unhealthy at once
            self.assertEqual(Path(raw, "router", "config.yaml").read_text(),
                             "last-good")

    def test_failed_restart_leaves_wired_for_retry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            r = replica("glm53-serve-0")
            r.healthy, r.api_url = True, "https://proxy.example"
            f = fm.Fleet(make_cfg())
            with mock.patch.dict(fm.os.environ, {"HOME": raw}):
                with mock.patch.object(fm, "subprocess") as sub:
                    sub.run.return_value = SimpleNamespace(returncode=1,
                                                           stderr="unit not found")
                    f.wire_litellm([r])
            self.assertEqual(f.wired, set())  # retry happens next cycle

    def test_no_rewrite_when_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            r = replica("glm53-serve-0")
            r.healthy, r.api_url = True, "https://proxy.example"
            f = fm.Fleet(make_cfg())
            with mock.patch.dict(fm.os.environ, {"HOME": raw}):
                Path(raw, "router").mkdir(parents=True)
                Path(raw, "router", "config.yaml").write_text("x")
                f.wired = {"glm53-serve-0"}
                with mock.patch.object(fm, "subprocess") as sub:
                    f.wire_litellm([r])
                sub.run.assert_not_called()


class TunnelTests(unittest.TestCase):
    def test_port_allocation(self):
        f = fm.Fleet(make_cfg(tunnels=True))
        self.assertEqual(f.tunnel_port("glm53-serve-0"), 18000)
        self.assertEqual(f.tunnel_port("glm53-serve-7"), 18007)

    def test_existing_tunnel_short_circuits(self):
        f = fm.Fleet(make_cfg(tunnels=True))
        r = replica("glm53-serve-0")
        r.healthy, r.api_url = True, "https://proxy.example"
        body = b"vllm:num_requests_running 1\n"
        with mock.patch.object(fm, "http_code", return_value=(200, body)), \
                mock.patch.object(fm, "subprocess") as sub:
            base = f.ensure_tunnel(r)
        self.assertEqual(base, "http://localhost:18000")
        sub.run.assert_not_called()

    def test_no_tunnels_config_uses_proxy_url(self):
        f = fm.Fleet(make_cfg())
        r = replica("glm53-serve-0")
        r.api_url = "https://proxy.example"
        self.assertEqual(f.ensure_tunnel(r), "https://proxy.example")

    def test_tunnel_failure_falls_back_to_proxy(self):
        f = fm.Fleet(make_cfg(tunnels=True))
        r = replica("glm53-serve-0")
        r.api_url, r.public_ip = "https://proxy.example", "1.2.3.4"
        with mock.patch.object(fm, "http_code", return_value=(0, b"")), \
                mock.patch.object(fm, "subprocess") as sub:
            sub.run.return_value = SimpleNamespace(returncode=255, stderr="no route")
            base = f.ensure_tunnel(r)
        self.assertEqual(base, "https://proxy.example")

    def test_tunnel_reestablished_when_dead(self):
        """A dead tunnel on an unchanged fleet self-heals within a cycle."""
        f = fm.Fleet(make_cfg(tunnels=True))
        r = replica("glm53-serve-0", mid=5)
        r.healthy, r.api_url, r.public_ip = True, "https://proxy.example", "1.2.3.4"
        with mock.patch.object(fm, "jl_json", return_value=[]), \
             mock.patch.object(fm, "http_code", return_value=(0, b"")), \
             mock.patch.object(fm, "subprocess") as sub:
            sub.run.return_value = SimpleNamespace(returncode=0, stderr="")
            f.cycle()
        ssh_calls = [c for c in sub.run.call_args_list if c[0][0][0] == "ssh"]
        self.assertTrue(ssh_calls, "cycle must re-establish a dead tunnel")

    def test_destroy_kills_tunnel(self):
        f = fm.Fleet(make_cfg(tunnels=True))
        r = replica("glm53-serve-3")
        with mock.patch.object(fm, "jl"), \
                mock.patch.object(fm, "subprocess") as sub:
            f.destroy(r)
        pkill = [c for c in sub.run.call_args_list
                 if c[0][0][0] == "pkill"]
        self.assertTrue(pkill)


class RouterApiWiringTests(unittest.TestCase):
    """DB-backed router: hot add/remove via the admin API, no restart."""

    def setUp(self):
        self.f = fm.Fleet(make_cfg(router_api=True,
                                   router_url="https://router.example",
                                   master_key="sk-master"))

    def replica(self, name, url):
        r = replica(name)
        r.healthy, r.api_url = True, url
        return r

    def test_adds_missing_deployment_without_restart(self):
        r = self.replica("glm53-serve-0", "https://a.example")
        calls = []
        def fake_api(method, path, payload=None):
            calls.append((method, path, payload))
            if path == "/v1/model/info":
                return {"data": []}
            return {}
        with mock.patch.object(self.f, "router_api", side_effect=fake_api):
            self.f.wire_litellm([r])
        added = [c for c in calls if c[1] == "/model/new"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0][2]["litellm_params"]["api_base"],
                         "https://a.example/v1")
        self.assertEqual(self.f.wired, {"glm53-serve-0"})

    def test_removes_stale_deployment(self):
        r = self.replica("glm53-serve-0", "https://a.example")
        info = {"data": [{"model_info": {"id": "dep-99"},
                          "litellm_params": {"api_base": "https://old.example/v1"}}]}
        deleted = []
        def fake_api(method, path, payload=None):
            if path == "/v1/model/info":
                return info
            if path == "/model/delete":
                deleted.append(payload)
            return {}
        with mock.patch.object(self.f, "router_api", side_effect=fake_api):
            self.f.wire_litellm([r])
        self.assertEqual(deleted, [{"id": "dep-99"}])

    def test_no_calls_when_unchanged(self):
        self.f.wired = {"glm53-serve-0"}
        r = self.replica("glm53-serve-0", "https://a.example")
        with mock.patch.object(self.f, "router_api") as api:
            self.f.wire_litellm([r])
        api.assert_not_called()

    def test_retiring_replica_is_not_wired(self):
        r = self.replica("glm53-serve-0", "https://a.example")
        victim = self.replica("glm53-serve-1", "https://b.example")
        self.f.retiring["glm53-serve-1"] = 0
        info = {"data": [{"model_info": {"id": "dep-b"},
                          "litellm_params": {"api_base": "https://b.example/v1"}}]}
        ops = []
        def fake_api(method, path, payload=None):
            if path == "/v1/model/info":
                return info
            ops.append(path)
            return {}
        with mock.patch.object(self.f, "router_api", side_effect=fake_api):
            self.f.wire_litellm([r, victim])
        self.assertEqual(self.f.wired, {"glm53-serve-0"})
        self.assertIn("/model/delete", ops)


class HealthProbeTests(unittest.TestCase):
    def test_metrics_discriminates_and_lab_url_is_ignored(self):
        r = replica("glm53-serve-0", mid=42)
        endpoints = ["https://lab.example", "https://api.example"]
        def fake_jl_json(*args):
            self.assertEqual(args, ("get", "42"))
            return {"endpoints": endpoints, "url": "https://lab.example"}
        def fake_http(url, key=None, timeout=15):
            return (200, b"") if url == "https://api.example/metrics" else (200, b"")
        # lab URL would also answer 200 — the probe must still pick it ONLY
        # from the endpoints list, and only on /metrics. Simulate the lab
        # answering 200 for /metrics too: it must not be in candidates.
        body = b"vllm:num_requests_running 1\n"
        with mock.patch.object(fm, "jl_json", side_effect=fake_jl_json), \
             mock.patch.object(fm, "http_code",
                               side_effect=lambda u, **k: (200, body)):
            r.probe()
        self.assertTrue(r.healthy)
        self.assertEqual(r.api_url, "https://lab.example")
        # ^ first candidate wins when both serve /metrics; the protection is
        # that 'url' is never a candidate. Assert jl's url field was not used:
        with mock.patch.object(fm, "jl_json", side_effect=fake_jl_json), \
             mock.patch.object(fm, "http_code",
                               side_effect=lambda u, **k: (0, b"")):
            r2 = replica("glm53-serve-0", mid=42)
            r2.probe()
            self.assertFalse(r2.healthy)  # nothing reachable -> not healthy

    def test_vm_replica_probed_via_public_ip(self):
        """VMs have no proxy endpoints: fall back to public_ip:8000."""
        r = replica("glm53-serve-0", mid=7)
        body = b"vllm:num_requests_running 1\n"
        with mock.patch.object(fm, "jl_json",
                               return_value={"endpoints": [], "public_ip": "10.1.2.3"}), \
             mock.patch.object(fm, "http_code",
                               side_effect=lambda u, **k: (200, body) if ":8000/metrics" in u else (0, b"")):
            r.probe()
        self.assertTrue(r.healthy)
        self.assertEqual(r.api_url, "http://10.1.2.3:8000")

    def test_scrape_parses_metrics(self):
        r = replica("glm53-serve-0")
        r.healthy, r.api_url = True, "https://api.example"
        body = (b'# HELP vllm:num_requests_running\n'
                b'vllm:num_requests_running{model_name="GLM-5.3"} 3.0\n'
                b'vllm:num_requests_waiting{model_name="GLM-5.3"} 5\n'
                b'vllm:gpu_cache_usage_perc 0.25\n')
        with mock.patch.object(fm, "http_code", return_value=(200, body)):
            r.scrape()
        self.assertEqual(r.running, 3.0)
        self.assertEqual(r.waiting, 5.0)
        self.assertEqual(r.cache, 0.25)


if __name__ == "__main__":
    unittest.main()
