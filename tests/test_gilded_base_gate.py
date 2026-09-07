#!/usr/bin/env python3
"""The image must compose the Gilded Gnosis base with pinned bytes only.

Every source this appliance installs on top of local-inference-lab's
Gilded Gnosis v20 r34 image is SHA-256 pinned in the Dockerfile. This gate
fails when the base reference, a pinned parent state, a maintenance payload,
or the read-lease patch drifts from what was reviewed, and when a VerdictAI
path or overlay reappears.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
MAINTENANCE = ROOT / "maintenance/r34-aibeast-20260815/context"

BASE_IMAGE = ("docker.io/voipmonitor/vllm@sha256:"
              "820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b")
SITE_PACKAGES = "/opt/venv/lib/python3.12/site-packages"

# Exact Gilded Gnosis r34 state, before this appliance changes anything.
PARENT_STATE = {
    "vllm/model_executor/layers/quantization/exl3.py":
        "42c0e9150a065c48e3780eebc8b3c89ea410d82610e2c14bc97546dee6866214",
    "vllm/v1/core/kv_cache_manager.py":
        "d540f800e948d4e94e72fb19e91ebb37f163e37a2a64d07051dfd7b5d87dfe2f",
    "vllm/v1/core/sched/scheduler.py":
        "1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce",
    "lmcache/integration/vllm/vllm_multi_process_adapter.py":
        "ea580badefb9a0fad5fa2ac1bdcff4f2b36fc85138d95a5be4b180f07dd2c874",
}

# Reviewed maintenance payloads carried from the running appliance's build.
MAINTENANCE_PAYLOADS = {
    "runtime/vllm/vllm/model_executor/layers/quantization/exl3.py":
        "78a732362077a715228ea096eb07fc9c074ddb712fc5e76e4f289bf4244f4918",
    "runtime/vllm/vllm/v1/core/kv_cache_manager.py":
        "e7b8bb464c5f4741f482e03bb28148cbec9d1d4438d8d70774eb95bf5f9cca78",
    "runtime/vllm/vllm/v1/core/sched/scheduler.py":
        "23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6",
    "runtime/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py":
        "0781f930e304992c75ebf596030b6a3f6d0bf697558de14168168531ee51fe21",
    "patches/0001-fix-mp-recover-expired-L1-read-leases.patch":
        "07c0e5e522c884847832d215ff72120a26df176c34a4c8777ca5096eed21a3f8",
}

# The nine files the expired-L1-read-lease patch rewrites, and the state the
# Dockerfile requires afterwards.
LEASE_STATE = {
    "lmcache/cli/commands/trace/_dispatch.py":
        "58cc7f828c2e65edbfe0d4720a849c931f9b217274e1dad511fcf2aa480aaa3a",
    "lmcache/v1/distributed/l1_manager.py":
        "bc98d1db99236847b5b5f88dc77c1a105dc796d206cf3d51e7940be642836e57",
    "lmcache/v1/distributed/storage_manager.py":
        "9472313897a53bf1c7af68c278921222e01cdfbf5e55198c0022bdf42e22872c",
    "lmcache/v1/multiprocess/config.py":
        "04ef2d9e1be5f986c1a93b0ab5acc7c87e47655eb9a239ca5d185ef809b1bed5",
    "lmcache/v1/multiprocess/engine_context.py":
        "89cfd0dc03e0e38fe6d29fc6b2775710d87fce46e800cfad6fe9f6b4b7f2cef0",
    "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py":
        "d62c23985d0b1f38ff9b7429b17b29931188512ca1c4e5c0f8e65325d2295e71",
    "lmcache/v1/multiprocess/modules/lookup.py":
        "ef51c00cd34a2ccc0840b4916e3c3d2d3e651eb5907200847e67972a67b8edf5",
    "lmcache/v1/multiprocess/server.py":
        "ee300edb1076b656f6148f362e0cdac6f836e7d48262bf143ac64338905a07c3",
    "lmcache/v1/multiprocess/session.py":
        "3e4f1f9e9e80ffd27dd7c8f00da5194897ea89e4227c6d18fe888044398793df",
}

MIRRORED = (
    "vllm/model_executor/layers/quantization/exl3.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/sched/scheduler.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GildedBaseGate(unittest.TestCase):
    def test_base_reference_is_the_reviewed_gilded_digest(self) -> None:
        self.assertIn(f"ARG BASE_IMAGE={BASE_IMAGE}", DOCKERFILE)
        self.assertIn("FROM ${BASE_IMAGE}", DOCKERFILE)
        # A tag or a floating reference would silently change the runtime.
        for match in re.finditer(r"^(?:FROM|ARG BASE_IMAGE=)\s*(\S+)", DOCKERFILE, re.M):
            reference = match.group(1)
            if reference.startswith("$"):
                continue
            self.assertRegex(reference, r"@sha256:[0-9a-f]{64}$", reference)

    def test_parent_state_is_pinned_for_runtime_and_source_tree(self) -> None:
        for relative, digest in PARENT_STATE.items():
            self.assertIn(f'{digest} "$SITE_PACKAGES/{relative}"', DOCKERFILE)
        for relative in MIRRORED:
            digest = PARENT_STATE[relative]
            self.assertIn(f"{digest} /opt/vllm/{relative}", DOCKERFILE)

    def test_maintenance_payloads_match_their_pins(self) -> None:
        for relative, digest in MAINTENANCE_PAYLOADS.items():
            payload = MAINTENANCE / relative
            self.assertTrue(payload.is_file(), f"missing build input: {payload}")
            self.assertEqual(sha256(payload), digest, str(payload))
            self.assertIn(f"maintenance/r34-aibeast-20260815/context/{relative}",
                          DOCKERFILE)

    def test_read_lease_patch_is_applied_and_its_result_pinned(self) -> None:
        self.assertIn('patch --batch --forward -p1 -d "$SITE_PACKAGES"', DOCKERFILE)
        patch_text = (MAINTENANCE
                      / "patches/0001-fix-mp-recover-expired-L1-read-leases.patch"
                      ).read_text(encoding="utf-8")
        touched = {line.split(" b/", 1)[1].strip()
                   for line in patch_text.splitlines() if line.startswith("diff --git ")}
        self.assertEqual(touched, set(LEASE_STATE))
        for relative, digest in LEASE_STATE.items():
            self.assertIn(f'{digest} "$SITE_PACKAGES/{relative}"', DOCKERFILE)

    def test_maintenance_result_is_pinned_before_release_overlays(self) -> None:
        installed = {
            "vllm/model_executor/layers/quantization/exl3.py":
                MAINTENANCE_PAYLOADS[
                    "runtime/vllm/vllm/model_executor/layers/quantization/exl3.py"],
            "vllm/v1/core/kv_cache_manager.py":
                MAINTENANCE_PAYLOADS["runtime/vllm/vllm/v1/core/kv_cache_manager.py"],
            "vllm/v1/core/sched/scheduler.py":
                MAINTENANCE_PAYLOADS["runtime/vllm/vllm/v1/core/sched/scheduler.py"],
            "lmcache/integration/vllm/vllm_multi_process_adapter.py":
                MAINTENANCE_PAYLOADS[
                    "runtime/lmcache/lmcache/integration/vllm/"
                    "vllm_multi_process_adapter.py"],
        }
        for relative, digest in installed.items():
            self.assertIn(f'{digest} "$SITE_PACKAGES/{relative}"', DOCKERFILE)
        # The overlay installers must run after the maintenance layer, because
        # their pinned before-states are the maintenance states.
        lease = DOCKERFILE.index("lmcache-read-lease-recovery.patch")
        refresh = DOCKERFILE.index("apply_glm53_refresh.py")
        cache = DOCKERFILE.index("patch_scopedlmcache_retrieve.py")
        self.assertLess(lease, refresh)
        self.assertLess(lease, cache)

    def test_no_verdict_lineage_remains(self) -> None:
        for token in ("verdictai", "/opt/infernal-invocation", "glm53-runtime",
                      "apply_glm53_runtime_overlays", "/opt/local-inference/nccl"):
            self.assertNotIn(token, DOCKERFILE, token)
        self.assertFalse((ROOT / "patches/glm53-runtime").exists())
        self.assertFalse((ROOT / "scripts/apply_glm53_runtime_overlays.py").exists())

    def test_gilded_specific_paths_are_asserted_at_build_time(self) -> None:
        for probe in ('test -d "$SITE_PACKAGES/b12x"',
                      "test -f /opt/exllamav3-python/exllamav3/modules/quant/"
                      "exl3_lib/quantize.py",
                      "test -f /opt/libnccl-local-inference.so.2.30.4",
                      "test -d /opt/vllm/vllm/parser"):
            self.assertIn(probe, DOCKERFILE, probe)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
