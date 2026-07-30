#!/usr/bin/env python3
"""patch_exl3_mixk.py (v5) — mixed-K (per-expert 3/4 bpw) rank-sliced EXL3 MoE
for the installed vLLM exl3.py (GG v20 r11-r14).

GG v20-r14 carries the reviewed one-grid mixed-K implementation natively. On
that lineage this compatibility installer verifies the complete native API and
leaves the immutable upstream source untouched. The appended v5 implementation
remains the r11-r13 fallback.

v5 over v4:
  * inherits r13's backend-declared minimum capturable Trellis row count
    instead of retaining the pre-r12 literal default of 4.

v4 over v3:
  * supports r13's consolidated ``sparkinfer.moe.fused_moe`` weight/runtime
    planning API while retaining the r11 API as an explicit compatibility
    branch;
  * keys target and draft arenas by r13's stamped runtime owner;
  * makes the planned batch capacity invariant, matching r13's fix for
    mid-serve arena allocation and first-request OOMs.

v3 over v2:
  * runtime cache mirrors the uniform path: module-global, keyed on dims and
    tier signature, shared across all target layers (v2 keyed per layer and
    allocated per-layer scratch -> OOM during the profile forward);
  * one decode arena + one prefill arena aliased across tiers via exact-shape
    views (tiers execute sequentially); net scratch ~= uniform + accumulator;
  * env knobs identical to the uniform path: VLLM_EXL3_TRELLIS_MIN_M/MAX_M,
    VLLM_EXL3_TRELLIS_BLOCK_M, VLLM_EXL3_PREFILL_TRELLIS/PREFILL_BLOCK_M;
    max_batched from layer.exl3_max_num_batched_tokens like uniform;
  * keeps the v2 mapped top-k-sum fix (route map passed as output_expert_map).

Idempotent + upgradeable: v5 marker short-circuits; any older mixk install is
rebuilt in place from the v1 marker. Backup at exl3.py.orig on first apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER_V1 = "# === EXL3-MIXK-PATCH v1 ==="
MARKER_V3 = "# === EXL3-MIXK-PATCH v3 (shared runtime, aliased arenas) ==="
MARKER_V4 = "# === EXL3-MIXK-PATCH v4 (r13 consolidated fused-MoE API) ==="
MARKER_V5 = "# === EXL3-MIXK-PATCH v5 (backend Trellis minimum) ==="
NATIVE_R14_MARKERS = (
    "def _load_sparkinfer_mixed_trellis(",
    "def _prepare_mixed_rank_sliced_weights(",
    "def _apply_mixed_rank_sliced(",
)


def find_exl3() -> Path:
    import vllm
    import os
    return Path(os.path.dirname(vllm.__file__)) / "model_executor/layers/quantization/exl3.py"


E1_OLD = '''        self.rank_sliced_metadata = dict(metadata)
        self.bits = float(metadata["bits"])
        self.codebook = str(metadata["codebook"])'''
E1_NEW = '''        self.rank_sliced_metadata = dict(metadata)
        bits_field = metadata["bits"]
        if isinstance(bits_field, str) and bits_field.strip().lower() == "mixed":
            k_values = sorted(int(k) for k in metadata.get("k_values", ()))
            if not k_values or any(k not in (3, 4, 5, 6) for k in k_values):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires k_values within 3..6, got "
                    f"{metadata.get('k_values')!r}"
                )
            self.bits = None
            self.mixed_k_values = tuple(k_values)
        else:
            self.bits = float(bits_field)
            self.mixed_k_values = None
        self.codebook = str(metadata["codebook"])'''

E2_OLD = 'preallocate=rank_sliced and suffix in {"suh", "svh", "trellis"},'
E2_NEW = ('preallocate=rank_sliced and suffix in '
          '({"suh", "svh"} if getattr(self.quant_config, "mixed_k_values", None) '
          'else {"suh", "svh", "trellis"}),')

E3_OLD = '''    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
'''
E3_NEW = '''    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        if getattr(self.quant_config, "mixed_k_values", None):
            return self._prepare_rank_sliced_weights_mixk(layer)
'''

E5_OLD = '''        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])'''
E5_NEW = '''        if getattr(layer, "exl3_mixk", None) is not None:
            return self._apply_rank_sliced_mixk(layer, x, topk_weights, topk_ids)
        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])'''


APPEND = (
    MARKER_V1 + "\n" + MARKER_V3 + "\n" + MARKER_V4 + "\n" + MARKER_V5
) + '''
# Mixed-K rank-sliced EXL3: experts partitioned by native trellis width into
# K-homogeneous tiers; one sparkinfer trellis_moe Weights per tier per layer;
# global routing with per-tier route/output expert maps (-1 filtered by the
# w4a16 route pack and its mapped top-k sum). Plans and scratch are shared
# across all layers with the same tier signature, exactly like the uniform
# runtime; the two tier arenas alias one allocation since tiers execute
# sequentially within a layer.

_MIXK_RUNTIMES: dict[tuple, dict] = {}


def _mixk_prepare_rank_sliced_weights(self, layer) -> None:
    consolidated_api = "_load_sparkinfer_fused_moe" in globals()
    api = (
        _load_sparkinfer_fused_moe()
        if consolidated_api
        else _load_sparkinfer_trellis()
    )
    num_experts = int(layer.local_num_experts)
    hidden_size = int(layer.exl3_hidden_size)
    intermediate_size = int(layer.exl3_intermediate_size_per_partition)
    allowed = set(self.quant_config.mixed_k_values)

    w13_p = layer.w13_trellis
    w2_p = layer.w2_trellis
    sids = list(w13_p.exl3_shard_ids)
    if len(sids) != 2 or len(w2_p.exl3_shard_ids) != 1:
        raise ValueError("mixed EXL3 expects two w13 shards and one w2 shard")
    w2_sid = w2_p.exl3_shard_ids[0]

    k_of = []
    for e in range(num_experts):
        widths = {
            int(w13_p.exl3_tensors[(e, sids[0])].shape[-1]),
            int(w13_p.exl3_tensors[(e, sids[1])].shape[-1]),
            int(w2_p.exl3_tensors[(e, w2_sid)].shape[-1]),
        }
        if len(widths) != 1:
            raise ValueError(f"expert {e}: inconsistent trellis widths {widths}")
        width = widths.pop()
        if width % 16 or width // 16 not in allowed:
            raise ValueError(
                f"expert {e}: trellis width {width} outside declared k_values "
                f"{sorted(allowed)}"
            )
        k_of.append(width // 16)
    tiers = {}
    for e, k in enumerate(k_of):
        tiers.setdefault(k, []).append(e)

    gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
    gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
    down_suh = self._rank_sliced_backing(layer, "w2_suh")
    down_svh = self._rank_sliced_backing(layer, "w2_svh")
    tile_config = self._trellis_tile_config(hidden_size, intermediate_size)
    marker = next(iter(layer.w13_mcg.exl3_tensors.values()))
    device = gate_suh.device

    tier_entries = []
    for k in sorted(tiers):
        experts = tiers[k]
        idx = torch.tensor(experts, dtype=torch.int64, device=device)
        w13_k = torch.stack([
            torch.stack([w13_p.exl3_tensors[(e, sid)] for e in experts])
            for sid in sids
        ]).contiguous()
        w2_k = torch.stack(
            [w2_p.exl3_tensors[(e, w2_sid)] for e in experts]
        ).contiguous()
        expected_w13 = (2, len(experts), hidden_size // 16,
                        intermediate_size // 16, 16 * k)
        expected_w2 = (len(experts), intermediate_size // 16,
                       hidden_size // 16, 16 * k)
        if tuple(w13_k.shape) != expected_w13 or tuple(w2_k.shape) != expected_w2:
            raise ValueError(
                f"mixed EXL3 tier K={k} slab geometry mismatch: "
                f"w13={tuple(w13_k.shape)}, w2={tuple(w2_k.shape)}"
            )
        g_suh = gate_suh.index_select(0, idx).contiguous()
        u_suh = up_suh.index_select(0, idx).contiguous()
        g_svh = gate_svh.index_select(0, idx).contiguous()
        u_svh = up_svh.index_select(0, idx).contiguous()
        d_suh = down_suh.index_select(0, idx).contiguous()
        d_svh = down_svh.index_select(0, idx).contiguous()
        rotations = torch.empty(
            (len(experts), 3 * intermediate_size),
            dtype=torch.float16, device=device,
        )
        rotations[:, :intermediate_size].copy_(g_svh)
        rotations[:, intermediate_size:2 * intermediate_size].copy_(u_svh)
        rotations[:, 2 * intermediate_size:].copy_(d_suh)
        if consolidated_api:
            weight_plan = api.plan_weights(
                quant_modes="w4a16",
                source_format="exl3_trellis_mcg",
                activation=layer.activation.value,
                params_dtype=layer.exl3_params_dtype,
                num_experts=len(experts),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                w13_layout="w13",
                trellis_bits=k,
                trellis_tile_config=tile_config,
            )
            weights = api.prepare_weights(
                plan=weight_plan,
                params_dtype=layer.exl3_params_dtype,
                w1_fp4=w13_k,
                w2_fp4=w2_k,
                gate_suh=g_suh,
                up_suh=u_suh,
                intermediate_rotations=rotations,
                down_svh=d_svh,
                trellis_mcg=marker,
            )
        else:
            weights = api.prepare_weights(
                w13_k, w2_k,
                gate_suh=g_suh, up_suh=u_suh,
                intermediate_rotations=rotations,
                down_svh=d_svh,
                codebook="mcg", mcg=marker,
                tile_config=tile_config,
            )
        route_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        route_map[idx] = torch.arange(len(experts), dtype=torch.int32, device=device)
        tier_entries.append({
            "k": k,
            "num_experts": len(experts),
            "weights": weights,
            "route_expert_map": route_map,
            "_slabs": (w13_k, w2_k, g_suh, u_suh, rotations, d_svh),
        })
        logger.info(
            "EXL3 mixk %s: tier K=%d holds %d/%d experts",
            getattr(layer, "prefix", "moe"), k, len(experts), num_experts,
        )

    w13_p.exl3_tensors.clear()
    w2_p.exl3_tensors.clear()
    layer.exl3_mixk = {
        "tiers": tier_entries,
        "signature": tuple((t["k"], t["num_experts"]) for t in tier_entries),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
    }
    layer.exl3_trellis_tile_config = tile_config

    # Rank-sliced scale parameters are preallocated as full contiguous slabs.
    # The tier-specific copies above are now the only payloads used by the
    # mixed-K runtime. Keeping the source slabs would duplicate them for every
    # routed layer; keeping their freed allocator blocks cached is even worse
    # because the per-expert trellis allocations cannot satisfy the next
    # tier-contiguous slab. Release both before preparing the next layer.
    for param_name in ("w13_suh", "w13_svh", "w2_suh", "w2_svh"):
        param = getattr(layer, param_name)
        param.exl3_tensors.clear()
        param.exl3_backing = None
    del gate_suh, up_suh, gate_svh, up_svh, down_suh, down_svh
    torch.cuda.empty_cache()


def _mixk_scratch_view(backing: torch.Tensor, spec) -> torch.Tensor:
    nbytes = 1
    for dim in spec.shape:
        nbytes *= int(dim)
    nbytes *= torch.empty((), dtype=spec.dtype).element_size()
    return backing.narrow(0, 0, nbytes).view(spec.dtype).view(tuple(spec.shape))


def _mixk_runtime(self, layer, x, topk_ids):
    mix = layer.exl3_mixk
    min_trellis_m = _positive_env_int(
        "VLLM_EXL3_TRELLIS_MIN_M", _DEFAULT_TRELLIS_MIN_M
    )
    max_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)
    block_m = _positive_env_int("VLLM_EXL3_TRELLIS_BLOCK_M", 8)
    prefill_trellis = os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") == "1"
    prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)
    if min_trellis_m > max_trellis_m:
        raise ValueError("VLLM_EXL3_TRELLIS_MIN_M cannot exceed VLLM_EXL3_TRELLIS_MAX_M")
    if not prefill_trellis:
        raise ValueError(
            "mixed-K EXL3 requires the planned prefill path "
            "(VLLM_EXL3_PREFILL_TRELLIS=1); the eager parity fallback has no "
            "mixed-K support"
        )
    # This capacity must not depend on the live batch. r13 fixed the uniform
    # path after a cache miss above the planned size silently allocated a new
    # ~1 GiB arena during serving; mixed-K must preserve the same invariant.
    max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
    topk = int(topk_ids.shape[1])
    key = (
        (
            _runtime_owner_token(self.quant_config, layer)
            if "_runtime_owner_token" in globals()
            else _runtime_scope_id(self.quant_config)
        ),
        x.device.index,
        x.dtype,
        mix["hidden_size"],
        mix["intermediate_size"],
        mix["num_experts"],
        mix["signature"],
        topk,
        max_batched_tokens,
        min_trellis_m,
        max_trellis_m,
        block_m,
        prefill_block_m,
        layer.exl3_trellis_tile_config,
    )
    runtime = _MIXK_RUNTIMES.get(key)
    if runtime is not None:
        return runtime
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "mixed-K EXL3 runtime must be planned during the eager profile "
            "pass before CUDA graph capture"
        )
    consolidated_api = "_load_sparkinfer_fused_moe" in globals()
    api = (
        _load_sparkinfer_fused_moe()
        if consolidated_api
        else _load_sparkinfer_trellis()
    )

    def make_plan(weights, k, tier_experts, plan_max_tokens, plan_block_m):
        if consolidated_api:
            caps = api.Caps(
                max_tokens=plan_max_tokens,
                num_topk=topk,
                route_num_experts=mix["num_experts"],
                device=x.device,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=plan_block_m,
            )
        else:
            caps = api.Caps(
                max_tokens=plan_max_tokens,
                num_topk=topk,
                num_experts=tier_experts,
                hidden_size=mix["hidden_size"],
                intermediate_size=mix["intermediate_size"],
                route_num_experts=mix["num_experts"],
                block_size_m=plan_block_m,
                trellis_bits=k,
                tile_config=layer.exl3_trellis_tile_config,
                input_dtype=x.dtype,
                device=x.device,
            )
        return api.plan(caps)

    tiers_rt = []
    for tier, (k, tier_experts) in zip(mix["tiers"], mix["signature"]):
        decode_plan = make_plan(
            tier["weights"], k, tier_experts, max_trellis_m, block_m
        )
        prefill_plan = (
            make_plan(
                tier["weights"], k, tier_experts,
                max_batched_tokens, prefill_block_m,
            )
            if max_batched_tokens > max_trellis_m else None
        )
        tiers_rt.append({"k": k, "num_experts": tier_experts,
                         "decode_plan": decode_plan, "prefill_plan": prefill_plan})

    def arena_for(plans):
        specs = [p.scratch_specs()[0] for p in plans if p is not None]
        if not specs:
            return None
        nbytes = 0
        for spec in specs:
            b = 1
            for dim in spec.shape:
                b *= int(dim)
            b *= torch.empty((), dtype=spec.dtype).element_size()
            nbytes = max(nbytes, b)
        return torch.empty(nbytes, dtype=torch.uint8, device=x.device)

    decode_arena = arena_for([t["decode_plan"] for t in tiers_rt])
    prefill_arena = arena_for([t["prefill_plan"] for t in tiers_rt])
    accum = torch.zeros(
        (max(max_batched_tokens, max_trellis_m), mix["hidden_size"]),
        dtype=torch.float32, device=x.device,
    )
    runtime = {
        "api": api,
        "tiers": tiers_rt,
        "decode_arena": decode_arena,
        "prefill_arena": prefill_arena,
        "accum": accum,
        "min_trellis_m": min_trellis_m,
        "max_trellis_m": max_trellis_m,
        "max_batched_tokens": max_batched_tokens,
    }
    _MIXK_RUNTIMES[key] = runtime
    arena_mib = sum(
        a.numel() / (1 << 20) for a in (decode_arena, prefill_arena) if a is not None
    )
    logger.info(
        "EXL3 mixk runtime: tiers %s, decode window [%d, %d], prefill_m=%d, "
        "shared arenas %.1f MiB",
        mix["signature"], min_trellis_m, max_trellis_m, max_batched_tokens,
        arena_mib,
    )
    return runtime


def _mixk_apply(self, layer, x, topk_weights, topk_ids):
    runtime = self._mixk_runtime(layer, x, topk_ids)
    m = int(x.shape[0])
    if m > runtime["max_batched_tokens"]:
        raise ValueError(
            f"mixed-K EXL3 batch exceeds planned capacity: m={m}, "
            f"capacity={runtime['max_batched_tokens']}"
        )
    decode = runtime["min_trellis_m"] <= m <= runtime["max_trellis_m"]
    arena = runtime["decode_arena"] if decode else runtime["prefill_arena"]
    accum = runtime["accum"][:m]
    mix_tiers = layer.exl3_mixk["tiers"]
    first = True
    for rt_tier, mix_tier in zip(runtime["tiers"], mix_tiers):
        if (rt_tier["k"], rt_tier["num_experts"]) != (
            mix_tier["k"], mix_tier["num_experts"]
        ):
            raise RuntimeError("mixed-K tier signature drifted between "
                               "runtime and layer weights")
        plan = rt_tier["decode_plan"] if decode else rt_tier["prefill_plan"]
        if plan is None:
            raise RuntimeError("mixed-K EXL3 prefill plan missing for large batch")
        scratch = _mixk_scratch_view(arena, plan.scratch_specs()[0])
        bind_kwargs = {
            "scratch": scratch,
            "a": x,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "route_expert_map": mix_tier["route_expert_map"],
            # The W4A16 top-k sum resolves sum_expert_map from
            # output_expert_map first; bind keys its mapped launch variant on
            # output_expert_map only, so pass the same map to align the
            # preplanned launch with run's requested contract.
            "output_expert_map": mix_tier["route_expert_map"],
            "output": accum if first else None,
        }
        bind_kwargs[
            "experts" if "_load_sparkinfer_fused_moe" in globals() else "weights"
        ] = mix_tier["weights"]
        binding = runtime["api"].bind(plan, **bind_kwargs)
        out = runtime["api"].run(binding=binding)
        if not first:
            accum.add_(out)
        first = False
    return accum.to(x.dtype)


Exl3MoEMethod._prepare_rank_sliced_weights_mixk = _mixk_prepare_rank_sliced_weights
Exl3MoEMethod._mixk_runtime = _mixk_runtime
Exl3MoEMethod._apply_rank_sliced_mixk = _mixk_apply
'''


def main() -> None:
    if "--print-path" in sys.argv:
        print(find_exl3())
        return
    path = find_exl3()
    src = path.read_text()
    native_markers = tuple(marker in src for marker in NATIVE_R14_MARKERS)
    if all(native_markers):
        print(f"upstream native mixed-K present; compatibility patch skipped: {path}")
        return
    if any(native_markers):
        present = [
            marker for marker, found in zip(NATIVE_R14_MARKERS, native_markers)
            if found
        ]
        sys.exit(
            f"INCOMPLETE NATIVE MIXED-K API {present!r} in {path}; "
            "image lineage differs from reviewed r14 — do not patch blindly."
        )
    if MARKER_V5 in src:
        print(f"already patched (v5): {path}")
        return
    if MARKER_V1 in src:
        # rebuild any older mixk install: keep the in-place E-edits, replace
        # the appended block wholesale from the v1 marker onward
        src = src[: src.index(MARKER_V1)].rstrip() + "\n\n" + APPEND + "\n"
        path.write_text(src)
        import py_compile
        py_compile.compile(str(path), doraise=True)
        print(f"upgraded to v5: {path}")
        return
    missing = [name for name, old in (("E1", E1_OLD), ("E2", E2_OLD),
                                      ("E3", E3_OLD), ("E5", E5_OLD))
               if old not in src]
    if missing:
        sys.exit(f"ANCHOR MISMATCH {missing} in {path}; image lineage differs "
                 "from v20final — do not apply blindly.")
    for old, new in ((E1_OLD, E1_NEW), (E2_OLD, E2_NEW),
                     (E3_OLD, E3_NEW), (E5_OLD, E5_NEW)):
        if src.count(old) != 1:
            sys.exit(f"anchor not unique ({src.count(old)}x): {old[:60]!r}")
        src = src.replace(old, new)
    src = src + "\n\n" + APPEND + "\n"
    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(src)
    import py_compile
    py_compile.compile(str(path), doraise=True)
    print(f"patched {path} (v5, backup at {backup})")


if __name__ == "__main__":
    main()
