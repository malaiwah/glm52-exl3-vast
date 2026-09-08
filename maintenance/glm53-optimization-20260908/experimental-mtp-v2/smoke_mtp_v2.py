#!/usr/bin/env python3
"""Main-owned active V2 MTP smoke; no model/checkpoint/engine is loaded.

CPU: python smoke_mtp_v2.py --device cpu --dependency-root ../../baseline
CUDA: python smoke_mtp_v2.py --device cuda

CPU executes exact candidate preprocessing/copy methods with portable sampler
helpers, returning at the CUDA Gumbel boundary; it does NOT emulate or qualify
native Gumbel/rejection kernels. CUDA imports the installed native MTP class,
uses real SamplingStates/UVA rotation, real seeded Gumbel and rejection kernels,
and captures/replays sampling while request parameters change outside replay.
Source hashes are required in both modes. This script has not been run by its
author; only Main is authorized to execute it in the experiment runtime.
"""

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def checked_source(root, relative, sha256):
    path = root / relative
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"Source mismatch: {path}: {actual} != {sha256}")
    return data.decode()


def exec_nodes(nodes, path, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def select_functions(source, names):
    found = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in found} != set(names):
        raise RuntimeError(f"Missing source functions: {names}")
    return found


def load_runtime(args):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if args.device == "cuda":
        import vllm
        installed_root = Path(vllm.__file__).resolve().parent.parent
        for entry in manifest["files"]:
            checked_source(installed_root, entry["path"], entry["candidate_sha256"])
        for entry in manifest["baseline_dependencies"]:
            checked_source(installed_root, entry["path"], entry["sha256"])
        mtp = importlib.import_module("vllm.v1.worker.gpu.spec_decode.mtp.speculator")
        ops = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
        if not ops.HAS_TRITON or not ops.current_platform.is_cuda():
            raise RuntimeError("Native CUDA smoke requires actual CUDA/Triton dispatch")
        reject = importlib.import_module("vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils")
        return SimpleNamespace(cls=mtp.MTPSpeculator, states_cls=mtp.SamplingStates,
                               gumbel=mtp.gumbel_sample, draft_pos=mtp.draft_gumbel_pos,
                               reject=reject.rejection_sample)

    if args.dependency_root is None:
        raise RuntimeError("CPU mode requires --dependency-root BASELINE")
    sources = {}
    for entry in manifest["files"]:
        sources[entry["path"]] = checked_source(ROOT, entry["path"], entry["candidate_sha256"])
    for entry in manifest["baseline_dependencies"]:
        sources[entry["path"]] = checked_source(args.dependency_root, entry["path"], entry["sha256"])
    namespace = {"torch": torch, "nn": torch.nn, "VllmConfig": object,
                 "SamplingStates": object, "HAS_TRITON": False,
                 "current_platform": SimpleNamespace(is_cpu=lambda: True)}
    ops_path = "vllm/v1/sample/ops/topk_topp_sampler.py"
    exec_nodes(select_functions(sources[ops_path], {"apply_top_k_top_p", "apply_top_k_top_p_pytorch", "apply_top_k_only"}), ops_path, namespace)
    utils_path = "vllm/v1/worker/gpu/spec_decode/utils.py"
    utils_tree = ast.parse(sources[utils_path])
    constants = [node for node in utils_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "DRAFT_GUMBEL_POS_OFFSET" for target in node.targets)]
    if len(constants) != 1:
        raise RuntimeError("Cannot resolve native draft stream offset")
    exec_nodes(constants + select_functions(sources[utils_path], {"draft_gumbel_pos"}), utils_path, namespace)
    base_path = "vllm/v1/worker/gpu/spec_decode/speculator.py"
    base_class = next(node for node in ast.parse(sources[base_path]).body
                      if isinstance(node, ast.ClassDef) and node.name == "DraftModelSpeculator")
    base_copy = next(node for node in base_class.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_copy_request_inputs")
    exec_nodes([ast.ClassDef(name="AutoRegressiveSpeculator", bases=[], keywords=[], body=[base_copy], decorator_list=[])], base_path, namespace)
    # Deliberate CPU boundary, not a fake native sampler: expose the actual
    # candidate's pre-noise logits for independent probability assertions.
    namespace["gumbel_sample"] = lambda logits, *_args, **_kwargs: logits
    mtp_path = "vllm/v1/worker/gpu/spec_decode/mtp/speculator.py"
    mtp_class = next(node for node in ast.parse(sources[mtp_path]).body
                     if isinstance(node, ast.ClassDef) and node.name == "MTPSpeculator")
    exec_nodes([mtp_class], mtp_path, namespace)
    return SimpleNamespace(cls=namespace["MTPSpeculator"], states_cls=None,
                           gumbel=None, draft_pos=namespace["draft_gumbel_pos"], reject=None)


def profile(capacity, vocab, variant):
    temperature = np.ones(capacity, dtype=np.float32)
    temperature[[0, 4]] = 0.0
    temperature[1] = 0.7
    temperature[3] = 1.3
    k = np.full(capacity, vocab, dtype=np.int32)
    p = np.ones(capacity, dtype=np.float32)
    if variant % 3 == 0:
        k[[1, 3, 4]] = [5, 11, 1]
        p[[1, 3, 4]] = [0.65, 0.9, 0.1]
    elif variant % 3 == 1:
        k[[1, 3, 4]] = [1, 7, 2]
        p[[1, 3, 4]] = [1.0, 0.55, 0.2]
    else:
        p[[1, 3, 4]] = [0.00001, 0.75, 0.3]
        # Reuse request-state slot 1 with a new temperature and seed.
        temperature[1] = 1.1
    seeds = np.arange(capacity, dtype=np.int64) * 31 + 997 + variant
    return temperature, k, p, seeds


def new_states(runtime, capacity, vocab, device):
    if device.type == "cuda":
        return runtime.states_cls(capacity, vocab, seed=71)
    return SimpleNamespace(**{name: SimpleNamespace(gpu=None) for name in ("temperature", "top_k", "top_p", "seeds")})


def update_states(states, values, device):
    temperature, top_k, top_p, seeds = values
    for name, data in zip(("temperature", "top_k", "top_p", "seeds"), values):
        if device.type == "cuda":
            getattr(states, name).np[:] = data
        else:
            # Replace source views between copies. Only CUDA mode qualifies
            # actual UvaBufferPool rotation and captured pointer lifetimes.
            getattr(states, name).gpu = torch.from_numpy(data.copy())
    if device.type == "cuda":
        states.min_p.np.fill(0.0)
        states.apply_staged_writes()


def new_proposer(runtime, device, capacity, vocab, fp64):
    # Avoid model allocation; exercise actual copy and sampling methods with
    # their real tensor state. Full runner/model initialization is Main's
    # serving experiment, not this isolated sampling smoke.
    proposer = object.__new__(runtime.cls)
    proposer.sampling_states = new_states(runtime, capacity, vocab, device)
    proposer.top_k = torch.full((capacity,), vocab, dtype=torch.int32, device=device)
    proposer.top_p = torch.ones(capacity, dtype=torch.float32, device=device)
    proposer.temperature = torch.zeros(capacity, dtype=torch.float32, device=device)
    proposer.seeds = torch.zeros(capacity, dtype=torch.int64, device=device)
    proposer.idx_mapping = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    proposer.draft_logits = torch.full((capacity, 3, vocab), -777.0, dtype=torch.float32, device=device)
    proposer.use_fp64_gumbel = fp64
    return proposer


def stage(proposer, mapping, active, values, device):
    update_states(proposer.sampling_states, values, device)
    proposer._copy_request_inputs(active, mapping[:active],
                                  proposer.sampling_states.temperature.gpu,
                                  proposer.sampling_states.seeds.gpu)


def fixture_logits(rows, vocab, dtype, device):
    generator = torch.Generator().manual_seed(73 + rows)
    logits = torch.full((rows, vocab), -20.0, dtype=dtype)
    active = ((torch.arange(32, dtype=torch.float32) - 16) / 4).to(dtype)
    for row in logits:
        row[torch.randperm(vocab, generator=generator)[:32]] = active
    # A greedy row with tied maxima and a small top-p must keep original argmax.
    logits[0].fill_(-4)
    logits[0, :2] = 4
    return logits.to(device)


def oracle_logits(logits, mapping, values):
    temperatures, top_k, top_p, _ = values
    result = logits.cpu().float().clone()
    for i, req in enumerate(mapping.cpu().tolist()):
        if req < 0 or temperatures[req] == 0:
            continue
        # Native Gumbel promotes head logits before temperature scaling.
        row = result[i] / float(temperatures[req])
        k = min(int(top_k[req]), row.numel())
        threshold = row.topk(k).values[-1]
        row.masked_fill_(row < threshold, -float("inf"))
        # FP64 independent descending nucleus oracle; fixtures deliberately
        # avoid ties at random-row cutoffs and probabilities near boundaries.
        values_desc, indices = row.double().sort(descending=True)
        probabilities = values_desc.softmax(-1)
        keep = probabilities.cumsum(-1) - probabilities < float(top_p[req])
        keep[0] = True
        result[i] = torch.empty_like(row).scatter_(0, indices, values_desc.masked_fill(~keep, -float("inf")).float())
    return result.to(logits.device)


def call_proposal(proposer, logits, mapping, positions, step, active):
    return proposer._sample_probabilistic_draft(
        logits.clone(), positions, mapping, proposer.temperature, proposer.seeds,
        step, proposer.draft_logits, active)


def assert_probabilities(actual, expected):
    actual_q, expected_q = actual.softmax(-1), expected.softmax(-1)
    assert torch.equal(actual_q != 0, expected_q != 0)
    torch.testing.assert_close(actual_q, expected_q, atol=3e-7, rtol=3e-6)
    assert torch.isfinite(actual_q).all()


def check_draw(runtime, proposer, result, logits, mapping, positions, step, active_count, values):
    expected = oracle_logits(logits, mapping, values)
    if runtime.gumbel is None:
        assert_probabilities(result, expected)
        return
    expected_ids = runtime.gumbel(expected, mapping, proposer.temperature, proposer.seeds,
                                  runtime.draft_pos(positions), apply_temperature=False,
                                  use_fp64=proposer.use_fp64_gumbel)
    assert torch.equal(result[:active_count], expected_ids[:active_count])
    step_values = step.cpu().tolist() if step.ndim else [int(step)] * mapping.numel()
    observed = set()
    for row, request in enumerate(mapping.cpu().tolist()):
        if request < 0 or row >= active_count:
            continue
        column = step_values[row]
        assert_probabilities(proposer.draft_logits[request, column][None], expected[row][None])
        sampled_q = proposer.draft_logits[request, column].softmax(-1)[result[row]]
        assert sampled_q > 0
        observed.add((request, column))
    # Invalid/padded rows, including a stale valid index beyond active_rows,
    # must not overwrite any other request or speculative-column cache entry.
    for request in range(proposer.draft_logits.shape[0]):
        for column in range(proposer.draft_logits.shape[1]):
            if (request, column) not in observed:
                assert torch.equal(proposer.draft_logits[request, column], torch.full_like(proposer.draft_logits[request, column], -777.0))


def eager_cases(runtime, device, vocab):
    results = []
    definitions = [
        ("small", [4, 1, 3], 3, False),
        ("large_padded", [4, 1, 3, 0, 2, 5, 6, 7, 8, 9, -1, 4], 10, False),
        ("request_major_slots", [4, 4, 4, 1, 1, 1, 3, 3, 3], 9, True),
    ]
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for fp64 in (False, True):
            for name, requests, active_count, repeated in definitions:
                proposer = new_proposer(runtime, device, 16, vocab, fp64)
                values = profile(16, vocab, 0)
                mapping = torch.tensor(requests, dtype=torch.int32, device=device)
                stage(proposer, mapping, active_count, values, device)
                positions = torch.arange(len(requests), dtype=torch.int64, device=device) + 41
                step = torch.arange(len(requests), dtype=torch.int64, device=device) % 3 if repeated else torch.tensor(1, device=device)
                active = torch.tensor(active_count, dtype=torch.int32, device=device)
                logits = fixture_logits(len(requests), vocab, dtype, device)
                result = call_proposal(proposer, logits, mapping, positions, step, active)
                check_draw(runtime, proposer, result, logits, mapping, positions, step, active_count, values)
                if runtime.gumbel is not None:
                    replay = call_proposal(proposer, logits, mapping, positions, step, active)
                    assert torch.equal(result, replay)
                results.append({"case": name, "input_dtype": str(dtype), "fp64_noise": fp64, "status": "pass"})
    return results


def graph_cases(runtime, device, vocab):
    results = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        proposer = new_proposer(runtime, device, 16, vocab, True)
        mapping = torch.tensor([4, 1, 3, 0, 2, 5, 6, 7, 8, 9, -1, 4], dtype=torch.int32, device=device)
        positions = torch.arange(12, dtype=torch.int64, device=device) + 101
        step = torch.tensor(1, device=device)
        active = torch.tensor(10, dtype=torch.int32, device=device)
        logits = fixture_logits(12, vocab, dtype, device)
        stage(proposer, mapping, 10, profile(16, vocab, 0), device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call_proposal(proposer, logits, mapping, positions, step, active)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampled = call_proposal(proposer, logits, mapping, positions, step, active)
        for variant in range(1, 5):
            # apply_staged_writes rotates the actual UVA pointers. The real
            # _copy_request_inputs must refresh persistent device parameters
            # outside replay; capturing a previous UVA view fails these q checks.
            values = profile(16, vocab, variant)
            mapping.copy_(torch.tensor(
                [4, 3, 1, 0, 2, 5, 6, 7, 8, 9, -1, 4] if variant % 2
                else [4, 1, 3, 0, 2, 5, 6, 7, 8, 9, -1, 4],
                dtype=torch.int32, device=device))
            active_count = 10 if variant % 2 else 8
            active.fill_(active_count)
            stage(proposer, mapping, active_count, values, device)
            proposer.draft_logits.fill_(-777.0)
            graph.replay()
            check_draw(runtime, proposer, sampled, logits, mapping, positions, step, active_count, values)
        results.append({"case": "graph_replay_rotating_uva", "input_dtype": str(dtype), "transitions": 4, "status": "pass"})
    return results


def rejection_cases(runtime, device, vocab, variant):
    proposer = new_proposer(runtime, device, 16, vocab, True)
    values = profile(16, vocab, variant)
    mapping = torch.tensor([4, 1, 3], dtype=torch.int32, device=device)
    stage(proposer, mapping, 3, values, device)
    positions = torch.tensor([31, 41, 51], dtype=torch.int64, device=device)
    step = torch.tensor(0, device=device)
    active = torch.tensor(3, dtype=torch.int32, device=device)
    logits = fixture_logits(3, vocab, torch.bfloat16, device)
    draft_ids = call_proposal(proposer, logits, mapping, positions, step, active)
    q_logits = proposer.draft_logits[mapping.long(), 0]
    assert int((q_logits[1].softmax(-1) > 0).sum()) == 1
    assert any(
        bool(torch.isneginf(q_logits[1, start:start + 8192]).all())
        for start in range(0, vocab, 8192)
    ), "Fixture must exercise an all--inf native rejection tile"
    target = q_logits.repeat_interleave(2, dim=0)
    draft_input = torch.stack((torch.zeros_like(draft_ids), draft_ids), dim=1).flatten()
    expanded_pos = torch.stack((positions, positions + 1), dim=1).flatten()
    expanded_mapping = mapping.repeat_interleave(2)
    local_pos = torch.tensor([0, 1] * 3, dtype=torch.int32, device=device)
    sampled, count = runtime.reject(
        target, proposer.draft_logits, draft_input,
        torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device), expanded_pos,
        mapping, expanded_mapping, local_pos, proposer.temperature, proposer.seeds,
        3, use_fp64=True)
    assert torch.equal(count, torch.full_like(count, 2))
    assert torch.equal(sampled[:, 0], draft_ids)
    # Top-k=1 or a tiny nucleus leaves one proposal token. Put all target mass
    # Correct cached q requires rejection and deterministic recovery at that token.
    replacement = (draft_ids[1] + 1) % vocab
    disjoint_target = torch.full((2, vocab), -float("inf"), device=device)
    disjoint_target[:, replacement] = 0
    recovered, recovered_count = runtime.reject(
        disjoint_target, proposer.draft_logits, draft_input[2:4].contiguous(),
        torch.tensor([0, 2], dtype=torch.int32, device=device), expanded_pos[2:4].contiguous(),
        mapping[1:2].contiguous(), expanded_mapping[2:4].contiguous(), local_pos[2:4].contiguous(),
        proposer.temperature, proposer.seeds, 3, use_fp64=True)
    assert int(recovered_count[0]) == 1 and recovered[0, 0] == replacement
    constraint = "topk1" if variant == 1 else "tiny_nucleus"
    return [{"case": "native_rejection_p_equals_q", "constraint": constraint, "status": "pass"},
            {"case": "native_rejection_disjoint_support", "constraint": constraint, "status": "pass"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--vocab-size", type=int, default=8193)
    args = parser.parse_args()
    if args.vocab_size < 32:
        parser.error("--vocab-size must be at least 32")
    device = torch.device(args.device)
    if device.type == "cuda":
        if args.vocab_size < 8193:
            parser.error("CUDA smoke requires --vocab-size >=8193 for multi-block rejection")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.set_device(0)
    runtime = load_runtime(args)
    with torch.inference_mode():
        results = eager_cases(runtime, device, args.vocab_size)
        if device.type == "cuda":
            results.extend(graph_cases(runtime, device, args.vocab_size))
            for variant in (1, 2):
                results.extend(rejection_cases(runtime, device, args.vocab_size, variant))
            torch.cuda.synchronize()
    print(json.dumps({"status": "pass", "device": str(device),
                      "scope": "native_cuda_sampling_graphs_rejection" if device.type == "cuda" else "cpu_probability_stage_only",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
