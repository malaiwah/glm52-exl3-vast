# Model families

The serve line used to answer every architectural question with "GLM-5.2".
Some of those answers were defaults; others were literals. `--tensor-parallel-size 4`
was a literal, which meant the image aborted at the GPU-count gate on any host
with fewer than four cards — before vLLM was ever invoked, for a model that
might have fitted on one.

A **family** owns everything true about an architecture rather than about this
deployment: which checkpoints exist, which engine flags are needed, which knobs
mean anything, and which of the measured failure rules apply.

| | |
|---|---|
| `scripts/glm_config.py` | `FAMILIES` registry, family-aware resolver, family-scoped validation |
| `entrypoint.sh` | family-guarded env block, generic serve line + `FAMILY_SERVE_ARGS` |
| `scripts/gpu_detect.py` | what the container can actually use, from nvidia-smi ∩ the visibility variables |
| `tests/test_families.py` | release defaults, Qwen/custom coherence, and rule scoping |
| `tests/test_gpu_detect.py` | injected visibility/device-list behavior |
| `tests/test_knob_wiring.py` | every knob has a consumer; the UI hardcodes nothing |

---

## 1. What is generic and what is not

Audited against the serve invocation as it stood. **GLM-5.2/MLA/EXL3-specific**,
now supplied by the family and absent for anything else:

| flag | why it is family-specific |
|---|---|
| `--decode-context-parallel-size`, `--dcp-comm-backend a2a`, `--dcp-kv-cache-interleave-size` | DCP shards the **MLA** KV cache. There is no MLA and no DCP in a dense model. |
| `--attention-backend B12X_MLA_SPARSE` | the MLA sparse-indexer kernel |
| `--moe-backend b12x` | GLM-5.2 is MoE; Qwen3.6-27B is dense |
| `--quantization exl3` | checkpoint-specific — and naming a method the checkpoint does not carry is a boot failure |
| `--tool-call-parser glm47`, `--reasoning-parser glm45` | GLM chat-template grammars |
| `--hf-overrides {"use_index_cache":true,"max_position_embeddings":<context>,"index_topk_pattern":"FFFSSS…"}` | sparse-indexer layout plus a clamp that avoids allocating the checkpoint's unused 1M-position BF16 RoPE tables when serving 512K; meaningless elsewhere |
| `--default-chat-template-kwargs {"reasoning_effort":"high"}` | a GLM chat-template kwarg |
| `custom_ops:["all"]`, `pass_config.fuse_allreduce_rms` in the compilation config | b12x fusion passes |
| the whole MTP78 apparatus (graft, overlay download, draft dir, `DRAFT_*`) | surgery on GLM-5.2's layer 78 |
| the Glm5v vision wrapper | a GLM-5.2 config.json wrapper + plugin |
| `VLLM_EXL3_*`, `VLLM_USE_B12X_*`, `B12X_MLA_*`, `VLLM_DCP_*`, PCIe all-reduce | b12x/MLA/EXL3 kernels |

**Generic**, and still on the shared serve line: `--served-model-name`, host/port,
`--trust-remote-code`, `--tensor-parallel-size` (now a knob), `--kv-cache-dtype`,
`--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`,
`--max-num-batched-tokens`, `--max-cudagraph-capture-size`,
`--enable-chunked-prefill`, `--enable-prefix-caching`,
`--enable-prompt-tokens-details`, `--enable-force-include-usage`,
`--no-async-scheduling`, the KV-offload connector, TLS, auth, and the KV pool pin.

### One thing deliberately did NOT move

Speculative decoding. It is easy to assume MTP is a GLM feature, but Qwen3.6's
checkpoint includes MTP weights and the bundled vLLM exposes the generic
`mtp` method. So `MTP_TOKENS` is a **generic** knob and each family supplies
its default; what is GLM-only is the MTP78 draft machinery —
`MTP_DRAFT`, `DRAFT_MODEL`, `DRAFT_QUANTIZATION`.

### Where each half lives, and why the split looks untidy

* **CLI args are in `glm_config.py`**, because the landing page must be able to
  show what a family implies and to refuse knobs that cannot apply to it.
* **The env block stays in `entrypoint.sh`**, guarded by
  `if [ "$FAMILY_ENV_BLOCK" = "glm52" ]`. That block is the "each one is a
  measured decision" list; it must run *after* the image's own `ENV` in order to
  override it, and the whole `TUNE_` override mechanism is defined in terms of
  that ordering. Moving it into Python would have created a second source of
  truth for the settings with the most expensive history in this repo.

---

## 2. The inheritance model gains a layer

```
built-in defaults  <  FAMILY defaults  <  VARIANT defaults  <  startup environment  <  JSON state file
```

The family layer is what "GLM-5.2 wants TP=4 and Qwen3.6 wants TP=1" means.
The variant layer is what "this GLM checkpoint needs its native draft and
an explicitly bounded 512K memory shape" means. Both sit below the environment
so an operator can still override them from the template, and below the state
file so the landing page can. `/config` shows the source of every value;
`family` and `variant` are both possible sources.

`minimize()` — the function that decides what actually gets written — compares
each knob against **the selected family's** baseline, so a value left at that
family's own default is not pinned into the file and the family stays free to
change it later. The family key itself is compared against the *no-file*
resolution instead; comparing it against a baseline built from itself is
circular, and an earlier revision did exactly that: the chosen family always
equalled its own baseline, was never written, and every apply silently reverted
to the previous family, taking its validation rules with it. There is a
regression test.

---

## 3. Inapplicable knobs are refused, not ignored

A knob scoped to a family it is not in resolves with source `n/a`:

* `/config` renders it **disabled**, with "not applicable to this model family"
  in place of the input, and an `n/a` badge next to the key.
* `minimize()` **drops** it instead of writing it, and the apply banner says
  which keys were dropped and why — switching family therefore cannot leave a
  stale `DCP` or `MTP_DRAFT` behind.
* A hand-edited state file that carries one is an **error**, naming the key —
  the same treatment the termination switches get.

GLM-only: `MTP_DRAFT`, `MTP_DRAFT_SAMPLE_METHOD`, `DRAFT_MODEL`,
`DRAFT_QUANTIZATION`, `DCP`, `VLLM_EXL3_TRELLIS_MAX_M`, `VISION`, and
`VISION_CHUNKS`.

This is the point of the exercise: nobody should be able to assemble a
configuration that hits the `m=3` capture class of bug on a model where the
trellis does not exist in the first place.

### The measured rules are scoped too

`concurrency-window`, `tr3-draft-needs-exl3`, `draft-quant-inherit`,
`kv-nvfp4-uncalibrated`, `vision-long-context`, `vision-kv-pressure`,
`dcp-divides-tp` and `dcp-reduces-pool` now fire **only** for `glm52`. They
describe EXL3 kernels, an MLA KV layout and a GLM vision wrapper; letting them
apply everywhere would reject configurations that are fine and, worse, teach
users to distrust the validator.

New, family-generic rules: `tp-exceeds-gpus` (error — the engine cannot start),
`variant-family-mismatch` (error), `kv-dtype-family` (error — `nvfp4_ds_mla` is
an MLA KV layout), `knob-inapplicable` (error), `family-untested` (warn),
`tp-off-measured` (warn).

### The long-context gate stays family-independent

`verify_serving.py` has no family branch and does not import the config
registry. The needle probe is the gate for **any** model: short prompts remain
insufficient regardless of architecture, and the probe simply sizes itself
against whatever `MAX_MODEL_LEN` the family resolved to (512K for GLM and the
measured 192K vision-enabled envelope for Qwen NVFP4). A family cannot opt out
of being verified.

---

## 4. The MadeBy561 GLM-5.2 hybrid variant

`madeby561-hybrid` is a checkpoint variant inside the measured `glm52` family,
not a custom model. It therefore keeps GLM's sparse MLA, DCP, calibrated NVFP4
KV, parsers and tool surface while replacing the target quantizer and draft:

| knob | variant default |
|---|---|
| repository | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`, release bundle `66f3623…` (byte-identical weights from `68babde…`) |
| quantization | `nvfp4_nf3_hybrid` with the checkpoint's MXFP8 dense/shared-expert overlay |
| MTP | 3 tokens, native serialized NVFP4 experts, probabilistic proposals |
| max context / pool | 524,288 / 2,048 blocks (exactly one maximum-length logical pool) |
| batch / DCP workspace | 2,048 / 512 MiB |
| utilization | 0.98, safe only because the pool is explicitly pinned |
| transport | FP8 ring; query split at 8,192 context tokens; DMA at 393,216 bytes; CKV prefetch off |

These values are atomic defaults but not locks. An explicit template or saved
value still wins. That distinction matters: v20 auto-sized 551,680 logical KV
tokens and passed startup admission, then OOMed in the first 32K MTP proposal.
A 3,072-token batch passed repeated 32K and C1–C8 but OOMed immediately at
520K. The variant default keeps the request limit while reducing transient
activation pressure.

On the all-NODE 4x96 GB AIBeast host at 280 W/card, this v20 turnkey profile
measures 2,701 tok/s at 8K, 2,176 at 32K, 1,987 at 66K and 121.6 tok/s C1;
aggregate decode reaches 269.7 tok/s at C8. It retrieved all five needles from
a 521,277-token haystack. The v19 control measured 2,299 at 8K, 2,192 at 64K
and 119.2 C1. The mixed-topology Vast host remains a correctness and relative-
tuning platform, not an absolute-performance proxy.

The published MadeBy561 vision wrapper is not part of this qualified profile.
With MTP disabled it passed short and 32K text retrieval but recovered only
1/18 fields from the appliance's 5120x2880 dashboard. A transient correction
matching the successful Jarrel wrapper's config delegation and load-only name
mapping reached only 2/18. PP/TG remained within about one percent, as
expected. This result rules out both MTP and EXL3 expert ordering as complete
explanations; it does not rule out the simpler vision tasks reported upstream.

---

## 5. The Qwen3.6-27B NVFP4 preset

Facts below are from the [model card](https://huggingface.co/Qwen/Qwen3.6-27B);
quantization details come from the
[NVIDIA NVFP4 checkpoint card](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4).
The full 27B profile completed download, serving, performance, long-context,
vision, structured-output, tool and speculative-decoding qualification on a
single RTX 5090 32 GB. The small Qwen3.5 representative remains the economical
provider-plumbing smoke model; it is no longer the evidence for this profile's
model behavior.

| | from the model card |
|---|---|
| architecture | **dense** 27B, hidden 5120, 64 layers, hybrid: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| attention | Gated DeltaNet (48 V heads / 16 QK heads, dim 128) + Gated Attention (24 Q / 4 KV heads, dim 256, RoPE). **Not MLA, not MoE.** |
| context | 262,144 native, ~1,010,000 with YaRN rope scaling |
| size | ~55.6 GB at BF16; the selected NVFP4 checkpoint downloads about 21 GiB |
| reasoning | thinking on by default; `chat_template_kwargs: {"enable_thinking": false}`, `{"preserve_thinking": true}` |
| card's vLLM line | `vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3` |
| appliance support | bundled vLLM recognizes `Qwen3_5ForConditionalGeneration`, ModelOpt/NVFP4, Qwen parsers, and the `mtp` speculative method |

Preset choices and why:

| knob | value | reasoning |
|---|---|---|
| repo | `nvidia/Qwen3.6-27B-NVFP4`, revision `0893e160…` | the vendor-published low-weight checkpoint, pinned to the current Hub revision rechecked on 2026-07-29 |
| `--quantization` | `modelopt` | matches the checkpoint metadata and the quantizer bundled in the pinned image |
| `TENSOR_PARALLEL_SIZE` | **1** | the point of this profile is an inexpensive single-Blackwell development rental |
| `MAX_MODEL_LEN` | **196608** | highest repeatably safe detailed-vision envelope; 200K/208K left only 177/29 MiB after the image and 256K did not leave a safe vision workspace |
| `MTP_TOKENS` | **0** | compiled MTP-off is faster; eager MTP2 measured 46/81/101 tok/s at C1/C2/C4 versus 68/121/222 without speculation |
| `KV_CACHE_DTYPE` | `auto` | resolves the checkpoint-declared FP8 E4M3 KV format; `nvfp4_ds_mla` is refused because Qwen is not MLA |
| `GPU_BLOCKS_OVERRIDE` | 0 | the 2048-block pin is a GLM-specific 512K trick |
| `GPU_MEMORY_UTILIZATION` | 0.90 | retained about 511 MiB after a 5K vision repetition following the near-maximum prefill; 0.91/0.92 were too tight |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | delivered 3.7K tok/s at 8K/64K; 8192 cut the profiled KV ceiling to about 168K and could not boot this context |
| parsers | `qwen3` reasoning, `qwen3_coder` tools | both passed full-checkpoint thinking, preserved multi-turn reasoning, tools, and structured output |
| multimodal | on, `max_pixels=8388608` | two 5120x2880 dashboard gates passed with 17–18/18 detail recall and image follow-up memory |

At the selected shape, uncached prefill measured 3,704/3,716/2,513 tok/s at
8K/64K/180K and aggregate decode measured 68.0/120.9/221.6 tok/s at C1/C2/C4.
The exact 192,290-token five-depth gate passed with no degeneration. See
`TEST_RESULTS.md` for the complete live evidence and rejected candidates.

---

## 6. Tensor parallelism follows the hardware

`--tensor-parallel-size 4` was a literal; then it was a knob defaulting to 4.
Both are wrong for an image people rent on arbitrary hardware — a 1-, 2- or
8-GPU host all got 4 and failed opaquely. **It now defaults to the number of
GPUs the container can actually use**, and the resolution shows up as source
`detected` in the boot log and on `/config` beside `default`/`family`/`env`/
`file`.

Counting is `scripts/gpu_detect.py`, and it is not `nvidia-smi -L | wc -l`:

* `NVIDIA_VISIBLE_DEVICES` is what the container runtime was told to expose
  (`all`, `none`, or a list of indices/UUIDs).
* `CUDA_VISIBLE_DEVICES` narrows it further inside the container, and CUDA's
  enumeration rule is unusual: **the list is honoured up to the first invalid or
  repeated entry and everything after it is dropped**. `0,5,1` on a 4-GPU box
  yields one device, not two. Getting this wrong sets TP higher than the number
  of devices torch will enumerate, and the engine dies during init.
* The two compose; the intersection is what torch sees.
* A provider's advertised count (`RUNPOD_GPU_COUNT`, `GPU_COUNT`) is
  **corroboration, never the source**. When it disagrees with what is visible,
  the boot log says so — a mismatch means the pod is not what it claims.
* If detection yields zero, the entrypoint **refuses to start** and names the
  knob, rather than silently falling back to 4. A wrong TP is a boot failure
  either way; the only thing that changes is whether the user can tell why.

`DCP` follows the detected count too, for the GLM family: one KV shard per rank
is what the 4-GPU configuration always did, and DCP must divide TP.

### What moves with the rank count, and what does not

Worked through rather than assumed, because the measured GLM numbers all come
from one 4× RTX PRO 6000 box:

| setting | moves with TP? |
|---|---|
| `DCP` | **yes** — it is a KV shard count, one per rank; defaults to TP |
| `GPU_BLOCKS_OVERRIDE` | **yes** when explicitly pinned. The release default is 0, so vLLM profiles the pool for the actual per-rank memory envelope. |
| the memory profile / KV headroom | **yes** — per-rank weights change, so the pool that fits changes |
| `MAX_NUM_SEQS × (1 + MTP_TOKENS) ≤ 32` | **NO.** This is a per-kernel batch width — the CUDA-graph capture window and the EXL3 trellis window — and has nothing to do with how many ranks exist. The rule applies unchanged at any TP, and the warning says so explicitly so nobody "fixes" it by scaling with GPUs |
| `dcp-kv-cache-interleave-size 64` | independent of rank count (it is an interleave granularity); left alone |

GLM-5.2 below TP=4 is now a hard **error**, not a warning: 753B is ~308 GiB of
weights, which is 154 GiB per rank at TP=2 — it cannot fit on any card that
exists. The message says so and points at `MODEL_FAMILY=qwen36`. TP above 4 is
allowed with a warning that the published numbers (835,584-token pool,
121.3 tok/s, needle 5/5 at 490K) were all measured at TP=4 and should not be
assumed to transfer.

### Running on one GPU

```
-e MODEL_FAMILY=qwen36
```

is enough. TP comes from detection, so nothing else needs setting.

---

## 7. Checking a configuration without renting anything

```
docker run --rm -e CONFIG_SMOKE=1 -e MODEL_FAMILY=qwen36 <image>
```

`CONFIG_SMOKE=1` resolves the full configuration, prints every knob with its
source, prints the repo and directory that WOULD be downloaded, and prints the
**exact argv** the engine would be given — then exits. It downloads nothing,
touches no GPU, starts no landing page, mints no API key, and issues no ACME or
DNS calls.

This exists because three separate "the knob was ignored" reports were
investigated on rented hardware, at real cost, when a local run of this would
have answered all three in seconds. It is the first thing to run after changing
anything in the config layer, and the first thing to ask for when someone
reports that a setting had no effect.

## 8. Remaining qualification

The generic one-GPU provider path remains cheaply smoke-tested with a small
Qwen3.5 checkpoint. The exact Qwen3.6-27B NVFP4 checkpoint is now qualified on
one RTX 5090 at the 192K vision-enabled profile. A Runpod repetition of the
full 27B performance rows remains useful for provider comparison, but is not a
model-profile blocker. GLM-5.2 remains the four-GPU flagship; custom
checkpoints remain compatibility-by-vLLM rather than a blanket support claim.
