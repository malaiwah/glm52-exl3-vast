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
| `tests/test_families.py` | 118 assertions: GLM unchanged, Qwen coherent, rules scoped |
| `tests/test_gpu_detect.py` | 20 assertions against injected device lists |
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
| `--hf-overrides {"use_index_cache":true,"index_topk_pattern":"FFFSSS…"}` | the sparse indexer's per-layer pattern; meaningless elsewhere |
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
own model card gives
`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
So `MTP_TOKENS` is a **generic** knob and the family supplies the *method*
(`mtp` vs `qwen3_next_mtp`); what is GLM-only is the MTP78 draft machinery —
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
built-in defaults  <  FAMILY defaults  <  startup environment  <  JSON state file
```

The family layer is what "GLM-5.2 wants TP=4 and Qwen3.6 wants TP=1" means. It
sits below the environment so an operator can still override it from the
template, and below the state file so the landing page can. `/config` shows the
source of every value, and `family` is now one of them.

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

GLM-only: `MTP_DRAFT`, `DRAFT_MODEL`, `DRAFT_QUANTIZATION`, `DCP`,
`VLLM_EXL3_TRELLIS_MAX_M`, `VLLM_EXL3_TRELLIS_MIN_M`, `VISION`,
`VISION_CHUNKS`, `BASE_GENERATION`.

This is the point of the exercise: nobody should be able to assemble a
configuration that hits the `m=3` capture class of bug on a model where the
trellis does not exist in the first place.

### The measured rules are scoped too

`concurrency-window`, `trellis-min-m`, `capture-below-trellis-min`,
`tr3-draft-on-v20`, `tr3-draft-needs-exl3`, `draft-quant-inherit`,
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
against whatever `MAX_MODEL_LEN` the family resolved to (512K for GLM, 256K for
Qwen). A family cannot opt out of being verified.

---

## 4. The Qwen3.6-27B preset — UNVALIDATED

**Nobody has booted this image with it.** It exists so the template can start on
one GPU and be iterated on. Treat a clean boot as the beginning of validation,
not the end.

Facts below are from the [model card](https://huggingface.co/Qwen/Qwen3.6-27B);
the serve line is derived from the card's own vLLM example.

| | from the model card |
|---|---|
| architecture | **dense** 27B, hidden 5120, 64 layers, hybrid: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| attention | Gated DeltaNet (48 V heads / 16 QK heads, dim 128) + Gated Attention (24 Q / 4 KV heads, dim 256, RoPE). **Not MLA, not MoE.** |
| context | 262,144 native, ~1,010,000 with YaRN rope scaling |
| size | ~55.6 GB at BF16 — fits one 96 GB card |
| reasoning | thinking on by default; `chat_template_kwargs: {"enable_thinking": false}`, `{"preserve_thinking": true}` |
| card's vLLM line | `vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3` |
| card's speculation | `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` |

Preset choices and why:

| knob | value | reasoning |
|---|---|---|
| repo | `Qwen/Qwen3.6-27B` | the upstream BF16 checkpoint; **override it with `MODEL_VARIANT`/`MODEL_DIR` if you want a quant** |
| `--quantization` | *omitted* | the checkpoint is BF16. Naming a method it does not carry is a boot failure, and guessing a quant repo that may not exist is worse than not guessing. |
| `TENSOR_PARALLEL_SIZE` | **1** | ~56 GB fits one 96 GB card. The card's own example says 8, but that is for their 1M-context reference; TP=1 is what makes this preset useful on a single-GPU rental. |
| `MAX_MODEL_LEN` | 262144 | the native length. Going beyond needs YaRN rope scaling, which this preset does not configure. |
| `MTP_TOKENS` | 2 | the card's number, with `method: qwen3_next_mtp` |
| `KV_CACHE_DTYPE` | `auto` | let vLLM choose; `fp8` is offered, `nvfp4_ds_mla` is refused as an MLA layout |
| `GPU_BLOCKS_OVERRIDE` | 0 | the 2048-block pin is a GLM-specific 512K trick |
| `GPU_MEMORY_UTILIZATION` | 0.90 | conservative; 0.93 was tuned for the GLM memory profile |
| tool-call parser | *none* | the card does not name one. `--enable-auto-tool-choice` without a valid parser fails at startup, so guessing (`hermes`?) would trade a missing feature for an unbootable engine. |

**Unknowns that only a boot can settle:** whether this vLLM fork implements
Gated DeltaNet at all, whether `qwen3_next_mtp` exists in it, whether the
`FULL_AND_PIECEWISE` cudagraph mode is right for a hybrid-attention model, and
what the actual memory profile looks like. If the engine refuses the
speculative config, set `MTP_TOKENS=0`; if it refuses the compilation config,
that is the next thing to strip.

---

## 5. Tensor parallelism follows the hardware

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
| `GPU_BLOCKS_OVERRIDE` (the 512K pool pin) | **yes, and it is not automatic**. 2048 blocks was chosen to mean exactly 512K tokens at TP=4/DCP=4. At another rank count that arithmetic no longer holds, so a warning fires and points at `GPU_BLOCKS_OVERRIDE=0` |
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

## 6. Checking a configuration without renting anything

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

## 7. Untested

- **The Qwen preset has never been booted**, by anyone, with this image. Every
  value above is either quoted from the model card or reasoned from it. A live
  RunPod pod did get as far as resolving it correctly and starting the download
  (verified: repo, directory, TP and argv are all Qwen), but no engine has run.
- **GPU detection has never seen a real multi-GPU narrowing.** The logic is
  tested against injected device lists — the development host's 4 GPUs are busy
  serving production traffic and were not touched. `CUDA_VISIBLE_DEVICES`
  semantics in particular are implemented from documentation, not observed.
- The family-guarded env block: the GLM branch is byte-identical to what shipped
  and is exercised daily, but the `generic` branch (what a non-GLM family gets)
  has never run an engine.
- `FAMILY_SERVE_ARGS` as a bash array: the quoting is exercised by
  `tests/test_families.py` against the exact argv, and `bash -n` + shellcheck
  pass, but no vLLM has consumed it.
- The 1-GPU path end to end. The gate arithmetic is tested; the engine is not.
- `MODEL_FAMILY` switching on a live instance (it triggers a fresh download and
  a full restart, both of which are implemented but unexercised for Qwen).
