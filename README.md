# GLM-5.2 EXL3 turnkey for vast.ai — 4x RTX PRO 6000 (96GB)

One-click 512K-context GLM-5.2 OpenAI-compatible endpoint on rented GPUs:
EXL3 trellis weights (~77 GiB/rank — fits commodity 95.01-GiB cards),
**fp8 KV cache** (correct on stock drivers — the nvfp4 default silently
corrupts >~150K context without a host driver P2P override; see Evidence),
MTP speculative decoding, and DRAM KV offload auto-sized to 70% of host RAM.
Weights auto-download on first boot (~332 GB — pick a fast-net host).

## vast.ai template settings
- **Image**: `ghcr.io/malaiwah/glm52-exl3-vast:latest`
- **Launch mode**: docker ENTRYPOINT (vLLM logs appear on the instance console;
  SSH works per vast standards)
- **Docker options**: `-p 8000:8000 --ipc=host --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576` (memlock is REQUIRED for DRAM offload)
- **Disk**: >= 400 GB
- **GPU filter**: 4x RTX PRO 6000 Blackwell (96 GB), CUDA >= 13.0
- **Env (all optional)**: `HF_TOKEN` (faster download), `OFFLOAD_FRACTION`
  (default 0.70; 0 disables), `MTP_TOKENS` (default 3; 0 disables),
  `MAX_NUM_SEQS`, `MAX_MODEL_LEN` (default 524288), `SERVED_MODEL_NAME`

Endpoint: `http://<instance>:8000/v1` once the console shows
`Application startup complete` (first boot: download + JIT, plan ~30-60 min;
later boots only pay JIT).

## Evidence / why these defaults
Root-cause investigation of the long-context corruption and the validated
config matrix (6 runs, 5 hosts, 4 driver families):
- Root cause (nvfp4 KV x host P2P state): gist cae272443a9817da72b6802a0b9a5d73
- Override-host proof 7/7 @505K: gist 7d5d7e685f7498a356fa2dd12b876f14
- fp8 clean to 440K on stock: same matrix gist; harness: gist 929d7d8e4ac94c43fe126c4b3f6a6ea6
- 512K fp8 KV via `--num-gpu-blocks-override 2048` validated at util 0.93.

Base image: `verdictai/glm52-exl3-sparkinfer@sha256:bfd6d667...` (pinned).
Checkpoint: `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`.
