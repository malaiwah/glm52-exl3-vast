# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.utils import draft_gumbel_pos


class MTPSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.sampling_states: SamplingStates | None = None
        self.top_k: torch.Tensor | None = None
        self.top_p: torch.Tensor | None = None
        if self.draft_logits is not None:
            # Persistent device buffers: SamplingStates rotates its UVA
            # backing pointers and must not be read directly inside a graph.
            self.top_k = torch.full(
                (self.max_num_reqs,), self.vocab_size, dtype=torch.int32, device=device
            )
            self.top_p = torch.ones(
                self.max_num_reqs, dtype=torch.float32, device=device
            )

    def _copy_request_inputs(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        super()._copy_request_inputs(num_reqs, idx_mapping, temperature, seeds)
        if self.draft_logits is not None:
            assert self.sampling_states is not None, "MTP sampling states are not bound"
            assert self.top_k is not None and self.top_p is not None
            # This runs before prefill/decode graph replay, just like the
            # inherited temperature/seed copies, using the current UVA views.
            self.top_k.copy_(self.sampling_states.top_k.gpu)
            self.top_p.copy_(self.sampling_states.top_p.gpu)

    def _sample_probabilistic_draft(
        self,
        logits: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor,
        active_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.sampling_states is not None, "MTP sampling states are not bound"
        assert self.top_k is not None and self.top_p is not None
        # Request-state indices can be reordered/repeated; -1 marks graph
        # padding. Gather only valid addresses, without host-side decisions.
        request_indices = idx_mapping.clamp_min(0)
        row_temperature = temperature[request_indices]
        unconstrained = (idx_mapping < 0) | (row_temperature == 0.0)
        row_temperature.masked_fill_(unconstrained, 1.0)
        # Native Gumbel sampling scales head logits in FP32, including when
        # the head returns BF16/FP16. Preserve that numerical contract.
        logits = logits.float()
        logits.div_(row_temperature.unsqueeze(-1))
        top_k = self.top_k[request_indices]
        top_k.clamp_max_(logits.shape[-1])
        top_p = self.top_p[request_indices]
        # Greedy rows retain their original argmax, including tied maxima.
        top_k.masked_fill_(unconstrained, logits.shape[-1])
        top_p.masked_fill_(unconstrained, 1.0)
        logits = apply_top_k_top_p(logits, top_k, top_p)
        # Store precisely the filtered logits used by the seeded draft draw.
        # Rejection sampling reconstructs q from this cache; do not apply
        # temperature a second time or alter the draft's positional stream.
        return gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            draft_gumbel_pos(positions),
            apply_temperature=False,
            output_processed_logits=draft_logits,
            output_processed_logits_col=draft_step,
            output_processed_logits_active_rows=active_rows,
            use_fp64=self.use_fp64_gumbel,
        )

    @property
    def model_returns_tuple(self) -> bool:
        # DeepSeek MTP recycles the post-final-norm hidden state between
        # draft steps, so forward() returns (logit_hidden, recycle_hidden).
        return "DeepSeekMTPModel" in (
            self.draft_model_config.hf_config.architectures or []
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
