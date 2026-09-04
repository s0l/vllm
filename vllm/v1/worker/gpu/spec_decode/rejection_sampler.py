# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.ag2_rejection_capture import (
    Ag2RejectionCapture,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4
_SPARSE_LOCAL_CANDIDATES = 64

logger = logging.getLogger(__name__)


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True
        self._ag2_rejection_capture = Ag2RejectionCapture.from_env()
        self._ag2_sparse_target_topk = (
            os.environ.get("AG2_VLLM_SPARSE_TARGET_TOPK", "0") == "1"
        )
        self._ag2_sparse_target_shadow = (
            os.environ.get("AG2_VLLM_SPARSE_TARGET_SHADOW", "0") == "1"
        )
        self._ag2_sparse_target_request_prefix = os.environ.get(
            "AG2_VLLM_SPARSE_TARGET_REQUEST_PREFIX", ""
        )
        self._ag2_sparse_target_min_reqs = int(
            os.environ.get("AG2_VLLM_SPARSE_TARGET_MIN_REQS", "1")
        )
        if self._ag2_sparse_target_min_reqs < 1:
            raise ValueError("AG2 sparse target minimum requests must be positive")
        self._ag2_sparse_steps = 0
        self._ag2_sparse_fallbacks = 0
        self._ag2_sparse_shadow_steps = 0
        self._ag2_sparse_threshold_fallbacks = 0
        if self._ag2_sparse_target_request_prefix:
            if not self._ag2_sparse_target_topk:
                raise ValueError(
                    "AG2 sparse target request-prefix routing requires sparse "
                    "target top-k"
                )
            logger.warning(
                "AG2 sparse target request-prefix routing is diagnostic-only: "
                "prefix=%r min_reqs=%d",
                self._ag2_sparse_target_request_prefix,
                self._ag2_sparse_target_min_reqs,
            )
        if self._ag2_sparse_target_shadow:
            if not self._ag2_sparse_target_topk:
                raise ValueError(
                    "AG2 sparse target shadow requires sparse target top-k"
                )
            logger.warning(
                "AG2 sparse target shadow enabled: full-gather oracle is active; "
                "results are diagnostic-only"
            )

    def can_use_sparse_target_topk(self, input_batch: InputBatch) -> bool:
        """Return whether the whole active batch is inside the proven domain."""
        if not self._ag2_sparse_target_topk:
            return False
        if not self._ag2_sparse_request_prefix_eligible(input_batch):
            return False
        if input_batch.num_reqs < self._ag2_sparse_target_min_reqs:
            self._record_sparse_threshold_fallback(input_batch.num_reqs)
            return False
        if self._ag2_rejection_capture is not None or self.sampler.compute_nans:
            return False

        idx = input_batch.idx_mapping_np
        states = self.sampler.sampling_states
        top_k = states.top_k.np[idx]
        temperature = states.temperature.np[idx]
        if np.any(top_k <= 0) or np.any(top_k > _SPARSE_LOCAL_CANDIDATES):
            return False
        if np.any(top_k != top_k[0]):
            return False
        if np.any(temperature <= 0.0) or np.any(states.min_p.np[idx] != 0.0):
            return False
        if states.max_num_logprobs(idx) != NO_LOGPROBS:
            return False
        if self.sampler.logprob_token_ids_state.max_num_token_ids(idx) > 0:
            return False
        if np.any(self.sampler.logit_bias_state.use_logit_bias[idx]):
            return False
        return not np.any(self.sampler.bad_words_state.num_bad_words.np[idx] > 0)

    def _ag2_sparse_request_prefix_eligible(self, input_batch: InputBatch) -> bool:
        prefix = self._ag2_sparse_target_request_prefix
        return not prefix or all(
            req_id.startswith(prefix)
            for req_id in input_batch.req_ids[: input_batch.num_reqs]
        )

    @property
    def sparse_target_shadow_enabled(self) -> bool:
        return self._ag2_sparse_target_shadow

    def _record_sparse_step(self, fallback: bool) -> None:
        self._ag2_sparse_steps += 1
        if fallback:
            self._ag2_sparse_fallbacks += 1
        if self._ag2_sparse_steps == 1 or self._ag2_sparse_steps % 100 == 0:
            logger.info(
                "AG2 sparse target top-k: steps=%d fallbacks=%d",
                self._ag2_sparse_steps,
                self._ag2_sparse_fallbacks,
            )

    def _record_sparse_threshold_fallback(self, num_reqs: int) -> None:
        self._ag2_sparse_threshold_fallbacks += 1
        if self._ag2_sparse_threshold_fallbacks == 1:
            logger.info(
                "AG2 sparse target full-tail fallback: events=%d "
                "active_reqs=%d min_reqs=%d",
                self._ag2_sparse_threshold_fallbacks,
                num_reqs,
                self._ag2_sparse_target_min_reqs,
            )

    def _ag2_capture_sampling_state(
        self,
        state_idx: torch.Tensor,
        num_logits: int,
    ) -> dict[str, object]:
        """Snapshot every persistent input consumed by target sampling."""
        del num_logits  # Kept in the observer ABI for old diagnostic callers.
        state_idx = state_idx.long()
        state_idx_cpu = state_idx.detach().cpu()
        state_idx_np = state_idx_cpu.numpy()
        states = self.sampler.sampling_states
        penalties = self.sampler.penalties_state
        req_states = self.sampler.req_states
        thinking = self.sampler.thinking_budget_state
        total_len = req_states.total_len.gpu[state_idx].detach().cpu()
        max_total_len = int(total_len.max().item()) if total_len.numel() else 0
        use_penalty = penalties.use_penalty[state_idx_np].copy()
        if thinking.enabled:
            thinking_use_budget = torch.from_numpy(
                thinking.use_thinking_budget[state_idx_np].copy()
            )
            thinking_token_budget = (
                thinking.thinking_token_budget.gpu[state_idx].detach().cpu()
            )
            thinking_cached_last_start = (
                thinking.cached_last_start[state_idx].detach().cpu()
            )
            thinking_cached_last_end = (
                thinking.cached_last_end[state_idx].detach().cpu()
            )
            thinking_cached_scan_pos = (
                thinking.cached_scan_pos[state_idx].detach().cpu()
            )
            thinking_start_token_ids = thinking.reasoning_start_token_ids.detach().cpu()
            thinking_natural_end_token_ids = (
                thinking.natural_reasoning_end_token_ids.detach().cpu()
            )
            thinking_end_token_ids = thinking.reasoning_end_token_ids.detach().cpu()
        else:
            num_reqs = state_idx.numel()
            thinking_use_budget = torch.zeros(num_reqs, dtype=torch.bool)
            thinking_token_budget = torch.full((num_reqs,), -1, dtype=torch.int32)
            thinking_cached_last_start = torch.full((num_reqs,), -1, dtype=torch.int32)
            thinking_cached_last_end = torch.full((num_reqs,), -1, dtype=torch.int32)
            thinking_cached_scan_pos = torch.zeros(num_reqs, dtype=torch.int32)
            thinking_start_token_ids = torch.empty(0, dtype=torch.int32)
            thinking_natural_end_token_ids = torch.empty(0, dtype=torch.int32)
            thinking_end_token_ids = torch.empty(0, dtype=torch.int32)
        payload: dict[str, object] = {
            "state_idx": state_idx_cpu,
            "temperature": states.temperature.gpu[state_idx].detach().cpu(),
            "top_k": states.top_k.gpu[state_idx].detach().cpu(),
            "top_p": states.top_p.gpu[state_idx].detach().cpu(),
            "min_p": states.min_p.gpu[state_idx].detach().cpu(),
            "seeds": states.seeds.gpu[state_idx].detach().cpu(),
            "seeds_set": states.seeds_set[state_idx_cpu.numpy()].copy(),
            "repetition_penalty": penalties.repetition_penalty.gpu[state_idx]
            .detach()
            .cpu(),
            "frequency_penalty": penalties.frequency_penalty.gpu[state_idx]
            .detach()
            .cpu(),
            "presence_penalty": penalties.presence_penalty.gpu[state_idx]
            .detach()
            .cpu(),
            "use_penalty": use_penalty,
            "prompt_len": req_states.prompt_len.gpu[state_idx].detach().cpu(),
            "prefill_len": req_states.prefill_len.gpu[state_idx].detach().cpu(),
            "total_len": total_len,
            "num_computed_tokens": req_states.num_computed_tokens.gpu[state_idx]
            .detach()
            .cpu(),
            "last_sampled_tokens": req_states.last_sampled_tokens[state_idx]
            .detach()
            .cpu(),
            "draft_tokens": req_states.draft_tokens[state_idx].detach().cpu(),
            "all_token_ids": req_states.all_token_ids.gpu[state_idx, :max_total_len]
            .detach()
            .cpu(),
            "thinking_budget_enabled": thinking.enabled,
            "thinking_use_budget": thinking_use_budget,
            "thinking_token_budget": thinking_token_budget,
            "thinking_cached_last_start": thinking_cached_last_start,
            "thinking_cached_last_end": thinking_cached_last_end,
            "thinking_cached_scan_pos": thinking_cached_scan_pos,
            "thinking_start_token_ids": thinking_start_token_ids,
            "thinking_natural_end_token_ids": thinking_natural_end_token_ids,
            "thinking_end_token_ids": thinking_end_token_ids,
        }
        if bool(use_penalty.any()):
            payload["prompt_bin_mask"] = penalties.prompt_bin_mask[state_idx]
            payload["prompt_bin_mask"] = payload["prompt_bin_mask"].detach().cpu()
            payload["output_bin_counts"] = penalties.output_bin_counts[state_idx]
            payload["output_bin_counts"] = payload["output_bin_counts"].detach().cpu()
        return payload

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> LogprobsTensors | None:
        max_per_req_token_ids = self.sampler.logprob_token_ids_state.max_num_token_ids(
            idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS and max_per_req_token_ids == 0:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0,
            flat_sampled,
            cu_num_generated_tokens,
            logprob_token_ids_state=self.sampler.logprob_token_ids_state,
            expanded_idx_mapping=expanded_idx_mapping,
            max_per_req_token_ids=max_per_req_token_ids,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        req_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        capture_rank = -1
        capture_stages: dict[str, torch.Tensor] | None = None
        if self._ag2_rejection_capture is not None:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            capture_rank = get_tensor_model_parallel_rank()
            if self._ag2_rejection_capture.should_capture(
                rank=capture_rank,
                positions=pos,
            ):
                capture_stages = {}
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            capture_stages=capture_stages,
        )
        if capture_stages is not None:
            capture_stages["after_thinking_budget"] = (
                processed_logits.detach().cpu().clone()
            )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        if self._ag2_rejection_capture is not None:
            self._ag2_rejection_capture.capture(
                rank=capture_rank,
                req_ids=req_ids,
                processed_target_logits=processed_logits,
                draft_logits=draft_logits,
                draft_sampled=draft_sampled,
                sampled=sampled,
                num_sampled=num_sampled,
                positions=pos,
                cu_num_logits=cu_num_logits,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                use_fp64=self.sampler.use_fp64_gumbel,
                use_block_verification=self.use_block_verification,
                sampling_stages=capture_stages,
                sampling_state=(
                    self._ag2_capture_sampling_state(
                        idx_mapping[: len(req_ids)],
                        processed_logits.shape[0],
                    )
                    if capture_stages is not None
                    else None
                ),
            )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                input_batch.req_ids[start:end],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.idx_mapping_np[start:end],
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def sample_sparse_target_topk(
        self,
        local_logits: torch.Tensor,
        vocab_start: int,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """Verify from processed TP-local candidates, with exact full fallback."""
        if not self.can_use_sparse_target_topk(input_batch):
            raise RuntimeError("sparse target top-k called outside proven domain")

        rows, local_vocab = local_logits.shape
        vocab_size = self.sampler.sampling_states.vocab_size
        full_logits_oracle = None
        if self._ag2_sparse_target_shadow:
            full_logits_oracle = tensor_model_parallel_all_gather(local_logits, dim=-1)[
                ..., :vocab_size
            ]

        if local_vocab * 3 < vocab_size:
            raise RuntimeError(
                "sparse target local vocabulary cannot cover the global vocabulary: "
                f"local={local_vocab} global={vocab_size}"
            )
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_local = self.sampler.apply_sampling_params(
            local_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
            skip_top_k_top_p=True,
            vocab_start=vocab_start,
        )

        candidate_k = _SPARSE_LOCAL_CANDIDATES + 1
        local_values, local_ids = torch.topk(processed_local, candidate_k, dim=-1)
        global_ids = local_ids + vocab_start
        local_pairs = torch.stack((local_values, global_ids.float()), dim=-1).flatten(1)
        gathered_flat = tensor_model_parallel_all_gather(local_pairs, dim=-1)
        tp_size = gathered_flat.shape[-1] // local_pairs.shape[-1]
        gathered = gathered_flat.view(rows, tp_size, candidate_k, 2)
        gathered_values = gathered[..., 0]
        top_k = int(
            self.sampler.sampling_states.top_k.np[input_batch.idx_mapping_np[0]]
        )
        left = gathered_values[..., top_k - 1 : _SPARSE_LOCAL_CANDIDATES]
        right = gathered_values[..., top_k : _SPARSE_LOCAL_CANDIDATES + 1]
        # Equal finite values may straddle the omitted local tail and require
        # the exact full path. Equal -inf values carry zero probability, so an
        # omitted -inf token cannot change top-p or rejection sampling.
        clear = ((left != right) | (torch.isneginf(left) & torch.isneginf(right))).any(
            dim=-1
        )

        # Sampling runs outside model CUDA graphs. This one synchronization is
        # the fail-closed decision; every rank sees the same gathered tensor.
        if not bool(clear.all().item()):
            self._record_sparse_step(fallback=True)
            full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
            full_logits = full_logits[..., : self.sampler.sampling_states.vocab_size]
            return self(full_logits, input_batch, draft_logits)

        padded_vocab = local_vocab * tp_size
        sparse = torch.full(
            (rows, padded_vocab),
            -float("inf"),
            dtype=processed_local.dtype,
            device=processed_local.device,
        )
        candidate_values = gathered_values.reshape(rows, -1)
        candidate_ids = gathered[..., 1].reshape(rows, -1).to(torch.int64)
        sparse.scatter_(1, candidate_ids, candidate_values)
        sparse = sparse[..., :vocab_size]
        processed_logits = self.sampler.sampling_states.apply_top_k_top_p(
            sparse,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
        )
        if full_logits_oracle is not None:
            oracle_processed = self.sampler.apply_sampling_params(
                full_logits_oracle,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            if not torch.equal(processed_logits, oracle_processed):
                sparse_finite = torch.isfinite(processed_logits)
                oracle_finite = torch.isfinite(oracle_processed)
                support_diff = sparse_finite != oracle_finite
                common = sparse_finite & oracle_finite
                value_diff = common & (processed_logits != oracle_processed)
                raise RuntimeError(
                    "AG2 sparse processed-logits mismatch: "
                    f"step={self._ag2_sparse_steps + 1} rows={rows} "
                    f"local_vocab={local_vocab} vocab_start={vocab_start} "
                    f"support_diff={int(support_diff.sum().item())} "
                    f"value_diff={int(value_diff.sum().item())} "
                    f"sparse_finite={int(sparse_finite.sum().item())} "
                    f"oracle_finite={int(oracle_finite.sum().item())}"
                )
            self._ag2_sparse_shadow_steps += 1
            if (
                self._ag2_sparse_shadow_steps == 1
                or self._ag2_sparse_shadow_steps % 100 == 0
            ):
                logger.info(
                    "AG2 sparse processed logits exact: steps=%d rows=%d "
                    "local_vocab=%d vocab_start=%d",
                    self._ag2_sparse_shadow_steps,
                    rows,
                    local_vocab,
                    vocab_start,
                )

        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        self._record_sparse_step(fallback=False)
        sparse_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sparse_output
