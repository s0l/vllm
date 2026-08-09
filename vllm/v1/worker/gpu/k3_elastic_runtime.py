# SPDX-License-Identifier: Apache-2.0
"""Default-off V2 runtime adapter for registered K3 elastic graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    create_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.k3_elastic_graph import (
    K3ElasticGraphPlan,
    K3ElasticGraphRegistry,
    K3ElasticWorkloadKey,
    context_bucket,
    kv_pressure_bucket,
    make_decode_slices,
    prefill_bucket,
    slice_input_batch,
)
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import UBatchSlice

logger = init_logger(__name__)


@dataclass(frozen=True)
class K3ElasticPreparedStep:
    plan: K3ElasticGraphPlan
    slices: tuple[UBatchSlice, ...]
    attn_metadata: list[dict[str, Any]]
    slot_mappings_by_layer: list[dict[str, torch.Tensor]]
    model_state: Any


class K3ElasticRuntime:
    def __init__(
        self,
        config_json: str,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.registry = K3ElasticGraphRegistry.from_json(config_json)
        identities = {plan.runtime_identity for plan in self.registry.plans}
        if len(identities) != 1:
            raise ValueError("K3 elastic registry must bind one runtime identity")
        self.runtime_identity = next(iter(identities))
        self.vllm_config = vllm_config
        self.device = device
        self._wrappers: dict[str, UBatchWrapper] = {}
        self._dcp_primed_plans: set[str] = set()
        self._warmed_plans: set[str] = set()
        self._capture_logged: set[str] = set()
        self._replay_logged: set[str] = set()
        logger.info(
            "K3 elastic runtime registered identity=%s plans=%s",
            self.runtime_identity,
            [plan.name for plan in self.registry.plans],
        )

    def select(
        self,
        scheduler_output: SchedulerOutput,
        input_batch: InputBatch,
        batch_desc: BatchExecutionDescriptor,
        decode_query_len: int,
    ) -> tuple[K3ElasticGraphPlan, tuple[UBatchSlice, ...]] | None:
        if self.vllm_config.parallel_config.data_parallel_size != 1:
            return None
        if not scheduler_output.is_pure_decode_step:
            return None
        if input_batch.num_reqs == 0 or np.any(input_batch.is_prefilling_np):
            return None
        rows = input_batch.num_scheduled_tokens
        if rows.size == 0 or np.any(rows != rows[0]):
            return None
        max_context = int(
            input_batch.seq_lens_cpu_upper_bound[: input_batch.num_reqs]
            .max()
            .item()
        )
        key = K3ElasticWorkloadKey(
            runtime_identity=self.runtime_identity,
            x=input_batch.num_reqs,
            rows_per_request=int(rows[0]),
            mode="decode",
            context_bucket=context_bucket(max_context),
            kv_pressure_bucket=kv_pressure_bucket(scheduler_output.kv_cache_usage),
            prefill_bucket=prefill_bucket(0),
            graph_family="captured-k3-target",
        )
        for plan in self.registry.candidates(key, state_bytes=0):
            slices = make_decode_slices(
                plan,
                input_batch,
                batch_desc,
                decode_query_len,
            )
            if slices is not None:
                logger.info_once(
                    "K3 elastic plan selected name=%s x=%d partition=%s "
                    "rows_per_request=%d context=%s kv=%s dispatch=%s",
                    plan.name,
                    key.x,
                    plan.partition_x,
                    key.rows_per_request,
                    key.context_bucket,
                    key.kv_pressure_bucket,
                    batch_desc.cg_mode.name,
                    scope="local",
                )
                return plan, slices
        return None

    def prepare(
        self,
        plan: K3ElasticGraphPlan,
        slices: tuple[UBatchSlice, ...],
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        model_state: Any,
        attn_groups: list[list[Any]],
        kv_cache_config: KVCacheConfig,
    ) -> K3ElasticPreparedStep:
        attn_metadata: list[dict[str, Any]] = []
        slot_mappings_by_layer: list[dict[str, torch.Tensor]] = []
        for wave, ubatch_slice in enumerate(slices):
            wave_batch = slice_input_batch(input_batch, ubatch_slice)
            wave_block_tables = tuple(
                table[ubatch_slice.request_slice] for table in block_tables
            )
            wave_slot_mappings = slot_mappings[:, ubatch_slice.token_slice]
            attn_metadata.append(
                model_state.prepare_attn(
                    wave_batch,
                    CUDAGraphMode.FULL,
                    wave_block_tables,
                    wave_slot_mappings,
                    attn_groups,
                    kv_cache_config,
                    metadata_builder_idx=wave,
                )
            )
            slot_mappings_by_layer.append(
                build_slot_mappings_by_layer(
                    wave_slot_mappings,
                    kv_cache_config,
                )
            )
        begin_joined = getattr(model_state, "begin_joined_mtp_replay_step", None)
        if begin_joined is None:
            raise RuntimeError("K3 elastic graph requires joined GDN replay")
        begin_joined(input_batch.num_reqs)
        return K3ElasticPreparedStep(
            plan=plan,
            slices=slices,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            model_state=model_state,
        )

    def _make_forward_context(
        self,
        prepared: K3ElasticPreparedStep,
        input_batch: InputBatch,
        mode: CUDAGraphMode,
    ) -> Any:
        return create_forward_context(
            prepared.attn_metadata,
            self.vllm_config,
            dp_metadata=None,
            cudagraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(num_tokens=input_batch.num_tokens),
            ubatch_slices=prepared.slices,
            slot_mapping=prepared.slot_mappings_by_layer,
            additional_kwargs={"k3_elastic_graph_plan": prepared.plan.name},
            is_padding=input_batch.is_padding,
            num_tokens_unpadded=input_batch.num_tokens,
        )

    def _run_uncaptured_warmups(
        self,
        wrapper: UBatchWrapper,
        prepared: K3ElasticPreparedStep,
        model_inputs: dict[str, Any],
        input_batch: InputBatch,
    ) -> None:
        if prepared.plan.name in self._warmed_plans:
            return
        forward_context = self._make_forward_context(
            prepared,
            input_batch,
            CUDAGraphMode.NONE,
        )
        begin_joined = prepared.model_state.begin_joined_mtp_replay_step
        for _ in range(2):
            with override_forward_context(forward_context):
                output = wrapper(**model_inputs)
            del output
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            # The target warmup writes the same KV slots idempotently but its
            # append-only GDN journal must not leak into the next warmup or the
            # one real captured target forward.
            begin_joined(input_batch.num_reqs)
        logger.info(
            "K3 elastic uncaptured warmups complete name=%s count=2 waves=%s",
            prepared.plan.name,
            prepared.plan.partition_x,
        )
        self._warmed_plans.add(prepared.plan.name)

    def _prime_dcp_capture_state(
        self,
        plan: K3ElasticGraphPlan,
        prepared: K3ElasticPreparedStep,
        model: Any,
    ) -> None:
        if plan.name in self._dcp_primed_plans:
            return
        index_sets = {
            tuple(int(index) for index in indices)
            for module in model.modules()
            if getattr(module, "dcp_full_kv_attention_heads", False)
            for indices in [getattr(module, "dcp_local_kv_head_indices", None)]
            if indices is not None
        }
        if not index_sets:
            self._dcp_primed_plans.add(plan.name)
            return
        wrappers: dict[int, Any] = {}
        for wave_metadata in prepared.attn_metadata:
            for metadata in wave_metadata.values():
                prefill = getattr(metadata, "prefill", None)
                wrapper = getattr(prefill, "wrapper", None)
                if wrapper is not None:
                    wrappers[id(wrapper)] = wrapper
        if len(wrappers) != len(prepared.slices):
            raise RuntimeError(
                "K3 elastic DCP capture requires one prefill wrapper per wave: "
                f"waves={len(prepared.slices)} wrappers={len(wrappers)}"
            )
        for wrapper in wrappers.values():
            prime = getattr(wrapper, "prime_dcp_local_kv_head_indices", None)
            if prime is None:
                raise RuntimeError(
                    "K3 elastic DCP prefill wrapper cannot prime KV-head indices"
                )
            for indices in sorted(index_sets):
                prime(indices, self.device)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        logger.info(
            "K3 elastic DCP capture state primed name=%s waves=%d index_sets=%s",
            plan.name,
            len(wrappers),
            sorted(index_sets),
        )
        self._dcp_primed_plans.add(plan.name)

    def run(
        self,
        prepared: K3ElasticPreparedStep,
        model: Any,
        model_inputs: dict[str, Any],
        input_batch: InputBatch,
    ) -> Any:
        self._prime_dcp_capture_state(prepared.plan, prepared, model)
        wrapper = self._wrappers.get(prepared.plan.name)
        if wrapper is None:
            wrapper = UBatchWrapper(
                model,
                self.vllm_config,
                CUDAGraphMode.FULL,
                self.device,
                num_ubatches_override=len(prepared.slices),
            )
            self._wrappers[prepared.plan.name] = wrapper
        self._run_uncaptured_warmups(
            wrapper,
            prepared,
            model_inputs,
            input_batch,
        )
        graph_key = (input_batch.num_tokens, prepared.plan.name)
        was_captured = graph_key in wrapper.cudagraphs
        forward_context = self._make_forward_context(
            prepared,
            input_batch,
            CUDAGraphMode.FULL,
        )
        with override_forward_context(forward_context):
            output = wrapper(**model_inputs)
        if not was_captured and prepared.plan.name not in self._capture_logged:
            logger.info(
                "K3 elastic graph captured name=%s tokens=%d waves=%s",
                prepared.plan.name,
                input_batch.num_tokens,
                prepared.plan.partition_x,
            )
            self._capture_logged.add(prepared.plan.name)
        elif was_captured and prepared.plan.name not in self._replay_logged:
            logger.info(
                "K3 elastic graph replay active name=%s tokens=%d",
                prepared.plan.name,
                input_batch.num_tokens,
            )
            self._replay_logged.add(prepared.plan.name)
        return output
