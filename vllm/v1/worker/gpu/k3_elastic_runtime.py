# SPDX-License-Identifier: Apache-2.0
"""Default-off captured K3 dataflow runtime for registered MAXx shapes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
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
from vllm.v1.worker.gpu.k3_dataflow_lane import (
    K3DataflowLane,
    K3DataflowTicket,
    activate_k3_dataflow_route,
)
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
from vllm.v1.worker.ubatch_utils import UBatchSlice

logger = init_logger(__name__)


@dataclass(frozen=True)
class K3ElasticPreparedStep:
    plan: K3ElasticGraphPlan
    slices: tuple[UBatchSlice, ...]
    wave_batches: tuple[InputBatch, ...]
    attn_metadata: list[dict[str, Any]]
    slot_mappings_by_layer: list[dict[str, torch.Tensor]]
    model_state: Any


@dataclass
class _CapturedPlan:
    graph: torch.cuda.CUDAGraph
    output: Any
    execution: "K3DataflowExecution"


class K3DataflowExecution:
    """Construct one static multi-cohort layer DAG on the calling host thread."""

    def __init__(
        self,
        prepared: K3ElasticPreparedStep,
        wave_contexts: tuple[Any, ...],
        *,
        num_layers: int,
        device: torch.device,
    ) -> None:
        self.prepared = prepared
        self.wave_contexts = wave_contexts
        self.device = device
        self.compute_streams = tuple(
            torch.cuda.Stream(device=device) for _ in prepared.slices
        )
        tickets: list[K3DataflowTicket] = []
        for stage in range(num_layers):
            for phase in ("attention", "mlp"):
                for cohort in range(len(prepared.slices)):
                    tickets.append(
                        K3DataflowTicket(
                            ordinal=len(tickets),
                            cohort=cohort,
                            stage=stage,
                            phase=phase,
                        )
                    )
        self.tickets = tuple(tickets)
        self.lane = K3DataflowLane(self.tickets, device=device)
        self.compute_tail_events = tuple(
            torch.cuda.Event() for _ in prepared.slices
        )

    def _ticket(self, stage: int, phase: str, cohort: int) -> K3DataflowTicket:
        phase_index = 0 if phase == "attention" else 1
        ordinal = (
            (stage * 2 + phase_index) * len(self.prepared.slices) + cohort
        )
        return self.tickets[ordinal]

    def run_core(
        self,
        core: Any,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise RuntimeError("K3 dataflow currently requires PP=1")
        if intermediate_tensors is not None:
            raise RuntimeError("K3 dataflow does not accept PP intermediate tensors")
        if getattr(core, "aux_hidden_state_layers", ()):
            raise RuntimeError("K3 dataflow does not support auxiliary traces")

        origin = torch.cuda.current_stream(self.device)
        self.lane.begin(origin_stream=origin, compute_streams=self.compute_streams)
        hidden: list[torch.Tensor] = []
        residual: list[torch.Tensor | None] = []
        wave_positions: list[torch.Tensor] = []
        try:
            for cohort, (ubatch_slice, stream, context) in enumerate(
                zip(
                    self.prepared.slices,
                    self.compute_streams,
                    self.wave_contexts,
                    strict=True,
                )
            ):
                token_slice = ubatch_slice.token_slice
                cohort_positions = positions[..., token_slice]
                wave_positions.append(cohort_positions)
                with torch.cuda.stream(stream), override_forward_context(context):
                    if inputs_embeds is not None:
                        cohort_hidden = inputs_embeds[token_slice]
                    else:
                        if input_ids is None:
                            raise RuntimeError("K3 dataflow received no model input")
                        cohort_hidden = core.embed_input_ids(input_ids[token_slice])
                hidden.append(cohort_hidden)
                residual.append(None)

            layers = tuple(islice(core.layers, core.start_layer, core.end_layer))
            if len(layers) * 2 * len(hidden) != len(self.tickets):
                raise RuntimeError("K3 dataflow layer/ticket geometry changed")
            for stage, layer in enumerate(layers):
                for cohort, (stream, context) in enumerate(
                    zip(self.compute_streams, self.wave_contexts, strict=True)
                ):
                    ticket = self._ticket(stage, "attention", cohort)
                    with (
                        torch.cuda.stream(stream),
                        override_forward_context(context),
                        activate_k3_dataflow_route(
                            self.lane, ticket, stream, wait_for_completion=True
                        ),
                    ):
                        hidden[cohort], residual[cohort] = (
                            layer.ag2_dataflow_attention_stage(
                                hidden[cohort],
                                residual[cohort],
                                wave_positions[cohort],
                            )
                        )
                for cohort, (stream, context) in enumerate(
                    zip(self.compute_streams, self.wave_contexts, strict=True)
                ):
                    ticket = self._ticket(stage, "mlp", cohort)
                    cohort_residual = residual[cohort]
                    if cohort_residual is None:
                        raise RuntimeError("K3 dataflow residual was not initialized")
                    with (
                        torch.cuda.stream(stream),
                        override_forward_context(context),
                        activate_k3_dataflow_route(
                            self.lane, ticket, stream, wait_for_completion=True
                        ),
                    ):
                        hidden[cohort], residual[cohort] = (
                            layer.ag2_dataflow_mlp_stage(
                                hidden[cohort], cohort_residual
                            )
                        )

            for cohort, (stream, context, tail) in enumerate(
                zip(
                    self.compute_streams,
                    self.wave_contexts,
                    self.compute_tail_events,
                    strict=True,
                )
            ):
                cohort_residual = residual[cohort]
                if cohort_residual is None:
                    raise RuntimeError("K3 dataflow residual was not initialized")
                with torch.cuda.stream(stream), override_forward_context(context):
                    hidden[cohort], _ = core.norm(
                        hidden[cohort], cohort_residual
                    )
                    tail.record(stream)
            self.lane.finish()
        except BaseException:
            self.lane.abort()
            raise

        for tail in self.compute_tail_events:
            origin.wait_event(tail)
        origin.wait_event(self.lane.done_event(self.tickets[-1]))
        return torch.cat(hidden, dim=0)


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
        self._captures: dict[str, _CapturedPlan] = {}
        self._dcp_primed_plans: set[str] = set()
        logger.info(
            "K3 dataflow runtime registered identity=%s plans=%s",
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
                plan, input_batch, batch_desc, decode_query_len
            )
            if slices is not None:
                logger.info_once(
                    "K3 dataflow plan selected name=%s x=%d partition=%s "
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
        wave_batches: list[InputBatch] = []
        attn_metadata: list[dict[str, Any]] = []
        slot_mappings_by_layer: list[dict[str, torch.Tensor]] = []
        for wave, ubatch_slice in enumerate(slices):
            wave_batch = slice_input_batch(input_batch, ubatch_slice)
            wave_batches.append(wave_batch)
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
                build_slot_mappings_by_layer(wave_slot_mappings, kv_cache_config)
            )
        begin_joined = getattr(model_state, "begin_joined_mtp_replay_step", None)
        if begin_joined is None:
            raise RuntimeError("K3 dataflow requires joined GDN replay")
        begin_joined(input_batch.num_reqs)
        return K3ElasticPreparedStep(
            plan=plan,
            slices=slices,
            wave_batches=tuple(wave_batches),
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            model_state=model_state,
        )

    def _wave_contexts(
        self, prepared: K3ElasticPreparedStep
    ) -> tuple[Any, ...]:
        contexts = []
        request_offset = 0
        token_offset = 0
        for wave, (batch, metadata, slots) in enumerate(
            zip(
                prepared.wave_batches,
                prepared.attn_metadata,
                prepared.slot_mappings_by_layer,
                strict=True,
            )
        ):
            contexts.append(
                create_forward_context(
                    metadata,
                    self.vllm_config,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                    batch_descriptor=BatchDescriptor(num_tokens=batch.num_tokens),
                    slot_mapping=slots,
                    additional_kwargs={
                        "k3_elastic_graph_plan": prepared.plan.name,
                        "k3_elastic_wave": wave,
                        "k3_elastic_request_offset": request_offset,
                        "k3_elastic_token_offset": token_offset,
                    },
                    is_padding=batch.is_padding,
                    num_tokens_unpadded=batch.num_tokens,
                    marlin_request_layout_cpu=batch.marlin_request_layout_cpu,
                )
            )
            request_offset += batch.num_reqs
            token_offset += batch.num_tokens
        return tuple(contexts)

    def _outer_context(
        self,
        prepared: K3ElasticPreparedStep,
        execution: K3DataflowExecution,
        input_batch: InputBatch,
    ) -> Any:
        return create_forward_context(
            prepared.attn_metadata[0],
            self.vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=input_batch.num_tokens),
            slot_mapping=prepared.slot_mappings_by_layer[0],
            additional_kwargs={
                "k3_dataflow_execution": execution,
                "k3_elastic_graph_plan": prepared.plan.name,
            },
            is_padding=input_batch.is_padding,
            num_tokens_unpadded=input_batch.num_tokens,
            marlin_request_layout_cpu=input_batch.marlin_request_layout_cpu,
        )

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
        wrappers: dict[int, Any] = {}
        for wave_metadata in prepared.attn_metadata:
            for metadata in wave_metadata.values():
                wrapper = getattr(getattr(metadata, "prefill", None), "wrapper", None)
                if wrapper is not None:
                    wrappers[id(wrapper)] = wrapper
        if index_sets and len(wrappers) != len(prepared.slices):
            raise RuntimeError(
                "K3 dataflow requires one FlashInfer wrapper per wave: "
                f"waves={len(prepared.slices)} wrappers={len(wrappers)}"
            )
        for wrapper in wrappers.values():
            prime = getattr(wrapper, "prime_dcp_local_kv_head_indices", None)
            if prime is None:
                raise RuntimeError("K3 dataflow cannot prime DCP KV-head indices")
            for indices in sorted(index_sets):
                prime(indices, self.device)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        self._dcp_primed_plans.add(plan.name)

    @staticmethod
    def _find_core(model: Any) -> Any:
        matches = [
            module
            for module in model.modules()
            if hasattr(module, "ag2_dataflow_marker")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "K3 dataflow requires exactly one marked Qwen3Next core: "
                f"found={len(matches)}"
            )
        return matches[0]

    def run(
        self,
        prepared: K3ElasticPreparedStep,
        model: Any,
        model_inputs: dict[str, Any],
        input_batch: InputBatch,
    ) -> Any:
        self._prime_dcp_capture_state(prepared.plan, prepared, model)
        captured = self._captures.get(prepared.plan.name)
        if captured is not None:
            captured.graph.replay()
            logger.info_once(
                "K3 dataflow graph replay active name=%s tokens=%d",
                prepared.plan.name,
                input_batch.num_tokens,
                scope="local",
            )
            return captured.output

        core = self._find_core(model)
        num_layers = int(core.end_layer - core.start_layer)
        execution = K3DataflowExecution(
            prepared,
            self._wave_contexts(prepared),
            num_layers=num_layers,
            device=self.device,
        )
        outer_context = self._outer_context(prepared, execution, input_batch)
        begin_joined = prepared.model_state.begin_joined_mtp_replay_step
        for _ in range(2):
            with override_forward_context(outer_context):
                output = model(**model_inputs)
            del output
            torch.cuda.synchronize(self.device)
            begin_joined(input_batch.num_reqs)

        graph = torch.cuda.CUDAGraph()
        capture_context = torch.cuda.graph(graph)
        with capture_context, override_forward_context(outer_context):
            output = model(**model_inputs)
        torch.cuda.synchronize(self.device)
        self._captures[prepared.plan.name] = _CapturedPlan(
            graph=graph, output=output, execution=execution
        )
        logger.info(
            "K3 dataflow graph captured name=%s tokens=%d waves=%s "
            "tickets=%d compile_path=required",
            prepared.plan.name,
            input_batch.num_tokens,
            prepared.plan.partition_x,
            len(execution.tickets),
        )
        return output
