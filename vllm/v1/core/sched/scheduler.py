# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.elastic_expert import ElasticExpertGrant, NativeExpertBudget
from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticGraphError,
    ElasticMaintenanceExecution,
    ElasticPlanKind,
    ElasticResidencyReceipt,
    ElasticRuntimeConfig,
    ElasticStepPlan,
    GraphExecutionPolicy,
    OwnerDispatch,
    PhysicalReplayKey,
    RuntimeGeneration,
    SemanticGraphStep,
    bind_runtime_generation_to_policy,
    build_execution_manifest,
    canonical_execution_request_order,
    canonical_graph_step_key,
    configured_compiled_piecewise_sizes,
    derive_short_decode_graph_inventory,
    execution_manifest_phase_from_step_key,
    resolve_step_physical_keys,
)
from vllm.v1.core.elastic_runtime import (
    compute_elastic_runtime_generation,
    elastic_auto_calibration_enabled,
)
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheBlockPoolRequirements,
)
from vllm.v1.core.kv_cache_manager import (
    KVCacheBlocks,
    KVCacheManager,
    PrefixCacheLease,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    get_num_blocks_per_request_for_kv_cache_config,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import (
    apply_force_non_speculative_override,
    build_dynamic_sd_schedule_lookup,
)
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


@dataclass(frozen=True)
class ElasticAdmissionGrant:
    step_key: tuple[int, ...]
    physical_keys: tuple[PhysicalReplayKey, ...]
    maintenance_transaction_id: str | None
    external_memory_bytes: int
    previous_external_memory_bytes: int
    minimum_free_primary_blocks: int
    requirements: KVCacheBlockPoolRequirements


@dataclass(frozen=True)
class DeferredMMWave:
    """Immutable scheduler snapshot spanning graph-only capture and MM replay."""

    step_key: tuple[int, ...]
    running_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    request_state: tuple[tuple[str, int, int, int, int], ...]
    scheduled_tokens: tuple[tuple[str, int], ...]
    scheduled_encoder_inputs: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass
class EncoderWaveOverlay:
    """Exact mutable simulation layered over an immutable live encoder cache."""

    cache_manager: EncoderCacheManager
    scheduled_identifiers: set[str] = field(default_factory=set)

    def clone(self) -> "EncoderWaveOverlay":
        return EncoderWaveOverlay(
            cache_manager=self.cache_manager.clone_for_preview(),
            scheduled_identifiers=set(self.scheduled_identifiers),
        )


def _nonnegative_env_int(name: str, default: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _elastic_catalog_cold_residency_envelope(row: Mapping[str, Any]) -> int:
    """Return the minimum safe loan for rebuilding a sealed owner set.

    A historical cold high-water can be smaller than the subsequently
    stabilized HOT residency when the calibration loan capped the KV-priced
    portion of capture.  Rebuilding the set from X0 must nevertheless fund
    every byte that remains resident after publication.  The physical floor
    is included for legacy rows whose resident sidecar did not carry it.
    """
    return max(
        int(row["cold_peak_bytes"]),
        int(row.get("resident_bytes", 0)),
        int(row.get("floor_bytes", 0)),
    )


class Scheduler(SchedulerInterface):
    _elastic_graph_execution_policy: GraphExecutionPolicy | None
    _elastic_preflight_admission_grant: ElasticAdmissionGrant | None
    _elastic_restore_retention_id: str | None
    _elastic_restore_execution_step_key: tuple[int, ...] | None
    _elastic_restore_wave_step_key: tuple[int, ...] | None
    _elastic_last_defer_reason: str | None

    @staticmethod
    def _validate_elastic_batch_lifecycle(
        async_scheduling: bool, max_concurrent_batches: int
    ) -> None:
        if async_scheduling or max_concurrent_batches > 1:
            raise RuntimeError(
                "elastic CUDA Graph/KV ownership requires one physical batch "
                "until an overlapping working-set union is measured"
            )

    @staticmethod
    def _resolve_elastic_mm_activation_loan(
        additional_config: object,
        *,
        env_value: str = "",
        elastic_on_demand_graphs: bool,
        multimodal_config: object | None,
        elastic_mapping_quantum: int,
    ) -> int:
        config_loan_bytes = (
            additional_config.get("elastic_mm_activation_loan_bytes", 0)
            if isinstance(additional_config, dict)
            else 0
        )
        if (
            isinstance(config_loan_bytes, bool)
            or not isinstance(config_loan_bytes, int)
            or config_loan_bytes < 0
        ):
            raise ValueError("elastic_mm_activation_loan_bytes must be an integer >= 0")
        try:
            env_loan_bytes = int(env_value) if env_value else 0
        except ValueError as error:
            raise ValueError(
                "AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES must be an integer"
            ) from error
        if env_loan_bytes < 0:
            raise ValueError("AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES must be >= 0")
        if config_loan_bytes and env_value and config_loan_bytes != env_loan_bytes:
            raise ValueError("elastic MM activation loan config/env disagree")
        loan_bytes = env_loan_bytes or config_loan_bytes
        skip_mm_profiling = bool(
            multimodal_config is not None
            and getattr(multimodal_config, "skip_mm_profiling", False)
        )
        if loan_bytes:
            if not elastic_on_demand_graphs:
                raise ValueError(
                    "elastic_mm_activation_loan_bytes requires elastic KV/Graph"
                )
            if multimodal_config is None or not skip_mm_profiling:
                raise ValueError(
                    "elastic_mm_activation_loan_bytes requires a multimodal "
                    "model with skip_mm_profiling"
                )
            if not elastic_mapping_quantum or loan_bytes % elastic_mapping_quantum:
                raise ValueError(
                    "elastic_mm_activation_loan_bytes must align to the elastic "
                    "mapping quantum"
                )
        elif elastic_on_demand_graphs and skip_mm_profiling:
            raise ValueError(
                "skip_mm_profiling requires a nonzero scheduler-visible "
                "elastic_mm_activation_loan_bytes"
            )
        return loan_bytes

    @staticmethod
    def _validate_elastic_graph_mm_overlap(
        *,
        worker_peak_bytes: int,
        granted_bytes: int,
        graph_granted_bytes: int,
        mm_activation_loan_bytes: int,
    ) -> None:
        """Validate the simultaneous HOT-Graph plus transient-MM endpoint."""
        if mm_activation_loan_bytes and worker_peak_bytes > granted_bytes:
            raise RuntimeError(
                "worker Graph+MM physical overlap exceeds its aggregate loan: "
                f"peak_bytes={worker_peak_bytes} granted_bytes={granted_bytes} "
                f"graph_bytes={graph_granted_bytes} "
                f"mm_bytes={mm_activation_loan_bytes}"
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.model_uses_mrope = vllm_config.model_config.uses_mrope
        self.model_uses_xdrope = vllm_config.model_config.uses_xdrope
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.spec_decode_metrics_level = (
            self.observability_config.per_request_spec_decode_metrics
        )
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        effective_max_resident_seqs = kv_cache_config.effective_max_resident_seqs
        if effective_max_resident_seqs:
            self.max_num_running_reqs = min(
                self.max_num_running_reqs,
                effective_max_resident_seqs,
            )
            logger.info(
                "Elastic scheduler resident cap enabled: configured=%d, effective=%d",
                self.scheduler_config.max_num_seqs,
                self.max_num_running_reqs,
            )
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        additional_config = self.vllm_config.additional_config
        self.elastic_on_demand_graphs = ElasticRuntimeConfig.from_vllm_config(
            self.vllm_config
        ).enabled
        self._elastic_native_budget = NativeExpertBudget.from_config(vllm_config)
        self._elastic_native_hot_rows = 0
        if self._elastic_native_budget is not None and (
            not self.elastic_on_demand_graphs
            or not kv_cache_config.elastic_mapping_quantum
        ):
            raise ValueError("native experts require the elastic scheduler budget")
        self._elastic_compiled_piecewise_sizes = configured_compiled_piecewise_sizes(
            self.vllm_config
        )
        if self._elastic_compiled_piecewise_sizes and not self.elastic_on_demand_graphs:
            raise ValueError(
                "elastic_compiled_piecewise_sizes requires elastic on-demand graphs"
            )
        if self.elastic_on_demand_graphs:
            if self.lora_config is not None:
                raise RuntimeError(
                    "elastic CUDA Graph execution does not yet support LoRA "
                    "identity-bound replay keys"
                )
            self._validate_elastic_batch_lifecycle(
                cast(bool, self.scheduler_config.async_scheduling),
                self.vllm_config.max_concurrent_batches,
            )
            policy_payload = kv_cache_config.elastic_graph_execution_policy
            if policy_payload is None:
                raise RuntimeError(
                    "elastic Graph execution policy was not resolved before "
                    "scheduler construction"
                )
            self._elastic_graph_execution_policy = GraphExecutionPolicy.from_payload(
                policy_payload
            )
        else:
            self._elastic_graph_execution_policy = None
        mm_config = self.vllm_config.model_config.multimodal_config
        self._elastic_mm_activation_loan_bytes = (
            self._resolve_elastic_mm_activation_loan(
                additional_config,
                env_value=os.environ.get(
                    "AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES", ""
                ),
                elastic_on_demand_graphs=self.elastic_on_demand_graphs,
                multimodal_config=mm_config,
                elastic_mapping_quantum=kv_cache_config.elastic_mapping_quantum,
            )
        )
        runtime_generation = compute_elastic_runtime_generation(vllm_config)
        if self._elastic_graph_execution_policy is not None:
            runtime_generation = bind_runtime_generation_to_policy(
                runtime_generation,
                self._elastic_graph_execution_policy,
            )
        self._elastic_admission_controller = ElasticAdmissionController(
            RuntimeGeneration(runtime_generation)
        )
        # SchedulerOutput is the single broadcast authority for whether every
        # rank enters a boundary vote. Never derive this decision from a
        # rank-local worker cache: divergent collective order would deadlock.
        self._elastic_accepted_decode_consensus_epoch: str | None = None
        self._elastic_deferred_mm_wave: DeferredMMWave | None = None
        # A zero-token staged replacement keeps its victims as physical
        # lifetime fences until its exact USER shape consumes the candidate.
        # If live request/cache state resolves a different next shape, abort the
        # unused candidate and restore the old set before replanning.
        # FIFO scheduler outputs may overlap under async scheduling. The
        # controller keeps every unsettled grant reserved until worker result.
        self._elastic_maintenance_started: dict[str, tuple[float, Any, int]] = {}
        self._elastic_useful_started: dict[str, float] = {}
        self._elastic_maintenance_wall_ms_total = 0.0
        self._elastic_maintenance_transactions_total = 0
        self._elastic_useful_wall_ms_total = 0.0
        self._elastic_useful_transactions_total = 0
        self._elastic_graph_key_outcomes: dict[str, int] = defaultdict(int)
        self._elastic_rate_limited_logs_total = 0
        # The last worker request is an execution shape. A live speculative
        # cohort separately owns the complete q=K+1 decode carrier needed by
        # its later verification steps, including while the current step is a
        # mixed/prefill shape.
        self._elastic_graph_carrier_step_key: tuple[int, ...] | None = None
        self._elastic_last_execution_shape: tuple[object, ...] | None = None
        # A sealed serving runtime reuses one terminal decode carrier for all
        # semantic cohorts up to MaxX. Calibration leaves this unset so every
        # declared exact shape remains independently measurable.
        self._elastic_terminal_decode_carrier_x: int | None = None
        self._elastic_serving_carrier_keys: tuple[PhysicalReplayKey, ...] = ()
        self._elastic_serving_carrier_resident_bytes = 0
        if os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION", "0") == "1":
            raise RuntimeError(
                "AG2_VLLM_ELASTIC_CALIBRATION is obsolete; use an explicit "
                "maintenance job with AG2_VLLM_ELASTIC_AUTO_CALIBRATE=1, "
                "AG2_VLLM_ELASTIC_CALIBRATION_ROLE=maintenance and an "
                "accepted AG2_VLLM_ELASTIC_CALIBRATION_SURFACE"
            )
        self._elastic_restore_mode = False
        self._elastic_require_catalog = (
            os.environ.get("AG2_VLLM_ELASTIC_REQUIRE_CATALOG", "0") == "1"
        )
        self._elastic_auto_calibrate = elastic_auto_calibration_enabled()
        self._elastic_graph_catalog: dict[tuple[int, ...], dict[str, Any]] = {}
        self._elastic_graph_catalog_coverage: dict[str, Any] = {}
        if self.elastic_on_demand_graphs:
            from vllm.v1.worker.startup_plan import (
                load_elastic_graph_catalog,
                load_elastic_graph_catalog_coverage,
            )

            catalog = load_elastic_graph_catalog(self.vllm_config, kv_cache_config)
            if catalog:
                self.activate_elastic_graph_catalog(
                    catalog,
                    load_elastic_graph_catalog_coverage(
                        self.vllm_config, kv_cache_config
                    ),
                )
        self._elastic_graph_idle_cleanup_timeout_s = 5.0
        self._elastic_last_graph_admission_rejection: tuple[object, ...] | None = None
        self._elastic_primary_blocks_per_max_request = (
            get_num_blocks_per_request_for_kv_cache_config(vllm_config, kv_cache_config)
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is None:
            manager_cls_obj = (
                EncoderDecoderCacheManager
                if self.is_encoder_decoder
                else EncoderCacheManager
            )
        self.encoder_cache_manager = manager_cls_obj.create_manager(
            cache_size=encoder_cache_size, vllm_config=vllm_config
        )
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        # Positions past the computed tokens that the drafter reads during
        # prefill. Multi-module MTP consumes the full speculative span.
        self.num_prefill_lookahead = 0
        self._elastic_short_decode_inventory: Mapping[
            int, tuple[PhysicalReplayKey, ...]
        ] = {}
        if self.elastic_on_demand_graphs:
            decode_max_x = int(
                self._elastic_graph_catalog_coverage.get(
                    "decode_max_x",
                    # An offline/discovery start has no sealed boundary yet.
                    # Its temporary inventory may span the scheduler width;
                    # serving still requires the exact sealed catalog.
                    self.max_num_running_reqs,
                )
            )
            self._rebuild_elastic_short_decode_inventory(decode_max_x)
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
                self.num_prefill_lookahead = (
                    self.num_spec_tokens
                    if speculative_config.use_multi_module_mtp()
                    else 1
                )
            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
            if self.use_eagle and not self.use_eagle_block_drop:
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self._elastic_prefix_hits: dict[
            str,
            tuple[
                Request,
                tuple[KVCacheBlocks, int, int, bool],
                PrefixCacheLease | None,
            ],
        ] = {}
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle_block_drop,
            num_prefill_lookahead=self.num_prefill_lookahead,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
            enable_mamba_fine_grained_prefix_cache=(
                self.cache_config.enable_mamba_fine_grained_prefix_cache
            ),
        )
        if self.elastic_on_demand_graphs and self._elastic_graph_catalog:
            self._publish_elastic_startup_capacity()
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )
        self.max_concurrent_partial_prefills = max(
            1,
            min(
                int(os.environ.get("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "1")),
                self.max_num_running_reqs,
            ),
        )
        self.long_prefill_cap_min_prompt_tokens = _nonnegative_env_int(
            "AG2_VLLM_LONG_PREFILL_CAP_MIN_PROMPT_TOKENS"
        )
        if self.long_prefill_cap_min_prompt_tokens > 0:
            if self.scheduler_config.long_prefill_token_threshold <= 0:
                raise ValueError(
                    "AG2_VLLM_LONG_PREFILL_CAP_MIN_PROMPT_TOKENS requires "
                    "long_prefill_token_threshold > 0"
                )
            logger.info(
                "Long-prefill cap eligibility POC enabled: min_prompt_tokens=%d, "
                "chunk_cap=%d",
                self.long_prefill_cap_min_prompt_tokens,
                self.scheduler_config.long_prefill_token_threshold,
            )
        self.prefill_admission_delay_s = max(
            0.0,
            float(os.environ.get("AG2_VLLM_PREFILL_ADMISSION_DELAY_MS", "0")) / 1000.0,
        )
        admission_max_delay_ms = os.environ.get(
            "AG2_VLLM_PREFILL_ADMISSION_MAX_DELAY_MS"
        )
        if not admission_max_delay_ms:
            admission_max_delay_ms = str(int(self.prefill_admission_delay_s * 4000))
        self.prefill_admission_max_delay_s = max(
            self.prefill_admission_delay_s,
            float(admission_max_delay_ms) / 1000.0,
        )
        if self.max_concurrent_partial_prefills > 1:
            logger.info(
                "Concurrent partial prefill POC enabled: max_prefills=%d, "
                "admission_delay_ms=%.1f, max_delay_ms=%.1f",
                self.max_concurrent_partial_prefills,
                self.prefill_admission_delay_s * 1000,
                self.prefill_admission_max_delay_s * 1000,
            )
        self.canonical_prefill_admission = (
            os.environ.get("AG2_VLLM_CANONICAL_PREFILL_ADMISSION", "0") == "1"
        )
        if self.canonical_prefill_admission:
            raise ValueError(
                "AG2_VLLM_CANONICAL_PREFILL_ADMISSION is retired: with K3, "
                "decode consumes 4 tokens from a 4096-token scheduler budget, "
                "so the 4096-token canonical waiting-prefill requirement "
                "serializes long-prefill workloads at x1. Fix shape-dependent "
                "math or routing without suppressing valid concurrency."
            )
        self.num_canonical_prefill_deferrals_since_last_stats = 0
        self.kv_tail_handoff_enabled = (
            os.environ.get("AG2_VLLM_KV_TAIL_HANDOFF", "0") == "1"
        )
        self.num_kv_tail_deferrals_since_last_stats = 0
        if self.kv_tail_handoff_enabled:
            logger.info(
                "KV tail handoff POC enabled: retain an in-progress prompt's "
                "KV on transient FCFS pressure while another running request "
                "can make progress."
            )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.
        self._skip_zero_block_ids: set[int] = set()
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # TODO: Support models with multiple Mamba specs that require different
        # prefill checkpoint alignments instead of selecting the first one.
        self.mamba_prefill_checkpoint_alignment = next(
            (
                group.kv_cache_spec.prefill_checkpoint_alignment
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            None,
        )
        self.mamba_has_prefill_checkpoint_blocks = self.has_mamba_layers and all(
            not isinstance(group.kv_cache_spec, MambaSpec)
            or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
            for group in kv_cache_config.kv_cache_groups
        )
        from math import lcm

        state_alignments = [
            group.kv_cache_spec.state_update_chunk_alignment
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        self.mamba_state_update_alignment = lcm(*state_alignments, 1)
        if self.need_mamba_block_aligned_split:
            if self.block_size % self.mamba_state_update_alignment:
                raise ValueError(
                    "The resolved scheduler block size must be divisible by "
                    "the recurrent state-update alignment: "
                    f"block_size={self.block_size}, "
                    f"alignment={self.mamba_state_update_alignment}"
                )
            if self.hash_block_size % self.mamba_state_update_alignment:
                raise ValueError(
                    "The prefix hash unit must be divisible by the recurrent "
                    "state-update alignment: "
                    f"hash_block_size={self.hash_block_size}, "
                    f"alignment={self.mamba_state_update_alignment}"
                )
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
        # Opt-in: also stop at the junction, where an eagle sibling resumes. The
        # manager decides whether it can check-point there (per-group eagle bit,
        # no MTP re-prefill tail); splitting for a stop it would refuse costs a
        # forward pass and displaces the block-boundary stop.
        self.mamba_fine_grained_prefix_cache = (
            self.mamba_partial_cache_hit
            and self.kv_cache_manager.mamba_fine_grained_prefix_cache
        )
        if os.environ.get("AG2_VLLM_DCP_FINE_PREFIX", "0") == "1":
            from vllm.v1.request import _parse_prefix_cache_hint_tokens

            configured_hint = os.environ.get("AG2_VLLM_DCP_FINE_PREFIX_HINT_TOKENS")
            _parse_prefix_cache_hint_tokens(
                {"ag2_prefix_cache_hint_tokens": configured_hint}
                if configured_hint
                else None,
                hash_block_size=self.hash_block_size,
                use_eagle=self.use_eagle,
            )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )
        self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

    def activate_elastic_graph_catalog(
        self,
        catalog: dict[tuple[int, ...], dict[str, Any]],
        coverage: dict[str, Any],
    ) -> None:
        """Activate one sealed catalog before any user request is admitted."""
        from vllm.v1.core.elastic_catalog import (
            validate_elastic_catalog_key_inventory,
        )

        lifecycle_initialized = hasattr(self, "requests")
        active_lifecycle = lifecycle_initialized and (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or bool(self.num_waiting_for_streaming_input)
            or self._elastic_admission_controller.pending_maintenance_plan is not None
            or getattr(self, "_elastic_restore_retention_id", None) is not None
            or getattr(self, "_elastic_graph_carrier_step_key", None) is not None
        )
        if not catalog:
            raise RuntimeError("elastic catalog activation requires a sealed catalog")
        if self._elastic_graph_catalog:
            raise RuntimeError(
                "elastic catalog activation requires no previously active catalog"
            )
        if active_lifecycle:
            raise RuntimeError(
                "elastic catalog activation requires a quiescent pre-READY lifecycle"
            )
        required, _restore = validate_elastic_catalog_key_inventory(
            coverage.get("required_step_keys"),
            coverage.get("restore_step_keys", []),
            label="elastic catalog activation",
        )
        if set(catalog) != set(required):
            raise RuntimeError(
                "elastic catalog rows and product boundary come from different "
                "required inventories"
            )
        row_digest = getattr(catalog, "source_sha256", None)
        coverage_digest = coverage.get("_catalog_source_sha256")
        if (
            not isinstance(row_digest, str)
            or len(row_digest) != 64
            or any(char not in "0123456789abcdef" for char in row_digest)
            or not isinstance(coverage_digest, str)
            or len(coverage_digest) != 64
            or any(char not in "0123456789abcdef" for char in coverage_digest)
            or row_digest != coverage_digest
        ):
            raise RuntimeError(
                "elastic catalog rows and product boundary differ at source bytes"
            )
        previous_catalog = self._elastic_graph_catalog
        previous_coverage = self._elastic_graph_catalog_coverage
        previous_max_num_running_reqs = self.max_num_running_reqs
        previous_short_decode_inventory = getattr(
            self, "_elastic_short_decode_inventory", None
        )
        previous_terminal_decode_carrier_x = getattr(
            self, "_elastic_terminal_decode_carrier_x", None
        )
        try:
            self._elastic_graph_catalog = catalog
            self._elastic_graph_catalog_coverage = coverage
            self.max_num_running_reqs = min(
                self.max_num_running_reqs,
                coverage["mixed_max_x"],
            )
            self._elastic_terminal_decode_carrier_x = int(coverage["decode_max_x"])
            if previous_short_decode_inventory is not None:
                self._rebuild_elastic_short_decode_inventory(coverage["decode_max_x"])
            logger.info(
                "Elastic sealed product boundary enabled: decode_max_x=%d "
                "mixed_max_x=%d full_context_max_x=%d scheduler_cap=%d",
                coverage["decode_max_x"],
                coverage["mixed_max_x"],
                coverage["full_context_max_x"],
                self.max_num_running_reqs,
            )
            capture_groups = defaultdict(list)
            for catalog_key, row in catalog.items():
                for owner_key in (
                    Scheduler._elastic_graph_owner_key(catalog_key),
                    Scheduler._elastic_graph_global_owner_key(catalog_key),
                ):
                    if owner_key is not None:
                        capture_groups[owner_key].append(row)
                # A bounded short PIECEWISE request has one prefill step and
                # must reclaim before a fresh cohort crosses admission. Its
                # repeated cold envelope is the conservative same-key price.
                self._elastic_admission_controller.record_measurement(
                    catalog_key,
                    (
                        max(row["cold_peak_bytes"], row["hot_peak_bytes"])
                        if (
                            coverage.get("representation") == "bounded_exact_hotset"
                            and catalog_key[0] == 0
                        )
                        else row["hot_peak_bytes"]
                    ),
                )
            for owner_key, rows in capture_groups.items():
                envelope, provenance = (
                    Scheduler._combine_elastic_catalog_envelope_evidence(tuple(rows))
                )
                assert envelope is not None
                self._elastic_admission_controller.record_capture_envelope(
                    owner_key, envelope, resident_key_bytes=provenance
                )
        except Exception:
            self._elastic_graph_catalog = previous_catalog
            self._elastic_graph_catalog_coverage = previous_coverage
            self.max_num_running_reqs = previous_max_num_running_reqs
            self._elastic_terminal_decode_carrier_x = previous_terminal_decode_carrier_x
            if previous_short_decode_inventory is not None:
                self._elastic_short_decode_inventory = previous_short_decode_inventory
            raise

    @property
    def _gdn_checkpoint_coordinator(self) -> HybridKVCacheCoordinator | None:
        coordinator = self.kv_cache_manager.coordinator
        if (
            isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.gdn_checkpoint_keys is not None
        ):
            return coordinator
        return None

    def _gdn_boundary_key(self, request: Request, boundary: int) -> bytes | None:
        coordinator = getattr(self, "_gdn_checkpoint_coordinator", None)
        if coordinator is None:
            return None
        key_block_size = (
            coordinator.hash_block_size
            if getattr(coordinator, "enable_dcp_fine_prefix", False)
            else self.block_size
        )
        if boundary <= 0 or boundary % key_block_size != 0:
            return None
        block_hashes = BlockHashListWithBlockSize(
            request.block_hashes,
            coordinator.hash_block_size,
            key_block_size,
        )
        block_idx = boundary // key_block_size - 1
        if block_idx >= len(block_hashes):
            return None
        return bytes(block_hashes[block_idx])

    def _should_save_gdn_checkpoint(self, request: Request, boundary: int) -> bool:
        coordinator = getattr(self, "_gdn_checkpoint_coordinator", None)
        if coordinator is None or not getattr(
            coordinator, "enable_dcp_fine_prefix", False
        ):
            return True
        if boundary == request.shared_prefix_boundary:
            return True
        if boundary == self._fine_prefix_resume_boundary(request):
            return True
        if boundary in self._fine_prefix_checkpoint_boundaries(request):
            return True
        prompt_tail = (
            request.execution_prefill_len
            // coordinator.hash_block_size
            * coordinator.hash_block_size
        )
        if boundary == prompt_tail:
            return True
        return any(
            isinstance(group.kv_cache_spec, MambaSpec)
            and group.kv_cache_spec.separate_pool
            and boundary
            % (
                group.kv_cache_spec.block_size
                * coordinator.dcp_world_size
                * coordinator.pcp_world_size
            )
            == 0
            for group in coordinator.kv_cache_config.kv_cache_groups
        )

    def _fine_prefix_checkpoint_boundaries(
        self, request: Request, *, shared_prefix_boundary: int | None = None
    ) -> tuple[int, ...]:
        """Materialize useful match/resume pairs, not every hash boundary."""
        coordinator = getattr(self, "_gdn_checkpoint_coordinator", None)
        if coordinator is None or not getattr(
            coordinator, "enable_dcp_fine_prefix", False
        ):
            return ()
        unit = self.hash_block_size
        prefill_end = request.execution_prefill_len
        tail = max(0, prefill_end - 1) // unit * unit
        shared = (
            (
                request.shared_prefix_boundary
                if shared_prefix_boundary is None
                else shared_prefix_boundary
            )
            // unit
            * unit
        )
        matches = {tail, shared, getattr(request, "prefix_cache_hint_tokens", 0)}
        boundaries = set()
        for match in matches:
            if 0 < match < prefill_end:
                boundaries.add(match)
                if self.use_eagle and match > unit:
                    boundaries.add(match - unit)
        return tuple(sorted(boundaries))

    def _fine_prefix_resume_boundary(self, request: Request) -> int:
        """Return the recurrent-state boundary after the EAGLE overlap.

        EAGLE/MTP must recompute one hash unit before generation. Attention
        therefore matches the explicitly hinted boundary, while GDN restores
        the state immediately before that overlap.
        """
        coordinator = getattr(self, "_gdn_checkpoint_coordinator", None)
        match_boundary = getattr(request, "prefix_cache_hint_tokens", 0)
        if (
            coordinator is None
            or not getattr(coordinator, "enable_dcp_fine_prefix", False)
            or match_boundary == 0
        ):
            return 0
        resume_boundary = match_boundary - (
            self.hash_block_size if self.use_eagle else 0
        )
        if resume_boundary <= 0:
            raise RuntimeError(
                "Fine-prefix hint must exceed the EAGLE/MTP overlap "
                f"({self.hash_block_size} tokens)"
            )
        return resume_boundary

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        *,
        shared_prefix_boundary: int | None = None,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only while the scheduler-owned execution stream is being
        # replayed.  A resumed request can carry emitted output tokens past the
        # semantic prompt, and its final replay token is still prefill even
        # though that same forward can produce the next autoregressive sample.
        prefill_end = request.execution_prefill_len
        if start >= prefill_end:
            return num_new_tokens

        if shared_prefix_boundary is None:
            shared_prefix_boundary = request.shared_prefix_boundary

        coordinator = getattr(self, "_gdn_checkpoint_coordinator", None)
        # Keep ordinary chunking on the proven global scheduler geometry.
        # Only the explicit shared junction may use the finer hash boundary.
        # Grouped recurrence must preserve its accepted internal split grid;
        # this also applies to standalone GDN planners without a live pool.
        preserve_recurrent_chunking = bool(
            (coordinator is not None and coordinator.mamba_block_pool is not None)
            or getattr(self, "mamba_state_update_alignment", 1) > 1
        )
        block_size = self.block_size
        shared_boundary_alignment = (
            self.hash_block_size
            if coordinator is not None
            and getattr(coordinator, "enable_dcp_fine_prefix", False)
            else self.block_size
        )
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = prefill_end - prefill_end % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens
        checkpoint_position = get_mamba_prefill_checkpoint_position(
            prefill_end,
            self.hash_block_size,
            drop_eagle_block=self.use_eagle_block_drop,
        )
        use_internal_checkpoint = (
            self.mamba_has_prefill_checkpoint_blocks
            and end >= prefill_end
            and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=checkpoint_position,
                hash_block_size=self.hash_block_size,
                mamba_block_size=block_size,
                checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
            )
        )
        if use_internal_checkpoint:
            last_cache_position = 0
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, whose slot decode advances
        # to the boundary. A block too wide for one chunk advances sub-block
        # and re-aligns at the next boundary.
        if end < (last_cache_position if preserve_recurrent_chunking else prefill_end):
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self._long_prefill_chunk_cap(request)
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit and not use_internal_checkpoint
            else 0
        )
        if tail_boundary and self.use_eagle_block_drop:
            # Eagle matches one hash unit past the candidate and drops it, so
            # nothing proves the prompt's own last hash boundary. Materialize
            # the state one unit lower, where the hit can actually land. Keyed on
            # the block-drop bit, not plain use_eagle: this shift exists only to
            # compensate for the drop, and the Mamba manager's matching gate
            # reads the same bit (the coordinator is handed use_eagle_block_drop).
            tail_boundary = max(tail_boundary - self.hash_block_size, 0)
        junction = shared_prefix_boundary
        # Block-floored: a sub-block junction's state is not separately cacheable.
        block_floored = start + (junction - start) // block_size * block_size
        # Past the prompt the manager writes nothing, so fall back to the
        # block-floored stop rather than dropping it: a resumed request replaying
        # output tokens can still observe a junction there.
        junction_stop = (
            junction
            if (
                self.mamba_fine_grained_prefix_cache
                or shared_boundary_alignment == self.hash_block_size
            )
            and junction <= request.num_prompt_tokens
            else block_floored
        )
        fine_resume_boundary = Scheduler._fine_prefix_resume_boundary(self, request)
        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
            # rather than running past it.
            next_block_boundary
            if start % block_size != 0
            and not use_internal_checkpoint
            and (
                not preserve_recurrent_chunking
                or next_block_boundary <= last_cache_position
            )
            else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            fine_resume_boundary if start < fine_resume_boundary < end else 0,
            # Marconi shared-prefix junction: cache its state so sibling
            # requests sharing the prefix can reuse it.
            junction_stop if start < junction < end else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        fine_stops = Scheduler._fine_prefix_checkpoint_boundaries(
            self, request, shared_prefix_boundary=shared_prefix_boundary
        )
        end = min((s for s in (*stops, *fine_stops) if start < s < end), default=end)

        # Preserve the backend's recurrence grouping across artificial prefill
        # splits. The natural prompt end is intentionally exempt: prompt to
        # decode is a real model boundary and may occur at any token. Prefix
        # restore and persistent checkpoint geometry are validated on the same
        # grid during scheduler construction.
        state_alignment = getattr(self, "mamba_state_update_alignment", 1)
        if state_alignment > 1 and end < prefill_end:
            if start % state_alignment:
                raise RuntimeError(
                    "Cannot resume an unfinished recurrent prefill from an "
                    "unaligned state boundary: "
                    f"start={start}, alignment={state_alignment}"
                )
            end = end // state_alignment * state_alignment
        return max(end - start, 0)

    def _resolved_shared_prefix_boundary(self, request: Request, cached: int) -> int:
        """Use the same junction in leased preview and ordinary commit."""
        hint = request.prefix_cache_hint_tokens
        if hint and hint % self.hash_block_size != 0:
            raise RuntimeError("Validated prefix-cache hint is not hash aligned")
        return max(cached, hint)

    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        bound = getattr(self, "_elastic_prefix_hits", {}).get(request.request_id)
        if bound is not None:
            if bound[0] is not request:
                raise RuntimeError("prefix lease request identity changed")
            return bound[1]
        connector = self.connector
        if connector is not None and connector.supports_divergent_local_hybrid_hits:
            return self.kv_cache_manager.get_computed_blocks_for_connector(request)

        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
        return blocks, num_local, shared_prefix_boundary, False

    def _lease_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        if not hasattr(self, "_elastic_prefix_hits"):
            self._elastic_prefix_hits = {}
        bound = self._elastic_prefix_hits.get(request.request_id)
        if bound is not None:
            return self._get_local_prefix_cache_hit(request)
        result = (
            (self.kv_cache_manager.empty_kv_cache_blocks, 0, 0, False)
            if getattr(self, "_elastic_prefix_pressure_fallback", False)
            else self._get_local_prefix_cache_hit(request)
        )
        lease = (
            self.kv_cache_manager.lease_computed_blocks(result[0])
            if result[1] > 0
            else None
        )
        self._elastic_prefix_hits[request.request_id] = (request, result, lease)
        return result

    def _release_elastic_prefix_hits(self, keep: Iterable[str] = ()) -> None:
        retained = frozenset(keep)
        hits = getattr(self, "_elastic_prefix_hits", {})
        for request_id in tuple(hits):
            if request_id not in retained:
                _request, _result, lease = hits.pop(request_id)
                if lease is not None:
                    lease.release()

    def _drop_elastic_prefix_hits_for_pressure(self) -> bool:
        """Allow one cold replan when pinned cache prevents wave admission."""
        hits = getattr(self, "_elastic_prefix_hits", {})
        if not any(result[1] > 0 for _, result, _ in hits.values()):
            return False
        requests = [request for request, _result, _lease in hits.values()]
        self._release_elastic_prefix_hits()
        self._elastic_prefix_pressure_fallback = True
        empty = self.kv_cache_manager.empty_kv_cache_blocks
        for request in requests:
            hits[request.request_id] = (request, (empty, 0, 0, False), None)
        logger.info("Elastic prefix reuse deferred for physical admission pressure")
        return True

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Never end a prefill chunk within num_prefill_lookahead of the
        prefill end.

        At a chunked-prefill boundary, the multi-module MTP drafter consumes
        the next num_prefill_lookahead known prefill tokens as draft inputs. A
        boundary closer to the end than that would make it fall back to
        sampled drafts, permanently polluting the trailing modules' KV caches.
        Either finish the prefill or leave at least num_prefill_lookahead for
        the next chunk. No-op for eagle-family drafters (lookahead 1).
        """
        prefill_end = request.execution_prefill_len
        if num_computed_tokens >= prefill_end:
            return num_new_tokens
        original_num_new_tokens = num_new_tokens
        remaining = prefill_end - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        if (
            num_new_tokens != original_num_new_tokens
            and self.need_mamba_block_aligned_split
            and num_computed_tokens + num_new_tokens < prefill_end
        ):
            state_alignment = self.mamba_state_update_alignment
            if num_computed_tokens % state_alignment:
                raise RuntimeError(
                    "Cannot reserve MTP prefill lookahead from an unaligned "
                    "recurrent state boundary: "
                    f"start={num_computed_tokens}, alignment={state_alignment}"
                )
            aligned_end = (
                (num_computed_tokens + num_new_tokens)
                // state_alignment
                * state_alignment
            )
            num_new_tokens = aligned_end - num_computed_tokens
        return max(num_new_tokens, 0)

    @staticmethod
    def _is_prefill_request(request: Request) -> bool:
        return request.num_computed_tokens < request.execution_prefill_len

    def _partial_prefill_target_count(self) -> int:
        """Return the number of visible prefills to share this step's budget."""
        if self.max_concurrent_partial_prefills <= 1:
            return 1

        prefills = sum(self._is_prefill_request(request) for request in self.running)
        remaining_slots = min(
            self.max_concurrent_partial_prefills - prefills,
            max(0, self.max_num_running_reqs - len(self.running)),
        )
        if remaining_slots > 0:
            for request in self.waiting:
                if not self._is_prefill_request(request):
                    continue
                prefills += 1
                remaining_slots -= 1
                if remaining_slots == 0:
                    break
        return max(prefills, 1)

    def _partial_prefill_chunk_cap(self, target_count: int) -> int:
        if self.max_concurrent_partial_prefills <= 1 or target_count <= 1:
            return 0
        return max(1, self.max_num_scheduled_tokens // target_count)

    def _cap_prefill_chunk(
        self,
        request: Request,
        num_new_tokens: int,
        prefill_chunk_cap: int,
    ) -> int:
        if (
            prefill_chunk_cap > 0
            and self._is_prefill_request(request)
            and num_new_tokens > prefill_chunk_cap
        ):
            return prefill_chunk_cap
        return num_new_tokens

    def _long_prefill_chunk_cap(self, request: Request) -> int:
        """Return the cap only for prompts in the configured long domain."""
        threshold = self.scheduler_config.long_prefill_token_threshold
        if threshold <= 0:
            return 0
        if (
            self.long_prefill_cap_min_prompt_tokens > 0
            and request.num_prompt_tokens < self.long_prefill_cap_min_prompt_tokens
        ):
            return 0
        return threshold

    def _should_delay_waiting_prefill_admission(self) -> bool:
        """Briefly wait for an initial burst without delaying it indefinitely."""
        if (
            self.prefill_admission_delay_s <= 0
            or self.max_concurrent_partial_prefills <= 1
            or self._elastic_restore_mode
            or self.running
            or self.num_waiting_for_streaming_input > 0
        ):
            return False

        target_count = min(
            self.max_concurrent_partial_prefills,
            self.max_num_running_reqs,
        )
        waiting_prefills = 0
        oldest_arrival_time: float | None = None
        newest_arrival_time: float | None = None
        for request in self.waiting:
            if not self._is_prefill_request(request):
                continue
            waiting_prefills += 1
            oldest_arrival_time = (
                request.arrival_time
                if oldest_arrival_time is None
                else min(oldest_arrival_time, request.arrival_time)
            )
            newest_arrival_time = (
                request.arrival_time
                if newest_arrival_time is None
                else max(newest_arrival_time, request.arrival_time)
            )
            if waiting_prefills >= target_count:
                return False

        if (
            waiting_prefills == 0
            or oldest_arrival_time is None
            or newest_arrival_time is None
        ):
            return False

        now = time.time()
        if now - oldest_arrival_time >= self.prefill_admission_max_delay_s:
            return False
        if waiting_prefills < target_count:
            return True
        return now - newest_arrival_time < self.prefill_admission_delay_s

    def _elastic_mm_wave_request_state(
        self, request_ids: Sequence[str]
    ) -> tuple[tuple[str, int, int, int, int], ...] | None:
        state = []
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                return None
            state.append(
                (
                    request_id,
                    int(request.status),
                    request.num_computed_tokens,
                    request.num_tokens,
                    request.execution_prefill_len,
                )
            )
        return tuple(state)

    def _validated_deferred_mm_wave(self) -> DeferredMMWave | None:
        """Return a still-exact capture-to-user wave, or invalidate it."""
        wave = self._elastic_deferred_mm_wave
        if wave is None:
            return None
        request_ids = (*wave.running_request_ids, *wave.waiting_request_ids)
        current_state = self._elastic_mm_wave_request_state(request_ids)
        if current_state != wave.request_state:
            logger.info(
                "Deferred MM wave invalidated before USER mutation: "
                "step_key=%r expected=%r actual=%r",
                wave.step_key,
                wave.request_state,
                current_state,
            )
            self._elastic_deferred_mm_wave = None
            return None
        return wave

    def _arm_deferred_mm_wave(
        self,
        *,
        step_key: tuple[int, ...],
        scheduled_tokens: Mapping[str, int],
        scheduled_encoder_inputs: Mapping[str, Sequence[int]],
    ) -> None:
        plan = self._elastic_admission_controller.pending_maintenance_plan
        if (
            plan is None
            or plan.kind != ElasticPlanKind.MAINTENANCE
            or plan.maintenance_execution != ElasticMaintenanceExecution.GRAPH_ONLY
            or self._elastic_admission_controller.pending_maintenance_step_key
            != step_key
        ):
            raise RuntimeError("deferred MM wave has no exact graph-only plan")
        if not any(scheduled_encoder_inputs.values()):
            raise RuntimeError("deferred MM wave has no encoder consumer")
        running_ids = {request.request_id for request in self.running}
        ordered_ids = tuple(scheduled_tokens)
        wave_running_ids = tuple(
            request_id for request_id in ordered_ids if request_id in running_ids
        )
        wave_waiting_ids = tuple(
            request_id for request_id in ordered_ids if request_id not in running_ids
        )
        request_state = self._elastic_mm_wave_request_state(ordered_ids)
        if request_state is None:
            raise RuntimeError("deferred MM wave lost a request before capture")
        candidate = DeferredMMWave(
            step_key=step_key,
            running_request_ids=wave_running_ids,
            waiting_request_ids=wave_waiting_ids,
            request_state=request_state,
            scheduled_tokens=tuple(scheduled_tokens.items()),
            scheduled_encoder_inputs=tuple(
                (request_id, tuple(input_ids))
                for request_id, input_ids in scheduled_encoder_inputs.items()
                if input_ids
            ),
        )
        prior = self._elastic_deferred_mm_wave
        if prior is not None and prior != candidate:
            raise RuntimeError("a different deferred MM wave is already armed")
        self._elastic_deferred_mm_wave = candidate

    def _deferred_mm_wave_matches_candidate(
        self,
        *,
        step_key: tuple[int, ...],
        scheduled_tokens: Mapping[str, int],
        scheduled_encoder_inputs: Mapping[str, Sequence[int]],
    ) -> bool:
        """Validate a tick-B candidate before any Graph/KV admission mutation.

        The graph-only capture from tick A is already committed physical state.
        A queue, prefix-cache, encoder-cache, or budget drift therefore drops
        only the immutable USER binding and lets the current wave be replanned;
        it must not discard the now-HOT graph receipt.
        """
        wave = self._validated_deferred_mm_wave()
        if wave is None:
            return True
        actual_encoder_inputs = tuple(
            (request_id, tuple(input_ids))
            for request_id, input_ids in scheduled_encoder_inputs.items()
            if input_ids
        )
        if (
            step_key != wave.step_key
            or tuple(scheduled_tokens.items()) != wave.scheduled_tokens
            or actual_encoder_inputs != wave.scheduled_encoder_inputs
        ):
            logger.info(
                "Deferred MM USER binding invalidated before admission: "
                "expected_key=%r actual_key=%r expected_tokens=%r "
                "actual_tokens=%r expected_encoder=%r actual_encoder=%r",
                wave.step_key,
                step_key,
                wave.scheduled_tokens,
                tuple(scheduled_tokens.items()),
                wave.scheduled_encoder_inputs,
                actual_encoder_inputs,
            )
            self._elastic_deferred_mm_wave = None
            return False
        return True

    def _preflight_elastic_running_text_wave(
        self,
        *,
        token_budget: int,
        prefill_chunk_cap: int,
        defer_prefills: bool,
        physical_quiescent: bool = True,
    ) -> tuple[tuple[int, ...] | None, bool]:
        """Price one complete visible execution wave before prefix admission.

        The model executes the final assembled batch, not each prefix visited
        by the scheduler loop. Preparing maintenance for those prefixes makes
        X1, X2, ... evict and recapture each other while assembling a larger
        steady cohort. This preflight covers the complete RUNNING wave,
        including partial-prefill, mixed, and encoder rows. It mirrors the
        running loop's token and encoder budgets. Cache hits acquire temporary
        references; request ownership, encoder state and drafts stay unchanged.

        Returns the final canonical key and whether request-free maintenance
        was prepared for the next scheduler output.
        """
        self._elastic_preflight_joint_waiting_request_ids: tuple[str, ...] = ()
        self._elastic_preflight_waiting_ignore_prefix_request_ids: tuple[str, ...] = ()
        if (
            not self.elastic_on_demand_graphs
            or getattr(self, "_elastic_restore_mode", False)
            or token_budget <= 0
            or self._pending_elastic_maintenance_requires_exclusive_tick()
            or self.num_waiting_for_streaming_input
        ):
            return None, False

        prospective_tokens: dict[str, int] = {}
        prospective_drafts: dict[str, list[int]] = {}
        prospective_encoder_inputs: dict[str, tuple[int, ...]] = {}
        computed_overrides: dict[str, int] = {}
        remaining_budget = token_budget
        remaining_encoder_budget = self.max_num_encoder_input_tokens
        encoder_cache_manager = getattr(self, "encoder_cache_manager", None)
        encoder_wave_overlay = EncoderWaveOverlay(
            encoder_cache_manager.clone_for_preview()
            if encoder_cache_manager is not None
            else EncoderCacheManager(0)
        )
        deferred_mm_wave = self._validated_deferred_mm_wave()
        bound_running_ids = (
            frozenset(deferred_mm_wave.running_request_ids)
            if deferred_mm_wave is not None
            else None
        )
        for request in self.running:
            if (
                bound_running_ids is not None
                and request.request_id not in bound_running_ids
            ):
                if bound_running_ids.issubset(prospective_tokens):
                    break
                continue
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            if self.current_step < request.next_decode_eligible_step:
                continue
            if defer_prefills and request.is_prefill_chunk:
                continue
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new_tokens = self._cap_prefill_chunk(
                request, num_new_tokens, prefill_chunk_cap
            )
            long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
            if 0 < long_prefill_chunk_cap < num_new_tokens:
                num_new_tokens = long_prefill_chunk_cap
            num_new_tokens = min(num_new_tokens, remaining_budget)
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            encoder_inputs: Sequence[int] | None = None
            candidate_encoder_budget = remaining_encoder_budget
            candidate_encoder_overlay = encoder_wave_overlay
            if request.has_encoder_inputs:
                candidate_encoder_overlay = encoder_wave_overlay.clone()
                (
                    encoder_inputs,
                    num_new_tokens,
                    candidate_encoder_budget,
                    _external_encoder_inputs,
                    _cached_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    remaining_encoder_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                    encoder_wave_overlay=candidate_encoder_overlay,
                )
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )
            if num_new_tokens <= 0:
                continue

            prospective_tokens[request.request_id] = num_new_tokens
            if encoder_inputs:
                prospective_encoder_inputs[request.request_id] = tuple(encoder_inputs)
                remaining_encoder_budget = candidate_encoder_budget
            if request.has_encoder_inputs:
                encoder_wave_overlay = candidate_encoder_overlay
            remaining_budget -= num_new_tokens
            if request.spec_token_ids:
                prospective_spec_count = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if prospective_spec_count > 0:
                    prospective_drafts[request.request_id] = request.spec_token_ids[
                        :prospective_spec_count
                    ]
            if remaining_budget <= 0:
                break

        running_wave_tokens = dict(prospective_tokens)
        running_wave_drafts = dict(prospective_drafts)
        running_wave_encoder_inputs = dict(prospective_encoder_inputs)
        running_wave_remaining_budget = remaining_budget
        running_wave_remaining_encoder_budget = remaining_encoder_budget
        running_wave_encoder_overlay = encoder_wave_overlay

        # A product retry can arrive while the preceding cohort is still
        # decoding (for example, repetition recovery changes X32 to
        # Running31+Waiting1).  Price the exact combined wave up front instead
        # of disabling complete-wave preflight and repeatedly probing a cold
        # X1 prefix.  Keep this fast path deliberately narrow: unsupported
        # connector/encoder/LoRA/streaming state falls back to running-only
        # progress and is reconsidered without mutating the waiting request.
        joint_waiting_tokens = dict(prospective_tokens)
        joint_waiting_drafts = dict(prospective_drafts)
        joint_encoder_inputs = dict(prospective_encoder_inputs)
        joint_computed_overrides: dict[str, int] = {}
        joint_remaining_budget = remaining_budget
        joint_remaining_encoder_budget = remaining_encoder_budget
        joint_encoder_overlay = encoder_wave_overlay
        joint_waiting_count = 0
        waiting_candidates: Sequence[Request] | None = ()
        joint_waiting_supported = (
            bool(prospective_tokens)
            and bool(self.waiting or self.skipped_waiting)
            and not (
                self.policy != SchedulingPolicy.FCFS
                or self.num_waiting_for_streaming_input
                or self.connector is not None
                or self.ec_connector is not None
                or self.lora_config is not None
                or self.kv_cache_manager.enable_kv_cache_events
                or self.canonical_prefill_admission
            )
        )
        if joint_waiting_supported:
            available_slots = max(
                0,
                self.max_num_running_reqs
                - len(self.running)
                - self.num_waiting_for_streaming_input,
            )
            bound_waiting_ids = (
                frozenset(deferred_mm_wave.waiting_request_ids)
                if deferred_mm_wave is not None
                else None
            )
            waiting_candidates = self._elastic_schedulable_waiting_snapshot(
                allowed_request_ids=bound_waiting_ids
            )
            running_wave_is_pure_decode = self._is_pure_decode_step(
                prospective_tokens,
                prospective_drafts,
            )
            for request in waiting_candidates or ():
                if joint_waiting_count >= available_slots:
                    break
                if (
                    request.num_computed_tokens != 0
                    or request.num_stale_output_tokens > 0
                ):
                    break
                # Keep the same physical hits alive through the joint commit.
                _, num_computed_tokens, shared_boundary, _ = (
                    self._lease_local_prefix_cache_hit(request)
                )
                if defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    break
                num_new_tokens = request.num_tokens - num_computed_tokens
                pad_spec_decode = bool(
                    self.num_spec_tokens > 0
                    and self.dynamic_sd_lookup is None
                    and self.num_sampled_tokens_per_step > 0
                    and num_new_tokens == 1
                    and prospective_tokens
                    and running_wave_is_pure_decode
                )
                if pad_spec_decode:
                    num_new_tokens = 1 + self.num_spec_tokens
                    if (
                        num_new_tokens > joint_remaining_budget
                        or num_computed_tokens + num_new_tokens > self.max_model_len
                    ):
                        break
                num_new_tokens = self._cap_prefill_chunk(
                    request, num_new_tokens, prefill_chunk_cap
                )
                long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
                if 0 < long_prefill_chunk_cap < num_new_tokens:
                    num_new_tokens = long_prefill_chunk_cap
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > joint_remaining_budget
                ):
                    break
                num_new_tokens = min(num_new_tokens, joint_remaining_budget)
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_computed_tokens,
                        0,
                        shared_prefix_boundary=self._resolved_shared_prefix_boundary(
                            request, shared_boundary
                        ),
                    )
                encoder_inputs = None
                candidate_encoder_budget = joint_remaining_encoder_budget
                candidate_encoder_overlay = joint_encoder_overlay
                if request.has_encoder_inputs:
                    candidate_encoder_overlay = joint_encoder_overlay.clone()
                    (
                        encoder_inputs,
                        num_new_tokens,
                        candidate_encoder_budget,
                        _external_encoder_inputs,
                        _cached_encoder_inputs,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        joint_remaining_encoder_budget,
                        shift_computed_tokens=self.num_prefill_lookahead,
                        encoder_wave_overlay=candidate_encoder_overlay,
                    )
                num_new_tokens = self._reserve_prefill_lookahead(
                    request, num_computed_tokens, num_new_tokens
                )
                if num_new_tokens <= 0:
                    break
                trial_waiting_tokens = dict(joint_waiting_tokens)
                trial_waiting_tokens[request.request_id] = num_new_tokens
                if getattr(self, "scheduler_reserve_full_isl", False):
                    trial_requirements = self._elastic_remaining_resource_requirements(
                        trial_waiting_tokens
                    )
                    fits_current_layout = (
                        self.kv_cache_manager.coordinator.can_allocate(
                            trial_requirements,
                            primary_watermark_blocks=(
                                self.kv_cache_manager.watermark_blocks
                            ),
                        )
                    )
                    if (
                        not fits_current_layout
                        and not self._elastic_wave_fits_after_idle_reclaim(
                            trial_requirements
                        )
                    ):
                        break
                joint_waiting_tokens[request.request_id] = num_new_tokens
                joint_computed_overrides[request.request_id] = num_computed_tokens
                if encoder_inputs:
                    joint_encoder_inputs[request.request_id] = tuple(encoder_inputs)
                joint_remaining_encoder_budget = candidate_encoder_budget
                if request.has_encoder_inputs:
                    joint_encoder_overlay = candidate_encoder_overlay
                if pad_spec_decode:
                    joint_waiting_drafts[request.request_id] = [
                        -1
                    ] * self.num_spec_tokens
                joint_remaining_budget -= num_new_tokens
                joint_waiting_count += 1
                if joint_remaining_budget <= 0:
                    break
            # A candidate rejected by token/encoder/capacity limits must not
            # pin cache capacity while pricing the selected wave.
            self._release_elastic_prefix_hits(joint_computed_overrides)
            if joint_waiting_count:
                prospective_tokens = joint_waiting_tokens
                prospective_drafts = joint_waiting_drafts
                prospective_encoder_inputs = joint_encoder_inputs
                computed_overrides = joint_computed_overrides
                remaining_budget = joint_remaining_budget
                remaining_encoder_budget = joint_remaining_encoder_budget
                self._elastic_preflight_joint_waiting_request_ids = tuple(
                    joint_computed_overrides
                )
                self._elastic_preflight_waiting_ignore_prefix_request_ids = ()

        def clear_joint_identity() -> None:
            self._elastic_preflight_joint_waiting_request_ids = ()
            self._elastic_preflight_waiting_ignore_prefix_request_ids = ()

        def retry_running_only() -> tuple[tuple[int, ...] | None, bool]:
            """Restore progress after an optional joint expansion fails."""
            clear_joint_identity()
            self._release_elastic_prefix_hits()
            if not running_wave_tokens:
                return None, False
            running_is_pure_decode = self._is_pure_decode_step(
                running_wave_tokens,
                running_wave_drafts,
            )
            running_num_spec_tokens = self._num_spec_tokens_for_step(
                running_wave_tokens,
                running_is_pure_decode,
            )
            running_key = self._canonical_elastic_graph_step_key(
                running_wave_tokens,
                running_num_spec_tokens,
                running_is_pure_decode,
            )
            running_requirements = self._elastic_remaining_resource_requirements(
                running_wave_tokens
            )
            running_minimum_free_primary_blocks = (
                self._elastic_successor_primary_headroom(running_wave_tokens)
                + running_requirements.primary
            )
            running_fits, running_required, running_available = (
                self._can_fund_elastic_graph_step(
                    running_key,
                    minimum_free_primary_blocks=(running_minimum_free_primary_blocks),
                    gdn_blocks=(
                        self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                            running_requirements.mamba
                        )
                    ),
                    allow_maintenance=True,
                    **(
                        cast(
                            dict[str, Any],
                            {
                                "mm_activation_loan_bytes": (
                                    self._elastic_mm_activation_loan_bytes
                                )
                            }
                            if running_wave_encoder_inputs
                            else {},
                        )
                    ),
                )
            )
            if running_fits and self._reserve_elastic_admission(
                cast(tuple[int, ...], running_key),
                external_memory_bytes=running_required,
                minimum_free_primary_blocks=running_minimum_free_primary_blocks,
                requirements=running_requirements,
            ):
                return running_key, False
            pending_plan = self._elastic_admission_controller.pending_maintenance_plan
            if pending_plan is not None:
                if (
                    pending_plan.maintenance_execution
                    == ElasticMaintenanceExecution.GRAPH_ONLY
                ):
                    self._arm_deferred_mm_wave(
                        step_key=cast(tuple[int, ...], running_key),
                        scheduled_tokens=running_wave_tokens,
                        scheduled_encoder_inputs=running_wave_encoder_inputs,
                    )
                    return running_key, True
                if self._reserve_pending_elastic_maintenance_admission(
                    cast(tuple[int, ...], running_key),
                    minimum_free_primary_blocks=(running_minimum_free_primary_blocks),
                    requirements=running_requirements,
                ):
                    return running_key, True
            self._observe_elastic_running_wave_defer(
                cast(tuple[int, ...], running_key),
                required_external=running_required,
                available_external=running_available,
                physical_quiescent=physical_quiescent,
            )
            if (
                not running_wave_encoder_inputs
                and self._prepare_elastic_cold_form_reclaim(
                    cast(tuple[int, ...], running_key),
                    required_external=running_required,
                    available_external=running_available,
                    physical_quiescent=physical_quiescent,
                )
            ):
                return None, True
            return None, False

        rejected_joint_graphs: list[tuple[tuple[int, ...], int, int]] = []

        def resolve_failed_joint() -> tuple[tuple[int, ...] | None, bool]:
            """Widen admission at a safe boundary, otherwise preserve RUNNING."""
            if deferred_mm_wave is not None:
                self._elastic_deferred_mm_wave = None
                clear_joint_identity()
                pending = self._elastic_admission_controller.pending_maintenance_plan
                if pending is not None:
                    self._clear_pre_mutation_serving_maintenance(
                        reason="deferred_mm_running_receipt_changed"
                    )
                else:
                    self._rollback_elastic_admission()
                return self._preflight_elastic_running_text_wave(
                    token_budget=token_budget,
                    prefill_chunk_cap=prefill_chunk_cap,
                    defer_prefills=defer_prefills,
                    physical_quiescent=physical_quiescent,
                )
            # The immutable candidate was not fundable in the current resource
            # epoch. Drop only its USER binding; a preceding GRAPH_ONLY capture
            # is already committed and remains valid HOT state.
            self._elastic_deferred_mm_wave = None
            self._clear_pre_mutation_serving_maintenance(
                reason="joint_wave_rejected_before_running_retry"
            )
            # RUNNING-only progress is not proof that retained Graph state
            # should indefinitely exclude a fundable WAITING cohort. The
            # existing reclaim planner proves capacity improvement and owns
            # physical-quiescence/lease/loan checks before any mutation.
            joint_waiting_primary = tuple(
                self.kv_cache_manager.estimate_uncached_full_sequence_requirements(
                    request
                ).primary
                for request in cast(Sequence[Request], waiting_candidates)[
                    :joint_waiting_count
                ]
            )
            if self._prepare_elastic_waiting_deficit_reclaim(
                joint_waiting_primary,
                physical_quiescent=physical_quiescent,
            ):
                clear_joint_identity()
                return None, True
            # KV capacity alone does not price a new COLD joint Graph. A HOT
            # RUNNING fallback must not strand otherwise fundable WAITING work.
            # Only after all read-only previews failed may we arm an X0
            # transaction, using that candidate's explicit byte evidence.
            for key, required, available in rejected_joint_graphs:
                if self._prepare_elastic_cold_form_reclaim(
                    key,
                    required_external=required,
                    available_external=available,
                    physical_quiescent=physical_quiescent,
                ):
                    clear_joint_identity()
                    return None, True
            if rejected_joint_graphs:
                rejection = (
                    tuple(rejected_joint_graphs),
                    self._elastic_admission_controller.resident_bytes,
                    self._elastic_pressure_floor_external_bytes(),
                )
                if rejection != getattr(
                    self, "_elastic_last_joint_graph_rejection", None
                ):
                    self._elastic_last_joint_graph_rejection = rejection
                    logger.info(
                        "Elastic joint Graph admission blocked: "
                        "candidates_key_required_available=%r resident=%d "
                        "pressure_floor=%d",
                        *rejection,
                    )
            running_result = retry_running_only()
            if running_result[0] is not None or running_result[1]:
                return running_result
            # A failed RUNNING maintenance reservation can leave a provisional
            # pre-mutation plan. It has no consumer and must not survive this
            # rejected wave.
            if self._elastic_admission_controller.pending_maintenance_plan is not None:
                self._clear_pre_mutation_serving_maintenance(
                    reason="running_retry_rejected_before_waiting_reclaim"
                )
            else:
                self._rollback_elastic_admission()
            return None, False

        selected_joint_prefix = False
        if joint_waiting_count and deferred_mm_wave is None:
            # Choose the largest feasible FCFS prefix in the current physical
            # epoch. Graph buckets and HOT/COLD residency are not monotonic in
            # X, so inspect every prefix from largest to smallest rather than
            # binary-searching. These calls are strict read-only previews; the
            # selected candidate is the only one allowed to create a plan or
            # reserve KV/Graph state below.
            waiting_ids = tuple(joint_computed_overrides)
            selected_prefix = 0
            selected_state: (
                tuple[
                    dict[str, int],
                    dict[str, list[int]],
                    dict[str, tuple[int, ...]],
                    dict[str, int],
                ]
                | None
            ) = None
            inspected_effective_prefixes: set[int] = set()
            preview_state = (
                self._elastic_admission_controller.snapshot,
                self.kv_cache_manager.coordinator.elastic_external_memory_bytes,
            )
            for prefix_size in range(joint_waiting_count, 0, -1):
                self._release_elastic_prefix_hits(waiting_ids[:prefix_size])
                prefix_ids = frozenset(waiting_ids[:prefix_size])
                candidate_tokens = dict(running_wave_tokens)
                candidate_tokens.update(
                    (request_id, joint_waiting_tokens[request_id])
                    for request_id in waiting_ids[:prefix_size]
                )
                candidate_drafts = dict(running_wave_drafts)
                candidate_drafts.update(
                    (request_id, joint_waiting_drafts[request_id])
                    for request_id in waiting_ids[:prefix_size]
                    if request_id in joint_waiting_drafts
                )
                candidate_encoder_inputs = dict(running_wave_encoder_inputs)
                candidate_encoder_inputs.update(
                    (request_id, joint_encoder_inputs[request_id])
                    for request_id in waiting_ids[:prefix_size]
                    if request_id in joint_encoder_inputs
                )
                candidate_overrides = {
                    request_id: value
                    for request_id, value in joint_computed_overrides.items()
                    if request_id in prefix_ids
                }
                candidate_is_pure_decode = self._is_pure_decode_step(
                    candidate_tokens,
                    candidate_drafts,
                    computed_token_overrides=candidate_overrides,
                )
                candidate_k = self._num_spec_tokens_for_step(
                    candidate_tokens, candidate_is_pure_decode
                )
                candidate_key = self._canonical_elastic_graph_step_key(
                    candidate_tokens, candidate_k, candidate_is_pure_decode
                )
                assert candidate_key is not None
                candidate_physical_keys = self._elastic_step_residency_intent(
                    candidate_key
                )[2]
                candidate_is_cold = any(
                    not (
                        (entry := self._elastic_admission_controller.entries.get(key))
                        is not None
                        and entry.hot
                    )
                    for key in candidate_physical_keys
                )
                effective_prefix_size = prefix_size
                if (
                    candidate_is_cold
                    and any(candidate_overrides.values())
                    and not getattr(self, "_elastic_prefix_hits", {})
                ):
                    # Independent prefix-cache peeks have no joint ownership
                    # lease. A COLD capture must therefore be selected against
                    # the exact conservative no-prefix wave that commit can
                    # reproduce, including encoder budget and MM intent.
                    candidate_tokens = dict(running_wave_tokens)
                    candidate_drafts = dict(running_wave_drafts)
                    candidate_encoder_inputs = dict(running_wave_encoder_inputs)
                    candidate_overrides = {}
                    candidate_budget = running_wave_remaining_budget
                    candidate_encoder_budget = running_wave_remaining_encoder_budget
                    candidate_encoder_overlay = running_wave_encoder_overlay
                    effective_prefix_size = 0
                    for request_id in waiting_ids[:prefix_size]:
                        request = self.requests[request_id]
                        num_new_tokens = self._cap_prefill_chunk(
                            request, request.num_tokens, prefill_chunk_cap
                        )
                        long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
                        if 0 < long_prefill_chunk_cap < num_new_tokens:
                            num_new_tokens = long_prefill_chunk_cap
                        if (
                            not self.scheduler_config.enable_chunked_prefill
                            and num_new_tokens > candidate_budget
                        ):
                            break
                        num_new_tokens = min(num_new_tokens, candidate_budget)
                        if self.need_mamba_block_aligned_split:
                            num_new_tokens = self._mamba_block_aligned_split(
                                request,
                                num_new_tokens,
                                0,
                                0,
                                shared_prefix_boundary=0,
                            )
                        encoder_inputs = None
                        next_encoder_budget = candidate_encoder_budget
                        next_encoder_overlay = candidate_encoder_overlay
                        if request.has_encoder_inputs:
                            next_encoder_overlay = candidate_encoder_overlay.clone()
                            (
                                encoder_inputs,
                                num_new_tokens,
                                next_encoder_budget,
                                _external_encoder_inputs,
                                _cached_encoder_inputs,
                            ) = self._try_schedule_encoder_inputs(
                                request,
                                0,
                                num_new_tokens,
                                candidate_encoder_budget,
                                shift_computed_tokens=self.num_prefill_lookahead,
                                encoder_wave_overlay=next_encoder_overlay,
                            )
                        num_new_tokens = self._reserve_prefill_lookahead(
                            request, 0, num_new_tokens
                        )
                        if num_new_tokens <= 0:
                            break
                        candidate_tokens[request_id] = num_new_tokens
                        candidate_overrides[request_id] = 0
                        if encoder_inputs:
                            candidate_encoder_inputs[request_id] = tuple(encoder_inputs)
                            candidate_encoder_budget = next_encoder_budget
                        if request.has_encoder_inputs:
                            candidate_encoder_overlay = next_encoder_overlay
                        candidate_budget -= num_new_tokens
                        effective_prefix_size += 1
                        if candidate_budget <= 0:
                            break
                    if effective_prefix_size == 0:
                        continue
                    candidate_is_pure_decode = self._is_pure_decode_step(
                        candidate_tokens,
                        candidate_drafts,
                        computed_token_overrides=candidate_overrides,
                    )
                    candidate_k = self._num_spec_tokens_for_step(
                        candidate_tokens, candidate_is_pure_decode
                    )
                    candidate_key = self._canonical_elastic_graph_step_key(
                        candidate_tokens, candidate_k, candidate_is_pure_decode
                    )
                    assert candidate_key is not None
                if effective_prefix_size in inspected_effective_prefixes:
                    continue
                inspected_effective_prefixes.add(effective_prefix_size)
                candidate_requirements = self._elastic_remaining_resource_requirements(
                    candidate_tokens
                )
                candidate_minimum_primary = (
                    self._elastic_successor_primary_headroom(
                        candidate_tokens,
                        computed_token_overrides=candidate_overrides,
                    )
                    + candidate_requirements.primary
                )
                candidate_fits, _required, _available = (
                    self._can_fund_elastic_graph_step(
                        candidate_key,
                        minimum_free_primary_blocks=candidate_minimum_primary,
                        gdn_blocks=(
                            self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                candidate_requirements.mamba
                            )
                        ),
                        allow_maintenance=True,
                        preview_only=True,
                        **(
                            cast(
                                dict[str, Any],
                                {
                                    "mm_activation_loan_bytes": (
                                        self._elastic_mm_activation_loan_bytes
                                    )
                                }
                                if candidate_encoder_inputs
                                else {},
                            )
                        ),
                    )
                )
                if candidate_fits:
                    selected_prefix = effective_prefix_size
                    selected_state = (
                        candidate_tokens,
                        candidate_drafts,
                        candidate_encoder_inputs,
                        candidate_overrides,
                    )
                    break
                if not candidate_encoder_inputs:
                    rejected_joint_graphs.append((candidate_key, _required, _available))
                if self._drop_elastic_prefix_hits_for_pressure():
                    clear_joint_identity()
                    return self._preflight_elastic_running_text_wave(
                        token_budget=token_budget,
                        prefill_chunk_cap=prefill_chunk_cap,
                        defer_prefills=defer_prefills,
                        physical_quiescent=physical_quiescent,
                    )
            if selected_state is None:
                return resolve_failed_joint()
            if preview_state != (
                self._elastic_admission_controller.snapshot,
                self.kv_cache_manager.coordinator.elastic_external_memory_bytes,
            ):
                raise RuntimeError("elastic FCFS prefix preview mutated runtime state")
            (
                prospective_tokens,
                prospective_drafts,
                prospective_encoder_inputs,
                computed_overrides,
            ) = selected_state
            joint_waiting_count = selected_prefix
            waiting_candidates = cast(Sequence[Request], waiting_candidates)[
                :selected_prefix
            ]
            self._elastic_preflight_joint_waiting_request_ids = waiting_ids[
                :selected_prefix
            ]
            self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
            selected_joint_prefix = True

        if not prospective_tokens:
            return None, False
        is_pure_decode = self._is_pure_decode_step(
            prospective_tokens,
            prospective_drafts,
            computed_token_overrides=computed_overrides,
        )
        num_spec_tokens = self._num_spec_tokens_for_step(
            prospective_tokens,
            is_pure_decode,
        )
        final_key = self._canonical_elastic_graph_step_key(
            prospective_tokens,
            num_spec_tokens,
            is_pure_decode,
        )
        assert final_key is not None
        if not self._deferred_mm_wave_matches_candidate(
            step_key=final_key,
            scheduled_tokens=prospective_tokens,
            scheduled_encoder_inputs=prospective_encoder_inputs,
        ):
            return self._preflight_elastic_running_text_wave(
                token_budget=token_budget,
                prefill_chunk_cap=prefill_chunk_cap,
                defer_prefills=defer_prefills,
                physical_quiescent=physical_quiescent,
            )
        remaining_requirements = self._elastic_remaining_resource_requirements(
            prospective_tokens
        )
        minimum_free_primary_blocks = (
            self._elastic_successor_primary_headroom(
                prospective_tokens,
                computed_token_overrides=computed_overrides,
            )
            + remaining_requirements.primary
        )
        fits, required_external, available_external = self._can_fund_elastic_graph_step(
            final_key,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            gdn_blocks=(
                self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                    remaining_requirements.mamba
                )
            ),
            allow_maintenance=True,
            **(
                cast(
                    dict[str, Any],
                    {"mm_activation_loan_bytes": self._elastic_mm_activation_loan_bytes}
                    if prospective_encoder_inputs
                    else {},
                )
            ),
        )
        if fits:
            if self._reserve_elastic_admission(
                final_key,
                external_memory_bytes=required_external,
                minimum_free_primary_blocks=minimum_free_primary_blocks,
                requirements=remaining_requirements,
            ):
                return final_key, False
            if selected_joint_prefix:
                raise RuntimeError(
                    "elastic FCFS prefix changed between preview and reservation"
                )
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            pending_plan = self._elastic_admission_controller.pending_maintenance_plan
            if (
                pending_plan.maintenance_execution
                == ElasticMaintenanceExecution.GRAPH_ONLY
            ):
                self._arm_deferred_mm_wave(
                    step_key=final_key,
                    scheduled_tokens=prospective_tokens,
                    scheduled_encoder_inputs=prospective_encoder_inputs,
                )
                return final_key, True
            if (
                joint_waiting_count
                and any(computed_overrides.values())
                and not getattr(self, "_elastic_prefix_hits", {})
            ):
                # Prefix-cache peeks for new arrivals are not jointly leased
                # with the running cohort. If the optimistic combined shape is
                # COLD, bind the same full request cohort to a conservative
                # no-prefix view rather than capturing an unstable owner set.
                self._clear_pre_mutation_serving_maintenance(
                    reason="prefix_optimistic_joint_replanned"
                )
                prospective_tokens = dict(running_wave_tokens)
                prospective_drafts = dict(running_wave_drafts)
                prospective_encoder_inputs = dict(running_wave_encoder_inputs)
                computed_overrides = {}
                remaining_budget = running_wave_remaining_budget
                remaining_encoder_budget = running_wave_remaining_encoder_budget
                encoder_wave_overlay = running_wave_encoder_overlay.clone()
                conservative_waiting_ids: list[str] = []
                available_slots = max(
                    0,
                    self.max_num_running_reqs
                    - len(self.running)
                    - self.num_waiting_for_streaming_input,
                )
                for request in waiting_candidates or ():
                    if len(conservative_waiting_ids) >= available_slots:
                        break
                    num_new_tokens = self._cap_prefill_chunk(
                        request, request.num_tokens, prefill_chunk_cap
                    )
                    long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
                    if 0 < long_prefill_chunk_cap < num_new_tokens:
                        num_new_tokens = long_prefill_chunk_cap
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > remaining_budget
                    ):
                        break
                    num_new_tokens = min(num_new_tokens, remaining_budget)
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            0,
                            0,
                            shared_prefix_boundary=0,
                        )
                    encoder_inputs = None
                    candidate_encoder_budget = remaining_encoder_budget
                    candidate_encoder_overlay = encoder_wave_overlay
                    if request.has_encoder_inputs:
                        candidate_encoder_overlay = encoder_wave_overlay.clone()
                        (
                            encoder_inputs,
                            num_new_tokens,
                            candidate_encoder_budget,
                            _external_encoder_inputs,
                            _cached_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            0,
                            num_new_tokens,
                            remaining_encoder_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                            encoder_wave_overlay=candidate_encoder_overlay,
                        )
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, 0, num_new_tokens
                    )
                    if num_new_tokens <= 0:
                        break
                    prospective_tokens[request.request_id] = num_new_tokens
                    if encoder_inputs:
                        prospective_encoder_inputs[request.request_id] = tuple(
                            encoder_inputs
                        )
                        remaining_encoder_budget = candidate_encoder_budget
                    if request.has_encoder_inputs:
                        encoder_wave_overlay = candidate_encoder_overlay
                    computed_overrides[request.request_id] = 0
                    conservative_waiting_ids.append(request.request_id)
                    remaining_budget -= num_new_tokens
                    if remaining_budget <= 0:
                        break
                is_pure_decode = self._is_pure_decode_step(
                    prospective_tokens,
                    prospective_drafts,
                    computed_token_overrides=computed_overrides,
                )
                num_spec_tokens = self._num_spec_tokens_for_step(
                    prospective_tokens, is_pure_decode
                )
                final_key = self._canonical_elastic_graph_step_key(
                    prospective_tokens, num_spec_tokens, is_pure_decode
                )
                remaining_requirements = self._elastic_remaining_resource_requirements(
                    prospective_tokens
                )
                minimum_free_primary_blocks = (
                    self._elastic_successor_primary_headroom(
                        prospective_tokens,
                        computed_token_overrides=computed_overrides,
                    )
                    + remaining_requirements.primary
                )
                fits, required_external, available_external = (
                    self._can_fund_elastic_graph_step(
                        final_key,
                        minimum_free_primary_blocks=minimum_free_primary_blocks,
                        gdn_blocks=(
                            self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                remaining_requirements.mamba
                            )
                        ),
                        allow_maintenance=True,
                        **(
                            cast(
                                dict[str, Any],
                                {
                                    "mm_activation_loan_bytes": (
                                        self._elastic_mm_activation_loan_bytes
                                    )
                                }
                                if prospective_encoder_inputs
                                else {},
                            )
                        ),
                    )
                )
                conservative_ids = tuple(conservative_waiting_ids)
                self._elastic_preflight_joint_waiting_request_ids = conservative_ids
                self._elastic_preflight_waiting_ignore_prefix_request_ids = (
                    conservative_ids
                )
                if fits and self._reserve_elastic_admission(
                    cast(tuple[int, ...], final_key),
                    external_memory_bytes=required_external,
                    minimum_free_primary_blocks=minimum_free_primary_blocks,
                    requirements=remaining_requirements,
                ):
                    return final_key, False
                if (
                    self._elastic_admission_controller.pending_maintenance_plan
                    is not None
                    and self._reserve_pending_elastic_maintenance_admission(
                        cast(tuple[int, ...], final_key),
                        minimum_free_primary_blocks=minimum_free_primary_blocks,
                        requirements=remaining_requirements,
                    )
                ):
                    return final_key, True
                if selected_joint_prefix:
                    raise RuntimeError(
                        "elastic FCFS conservative prefix changed during reservation"
                    )
                self._elastic_preflight_joint_waiting_request_ids = ()
                self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
                return resolve_failed_joint()
            if self._reserve_pending_elastic_maintenance_admission(
                final_key,
                minimum_free_primary_blocks=minimum_free_primary_blocks,
                requirements=remaining_requirements,
            ):
                return final_key, True
            if selected_joint_prefix:
                # The preview is deliberately read-only, so a later commit
                # refusal is a stale resource epoch rather than corruption.
                # Drop the optional WAITING expansion and preserve liveness
                # through the already-visible RUNNING cohort.
                return resolve_failed_joint()
            if joint_waiting_count:
                return resolve_failed_joint()
            return None, False
        if joint_waiting_count:
            return resolve_failed_joint()
        self._observe_elastic_running_wave_defer(
            final_key,
            required_external=required_external,
            available_external=available_external,
            physical_quiescent=physical_quiescent,
        )
        if deferred_mm_wave is not None:
            self._elastic_deferred_mm_wave = None
            clear_joint_identity()
            self._rollback_elastic_admission()
            return self._preflight_elastic_running_text_wave(
                token_budget=token_budget,
                prefill_chunk_cap=prefill_chunk_cap,
                defer_prefills=defer_prefills,
                physical_quiescent=physical_quiescent,
            )
        if not prospective_encoder_inputs and self._prepare_elastic_cold_form_reclaim(
            final_key,
            required_external=required_external,
            available_external=available_external,
            physical_quiescent=physical_quiescent,
        ):
            return None, True
        return None, False

    def _observe_elastic_running_wave_defer(
        self,
        step_key: tuple[int, ...],
        *,
        required_external: int,
        available_external: int,
        physical_quiescent: bool,
    ) -> None:
        """Report the same deduplicated funding boundary on both RUNNING paths."""
        defer_identity = (
            step_key,
            getattr(self, "_elastic_last_defer_reason", None),
            required_external,
            available_external,
        )
        if defer_identity != getattr(self, "_elastic_last_wave_defer", None):
            self._elastic_last_wave_defer = defer_identity
            logger.warning(
                "Complete RUNNING wave deferred before KV mutation: "
                "step_key=%r reason=%s required_bytes=%d available_bytes=%d "
                "resident_bytes=%d floor_bytes=%d pending_loans=%d "
                "physical_quiescent=%s retained_carrier=%r",
                *defer_identity,
                self._elastic_admission_controller.resident_bytes,
                self._elastic_admission_controller.floor_bytes,
                len(self._elastic_admission_controller.pending_loans),
                physical_quiescent,
                getattr(self, "_elastic_graph_carrier_step_key", None),
            )

    def _prepare_elastic_cold_form_reclaim(
        self,
        step_key: tuple[int, ...],
        *,
        required_external: int,
        available_external: int,
        physical_quiescent: bool,
    ) -> bool:
        """Retire optional residency when it prevents a priced text wave.

        This is a request-free transaction, not a shape fallback. The next
        tick reprices the same work against the worker's post-reclaim receipt.
        """
        controller = self._elastic_admission_controller
        if (
            not physical_quiescent
            or getattr(self, "_elastic_restore_mode", False)
            or controller.pending_loans
            or controller.pending_maintenance_plan is not None
            or required_external <= available_external
        ):
            return False
        destination, cold = self._estimate_elastic_graph_step_bytes(step_key)
        if not cold or destination <= 0:
            return False
        # Do not assume allocator slack or future KV compaction. The same
        # pinned-KV capacity must fund the destination and mandatory carrier.
        # Only the protected destination intersection survives this teardown;
        # optional victims cannot contribute shared bytes to the next capture.
        protected = getattr(self, "_elastic_serving_carrier_keys", ())
        shared = self._elastic_capture_shared_resident_bytes(
            protected,
            self._elastic_capture_shared_evidence(step_key, destination),
        )
        post_reclaim_bound = controller.compose_destination_capture_loan(
            current_residency_bytes=self._elastic_pressure_floor_external_bytes(),
            destination_capture_endpoint_bytes=destination,
            shared_resident_bytes=shared,
        )
        if post_reclaim_bound > available_external:
            return False
        reclaim = controller.plan_pressure_reclaim_all(
            self._next_elastic_transaction_id(),
            request_bytes=controller.resident_bytes,
            available_bytes=available_external,
            protected_keys=protected,
        )
        if reclaim.kind != ElasticPlanKind.PRESSURE_RECLAIM:
            return False
        controller.arm_maintenance(reclaim, None)
        logger.info(
            "Elastic cold form recovery: step_key=%r post_reclaim_bound=%d "
            "available_bytes=%d victims=%d protected=%d",
            step_key,
            post_reclaim_bound,
            available_external,
            len(reclaim.victim_keys),
            len(reclaim.protected_keys),
        )
        return True

    def _preflight_elastic_waiting_text_wave(
        self,
        *,
        token_budget: int,
        prefill_chunk_cap: int,
        defer_prefills: bool,
        physical_quiescent: bool = True,
    ) -> tuple[tuple[int, ...] | None, bool, tuple[str, ...]]:
        """Price one complete idle admission wave before its prefixes.

        This is intentionally narrower than the general waiting scheduler: a
        connector, LoRA conflict, stale output, blocked status, or KV-event
        side effect remains fail-closed. Encoder intent is planned read-only;
        a COLD graph is captured in a separate transaction before MM. The supported
        product lane leases the local prefix cache, derives every candidate's
        exact token contribution, and prepares at most one final owner set.
        No request status or request block-table changes occur here. Temporary
        cache references survive through commit and release in schedule's finally.
        Ready structured grammars are viewed semantically as WAITING while
        remaining in ``skipped_waiting`` until the normal commit loop promotes
        them. The returned IDs bind the reservation to that exact readiness
        snapshot, so a newly-ready request cannot substitute a different shape.
        """
        if (
            not self.elastic_on_demand_graphs
            or getattr(self, "_elastic_restore_mode", False)
            or token_budget <= 0
            or self._pending_elastic_maintenance_requires_exclusive_tick()
            or self.running
            or not (self.waiting or self.skipped_waiting)
            or self.policy != SchedulingPolicy.FCFS
            or self.num_waiting_for_streaming_input
            or defer_prefills
            or self.connector is not None
            or self.ec_connector is not None
            or self.lora_config is not None
            or self.kv_cache_manager.enable_kv_cache_events
        ):
            return None, False, ()

        prospective_tokens: dict[str, int] = {}
        prospective_encoder_inputs: dict[str, tuple[int, ...]] = {}
        computed_overrides: dict[str, int] = {}
        remaining_budget = token_budget
        remaining_encoder_budget = self.max_num_encoder_input_tokens
        encoder_cache_manager = getattr(self, "encoder_cache_manager", None)
        encoder_wave_overlay = EncoderWaveOverlay(
            encoder_cache_manager.clone_for_preview()
            if encoder_cache_manager is not None
            else EncoderCacheManager(0)
        )
        deferred_mm_wave = self._validated_deferred_mm_wave()
        bound_waiting_ids = (
            frozenset(deferred_mm_wave.waiting_request_ids)
            if deferred_mm_wave is not None
            else None
        )
        waiting_candidates = self._elastic_schedulable_waiting_snapshot(
            allowed_request_ids=bound_waiting_ids
        )
        if waiting_candidates is None:
            return None, False, ()
        for request in waiting_candidates:
            if len(prospective_tokens) >= self.max_num_running_reqs:
                break
            if request.num_computed_tokens != 0 or request.num_stale_output_tokens > 0:
                return None, False, ()
            _, num_computed_tokens, shared_boundary, _ = (
                self._lease_local_prefix_cache_hit(request)
            )
            num_new_tokens = request.num_tokens - num_computed_tokens
            num_new_tokens = self._cap_prefill_chunk(
                request, num_new_tokens, prefill_chunk_cap
            )
            long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
            if 0 < long_prefill_chunk_cap < num_new_tokens:
                num_new_tokens = long_prefill_chunk_cap
            if (
                not self.scheduler_config.enable_chunked_prefill
                and num_new_tokens > remaining_budget
            ):
                break
            num_new_tokens = min(num_new_tokens, remaining_budget)
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    num_computed_tokens,
                    0,
                    shared_prefix_boundary=self._resolved_shared_prefix_boundary(
                        request, shared_boundary
                    ),
                )
            encoder_inputs: Sequence[int] | None = None
            candidate_encoder_budget = remaining_encoder_budget
            candidate_encoder_overlay = encoder_wave_overlay
            if request.has_encoder_inputs:
                candidate_encoder_overlay = encoder_wave_overlay.clone()
                (
                    encoder_inputs,
                    num_new_tokens,
                    candidate_encoder_budget,
                    _external_encoder_inputs,
                    _cached_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    num_computed_tokens,
                    num_new_tokens,
                    remaining_encoder_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                    encoder_wave_overlay=candidate_encoder_overlay,
                )
            num_new_tokens = self._reserve_prefill_lookahead(
                request, num_computed_tokens, num_new_tokens
            )
            if num_new_tokens <= 0:
                return None, False, ()
            prospective_tokens[request.request_id] = num_new_tokens
            if encoder_inputs:
                prospective_encoder_inputs[request.request_id] = tuple(encoder_inputs)
                remaining_encoder_budget = candidate_encoder_budget
            if request.has_encoder_inputs:
                encoder_wave_overlay = candidate_encoder_overlay
            computed_overrides[request.request_id] = num_computed_tokens
            remaining_budget -= num_new_tokens
            if remaining_budget <= 0:
                break

        self._release_elastic_prefix_hits(prospective_tokens)
        if not prospective_tokens:
            return None, False, ()
        selected_waiting_prefix = False
        if deferred_mm_wave is None:
            # Idle admission follows the same FCFS contract as a mixed
            # RUNNING+WAITING wave: execute the largest feasible prefix and
            # leave only the suffix WAITING. Graph residency is non-monotonic
            # in X, so every prefix is inspected from largest to smallest.
            waiting_ids = tuple(prospective_tokens)
            selected_state: (
                tuple[
                    dict[str, int],
                    dict[str, tuple[int, ...]],
                    dict[str, int],
                ]
                | None
            ) = None
            preview_state = (
                self._elastic_admission_controller.snapshot,
                self.kv_cache_manager.coordinator.elastic_external_memory_bytes,
            )
            for prefix_size in range(len(waiting_ids), 0, -1):
                prefix_ids = waiting_ids[:prefix_size]
                self._release_elastic_prefix_hits(prefix_ids)
                candidate_tokens = {
                    request_id: prospective_tokens[request_id]
                    for request_id in prefix_ids
                }
                candidate_encoder_inputs = {
                    request_id: prospective_encoder_inputs[request_id]
                    for request_id in prefix_ids
                    if request_id in prospective_encoder_inputs
                }
                candidate_overrides = {
                    request_id: computed_overrides[request_id]
                    for request_id in prefix_ids
                }
                candidate_is_pure_decode = self._is_pure_decode_step(
                    candidate_tokens,
                    {},
                    computed_token_overrides=candidate_overrides,
                )
                candidate_k = self._num_spec_tokens_for_step(
                    candidate_tokens, candidate_is_pure_decode
                )
                candidate_key = self._canonical_elastic_graph_step_key(
                    candidate_tokens, candidate_k, candidate_is_pure_decode
                )
                assert candidate_key is not None
                candidate_requirements = self._elastic_remaining_resource_requirements(
                    candidate_tokens
                )
                candidate_minimum_primary = (
                    self._elastic_successor_primary_headroom(
                        candidate_tokens,
                        computed_token_overrides=candidate_overrides,
                    )
                    + candidate_requirements.primary
                )
                candidate_fits, _required, _available = (
                    self._can_fund_elastic_graph_step(
                        candidate_key,
                        minimum_free_primary_blocks=candidate_minimum_primary,
                        gdn_blocks=(
                            self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                candidate_requirements.mamba
                            )
                        ),
                        allow_maintenance=True,
                        preview_only=True,
                        **(
                            cast(
                                dict[str, Any],
                                {
                                    "mm_activation_loan_bytes": (
                                        self._elastic_mm_activation_loan_bytes
                                    )
                                }
                                if candidate_encoder_inputs
                                else {},
                            )
                        ),
                    )
                )
                if candidate_fits:
                    selected_state = (
                        candidate_tokens,
                        candidate_encoder_inputs,
                        candidate_overrides,
                    )
                    waiting_candidates = waiting_candidates[:prefix_size]
                    break
                if self._drop_elastic_prefix_hits_for_pressure():
                    return self._preflight_elastic_waiting_text_wave(
                        token_budget=token_budget,
                        prefill_chunk_cap=prefill_chunk_cap,
                        defer_prefills=defer_prefills,
                        physical_quiescent=physical_quiescent,
                    )
            if selected_state is None:
                return None, False, ()
            if preview_state != (
                self._elastic_admission_controller.snapshot,
                self.kv_cache_manager.coordinator.elastic_external_memory_bytes,
            ):
                raise RuntimeError(
                    "elastic idle FCFS prefix preview mutated runtime state"
                )
            (
                prospective_tokens,
                prospective_encoder_inputs,
                computed_overrides,
            ) = selected_state
            selected_waiting_prefix = True
        self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
        is_pure_decode = self._is_pure_decode_step(
            prospective_tokens,
            {},
            computed_token_overrides=computed_overrides,
        )
        num_spec_tokens = self._num_spec_tokens_for_step(
            prospective_tokens,
            is_pure_decode,
        )
        final_key = self._canonical_elastic_graph_step_key(
            prospective_tokens,
            num_spec_tokens,
            is_pure_decode,
        )
        assert final_key is not None
        if not self._deferred_mm_wave_matches_candidate(
            step_key=final_key,
            scheduled_tokens=prospective_tokens,
            scheduled_encoder_inputs=prospective_encoder_inputs,
        ):
            return self._preflight_elastic_waiting_text_wave(
                token_budget=token_budget,
                prefill_chunk_cap=prefill_chunk_cap,
                defer_prefills=defer_prefills,
                physical_quiescent=physical_quiescent,
            )
        remaining_requirements = self._elastic_remaining_resource_requirements(
            prospective_tokens
        )
        minimum_free_primary_blocks = (
            self._elastic_successor_primary_headroom(
                prospective_tokens,
                computed_token_overrides=computed_overrides,
            )
            + remaining_requirements.primary
        )
        fits, required_external, available_external = self._can_fund_elastic_graph_step(
            final_key,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            gdn_blocks=(
                self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                    remaining_requirements.mamba
                )
            ),
            allow_maintenance=True,
            **(
                cast(
                    dict[str, Any],
                    {"mm_activation_loan_bytes": self._elastic_mm_activation_loan_bytes}
                    if prospective_encoder_inputs
                    else {},
                )
            ),
        )
        if fits:
            if self._reserve_elastic_admission(
                final_key,
                external_memory_bytes=required_external,
                minimum_free_primary_blocks=minimum_free_primary_blocks,
                requirements=remaining_requirements,
            ):
                return final_key, False, tuple(prospective_tokens)
            if deferred_mm_wave is not None:
                self._elastic_deferred_mm_wave = None
                self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
                self._rollback_elastic_admission()
                return self._preflight_elastic_waiting_text_wave(
                    token_budget=token_budget,
                    prefill_chunk_cap=prefill_chunk_cap,
                    defer_prefills=defer_prefills,
                    physical_quiescent=physical_quiescent,
                )
            if selected_waiting_prefix:
                raise RuntimeError(
                    "elastic idle FCFS prefix changed between preview and reservation"
                )
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            pending_plan = self._elastic_admission_controller.pending_maintenance_plan
            if (
                pending_plan.maintenance_execution
                == ElasticMaintenanceExecution.GRAPH_ONLY
            ):
                self._arm_deferred_mm_wave(
                    step_key=final_key,
                    scheduled_tokens=prospective_tokens,
                    scheduled_encoder_inputs=prospective_encoder_inputs,
                )
                return final_key, True, tuple(prospective_tokens)
            if any(computed_overrides.values()) and not getattr(
                self, "_elastic_prefix_hits", {}
            ):
                # Independent prefix-cache peeks can overbook the same
                # reclaimable blocks across a joint waiting wave. The normal
                # allocator mutates ownership request by request and may then
                # commit a radically different prefill shape. For a COLD graph
                # transition only, reprice the exact wave without speculative
                # prefix adoption and bind normal admission to that view.
                self._clear_pre_mutation_serving_maintenance(
                    reason="prefix_optimistic_waiting_replanned"
                )
                prospective_tokens = {}
                prospective_encoder_inputs = {}
                computed_overrides = {}
                remaining_budget = token_budget
                remaining_encoder_budget = self.max_num_encoder_input_tokens
                encoder_wave_overlay = EncoderWaveOverlay(
                    encoder_cache_manager.clone_for_preview()
                    if encoder_cache_manager is not None
                    else EncoderCacheManager(0)
                )
                for request in waiting_candidates:
                    if len(prospective_tokens) >= self.max_num_running_reqs:
                        break
                    num_new_tokens = self._cap_prefill_chunk(
                        request, request.num_tokens, prefill_chunk_cap
                    )
                    long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
                    if 0 < long_prefill_chunk_cap < num_new_tokens:
                        num_new_tokens = long_prefill_chunk_cap
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > remaining_budget
                    ):
                        break
                    num_new_tokens = min(num_new_tokens, remaining_budget)
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            0,
                            0,
                            shared_prefix_boundary=0,
                        )
                    encoder_inputs = None
                    candidate_encoder_budget = remaining_encoder_budget
                    candidate_encoder_overlay = encoder_wave_overlay
                    if request.has_encoder_inputs:
                        candidate_encoder_overlay = encoder_wave_overlay.clone()
                        (
                            encoder_inputs,
                            num_new_tokens,
                            candidate_encoder_budget,
                            _external_encoder_inputs,
                            _cached_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            0,
                            num_new_tokens,
                            remaining_encoder_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                            encoder_wave_overlay=candidate_encoder_overlay,
                        )
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, 0, num_new_tokens
                    )
                    if num_new_tokens <= 0:
                        break
                    prospective_tokens[request.request_id] = num_new_tokens
                    if encoder_inputs:
                        prospective_encoder_inputs[request.request_id] = tuple(
                            encoder_inputs
                        )
                        remaining_encoder_budget = candidate_encoder_budget
                    if request.has_encoder_inputs:
                        encoder_wave_overlay = candidate_encoder_overlay
                    computed_overrides[request.request_id] = 0
                    remaining_budget -= num_new_tokens
                    if remaining_budget <= 0:
                        break
                if not prospective_tokens:
                    return None, False, ()
                is_pure_decode = self._is_pure_decode_step(
                    prospective_tokens,
                    {},
                    computed_token_overrides=computed_overrides,
                )
                num_spec_tokens = self._num_spec_tokens_for_step(
                    prospective_tokens, is_pure_decode
                )
                final_key = self._canonical_elastic_graph_step_key(
                    prospective_tokens, num_spec_tokens, is_pure_decode
                )
                remaining_requirements = self._elastic_remaining_resource_requirements(
                    prospective_tokens
                )
                minimum_free_primary_blocks = (
                    self._elastic_successor_primary_headroom(
                        prospective_tokens,
                        computed_token_overrides=computed_overrides,
                    )
                    + remaining_requirements.primary
                )
                fits, required_external, available_external = (
                    self._can_fund_elastic_graph_step(
                        final_key,
                        minimum_free_primary_blocks=minimum_free_primary_blocks,
                        gdn_blocks=(
                            self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                remaining_requirements.mamba
                            )
                        ),
                        allow_maintenance=True,
                        **(
                            cast(
                                dict[str, Any],
                                {
                                    "mm_activation_loan_bytes": (
                                        self._elastic_mm_activation_loan_bytes
                                    )
                                }
                                if prospective_encoder_inputs
                                else {},
                            )
                        ),
                    )
                )
                request_ids = tuple(prospective_tokens)
                if fits and self._reserve_elastic_admission(
                    cast(tuple[int, ...], final_key),
                    external_memory_bytes=required_external,
                    minimum_free_primary_blocks=minimum_free_primary_blocks,
                    requirements=remaining_requirements,
                ):
                    self._elastic_preflight_waiting_ignore_prefix_request_ids = (
                        request_ids
                    )
                    return final_key, False, request_ids
                if (
                    self._elastic_admission_controller.pending_maintenance_plan
                    is not None
                ):
                    pending_plan = (
                        self._elastic_admission_controller.pending_maintenance_plan
                    )
                    assert pending_plan is not None
                    if (
                        pending_plan.maintenance_execution
                        == ElasticMaintenanceExecution.GRAPH_ONLY
                    ):
                        self._arm_deferred_mm_wave(
                            step_key=cast(tuple[int, ...], final_key),
                            scheduled_tokens=prospective_tokens,
                            scheduled_encoder_inputs=prospective_encoder_inputs,
                        )
                        self._elastic_preflight_waiting_ignore_prefix_request_ids = (
                            request_ids
                        )
                        return final_key, True, request_ids
                    if self._reserve_pending_elastic_maintenance_admission(
                        cast(tuple[int, ...], final_key),
                        minimum_free_primary_blocks=minimum_free_primary_blocks,
                        requirements=remaining_requirements,
                    ):
                        self._elastic_preflight_waiting_ignore_prefix_request_ids = (
                            request_ids
                        )
                        return final_key, True, request_ids
                return None, False, ()
            if self._reserve_pending_elastic_maintenance_admission(
                final_key,
                minimum_free_primary_blocks=minimum_free_primary_blocks,
                requirements=remaining_requirements,
            ):
                return final_key, True, tuple(prospective_tokens)
            if deferred_mm_wave is not None:
                self._elastic_deferred_mm_wave = None
                self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
                self._clear_pre_mutation_serving_maintenance(
                    reason="deferred_mm_waiting_receipt_changed"
                )
                return self._preflight_elastic_waiting_text_wave(
                    token_budget=token_budget,
                    prefill_chunk_cap=prefill_chunk_cap,
                    defer_prefills=defer_prefills,
                    physical_quiescent=physical_quiescent,
                )
            if selected_waiting_prefix:
                raise RuntimeError(
                    "elastic idle FCFS cold prefix changed during reservation"
                )
            return None, False, ()
        logger.debug(
            "Complete WAITING text wave deferred before KV mutation: "
            "required_bytes=%d available_bytes=%d step_key=%r reason=%s",
            required_external,
            available_external,
            final_key,
            getattr(self, "_elastic_last_defer_reason", None),
        )
        if deferred_mm_wave is not None:
            self._elastic_deferred_mm_wave = None
            self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
            self._rollback_elastic_admission()
            return self._preflight_elastic_waiting_text_wave(
                token_budget=token_budget,
                prefill_chunk_cap=prefill_chunk_cap,
                defer_prefills=defer_prefills,
                physical_quiescent=physical_quiescent,
            )
        if selected_waiting_prefix:
            raise RuntimeError("elastic idle FCFS prefix became unfundable")
        return None, False, ()

    def _elastic_schedulable_waiting_snapshot(
        self,
        *,
        allowed_request_ids: frozenset[str] | None = None,
    ) -> tuple[Request, ...] | None:
        """Return the exact FCFS text candidates schedulable in this tick.

        Structured grammar readiness is observed without moving queue entries.
        Other blocked states are skipped exactly as the normal waiting loop
        skips them. Unsupported ready state fails closed instead of pricing a
        wave different from the one the commit loop can assemble.
        """
        if self.policy != SchedulingPolicy.FCFS:
            return None
        candidates: list[Request] = []
        for request in itertools.chain(self.skipped_waiting, self.waiting):
            if (
                allowed_request_ids is not None
                and request.request_id not in allowed_request_ids
            ):
                continue
            status = request.status
            if status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                structured = request.structured_output_request
                grammar = structured.grammar if structured is not None else None
                if grammar is None:
                    continue
                if isinstance(grammar, Exception):
                    self.grammar_compile_error_reqs.add(request.request_id)
                    continue
                status = RequestStatus.WAITING
            elif self._is_blocked_waiting_status(status):
                continue
            if status != RequestStatus.WAITING:
                return None
            if request.num_computed_tokens != 0 or request.num_stale_output_tokens > 0:
                return None
            candidates.append(request)
        return tuple(candidates)

    def schedule(
        self,
        throttle_prefills: bool = False,
        *,
        physical_quiescent: bool = False,
    ) -> SchedulerOutput:
        try:
            # Re-offer the soft grant to admission. Only the final output
            # below changes worker mappings, so a stable grant is a no-op.
            if getattr(self, "_elastic_native_budget", None) is not None and not (
                self.kv_cache_manager.coordinator.set_elastic_expert_memory(0)
            ):
                raise RuntimeError("could not reclaim the logical expert grant")
            self._refresh_elastic_cache_frontier()
            return self._schedule_with_prefix_leases(
                throttle_prefills, physical_quiescent=physical_quiescent
            )
        finally:
            # Admitted requests acquired their own references in allocate_slots.
            # Failure, cancellation and maintenance-only steps retain none.
            self._release_elastic_prefix_hits()
            self._elastic_prefix_pressure_fallback = False

    def _plan_elastic_expert_grant(
        self, *, has_user_tokens: bool, minimum_free_primary_blocks: int
    ) -> ElasticExpertGrant | None:
        budget = getattr(self, "_elastic_native_budget", None)
        if budget is None:
            return None
        coordinator = self.kv_cache_manager.coordinator
        available = tuple(
            max(value - coordinator.elastic_external_memory_bytes, 0)
            for value in coordinator.max_elastic_external_memory_by_rank(
                minimum_free_primary_blocks=minimum_free_primary_blocks
            )
        )
        max_rows = None if has_user_tokens else self._elastic_native_hot_rows
        ranked = len(available) == budget.geometry.tp
        grant = (
            budget.fit_by_rank(available, max_rows=max_rows)
            if ranked
            else budget.fit(min(available), max_rows=max_rows)
        )
        if not coordinator.set_elastic_expert_memory(
            grant.borrowed_bytes,
            minimum_free_primary_blocks,
            rank_requested_bytes=budget.rank_borrowed_bytes(grant.hot_rows)
            if ranked
            else (),
        ):
            raise RuntimeError("expert grant disagrees with admitted free KV tail")
        self._elastic_native_hot_rows = grant.hot_rows
        return grant

    def _refresh_elastic_cache_frontier(self) -> None:
        """Price cache-preserving holes against future executable residency."""
        if not getattr(self, "elastic_on_demand_graphs", False):
            return
        coordinator = self.kv_cache_manager.coordinator
        if not isinstance(coordinator, HybridKVCacheCoordinator):
            return
        config = coordinator.kv_cache_config
        if not config.elastic_mapping_quantum or coordinator.mamba_block_pool is None:
            return
        catalog = getattr(self, "_elastic_graph_catalog", {})
        coverage = getattr(self, "_elastic_graph_catalog_coverage", {})
        pool = coordinator.block_pool
        source_digest = getattr(catalog, "source_sha256", None)
        if (
            getattr(self, "_elastic_restore_mode", False)
            or not catalog
            or not isinstance(source_digest, str)
            or len(source_digest) != 64
            or coverage.get("_catalog_source_sha256") != source_digest
        ):
            pool.cache_preservation_num_blocks = 1
            return
        max_gdn = max(
            config.elastic_gdn_initial_blocks,
            1 + self.max_num_running_reqs * config.elastic_gdn_blocks_per_request,
        )
        cold = max(
            _elastic_catalog_cold_residency_envelope(row) for row in catalog.values()
        )
        hot = max(int(row["hot_peak_bytes"]) for row in catalog.values())
        # Capture and vision execute in separate phases. Optional Graph owners
        # may be reclaimed; the mandatory carrier and physical floor may not.
        external = self._elastic_pressure_floor_external_bytes() + max(
            cold, hot + self._elastic_mm_activation_loan_bytes
        )
        if not 1 <= max_gdn <= coordinator.mamba_block_pool.num_gpu_blocks:
            raise RuntimeError("cache frontier GDN envelope exceeds virtual capacity")
        external = coordinator.normalize_elastic_external_memory(external)
        frontier = max(
            1,
            min(
                pool.num_gpu_blocks,
                coordinator._elastic_attention_capacity(max_gdn, external),
            ),
        )
        identity = (frontier, max_gdn, external)
        if identity != getattr(self, "_elastic_cache_frontier_identity", None):
            logger.info(
                "Elastic cache preservation frontier: blocks=%d max_gdn=%d "
                "post_reclaim_external_bytes=%d; no memory reservation",
                *identity,
            )
            self._elastic_cache_frontier_identity = identity
        pool.cache_preservation_num_blocks = frontier

    def _schedule_with_prefix_leases(
        self,
        throttle_prefills: bool = False,
        *,
        physical_quiescent: bool = False,
    ) -> SchedulerOutput:
        self.current_step += 1
        if not self.running and not self.waiting and not self.skipped_waiting:
            self._elastic_graph_carrier_step_key = None
            self._elastic_last_execution_shape = None
            # A completed GRAPH_ONLY capture remains HOT, but an orphaned USER
            # binding has no request lifecycle left to consume it.
            self._elastic_deferred_mm_wave = None
            pending_plan = self._elastic_admission_controller.pending_maintenance_plan
            if (
                pending_plan is not None
                and not self._pending_elastic_maintenance_requires_exclusive_tick()
            ):
                # A serving capture proposal has no consumer after the last
                # request is cancelled. It is pre-mutation state, so return its
                # tentative KV grant and discard it instead of retaining an
                # orphan plan into the next request epoch.
                self._clear_pre_mutation_serving_maintenance(
                    reason="orphan_maintenance_after_request_cancellation"
                )
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        gdn_checkpoint_restore: dict[str, bytes] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0
        elif self._pending_elastic_maintenance_requires_exclusive_tick():
            # Startup calibration/restore and explicit reclaim have no USER
            # consumer and therefore retain a request-free transaction. A
            # serving COLD plan must instead be rebound to an exact USER wave
            # below so capture and first replay share one resource commit.
            token_budget = 0
        prefill_target_count = self._partial_prefill_target_count()
        prefill_chunk_cap = self._partial_prefill_chunk_cap(prefill_target_count)

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether any admitted request consumes a synchronous connector load.
        has_sync_kv_loads = False
        # For logging.
        scheduled_timestamp = time.monotonic()
        calibration_execution_step_key = getattr(
            self, "_elastic_restore_execution_step_key", None
        )

        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        elastic_preplanned_running_step_key, _prepared_running_maintenance = (
            self._preflight_elastic_running_text_wave(
                token_budget=token_budget,
                prefill_chunk_cap=prefill_chunk_cap,
                defer_prefills=defer_prefills,
                physical_quiescent=physical_quiescent,
            )
        )
        deferred_mm_wave_for_commit = self._elastic_deferred_mm_wave
        bound_running_request_ids = (
            frozenset(deferred_mm_wave_for_commit.running_request_ids)
            if deferred_mm_wave_for_commit is not None
            else None
        )
        if _prepared_running_maintenance and (
            self._elastic_admission_controller.pending_maintenance_plan is None
            or self._pending_elastic_maintenance_requires_exclusive_tick()
        ):
            token_budget = 0
        serving_pending_maintenance = bool(
            self._elastic_admission_controller.pending_maintenance_plan is not None
            and not self._pending_elastic_maintenance_requires_exclusive_tick()
        )
        if (
            serving_pending_maintenance
            and self.running
            and elastic_preplanned_running_step_key is None
        ):
            # The pending owner set is not authority to mutate a different or
            # not-yet-observable wave. Preserve it until exact preflight can
            # bind and reserve the first USER consumer.
            token_budget = 0
        elastic_preplanned_waiting_step_key: tuple[int, ...] | None = None
        elastic_preplanned_waiting_request_ids: tuple[str, ...] = ()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if input_budget <= draft_slots:
                break
            if bound_running_request_ids and bound_running_request_ids.issubset(
                num_scheduled_tokens
            ):
                break
            request = self.running[req_index]
            if (
                bound_running_request_ids is not None
                and request.request_id not in bound_running_request_ids
            ):
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new_tokens = self._cap_prefill_chunk(
                request, num_new_tokens, prefill_chunk_cap
            )
            long_prefill_chunk_cap = self._long_prefill_chunk_cap(request)
            if 0 < long_prefill_chunk_cap < num_new_tokens:
                num_new_tokens = long_prefill_chunk_cap
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Apply recurrent-state boundaries before encoder caps so the
            # encoder preview cannot commit work beyond the executed chunk.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            cached_encoder_inputs: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                    cached_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # A RUNNING request can change the physical owner set at the
            # prefill -> decode boundary.  Waiting admission already performs
            # this preflight, but historically the running loop reached
            # allocate_slots first and discovered a COLD FULL owner only at
            # the user commit boundary.  Price the exact prospective shape
            # before any request/KV/spec mutation.  An empty-prefix miss emits
            # the scheduler-owned maintenance step; a later miss simply stops
            # this batch at the already executable prefix.
            if (
                self.elastic_on_demand_graphs
                and calibration_execution_step_key is None
                and elastic_preplanned_running_step_key is None
            ):
                prospective_tokens = dict(num_scheduled_tokens)
                prospective_tokens[request.request_id] = num_new_tokens
                prospective_drafts = dict(scheduled_spec_decode_tokens)
                if request.spec_token_ids:
                    prospective_spec_count = (
                        num_new_tokens
                        + request.num_computed_tokens
                        - request.num_tokens
                        - request.num_output_placeholders
                    )
                    if prospective_spec_count > 0:
                        prospective_drafts[request.request_id] = request.spec_token_ids[
                            :prospective_spec_count
                        ]
                prospective_is_pure_decode = self._is_pure_decode_step(
                    prospective_tokens,
                    prospective_drafts,
                )
                prospective_k = self._num_spec_tokens_for_step(
                    prospective_tokens,
                    prospective_is_pure_decode,
                )
                prospective_step_key = self._canonical_elastic_graph_step_key(
                    prospective_tokens,
                    prospective_k,
                    prospective_is_pure_decode,
                )
                remaining_requirements = self._elastic_remaining_resource_requirements(
                    prospective_tokens
                )
                graph_fits, required_external, available_external = (
                    self._can_fund_elastic_graph_step(
                        prospective_step_key,
                        minimum_free_primary_blocks=(
                            self._elastic_successor_primary_headroom(prospective_tokens)
                            + remaining_requirements.primary
                        ),
                        gdn_blocks=(
                            self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                remaining_requirements.mamba
                            )
                        ),
                        allow_maintenance=(
                            not num_scheduled_tokens and len(self.running) == 1
                        ),
                        mm_activation_loan_bytes=(
                            self._elastic_mm_activation_loan_bytes
                            if encoder_inputs_to_schedule
                            else 0
                        ),
                    )
                )
                if not graph_fits:
                    if not num_scheduled_tokens:
                        logger.debug(
                            "Minimum RUNNING Graph step deferred: "
                            "required_bytes=%d available_bytes=%d "
                            "step_key=%r reason=%s",
                            required_external,
                            available_external,
                            prospective_step_key,
                            getattr(self, "_elastic_last_defer_reason", None),
                        )
                    if not num_scheduled_tokens:
                        token_budget = 0
                    break

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Under FCFS tail pressure, first try another already-running
                    # prompt instead of immediately destroying this request's KV.
                    # This is safe only while this step has made progress already,
                    # or there is another running request left to try. If every
                    # request is blocked, the last one reaches the normal
                    # preemption path below and preserves the existing liveness
                    # fallback.
                    can_try_running_peer = req_index + 1 < len(self.running)
                    if (
                        self.kv_tail_handoff_enabled
                        and self.policy == SchedulingPolicy.FCFS
                        and self._is_prefill_request(request)
                        and (scheduled_running_reqs or can_try_running_peer)
                    ):
                        self.num_kv_tail_deferrals_since_last_stats += 1
                        req_index += 1
                        break

                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        # Record the index of the preemption victim to
                        # maintain accurate loop state.
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        # Decrement the loop cursor if the removed request
                        # preceded the current iteration, preventing the
                        # silent omission of the subsequent request.
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                if req_index < len(self.running) and request in self.running:
                    # Tail handoff retained this request's KV and advanced to
                    # another running peer. Continue scanning the running set.
                    continue
                # Cannot schedule this request after the liveness fallback.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            self._commit_encoder_cache_plan(
                request,
                cached_encoder_inputs,
                encoder_inputs_to_schedule,
                external_load_encoder_input,
            )
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        calibration_wave_step_key = getattr(
            self, "_elastic_restore_wave_step_key", None
        )
        calibration_wave_stop_reason: str | None = None
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)
            delay_waiting_prefills = self._should_delay_waiting_prefill_admission()
            joint_waiting_request_ids = getattr(
                self, "_elastic_preflight_joint_waiting_request_ids", ()
            )
            if elastic_preplanned_running_step_key is not None:
                if num_scheduled_tokens and joint_waiting_request_ids:
                    elastic_preplanned_waiting_step_key = (
                        elastic_preplanned_running_step_key
                    )
                    elastic_preplanned_waiting_request_ids = joint_waiting_request_ids
                    delay_waiting_prefills = False
                else:
                    # The running reservation cannot be expanded by an
                    # unpriced waiting candidate in this commit.
                    delay_waiting_prefills = True
            if not num_scheduled_tokens and not delay_waiting_prefills:
                (
                    elastic_preplanned_waiting_step_key,
                    _prepared_waiting_maintenance,
                    elastic_preplanned_waiting_request_ids,
                ) = self._preflight_elastic_waiting_text_wave(
                    token_budget=token_budget,
                    prefill_chunk_cap=prefill_chunk_cap,
                    defer_prefills=defer_prefills,
                    physical_quiescent=physical_quiescent,
                )
                # Idle WAITING preflight is the point that may arm, validate,
                # invalidate, or replace a deferred MM USER binding.
                deferred_mm_wave_for_commit = self._elastic_deferred_mm_wave
                if _prepared_waiting_maintenance and (
                    self._elastic_admission_controller.pending_maintenance_plan is None
                    or self._pending_elastic_maintenance_requires_exclusive_tick()
                ):
                    token_budget = 0
                if (
                    serving_pending_maintenance
                    and elastic_preplanned_waiting_step_key is None
                ):
                    token_budget = 0

            while (
                not delay_waiting_prefills
                and (self.waiting or self.skipped_waiting)
                and token_budget > 0
                and input_budget > draft_slots
            ):
                calibration_wave_target = getattr(
                    self, "_elastic_restore_wave_target", 0
                )
                if elastic_preplanned_waiting_request_ids and all(
                    request_id in num_scheduled_tokens
                    for request_id in elastic_preplanned_waiting_request_ids
                ):
                    calibration_wave_stop_reason = "preplanned_ids_complete"
                    break
                if (
                    self._elastic_restore_mode
                    and calibration_wave_target
                    and len(num_scheduled_tokens) >= calibration_wave_target
                ):
                    # ``prepare_elastic_restore_admission`` has committed
                    # one exact finite prefix. Incremental candidate leases can
                    # become less restrictive after that layout transition,
                    # but they may not expand the declared transaction. The
                    # next probe will redistribute the same workload over the
                    # returned prefix and price that exact final shape.
                    calibration_wave_stop_reason = "prepared_target_complete"
                    break
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    calibration_wave_stop_reason = (
                        "resident_cap:"
                        f"running={num_running}:cap={self.max_num_running_reqs}"
                    )
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if (
                    elastic_preplanned_waiting_step_key is not None
                    and request_id not in elastic_preplanned_waiting_request_ids
                ):
                    # Readiness may change after preflight. Never substitute a
                    # new request for one of the exact IDs whose Graph/KV wave
                    # was priced before mutation.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if self._has_prepared_elastic_waiting_admission(
                    elastic_preplanned_waiting_step_key,
                    calibration_wave_target,
                ):
                    # The joint product grant or explicit calibration-wave
                    # grant already mapped the complete KV/GDN/Graph envelope.
                    # A per-candidate wave would replace that atomic layout
                    # with current+lookahead and can underfill the declared
                    # X before the final Graph commit (observed as X43 -> X42
                    # at K3/B4096). Consume the prepared reservation directly.
                    elastic_candidate_lease = 2
                else:
                    elastic_candidate_lease = self._apply_elastic_waiting_candidate(
                        request,
                        token_budget=token_budget,
                        waiting_count=len(self.waiting) + len(self.skipped_waiting),
                        physical_quiescent=(
                            physical_quiescent and not num_scheduled_tokens
                        ),
                    )
                if (
                    self.kv_cache_manager.kv_cache_config.elastic_mapping_quantum
                    and elastic_candidate_lease == 0
                ):
                    calibration_wave_stop_reason = "elastic_candidate_lease_zero"
                    break
                stop_after_elastic_candidate = elastic_candidate_lease == 1

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    ignore_prefix = request_id in (
                        self._elastic_preflight_waiting_ignore_prefix_request_ids
                    )
                    did_prefix_cache_lookup = not ignore_prefix
                    hit_diverged = False
                    # Get locally-cached tokens.
                    if ignore_prefix:
                        new_computed_blocks = (
                            self.kv_cache_manager.empty_kv_cache_blocks
                        )
                        num_new_local_computed_tokens = 0
                        request.shared_prefix_boundary = 0
                    else:
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                            hit_diverged,
                        ) = self._get_local_prefix_cache_hit(request)

                    if not ignore_prefix:
                        request.shared_prefix_boundary = (
                            self._resolved_shared_prefix_boundary(
                                request, request.shared_prefix_boundary
                            )
                        )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
                    if self._ec_transfer_pending(request, num_computed_tokens):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed. A streaming-input
                    # session resumes here too, carrying whatever media its
                    # latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                    if self._ec_transfer_pending(request, num_computed_tokens):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                cached_encoder_inputs = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True

                    threshold = self._long_prefill_chunk_cap(request)
                    num_new_tokens = self._cap_prefill_chunk(
                        request, num_new_tokens, prefill_chunk_cap
                    )
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # A waiting text prefill must not acquire an additional
                    # numerical boundary merely because earlier requests left
                    # less than its normal aligned chunk in this step.  The
                    # normal chunk is derived with the full scheduler budget,
                    # then clipped by the existing Mamba/GDN boundary logic.
                    # Encoder requests retain their existing joint-budget
                    # contract; already-running prefills have a separate
                    # liveness/ITL contract and are deliberately out of scope.
                    if (
                        self.canonical_prefill_admission
                        and not load_kv_async
                        and not request.has_encoder_inputs
                        and self._is_prefill_request(request)
                    ):
                        canonical_num_new_tokens = min(
                            num_new_tokens,
                            self.max_num_scheduled_tokens,
                        )
                        if self.need_mamba_block_aligned_split:
                            canonical_num_new_tokens = self._mamba_block_aligned_split(
                                request,
                                canonical_num_new_tokens,
                                num_new_local_computed_tokens,
                                num_external_computed_tokens,
                            )
                        if canonical_num_new_tokens > token_budget:
                            self.num_canonical_prefill_deferrals_since_last_stats += 1
                            break

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(
                        num_new_tokens, token_budget, input_budget - draft_slots
                    )
                    assert num_new_tokens > 0

                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
                            # Recurrent alignment applies to prefill tokens, not
                            # a partial speculative tail. Drop the padding if the
                            # complete target+draftee row set no longer fits.
                            num_new_tokens = 1
                            pad_spec_decode = False

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                            cached_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                        )
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, num_computed_tokens, num_new_tokens
                    )
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks: int | KVCacheBlockPoolRequirements = 0
                if self.scheduler_reserve_full_isl or load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Ordinary
                    # chunked prefills also need their unallocated full-ISL tail
                    # protected: otherwise two requests can both pass the same
                    # point-in-time admission check, become RUNNING, and then
                    # serialize invisibly when their combined tails do not fit.
                    reserved_blocks = self._inflight_prefill_reserved_blocks(
                        exclude=request
                    )

                if (
                    self.elastic_on_demand_graphs
                    and not load_kv_async
                    and elastic_preplanned_waiting_step_key is None
                ):
                    prospective_tokens = dict(num_scheduled_tokens)
                    prospective_tokens[request_id] = num_new_tokens
                    prospective_drafts = dict(scheduled_spec_decode_tokens)
                    if pad_spec_decode:
                        prospective_drafts[request_id] = [-1] * self.num_spec_tokens
                    prospective_is_pure_decode = self._is_pure_decode_step(
                        prospective_tokens,
                        prospective_drafts,
                        computed_token_overrides={request_id: num_computed_tokens},
                    )
                    prospective_k = self._num_spec_tokens_for_step(
                        prospective_tokens,
                        prospective_is_pure_decode,
                    )
                    prospective_step_key = self._canonical_elastic_graph_step_key(
                        prospective_tokens,
                        prospective_k,
                        prospective_is_pure_decode,
                    )
                    remaining_requirements = self._request_remaining_blocks(request)
                    reserved_requirements = (
                        reserved_blocks
                        if isinstance(reserved_blocks, KVCacheBlockPoolRequirements)
                        else KVCacheBlockPoolRequirements(primary=reserved_blocks)
                    )
                    joint_remaining = remaining_requirements + reserved_requirements
                    minimum_attention_blocks = max(
                        (
                            block.block_id + 1
                            for blocks in new_computed_blocks.blocks
                            for block in blocks
                        ),
                        default=0,
                    )
                    prepared_calibration_wave = bool(
                        self._elastic_restore_mode
                        and calibration_wave_target
                        and calibration_wave_step_key is not None
                    )
                    # The request-free calibration preflight already admitted
                    # the exact final Graph envelope, full-sequence KV, decode
                    # successor and shared watermark as one atomic layout.
                    # Rechecking only the final member through the incremental
                    # serving estimator used a different headroom ledger: X1
                    # through X42 consumed the prepared grant, then X43 was
                    # spuriously stopped even though 447 tokens and one resident
                    # slot remained and allocate_slots had not rejected it.
                    # Commit identity below remains fail-closed, so a changed
                    # K/X/M can never consume this calibration-only grant.
                    if not prepared_calibration_wave:
                        graph_fits, required_external, available_external = (
                            self._can_fund_elastic_graph_step(
                                prospective_step_key,
                                minimum_free_primary_blocks=(
                                    self._elastic_successor_primary_headroom(
                                        prospective_tokens,
                                        computed_token_overrides={
                                            request_id: num_computed_tokens
                                        },
                                    )
                                    + joint_remaining.primary
                                ),
                                minimum_attention_blocks=minimum_attention_blocks,
                                gdn_blocks=(
                                    self.kv_cache_manager.coordinator.elastic_gdn_blocks_after_allocation(
                                        joint_remaining.mamba,
                                        new_computed_blocks.blocks,
                                    )
                                ),
                                allow_maintenance=(
                                    not num_scheduled_tokens
                                    and len(self.waiting) + len(self.skipped_waiting)
                                    == 1
                                ),
                                mm_activation_loan_bytes=(
                                    self._elastic_mm_activation_loan_bytes
                                    if encoder_inputs_to_schedule
                                    else 0
                                ),
                            )
                        )
                    if not prepared_calibration_wave and not graph_fits:
                        rejection = (
                            prospective_step_key,
                            required_external,
                            available_external,
                        )
                        if rejection != self._elastic_last_graph_admission_rejection:
                            logger.debug(
                                "Elastic CUDA Graph MaxX admission deferred: "
                                "feasible_x=%d rejected_x=%d required_bytes=%d "
                                "available_bytes=%d step_key=%s",
                                len(num_scheduled_tokens),
                                len(prospective_tokens),
                                required_external,
                                available_external,
                                prospective_step_key,
                            )
                        self._elastic_last_graph_admission_rejection = rejection
                        calibration_wave_stop_reason = "graph_funding_rejected"
                        break
                    self._elastic_last_graph_admission_rejection = None

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    calibration_wave_stop_reason = (
                        "allocate_slots_none:"
                        f"request={request_id}:"
                        f"rejection={self.kv_cache_manager.last_allocation_rejection!r}"
                    )
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    if self.needs_kv_cache_zeroing:
                        # Skip zeroing of the blocks the async load will
                        # overwrite; the zeroing could race the write.
                        self._skip_zero_block_ids.update(
                            self.kv_cache_manager.get_zeroing_block_ids_in_range(
                                request.request_id,
                                num_new_local_computed_tokens,
                                num_computed_tokens,
                            )
                        )
                    continue

                self.running.append(request)
                if num_external_computed_tokens > 0:
                    # ``load_kv_async`` is false on this path; the worker must
                    # complete the connector load before executing the step.
                    has_sync_kv_loads = True
                if num_new_local_computed_tokens > 0:
                    checkpoint_key = self._gdn_boundary_key(
                        request, num_computed_tokens
                    )
                    coordinator = self._gdn_checkpoint_coordinator
                    if (
                        checkpoint_key is not None
                        and coordinator is not None
                        and coordinator.has_gdn_checkpoint(
                            cast(BlockHash, checkpoint_key), touch=True
                        )
                    ):
                        gdn_checkpoint_restore[request_id] = checkpoint_key
                        logger.debug(
                            "Exact GDN prefix hit request=%s tokens=%d key=%s",
                            request_id,
                            num_computed_tokens,
                            checkpoint_key.hex()[:16],
                        )
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                self._commit_encoder_cache_plan(
                    request,
                    cached_encoder_inputs,
                    encoder_inputs_to_schedule,
                    external_load_encoder_input,
                )
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    encoder_compute_budget = new_encoder_compute_budget

                if stop_after_elastic_candidate:
                    break

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)
            self._elastic_restore_wave_target = 0
            self._elastic_restore_wave_step_key = None
            self._elastic_restore_execution_step_key = None

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    uses_mrope=self.model_uses_mrope,
                    uses_xdrope=self.model_uses_xdrope,
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Mamba "align" boundary states must carry their exact block identity
        # into connector metadata. Drain every step so stale offers cannot
        # survive a request cancellation; the connector pins accepted blocks
        # before the CoW retention below is released.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()
        kv_connector_block_state = None
        if self.connector is not None:
            # Any request scheduled this step can become a connector job now,
            # not only the ones that were allocated blocks: a store save lands
            # on the step that fills a block, which allocated none.
            block_state_req_ids = set(num_scheduled_tokens)
            block_state_req_ids.update(
                req_id for req_id in boundary_state_offloads if req_id in self.requests
            )
            kv_connector_block_state = KVConnectorBlockState(
                req_ids=block_state_req_ids,
                resolve_block_ids=self.kv_cache_manager.get_block_ids,
                boundary_state_offloads=boundary_state_offloads,
            )

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Compute the execution phase once, before policy/request overrides.
        # K=0 target decode and K>0 target verification must share this lane.
        is_pure_decode_step = self._is_pure_decode_step(
            num_scheduled_tokens, scheduled_spec_decode_tokens
        )

        # Dynamic speculative decoding: compute optimal K.
        num_spec_tokens_to_schedule = self._num_spec_tokens_for_step(
            num_scheduled_tokens,
            is_pure_decode_step,
        )

        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )

        gdn_checkpoint_save: dict[str, bytes] = {}
        if self._gdn_checkpoint_coordinator is not None:
            checkpoint_keys_to_save: set[bytes] = set()
            for req_id, scheduled_tokens in num_scheduled_tokens.items():
                request = self.requests[req_id]
                boundary = request.num_computed_tokens + scheduled_tokens
                # The hash covers only finalized input tokens. A sampled token
                # is not part of this state until it is scheduled next step.
                if boundary <= request.num_tokens and self._should_save_gdn_checkpoint(
                    request, boundary
                ):
                    checkpoint_key = self._gdn_boundary_key(request, boundary)
                    if (
                        checkpoint_key is not None
                        and checkpoint_key not in checkpoint_keys_to_save
                    ):
                        gdn_checkpoint_save[req_id] = checkpoint_key
                        checkpoint_keys_to_save.add(checkpoint_key)

        elastic_graph_step_key = self._canonical_elastic_graph_step_key(
            num_scheduled_tokens,
            num_spec_tokens_to_schedule,
            is_pure_decode_step,
        )
        if deferred_mm_wave_for_commit is not None and num_scheduled_tokens:
            actual_encoder_inputs = tuple(
                (request_id, tuple(input_ids))
                for request_id, input_ids in scheduled_encoder_inputs.items()
                if input_ids
            )
            if (
                elastic_graph_step_key != deferred_mm_wave_for_commit.step_key
                or tuple(num_scheduled_tokens.items())
                != deferred_mm_wave_for_commit.scheduled_tokens
                or actual_encoder_inputs
                != deferred_mm_wave_for_commit.scheduled_encoder_inputs
            ):
                raise RuntimeError(
                    "deferred MM wave crossed admission with a different USER shape"
                )
            # Tick B now owns the exact USER mutation. Clear the binding only
            # after the normal scheduler has reproduced every frozen field.
            self._elastic_deferred_mm_wave = None
        if (
            calibration_wave_step_key is not None
            and num_scheduled_tokens
            and elastic_graph_step_key != calibration_wave_step_key
        ):
            raise RuntimeError(
                "prepared elastic restore wave changed before commit: "
                f"prepared={calibration_wave_step_key!r} "
                f"commit={elastic_graph_step_key!r} "
                f"scheduled_requests={len(num_scheduled_tokens)} "
                f"scheduled_tokens={sum(num_scheduled_tokens.values())} "
                f"remaining_token_budget={token_budget} "
                f"scheduler_cap={self.max_num_running_reqs} "
                f"running={len(self.running)} waiting={len(self.waiting)} "
                f"skipped_waiting={len(self.skipped_waiting)} "
                f"stop_reason={calibration_wave_stop_reason!r} "
                "last_allocation_rejection="
                f"{self.kv_cache_manager.last_allocation_rejection!r}"
            )
        if (
            calibration_execution_step_key is not None
            and num_scheduled_tokens
            and elastic_graph_step_key != calibration_execution_step_key
        ):
            diagnostic = self._elastic_restore_admission_diagnostic(
                calibration_wave_stop_reason, token_budget, num_scheduled_tokens
            )
            raise RuntimeError(
                "prepared elastic restore execution changed before commit: "
                f"prepared={calibration_execution_step_key!r} "
                f"commit={elastic_graph_step_key!r} "
                f"admission={diagnostic!r}"
            )
        if (
            elastic_preplanned_running_step_key is not None
            and num_scheduled_tokens
            and elastic_graph_step_key != elastic_preplanned_running_step_key
        ):
            raise RuntimeError(
                "preflighted RUNNING decode wave changed before commit: "
                f"preflight={elastic_preplanned_running_step_key!r} "
                f"commit={elastic_graph_step_key!r}"
            )
        if (
            elastic_preplanned_waiting_step_key is not None
            and num_scheduled_tokens
            and elastic_graph_step_key != elastic_preplanned_waiting_step_key
        ):
            raise RuntimeError(
                "preflighted WAITING text wave changed before commit: "
                f"preflight={elastic_preplanned_waiting_step_key!r} "
                f"commit={elastic_graph_step_key!r}"
            )
        admission_grant = getattr(self, "_elastic_preflight_admission_grant", None)
        if admission_grant is not None:
            if elastic_graph_step_key != admission_grant.step_key:
                self._rollback_elastic_admission()
            elif (
                self._elastic_step_residency_intent(elastic_graph_step_key)[2]
                != admission_grant.physical_keys
            ):
                self._rollback_elastic_admission()
                raise RuntimeError(
                    "preflighted elastic physical owner set changed before commit"
                )
        maintenance_plan = None
        if self._should_commit_pending_elastic_maintenance(
            has_user_tokens=bool(num_scheduled_tokens)
        ):
            maintenance_plan = (
                self._elastic_admission_controller.pending_maintenance_plan
            )
            maintenance_step_key = (
                self._elastic_admission_controller.pending_maintenance_step_key
            )
            if num_scheduled_tokens:
                if admission_grant is None:
                    raise RuntimeError(
                        "serving elastic maintenance reached USER commit "
                        "without an atomic admission grant"
                    )
                if elastic_graph_step_key is None:
                    raise RuntimeError(
                        "scheduled elastic maintenance lost its execution shape"
                    )
                maintenance_step_key = self._bind_pending_elastic_maintenance_commit(
                    elastic_graph_step_key,
                    maintenance_step_key,
                )
            elastic_graph_step_key = maintenance_step_key
            if elastic_graph_step_key is None and cast(
                ElasticStepPlan, maintenance_plan
            ).kind not in {
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }:
                raise RuntimeError("elastic maintenance lost its scheduler shape")
        # Candidate admission may keep one immediate GDN lookahead mapped so a
        # cache hit cannot pin the tail needed by the next request. Return it
        # only at a real resource commit. An administrative zero-token output
        # has no consumer or physical admission and must preserve KV geometry.
        elastic_resource_commit = self._rebalance_elastic_capacity_before_commit(
            has_user_tokens=bool(num_scheduled_tokens),
            maintenance_plan=maintenance_plan,
        )
        elastic_mm_activation_loan = (
            self._elastic_mm_activation_loan_bytes if scheduled_encoder_inputs else 0
        )
        elastic_successor_primary_headroom = self._elastic_successor_primary_headroom(
            num_scheduled_tokens
        )
        # Administrative outputs have no admitted reclaim plan. Preserve both
        # Graph ownership and its loan until the explicit maintenance tick.
        elastic_preserve_graph_residency = (
            self.elastic_on_demand_graphs
            and elastic_graph_step_key is None
            and maintenance_plan is None
        )
        elastic_graph_step_grant = self._plan_elastic_graph_loan(
            elastic_graph_step_key,
            minimum_free_primary_blocks=elastic_successor_primary_headroom,
            mm_activation_loan_bytes=elastic_mm_activation_loan,
        )
        self._elastic_preflight_admission_grant = None
        if maintenance_plan is not None and maintenance_plan.kind in {
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            maintenance_coordinator = self.kv_cache_manager.coordinator
            retained_pinned = (
                self._elastic_pressure_floor_external_bytes()
                if maintenance_plan.kind == ElasticPlanKind.PRESSURE_RECLAIM
                else self._elastic_admission_controller.pinned_resident_bytes
            )
            elastic_graph_step_grant = (
                maintenance_coordinator.normalize_elastic_external_memory(
                    max(
                        retained_pinned,
                        maintenance_plan.capture_loan_bytes,
                    )
                )
            )
            if not maintenance_coordinator.set_elastic_external_memory(
                elastic_graph_step_grant,
                minimum_free_primary_blocks=elastic_successor_primary_headroom,
            ):
                raise RuntimeError("scheduler could not commit planned graph reclaim")
            self._elastic_admission_controller.replace_latest_loan(
                None,
                elastic_graph_step_grant,
            )

        elastic_expert_grant = self._plan_elastic_expert_grant(
            has_user_tokens=bool(num_scheduled_tokens),
            minimum_free_primary_blocks=elastic_successor_primary_headroom,
        )
        elastic_kv_transition = (
            self.kv_cache_manager.coordinator.take_elastic_transition()
            if elastic_resource_commit or elastic_expert_grant is not None
            else None
        )
        elastic_transaction_id = (
            maintenance_plan.transaction_id
            if maintenance_plan is not None
            else self._next_elastic_transaction_id()
        )
        execution_manifest = None
        current_dispatch: tuple[OwnerDispatch, ...] = ()
        successor_keys: tuple[PhysicalReplayKey, ...] = ()
        if elastic_graph_step_key is not None and num_scheduled_tokens:
            policy = self._elastic_graph_execution_policy
            if policy is None:
                raise RuntimeError("elastic execution requires a Graph policy")
            is_prefilling_by_request = {
                request_id: self._is_prefill_request(self.requests[request_id])
                for request_id in num_scheduled_tokens
            }
            ordered_execution_ids = canonical_execution_request_order(
                num_scheduled_tokens,
                is_prefilling_by_request=is_prefilling_by_request,
                # Match GPUModelRunner.decode_query_len and InputBatch sorting.
                # This is the physical target verification width, not merely
                # the number of newly sampled output tokens for this step.
                decode_query_len=(
                    self.num_sampled_tokens_per_step + self.num_spec_tokens
                ),
            )
            manifest_physical_keys = self._resolve_elastic_step_physical_keys(
                elastic_graph_step_key
            )
            execution_manifest, current_dispatch = build_execution_manifest(
                step_key=cast(tuple[int, int, int, int, int], elastic_graph_step_key),
                request_ids=ordered_execution_ids,
                per_request_query_lens=tuple(
                    num_scheduled_tokens[request_id]
                    for request_id in ordered_execution_ids
                ),
                per_request_is_prefilling=tuple(
                    is_prefilling_by_request[request_id]
                    for request_id in ordered_execution_ids
                ),
                scheduled_draft_rows=tuple(
                    len(scheduled_spec_decode_tokens.get(request_id, ()))
                    for request_id in ordered_execution_ids
                ),
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                requested_output_k=num_spec_tokens_to_schedule,
                executed_drafter_k=(
                    self.num_spec_tokens if num_spec_tokens_to_schedule > 0 else 0
                ),
                phase=cast(
                    str,
                    execution_manifest_phase_from_step_key(
                        cast(tuple[int, int, int, int, int], elastic_graph_step_key)
                    ),
                ),
                generation=self._elastic_admission_controller.generation,
                policy=policy,
                max_num_batched_tokens=(self.scheduler_config.max_num_batched_tokens),
                physical_keys=manifest_physical_keys,
            )
            _current_keys, successor_keys, _residency_union = (
                self._elastic_step_residency_intent(elastic_graph_step_key)
            )
        elastic_step_plan = maintenance_plan
        if elastic_graph_step_key is not None:
            residency_step_key = elastic_graph_step_key
            current_keys, successor_keys, physical_keys = (
                self._elastic_step_residency_intent(residency_step_key)
            )
            protected_successor_keys = tuple(
                key for key in physical_keys if key not in set(current_keys)
            )
            if elastic_step_plan is None:
                available_external = (
                    self.kv_cache_manager.coordinator.max_elastic_external_memory(
                        minimum_free_primary_blocks=(elastic_successor_primary_headroom)
                    )
                )
                desired_external, _ = self._estimate_elastic_graph_step_bytes(
                    residency_step_key
                )
                elastic_step_plan = self._elastic_admission_controller.plan(
                    elastic_transaction_id,
                    physical_keys,
                    request_bytes=self._elastic_admission_controller.resident_bytes,
                    available_bytes=available_external,
                    kv_transition=elastic_kv_transition,
                )
                if elastic_step_plan.kind == ElasticPlanKind.MAINTENANCE:
                    raise RuntimeError(
                        "COLD elastic promotion crossed the user commit boundary"
                    )
                if elastic_step_plan.kind == ElasticPlanKind.DEFER:
                    entry_states = tuple(
                        (
                            key.identity,
                            (
                                None
                                if (
                                    entry
                                    := self._elastic_admission_controller.entries.get(
                                        key
                                    )
                                )
                                is None
                                else entry.state.value
                            ),
                            None if entry is None else bool(entry.leases),
                        )
                        for key in physical_keys
                    )
                    hot_hit_ids = tuple(
                        key.identity for key in elastic_step_plan.hot_hits
                    )
                    cold_miss_ids = tuple(
                        key.identity for key in elastic_step_plan.cold_misses
                    )
                    raise RuntimeError(
                        "deferred elastic plan crossed the scheduler commit boundary: "
                        f"reason={elastic_step_plan.defer_reason} "
                        f"step_key={elastic_graph_step_key!r} "
                        f"residency_step_key={residency_step_key!r} "
                        f"entries={entry_states!r} "
                        f"hot_hits={hot_hit_ids!r} "
                        f"cold_misses={cold_miss_ids!r} "
                        f"request_bytes={elastic_step_plan.request_bytes} "
                        f"available_bytes={elastic_step_plan.available_bytes} "
                        f"capture_loan_bytes={elastic_step_plan.capture_loan_bytes} "
                        f"desired_external={desired_external} "
                        f"resident_bytes={self._elastic_admission_controller.resident_bytes}"
                    )
                elastic_step_plan = replace(
                    elastic_step_plan,
                    expert_grant=elastic_expert_grant,
                    kv_transition=elastic_kv_transition,
                    capture_loan_bytes=elastic_graph_step_grant,
                    protected_keys=tuple(
                        dict.fromkeys(
                            (
                                *elastic_step_plan.protected_keys,
                                *protected_successor_keys,
                            )
                        )
                    ),
                    execution_manifest=execution_manifest,
                    current_dispatch=current_dispatch,
                    successor_keys=successor_keys,
                )
            else:
                if elastic_step_plan.physical_keys != physical_keys:
                    raise RuntimeError(
                        "maintenance physical owner set changed after defer"
                    )
                elastic_step_plan = replace(
                    elastic_step_plan,
                    expert_grant=elastic_expert_grant,
                    kv_transition=elastic_kv_transition,
                    capture_loan_bytes=elastic_graph_step_grant,
                    protected_keys=tuple(
                        dict.fromkeys(
                            (
                                *elastic_step_plan.protected_keys,
                                *protected_successor_keys,
                            )
                        )
                    ),
                    execution_manifest=execution_manifest,
                    current_dispatch=current_dispatch,
                    successor_keys=successor_keys,
                )
                self._elastic_admission_controller.clear_maintenance()
            retained_physical_keys = getattr(
                self, "_elastic_restore_retained_physical_keys", ()
            )
            if retained_physical_keys:
                elastic_step_plan = replace(
                    elastic_step_plan,
                    protected_keys=tuple(
                        dict.fromkeys(
                            (*elastic_step_plan.protected_keys, *retained_physical_keys)
                        )
                    ),
                )
            decode_consensus_epoch = (
                elastic_step_plan.execution_epoch_fingerprint
                if elastic_step_plan.reusable_decode_consensus_epoch
                else None
            )
            elastic_step_plan = replace(
                elastic_step_plan,
                reuse_rank_consensus=bool(
                    decode_consensus_epoch is not None
                    and decode_consensus_epoch
                    == self._elastic_accepted_decode_consensus_epoch
                ),
            )
            if elastic_step_plan.kind == ElasticPlanKind.USER:
                self._elastic_admission_controller.commit_user(elastic_step_plan)
                useful_started = getattr(self, "_elastic_useful_started", None)
                if useful_started is None:
                    useful_started = self._elastic_useful_started = {}
                useful_started[elastic_step_plan.transaction_id] = time.monotonic()
            elif elastic_step_plan.kind == ElasticPlanKind.MAINTENANCE:
                stats_before = self._elastic_admission_controller.stats
                external_before = self._elastic_admission_controller.resident_bytes
                self._elastic_admission_controller.begin_maintenance(elastic_step_plan)
                self._elastic_maintenance_started[elastic_step_plan.transaction_id] = (
                    time.monotonic(),
                    stats_before,
                    external_before,
                )
                logger.info(
                    "Elastic Graph transaction start: tx=%s kind=%s final_key=%s "
                    "captures=%d evictions=%d external_before=%d loan=%d reason=cold",
                    elastic_step_plan.transaction_id,
                    elastic_step_plan.kind.value,
                    elastic_graph_step_key,
                    len(elastic_step_plan.cold_misses),
                    len(elastic_step_plan.victim_keys),
                    external_before,
                    elastic_step_plan.capture_loan_bytes,
                )
        elif elastic_step_plan is not None:
            if elastic_step_plan.kind not in {
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }:
                raise RuntimeError("shape-less elastic plan is not a reclaim")
            elastic_step_plan = replace(
                elastic_step_plan,
                expert_grant=elastic_expert_grant,
                kv_transition=elastic_kv_transition,
                capture_loan_bytes=elastic_graph_step_grant,
            )
            stats_before = self._elastic_admission_controller.stats
            external_before = self._elastic_admission_controller.resident_bytes
            self._elastic_admission_controller.begin_reclaim(elastic_step_plan)
            self._elastic_maintenance_started[elastic_step_plan.transaction_id] = (
                time.monotonic(),
                stats_before,
                external_before,
            )
            logger.info(
                "Elastic Graph transaction start: tx=%s kind=%s final_key=None "
                "captures=0 evictions=%d external_before=%d loan=%d "
                "reason=explicit_pressure",
                elastic_step_plan.transaction_id,
                elastic_step_plan.kind.value,
                len(elastic_step_plan.victim_keys),
                external_before,
                elastic_step_plan.capture_loan_bytes,
            )
            self._elastic_admission_controller.clear_maintenance()

        if (
            elastic_step_plan is not None
            and os.environ.get("AG2_VLLM_GRAPH_MODE_RECEIPT", "0") == "1"
        ):
            logger.debug(
                "AG2 elastic plan receipt: transaction=%s kind=%s step_key=%s "
                "hot_hits=%s cold_misses=%s victims=%s capture_order=%s",
                elastic_step_plan.transaction_id,
                elastic_step_plan.kind.value,
                elastic_graph_step_key,
                tuple(key.identity for key in elastic_step_plan.hot_hits),
                tuple(key.identity for key in elastic_step_plan.cold_misses),
                tuple(key.identity for key in elastic_step_plan.victim_keys),
                tuple(key.identity for key in elastic_step_plan.capture_order),
            )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            is_pure_decode_step=is_pure_decode_step,
            gdn_checkpoint_save=gdn_checkpoint_save or None,
            gdn_checkpoint_restore=gdn_checkpoint_restore or None,
            elastic_kv_transition=elastic_kv_transition,
            elastic_expert_grant=elastic_expert_grant,
            elastic_external_memory_bytes=(elastic_graph_step_grant),
            elastic_graph_external_memory_bytes=(
                elastic_graph_step_grant - elastic_mm_activation_loan
            ),
            elastic_mm_activation_loan_bytes=elastic_mm_activation_loan,
            elastic_successor_primary_headroom=(elastic_successor_primary_headroom),
            elastic_transaction_id=elastic_transaction_id,
            elastic_graph_step_key=elastic_graph_step_key,
            elastic_step_plan=elastic_step_plan,
            elastic_plan_fingerprint=(
                elastic_step_plan.fingerprint if elastic_step_plan is not None else None
            ),
            elastic_preserve_graph_residency=elastic_preserve_graph_residency,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
        )
        self._elastic_preflight_waiting_ignore_prefix_request_ids = ()
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # This scheduler-local snapshot is consumed by connector metadata
        # construction and must not cross the scheduler/worker ABI.
        scheduler_output.kv_connector_block_state = None

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _is_pure_decode_step(
        self,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        computed_token_overrides: dict[str, int] | None = None,
    ) -> bool:
        """Return whether every scheduled row is an autoregressive decode row.

        A request is in pure decode only after its prompt has already been
        computed and the current target query consists solely of the normal
        sampled-token positions plus previously drafted positions. Resumed
        output replay and chunked prefill therefore remain non-decode even
        when their computed position is already past the original prompt.
        """
        if not num_scheduled_tokens:
            return False
        for req_id, num_tokens in num_scheduled_tokens.items():
            request = self.requests[req_id]
            num_computed_tokens = (
                computed_token_overrides.get(req_id, request.num_computed_tokens)
                if computed_token_overrides is not None
                else request.num_computed_tokens
            )
            if num_computed_tokens < request.execution_prefill_len:
                return False
            num_draft_tokens = len(scheduled_spec_decode_tokens.get(req_id, ()))
            if num_tokens != self.num_sampled_tokens_per_step + num_draft_tokens:
                return False
        return True

    def _num_spec_tokens_for_step(
        self,
        num_scheduled_tokens: dict[str, int],
        is_pure_decode_step: bool,
    ) -> int:
        num_spec_tokens = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and num_scheduled_tokens:
            num_spec_tokens = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None:
            if num_spec_tokens != 0:
                raise RuntimeError(
                    "scheduler has speculative tokens without a speculative config"
                )
        elif (
            speculative_config.disable_speculation_on_non_decode
            and not is_pure_decode_step
        ):
            num_spec_tokens = 0
        return apply_force_non_speculative_override(
            num_spec_tokens,
            [
                self.requests[request_id].force_non_speculative
                for request_id in num_scheduled_tokens
            ],
        )

    def _rebuild_elastic_short_decode_inventory(self, max_x: int) -> None:
        """Bind runtime-derived decode carriers to one proved terminal MaxX."""
        if max_x <= 0:
            raise ValueError("short-decode terminal X must be positive")
        if not self.elastic_on_demand_graphs or self.num_spec_tokens < 0:
            raise RuntimeError("short-decode inventory requires elastic K>=0")
        self._elastic_short_decode_inventory = dict(
            derive_short_decode_graph_inventory(
                max_x=max_x,
                num_spec_tokens=self.num_spec_tokens,
                generation=self._elastic_admission_controller.generation,
                max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
                compiled_piecewise_sizes=self._elastic_compiled_piecewise_sizes,
                policy=self._elastic_graph_execution_policy,
            )
        )
        logger.info(
            "Elastic short-decode Graph inventory: K=%d MaxX=%d X/M=%s",
            self.num_spec_tokens,
            max_x,
            ",".join(
                f"{x}/{x * (self.num_spec_tokens + 1)}"
                for x in self._elastic_short_decode_inventory
            ),
        )

    @staticmethod
    def _make_elastic_graph_step_key(
        num_scheduled_tokens: dict[str, int],
        num_spec_tokens_to_schedule: int,
        is_pure_decode_step: bool,
    ) -> SemanticGraphStep | None:
        if not num_scheduled_tokens:
            return None
        token_counts = tuple(num_scheduled_tokens.values())
        num_reqs = len(token_counts)
        total_num_tokens = sum(token_counts)
        if total_num_tokens <= 0:
            return None
        uniform_token_count = token_counts[0] if len(set(token_counts)) == 1 else 0
        uniform_decode = is_pure_decode_step and uniform_token_count > 0
        owners = (
            ("target", "mtp_prefill", "mtp_decode")
            if num_spec_tokens_to_schedule > 0
            else ("target",)
        )
        return SemanticGraphStep(
            num_spec_tokens=num_spec_tokens_to_schedule,
            num_reqs=num_reqs,
            num_tokens=total_num_tokens,
            uniform_query_len=uniform_token_count if uniform_decode else None,
            phase="decode" if uniform_decode else "mixed",
            active_owners=owners,
        )

    def _canonical_elastic_graph_step_key(
        self,
        num_scheduled_tokens: dict[str, int],
        num_spec_tokens_to_schedule: int,
        is_pure_decode_step: bool,
    ) -> tuple[int, ...] | None:
        policy = getattr(self, "_elastic_graph_execution_policy", None)
        if not getattr(self, "elastic_on_demand_graphs", False) and policy is None:
            return None
        semantic = self._make_elastic_graph_step_key(
            num_scheduled_tokens,
            num_spec_tokens_to_schedule,
            is_pure_decode_step,
        )
        if semantic is None:
            return None
        if policy is None:
            raise RuntimeError("elastic step routing requires an execution policy")
        # Semantic X remains in scheduler_output for request metadata, sampling
        # and output cropping. CUDA Graph identity uses the smallest declared
        # physical cohort covering it, so X8 -> X7 reuses HOT X8/M32. DCP is
        # intentionally absent: it partitions attention context, not rows.
        physical_x = semantic.num_reqs
        restore_mode = getattr(self, "_elastic_restore_mode", False)
        if semantic.phase == "decode" and not restore_mode:
            physical_x = self._elastic_short_decode_physical_x(semantic.num_reqs)
        # Residency may retain a larger cohort carrier until drain, but current
        # execution always uses the smallest declared bucket covering semantic
        # X. _elastic_step_residency_intent() protects the retained carrier
        # without making it a dispatch candidate.
        scheduler_config = getattr(self, "scheduler_config", None)
        return canonical_graph_step_key(
            semantic,
            policy,
            physical_num_reqs=physical_x,
            max_num_batched_tokens=(
                None
                if scheduler_config is None
                else scheduler_config.max_num_batched_tokens
            ),
        )

    def _elastic_short_decode_physical_x(self, actual_x: int) -> int:
        """Resolve one semantic cohort against the accepted bounded X set."""
        from vllm.v1.core.elastic_graph import (
            select_short_decode_physical_x,
            short_decode_inventory_xs,
        )

        terminal_x = getattr(self, "_elastic_terminal_decode_carrier_x", None)
        if (
            not getattr(self, "_elastic_restore_mode", False)
            and terminal_x is not None
            and actual_x > terminal_x
        ):
            raise ElasticGraphError(
                "semantic short-decode X exceeds the retained terminal carrier: "
                f"actual_x={actual_x} max_x={terminal_x}"
            )
        inventory_xs = tuple(getattr(self, "_elastic_short_decode_inventory", {}))
        if not inventory_xs:
            coverage = getattr(self, "_elastic_graph_catalog_coverage", {})
            max_x = terminal_x or coverage.get("decode_max_x")
            if isinstance(max_x, bool) or not isinstance(max_x, int) or max_x <= 0:
                configured_max_x = getattr(self.scheduler_config, "max_num_seqs", None)
                max_x = (
                    configured_max_x
                    if isinstance(configured_max_x, int)
                    and not isinstance(configured_max_x, bool)
                    and configured_max_x > 0
                    else actual_x
                )
            inventory_xs = short_decode_inventory_xs(max_x)
        if (
            terminal_x is not None
            and not getattr(self, "_elastic_restore_mode", False)
            and inventory_xs[-1] != terminal_x
        ):
            raise RuntimeError(
                "short-decode inventory disagrees with the sealed terminal carrier: "
                f"inventory_max={inventory_xs[-1]} terminal_x={terminal_x}"
            )
        return select_short_decode_physical_x(actual_x, inventory_xs)

    def _elastic_graph_carrier_closure_step_key(
        self, step_key: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        """Return the phase-correct full-owner closure for an execution key."""
        if step_key is None or step_key[1] <= 0:
            return step_key
        if getattr(self, "_elastic_restore_mode", False):
            # Calibration measures each declared execution DAG. Product
            # residency coalesces those observations only after the catalog is
            # sealed; collapsing here would erase the evidence used to price
            # the closure itself.
            return step_key
        if step_key[0] == 1 and step_key[4] == 1:
            # A shrinking pure-decode tail may reuse the cohort's physical X,
            # but it cannot execute the target's q=1 invocation through a
            # mixed q=K+1 PIECEWISE descriptor. Keep the phase identity while
            # the canonicalizer above preserves the carrier cardinality.
            return step_key
        num_spec_tokens = step_key[1]
        physical_x = self._elastic_short_decode_physical_x(step_key[2])
        query_len = num_spec_tokens + 1
        policy = self._elastic_graph_execution_policy
        if policy is None:
            raise RuntimeError("elastic carrier routing requires an execution policy")
        target_full = int(policy.mode_for("target", query_len) == "FULL")
        return (
            target_full,
            num_spec_tokens,
            physical_x,
            physical_x * query_len,
            query_len,
        )

    def _commit_elastic_graph_carrier_step_key(
        self, execution_step_key: tuple[int, ...] | None
    ) -> None:
        """Commit the speculative residency high-water mark until drain."""
        if execution_step_key is None:
            return
        current = getattr(self, "_elastic_graph_carrier_step_key", None)
        if execution_step_key[1] == 0 and current is not None and current[1] > 0:
            # A target-only prefill/acceptance tail must not replace the live
            # speculative carrier.  K0 runtimes have no such owner and commit
            # their target carrier normally.
            return
        candidate = self._elastic_graph_carrier_closure_step_key(execution_step_key)
        assert candidate is not None
        if (
            current is not None
            and current[1] == candidate[1]
            and current[2] >= candidate[2]
        ):
            # A smaller execution bucket and transient q=1 tail must not shrink
            # or replace the q=K+1 residency closure. The scheduler clears this
            # high-water state only after the complete request epoch drains.
            current_query_len = current[1] + 1
            candidate_query_len = candidate[1] + 1
            current_is_closure = (
                current[4] == current_query_len
                and current[3] == current[2] * current_query_len
            )
            candidate_is_closure = (
                candidate[4] == candidate_query_len
                and candidate[3] == candidate[2] * candidate_query_len
            )
            if current[2] > candidate[2] or (
                current_is_closure and not candidate_is_closure
            ):
                return
        self._elastic_graph_carrier_step_key = candidate

    def _resolve_elastic_step_physical_keys(
        self, step_key: tuple[int, ...] | None
    ) -> tuple[PhysicalReplayKey, ...]:
        resolved = resolve_step_physical_keys(
            cast(tuple[int, int, int, int, int] | None, step_key),
            self._elastic_admission_controller.generation,
            self.scheduler_config.max_num_batched_tokens,
            getattr(self, "_elastic_compiled_piecewise_sizes", frozenset()),
            self._elastic_graph_execution_policy,
        )
        terminal_x = getattr(self, "_elastic_terminal_decode_carrier_x", None)
        if (
            step_key is None
            or step_key[1] <= 0
            or getattr(self, "_elastic_restore_mode", False)
            or terminal_x is None
        ):
            return resolved
        if step_key[2] > terminal_x:
            raise ElasticGraphError(
                "execution X exceeds the retained terminal carrier: "
                f"actual_x={step_key[2]} max_x={terminal_x}"
            )
        # The terminal MTP carrier is a pinned recovery/residency anchor, not
        # an execution shape.  Current dispatch must use the smallest declared
        # physical bucket that covers semantic X; residency intent below keeps
        # the terminal key protected without padding every MTP invocation to
        # MaxX.
        return resolved

    def resolve_elastic_serving_carrier_physical_keys(
        self, step_keys: Sequence[tuple[int, ...]]
    ) -> tuple[PhysicalReplayKey, ...]:
        """Resolve the one persistent owner shared by all restore recipes."""
        owner = self._elastic_graph_catalog_coverage.get("serving_carrier_owner")
        if owner != "mtp_decode":
            raise RuntimeError(
                "bounded elastic catalog has an unsupported serving carrier owner"
            )
        carrier = tuple(
            dict.fromkeys(
                key
                for step_key in step_keys
                for key in self._resolve_elastic_step_physical_keys(step_key)
                if key.logical.owner == owner
            )
        )
        if len(carrier) != 1:
            raise RuntimeError(
                "bounded elastic restore recipes do not share exactly one "
                f"terminal MTP carrier: keys={tuple(key.identity for key in carrier)!r}"
            )
        return carrier

    def _elastic_step_residency_intent(
        self, step_key: tuple[int, ...] | None
    ) -> tuple[
        tuple[PhysicalReplayKey, ...],
        tuple[PhysicalReplayKey, ...],
        tuple[PhysicalReplayKey, ...],
    ]:
        """Return current keys, successor intent, and the protected union.

        A cold successor remains future intent because the manager ABI cannot
        capture two different keys for one owner in one transaction. An
        already-HOT successor is retained and priced without becoming a
        current dispatch candidate.
        """
        current = self._resolve_elastic_step_physical_keys(step_key)
        if step_key is None:
            return current, (), current
        successor = self._resolve_elastic_step_physical_keys(
            self._elastic_graph_carrier_closure_step_key(step_key)
        )
        current_set = set(current)
        protected_successor = tuple(
            key
            for key in successor
            if key not in current_set
            and (
                (entry := self._elastic_admission_controller.entries.get(key))
                is not None
                and entry.hot
            )
        )
        serving_carrier = tuple(
            key
            for key in getattr(self, "_elastic_serving_carrier_keys", ())
            if key not in current_set
            and (
                (entry := self._elastic_admission_controller.entries.get(key))
                is not None
                and entry.hot
            )
        )
        retained_carrier = tuple(
            key
            for key in self._resolve_elastic_step_physical_keys(
                getattr(self, "_elastic_graph_carrier_step_key", None)
            )
            if key not in current_set
            and (
                (entry := self._elastic_admission_controller.entries.get(key))
                is not None
                and entry.hot
            )
        )
        successor_intent = tuple(dict.fromkeys((*successor, *serving_carrier)))
        successor_intent = tuple(dict.fromkeys((*successor_intent, *retained_carrier)))
        return (
            current,
            successor_intent,
            tuple(
                dict.fromkeys(
                    (
                        *current,
                        *protected_successor,
                        *serving_carrier,
                        *retained_carrier,
                    )
                )
            ),
        )

    def _elastic_successor_primary_headroom(
        self,
        num_scheduled_tokens: dict[str, int],
        *,
        computed_token_overrides: dict[str, int] | None = None,
    ) -> int:
        """Return exact primary blocks needed by the next sampled boundary."""
        overrides = computed_token_overrides or {}
        headroom = 0
        for request_id, scheduled_tokens in num_scheduled_tokens.items():
            request = self.requests[request_id]
            computed_tokens = overrides.get(request_id, request.num_computed_tokens)
            boundary = min(
                computed_tokens + scheduled_tokens,
                self.max_model_len,
            )
            if boundary >= self.max_model_len:
                continue
            current_blocks = (boundary + self.block_size - 1) // self.block_size
            successor_boundary = min(
                boundary + self.num_sampled_tokens_per_step,
                self.max_model_len,
            )
            successor_blocks = (
                successor_boundary + self.block_size - 1
            ) // self.block_size
            headroom += successor_blocks - current_blocks
        return headroom

    def _elastic_remaining_resource_requirements(
        self,
        request_ids: Iterable[str],
    ) -> KVCacheBlockPoolRequirements:
        """Return full-sequence KV/GDN demand not yet allocated.

        Elastic serving reserves the declared sequence before admitting a
        request. Graph admission must price the same future state; using only
        this tick's chunk lets a long request consume the headroom between the
        graph preflight and its final commit.
        """
        requirements = KVCacheBlockPoolRequirements()
        for request_id in request_ids:
            requirements += self._request_remaining_blocks(self.requests[request_id])
        return requirements

    def _elastic_wave_fits_after_idle_reclaim(
        self,
        requirements: KVCacheBlockPoolRequirements,
    ) -> bool:
        """Return whether KV can coexist after administrative Graph reclaim.

        ``can_allocate`` prices the currently mapped HOT set. A cold larger
        wave must still reach Graph planning when evicting idle PIECEWISE
        entries would make it feasible. Legacy pinned captures are also
        reclaimable at quiescence; only the mandatory serving carrier and
        measured allocator floor survive. This is a read-only feasibility
        probe, not permission to mutate: the planner still proves quiescence,
        leases, loans and the destination capture before admission.
        """
        coordinator = self.kv_cache_manager.coordinator
        gdn_blocks = coordinator.elastic_gdn_blocks_after_allocation(requirements.mamba)
        available_external = coordinator.max_elastic_external_memory(
            minimum_free_primary_blocks=(
                requirements.primary + self.kv_cache_manager.watermark_blocks
            ),
            gdn_blocks=gdn_blocks,
        )
        return available_external >= self._elastic_pressure_floor_external_bytes()

    def _reserve_elastic_admission(
        self,
        step_key: tuple[int, ...],
        *,
        external_memory_bytes: int,
        minimum_free_primary_blocks: int,
        requirements: KVCacheBlockPoolRequirements,
    ) -> bool:
        """Atomically map the joint Graph/KV/GDN envelope before allocation."""
        physical_keys = self._elastic_step_residency_intent(step_key)[2]
        pending_plan = self._elastic_admission_controller.pending_maintenance_plan
        maintenance_transaction_id = (
            pending_plan.transaction_id
            if pending_plan is not None
            and pending_plan.kind == ElasticPlanKind.MAINTENANCE
            and self._elastic_admission_controller.pending_maintenance_step_key
            == step_key
            and pending_plan.physical_keys == physical_keys
            else None
        )
        existing = getattr(self, "_elastic_preflight_admission_grant", None)
        if existing is not None:
            if (
                existing.step_key == step_key
                and existing.physical_keys == physical_keys
                and existing.maintenance_transaction_id == maintenance_transaction_id
                and existing.external_memory_bytes == external_memory_bytes
                and existing.minimum_free_primary_blocks == minimum_free_primary_blocks
                and existing.requirements == requirements
            ):
                return True
            raise RuntimeError("elastic admission attempted two different grants")

        coordinator = self.kv_cache_manager.coordinator
        previous_external = coordinator.elastic_external_memory_bytes
        if not coordinator.set_elastic_external_memory(
            external_memory_bytes,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
        ):
            return False
        if not coordinator.ensure_elastic_capacity(requirements):
            coordinator.rebalance_elastic_capacity()
            if not coordinator.set_elastic_external_memory(previous_external):
                raise RuntimeError(
                    "failed elastic admission could not restore external memory"
                )
            return False
        self._elastic_preflight_admission_grant = ElasticAdmissionGrant(
            step_key=step_key,
            physical_keys=physical_keys,
            maintenance_transaction_id=maintenance_transaction_id,
            external_memory_bytes=external_memory_bytes,
            previous_external_memory_bytes=previous_external,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            requirements=requirements,
        )
        return True

    def _reserve_pending_elastic_maintenance_admission(
        self,
        step_key: tuple[int, ...],
        *,
        minimum_free_primary_blocks: int,
        requirements: KVCacheBlockPoolRequirements,
    ) -> bool:
        """Reserve a COLD capture and its complete user wave before mutation."""
        plan = self._elastic_admission_controller.pending_maintenance_plan
        if plan is None or plan.kind != ElasticPlanKind.MAINTENANCE:
            return False
        residency_step_key = step_key
        if (
            self._elastic_admission_controller.pending_maintenance_step_key
            != residency_step_key
        ):
            self._clear_pre_mutation_serving_maintenance(
                reason="stale_maintenance_shape_replanned"
            )
            return False
        existing_grant = getattr(self, "_elastic_preflight_admission_grant", None)
        if existing_grant is not None:
            if (
                existing_grant.step_key != step_key
                or existing_grant.physical_keys != plan.physical_keys
                or existing_grant.maintenance_transaction_id != plan.transaction_id
            ):
                raise RuntimeError(
                    "pending maintenance is bound to a different admission grant"
                )
            if not (
                existing_grant.external_memory_bytes == plan.capture_loan_bytes
                and existing_grant.minimum_free_primary_blocks
                == minimum_free_primary_blocks
                and existing_grant.requirements == requirements
            ):
                # The physical capture proposal remains valid, but queue/KV
                # requirements changed within the same canonical shape.
                # Restore and atomically reserve the exact current contract.
                self._rollback_elastic_admission()
        if self._reserve_elastic_admission(
            step_key,
            external_memory_bytes=plan.capture_loan_bytes,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            requirements=requirements,
        ):
            return True
        self._elastic_admission_controller.discard_armed_maintenance(
            plan.transaction_id
        )
        self._elastic_last_defer_reason = "maintenance_admission_reservation_failed"
        return False

    def _clear_pre_mutation_serving_maintenance(self, *, reason: str) -> bool:
        """Discard only an unstarted serving proposal and restore its KV grant."""
        controller = self._elastic_admission_controller
        plan = controller.pending_maintenance_plan
        if plan is None:
            return False
        if plan.kind != ElasticPlanKind.MAINTENANCE:
            raise RuntimeError("cannot discard exclusive elastic maintenance")
        if plan.maintenance_execution != ElasticMaintenanceExecution.COUPLED_USER:
            raise RuntimeError("cannot discard graph-only elastic maintenance")
        already_started = plan.transaction_id in self._elastic_maintenance_started
        if already_started:
            raise RuntimeError(
                "cannot discard elastic maintenance after physical mutation: "
                f"transaction={plan.transaction_id!r} "
                "scheduler transaction already started"
            )
        controller.require_armed_maintenance_discardable(plan.transaction_id)
        grant = getattr(self, "_elastic_preflight_admission_grant", None)
        if grant is not None and (
            grant.step_key != controller.pending_maintenance_step_key
            or grant.physical_keys != plan.physical_keys
            or grant.maintenance_transaction_id != plan.transaction_id
        ):
            raise RuntimeError(
                "stale elastic grant is not bound to pending maintenance"
            )
        self._rollback_elastic_admission()
        controller.discard_armed_maintenance(plan.transaction_id)
        self._elastic_last_defer_reason = reason
        return True

    def _bind_pending_elastic_maintenance_commit(
        self,
        execution_step_key: tuple[int, ...],
        prepared_residency_step_key: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Bind a committed execution to its prepared physical carrier."""
        committed_residency_step_key = execution_step_key
        if (
            prepared_residency_step_key is None
            or committed_residency_step_key != prepared_residency_step_key
        ):
            raise RuntimeError(
                "prepared elastic maintenance shape changed before commit: "
                f"execution={execution_step_key!r} "
                f"committed_residency={committed_residency_step_key!r} "
                f"prepared_residency={prepared_residency_step_key!r}"
            )
        return prepared_residency_step_key

    def _rebalance_elastic_capacity_before_commit(
        self,
        *,
        has_user_tokens: bool,
        maintenance_plan: ElasticStepPlan | None,
    ) -> bool:
        """Rebalance KV geometry only when this output owns a real commit."""
        if not (has_user_tokens or maintenance_plan is not None):
            return False
        self.kv_cache_manager.coordinator.rebalance_elastic_capacity()
        return True

    def _pending_elastic_maintenance_requires_exclusive_tick(self) -> bool:
        """Return whether pending work has no serving USER consumer."""
        plan = self._elastic_admission_controller.pending_maintenance_plan
        if plan is None:
            return False
        return bool(
            plan.kind
            in {
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }
            or plan.maintenance_execution == ElasticMaintenanceExecution.GRAPH_ONLY
            or getattr(self, "_elastic_restore_mode", False)
        )

    def _should_commit_pending_elastic_maintenance(
        self, *, has_user_tokens: bool
    ) -> bool:
        """Keep serving capture pending until its first USER can execute."""
        if self._elastic_admission_controller.pending_maintenance_plan is None:
            return False
        plan = self._elastic_admission_controller.pending_maintenance_plan
        assert plan is not None
        if (
            has_user_tokens
            and plan.kind == ElasticPlanKind.MAINTENANCE
            and plan.maintenance_execution == ElasticMaintenanceExecution.GRAPH_ONLY
        ):
            raise RuntimeError("graph-only maintenance cannot carry USER tokens")
        return bool(
            has_user_tokens
            or self._pending_elastic_maintenance_requires_exclusive_tick()
        )

    def _rollback_elastic_admission(self) -> None:
        grant = getattr(self, "_elastic_preflight_admission_grant", None)
        if grant is None:
            return
        coordinator = self.kv_cache_manager.coordinator
        coordinator.rebalance_elastic_capacity()
        if not coordinator.set_elastic_external_memory(
            grant.previous_external_memory_bytes
        ):
            raise RuntimeError(
                "cancelled elastic admission could not restore external memory"
            )
        self._elastic_preflight_admission_grant = None

    @staticmethod
    def _elastic_graph_owner_key(
        step_key: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if step_key is None:
            return None
        # FULL is exact-X. PIECEWISE executable identity remains exact-X too,
        # but its first-capture price may use a conservative class envelope
        # proven for the same K/owner-set and padded token carrier.
        if step_key[0]:
            return step_key
        return (0, step_key[1], 0, step_key[3], 0)

    @staticmethod
    def _elastic_graph_global_owner_key(
        step_key: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        """Return the worst-form PIECEWISE envelope key for one owner set."""
        if step_key is None or step_key[0]:
            return None
        return (0, step_key[1], 0, 0, 0)

    def _elastic_capture_envelope(
        self,
        step_key: tuple[int, ...] | None,
    ) -> tuple[int, int, int] | None:
        return self._elastic_capture_envelope_with_provenance(step_key)[0]

    def _elastic_capture_envelope_with_provenance(
        self,
        step_key: tuple[int, ...] | None,
    ) -> tuple[
        tuple[int, int, int] | None,
        tuple[tuple[str, int], ...] | None,
    ]:
        """Resolve an envelope and receipt-key provenance for its byte bound."""
        owner_key = self._elastic_graph_owner_key(step_key)
        if owner_key is None:
            return None, None
        physical_keys: tuple[PhysicalReplayKey, ...] = ()
        if step_key is not None and hasattr(self, "_elastic_admission_controller"):
            physical_keys = self._resolve_elastic_step_physical_keys(step_key)
            # One semantic M class may contain a compiled-only target plus
            # one FULL MTP Graph, or a three-Graph batched-decode DAG. Resolve
            # the complete physical set before the class fallback. Individual
            # GraphPrice entries omit capture/floor workspace and cannot price
            # the independent aggregate endpoint.
            # PIECEWISE calibration replaces provisional class envelopes while
            # mutating rows. Those partial rows are not sealed observations.
            if physical_keys and (
                not getattr(self, "_elastic_restore_mode", False)
                or not any(key.logical.mode == "PIECEWISE" for key in physical_keys)
            ):
                requested = frozenset(physical_keys)
                exact_rows = tuple(
                    row
                    for catalog_key, row in getattr(
                        self, "_elastic_graph_catalog", {}
                    ).items()
                    if frozenset(self._resolve_elastic_step_physical_keys(catalog_key))
                    == requested
                    and int(row.get("cold_peak_bytes", 0)) > 0
                )
                combined = self._combine_elastic_catalog_envelope_evidence(exact_rows)
                if combined[0] is not None:
                    return combined
        envelopes = self._elastic_admission_controller.capture_envelopes
        envelope = envelopes.get(owner_key)
        selected_owner_key: tuple[int, ...] | None = owner_key
        if envelope is None and (
            not physical_keys
            or any(key.logical.mode == "PIECEWISE" for key in physical_keys)
        ):
            selected_owner_key = self._elastic_graph_global_owner_key(step_key)
            envelope = envelopes.get(cast(tuple[int, ...], selected_owner_key))
        if envelope is not None:
            assert selected_owner_key is not None
            return (
                envelope,
                self._elastic_admission_controller.capture_envelope_resident_key_bytes(
                    selected_owner_key
                ),
            )

        # A diagnostic/product request may disable speculative decoding for
        # one request.  Its K0 FULL target descriptor is byte-for-byte the
        # target member of the sealed K3 FULL owner set; K is not part of the
        # target PhysicalReplayKey.  Requiring a second logical catalog row for
        # that strict subset made the scheduler call the physical planner with
        # an exact price but no step envelope.  The planner correctly produced
        # MAINTENANCE and the contradictory scheduler assertion killed the
        # engine.  Reuse only a sealed row whose complete physical owner set is
        # a superset of the requested set.  Its aggregate cold envelope is a
        # conservative loan for the identical subset; unrelated topology,
        # geometry and generation remain UNKNOWN.
        if (
            not getattr(self, "_elastic_restore_mode", False)
            and step_key is not None
            and hasattr(self, "_elastic_admission_controller")
        ):
            requested = frozenset(
                physical_keys or self._resolve_elastic_step_physical_keys(step_key)
            )
            if requested:
                compatible_rows = []
                for catalog_key, row in getattr(
                    self, "_elastic_graph_catalog", {}
                ).items():
                    catalog_owners = frozenset(
                        self._resolve_elastic_step_physical_keys(catalog_key)
                    )
                    if requested < catalog_owners:
                        compatible_rows.append(row)
                combined = self._combine_elastic_catalog_envelope_evidence(
                    tuple(compatible_rows)
                )
                if combined[0] is not None:
                    return combined
        return None, None

    @staticmethod
    def _elastic_catalog_resident_key_bytes(
        row: Mapping[str, Any],
    ) -> tuple[tuple[str, int], ...] | None:
        raw = row.get("resident_key_bytes")
        if not isinstance(raw, (list, tuple)):
            return None
        pairs: list[tuple[str, int]] = []
        for item in raw:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
            ):
                return None
            pairs.append((item[0], item[1]))
        if len({identity for identity, _value in pairs}) != len(pairs):
            return None
        return tuple(sorted(pairs))

    @classmethod
    def _combine_elastic_catalog_envelope_evidence(
        cls,
        rows: tuple[Mapping[str, Any], ...],
    ) -> tuple[
        tuple[int, int, int] | None,
        tuple[tuple[str, int], ...] | None,
    ]:
        if not rows:
            return None, None
        envelope = (
            max(_elastic_catalog_cold_residency_envelope(row) for row in rows),
            max(int(row.get("floor_bytes", 0)) for row in rows),
            0,
        )
        proven_rows = tuple(
            cls._elastic_catalog_resident_key_bytes(row) for row in rows
        )
        if any(proven is None for proven in proven_rows):
            return envelope, None
        assert proven_rows[0] is not None
        common = dict(proven_rows[0])
        for proven in proven_rows[1:]:
            assert proven is not None
            proven_map = dict(proven)
            common = {
                identity: min(value, proven_map[identity])
                for identity, value in common.items()
                if identity in proven_map
            }
        return envelope, tuple(sorted(common.items()))

    def _record_elastic_capture_envelope(
        self,
        step_key: tuple[int, ...],
        envelope: tuple[int, int, int],
        *,
        publish_global: bool = True,
        resident_key_bytes: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
    ) -> None:
        """Publish exact-M and cross-M worst-form PIECEWISE envelopes."""
        owner_key = self._elastic_graph_owner_key(step_key)
        assert owner_key is not None
        self._elastic_admission_controller.record_capture_envelope(
            owner_key,
            envelope,
            resident_key_bytes=resident_key_bytes,
        )
        global_key = self._elastic_graph_global_owner_key(step_key)
        if global_key is not None and publish_global:
            self._elastic_admission_controller.record_capture_envelope(
                global_key,
                envelope,
                merge_max=True,
                resident_key_bytes=resident_key_bytes,
            )

    def _elastic_piecewise_physical_floor_bytes(
        self,
        step_key: tuple[int, ...] | None,
    ) -> int:
        """Return a proven lower bound for the same PIECEWISE physical DAG.

        PIECEWISE target/speculator entries are keyed by their padded token
        boundary and are recaptured when the runtime request count changes.
        A settled cold observation at one X is therefore a real lower bound
        for the same K/M/query lane at another X.  Reusing that fact for
        admission prevents a larger-X calibration wave from reaching the
        worker with a loan smaller than an already resident descriptor.

        This is not a reserve: calibration still borrows the exact remaining
        KV tail for the step and settlement returns the excess immediately.
        FULL graphs retain exact-X pricing.
        """
        return self._elastic_piecewise_physical_floor_evidence(step_key)[0]

    def _elastic_piecewise_physical_floor_evidence(
        self,
        step_key: tuple[int, ...] | None,
    ) -> tuple[int, tuple[tuple[str, int], ...] | None]:
        """Return the lower bound plus evidence shared by every max source."""
        if step_key is None or step_key[0]:
            return 0, None
        if hasattr(self, "_elastic_admission_controller") and not any(
            key.logical.mode == "PIECEWISE"
            for key in self._resolve_elastic_step_physical_keys(step_key)
        ):
            return 0, None
        physical_lane = (step_key[0], step_key[1], step_key[3], step_key[4])
        candidates = tuple(
            (catalog_key, row, int(row.get("cold_peak_bytes", 0)))
            for catalog_key, row in getattr(self, "_elastic_graph_catalog", {}).items()
            if (catalog_key[0], catalog_key[1], catalog_key[3], catalog_key[4])
            == physical_lane
        )
        value = max((candidate[2] for candidate in candidates), default=0)
        if value <= 0:
            return 0, None
        winners = tuple(candidate for candidate in candidates if candidate[2] == value)
        proven_rows = tuple(
            self._elastic_catalog_resident_key_bytes(row)
            for _key, row, _value in winners
        )
        if any(proven is None for proven in proven_rows):
            return value, None
        assert proven_rows[0] is not None
        proven_key_bytes = dict(proven_rows[0])
        for proven in proven_rows[1:]:
            assert proven is not None
            proven_map = dict(proven)
            proven_key_bytes = {
                identity: min(value, proven_map[identity])
                for identity, value in proven_key_bytes.items()
                if identity in proven_map
            }
        return value, tuple(sorted(proven_key_bytes.items()))

    @staticmethod
    def _adjust_elastic_capture_envelope(
        prior_capture: tuple[int, int, int],
        current_floor: int,
        sampling_workspace: int,
    ) -> tuple[int, int, int]:
        prior_grant, prior_floor, prior_sampling_workspace = prior_capture
        # A destination catalog row already includes the floor observed while
        # measuring that row.  A smaller pre-transition floor is not proof that
        # eviction/capture of the next descriptor will retain fewer bytes; the
        # first live K3 PIECEWISE->FULL transition demonstrated the opposite.
        # Add newly observed floor growth, but never subtract historical floor
        # from a measured destination envelope.
        floor_delta = max(0, current_floor - prior_floor)
        sampling_delta = sampling_workspace - prior_sampling_workspace
        desired_external = max(
            0,
            prior_grant + floor_delta + sampling_delta,
        )
        return desired_external, floor_delta, sampling_delta

    def _elastic_capture_shared_evidence(
        self,
        step_key: tuple[int, ...] | None,
        desired_external: int,
    ) -> tuple[tuple[str, int], ...] | None:
        """Bind sharing only to the evidence source that sets the byte bound."""
        prior, prior_keys = self._elastic_capture_envelope_with_provenance(step_key)
        prior_value: int | None = None
        if prior is not None:
            prior_value = self._adjust_elastic_capture_envelope(
                prior,
                0,
                0,
            )[0]
        floor_value, floor_keys = self._elastic_piecewise_physical_floor_evidence(
            step_key
        )
        measured = self._elastic_admission_controller.measured_bytes.get(
            cast(tuple[int, ...], step_key)
        )
        prior_selected = prior_value is not None and prior_value == desired_external
        floor_selected = floor_value > 0 and floor_value == desired_external
        unknown_selected = (
            prior is None and measured is not None and measured == desired_external
        )
        if unknown_selected:
            return None
        if prior_selected and floor_selected:
            if prior_keys is None or floor_keys is None:
                return None
            floor_map = dict(floor_keys)
            return tuple(
                sorted(
                    (
                        identity,
                        min(value, floor_map[identity]),
                    )
                    for identity, value in prior_keys
                    if identity in floor_map
                )
            )
        if prior_selected:
            return prior_keys
        if floor_selected:
            return floor_keys
        return None

    def _elastic_capture_shared_resident_bytes(
        self,
        destination_keys: Iterable[PhysicalReplayKey],
        evidence: tuple[tuple[str, int], ...] | None,
    ) -> int:
        """Intersect a priced destination with completed, still-HOT residency."""
        controller = self._elastic_admission_controller
        receipt = controller.last_receipt
        observed = (
            {}
            if receipt is None
            else {entry.key.identity: entry.resident_bytes for entry in receipt.entries}
        )
        proven = dict(evidence or ())
        return sum(
            min(observed.get(key.identity, 0), proven.get(key.identity, 0))
            for key in set(destination_keys)
            if (entry := controller.entries.get(key)) is not None and entry.hot
        )

    def _estimate_elastic_graph_step_bytes(
        self,
        step_key: tuple[int, ...] | None,
    ) -> tuple[int, bool]:
        """Return a HOT grant or a COLD destination-set endpoint.

        This is deliberately side-effect free so waiting admission can test the
        next runtime X/M before allocating KV blocks or changing request state.
        For ``capture_planned=True`` the byte value excludes current residency
        and source-workspace overlap. ``ElasticAdmissionController.plan`` is the
        sole owner that composes those simultaneously-live terms and deducts
        worker-proved sharing. For ``capture_planned=False`` the byte value is
        the complete HOT/compiled step grant. The worker-facing planner remains
        the sole owner of committing the loan, publishing a provisional
        envelope and advancing FIFO state.
        """
        pending_same_key_grant = next(
            (
                grant
                for loan in reversed(self._elastic_admission_controller.pending_loans)
                if loan.step_key == step_key
                for grant in (loan.grant_bytes,)
            ),
            0,
        )
        current_residency_step_key = self._elastic_admission_controller.step_key
        physical_owner_set_cold = False
        if step_key is None:
            # X0 is itself the next-step decision. The worker evicts every
            # dynamic owner before applying this grant, so carrying the prior
            # step's residency through an extra empty output only delays the
            # return to KV and creates a false two-step cleanup protocol.
            return 0, False
        if hasattr(self, "_elastic_admission_controller") and hasattr(
            self, "_elastic_admission_controller"
        ):
            physical_keys = self._resolve_elastic_step_physical_keys(step_key)
            if not physical_keys:
                # Some semantic PIECEWISE carriers (notably pure K0/B4096)
                # intentionally execute through torch.compile and own no
                # persistent CUDA Graph.  This consumed contract is identical
                # during calibration and serving.  Falling through to a
                # logical/global capture envelope in calibration can turn the
                # full prospective KV tail from an earlier witness into a
                # fictitious multi-GiB Graph floor after request allocation.
                return self._elastic_admission_controller.resident_bytes, False
        if (
            getattr(self, "_elastic_restore_mode", False)
            and hasattr(self, "_elastic_admission_controller")
            and hasattr(self, "_elastic_admission_controller")
            and all(
                (entry := self._elastic_admission_controller.entries.get(key))
                is not None
                and entry.hot
                for key in physical_keys
            )
        ):
            # Request-free maintenance publishes the Graph owners before the
            # first real calibration replay. That replay can materialize one
            # measured cuBLAS/runtime workspace just as it can in serving.
            # Include the unit in finite-wave preflight so an oversized cohort
            # contracts before KV/request mutation instead of overrunning its
            # committed loan during settlement.
            first_replay_workspace = (
                self._elastic_admission_controller.cublas_workspace_bytes
                if self._elastic_admission_controller.last_maintenance_step_key
                == step_key
                else 0
            )
            return max(
                self._elastic_admission_controller.resident_bytes
                + first_replay_workspace,
                self._elastic_admission_controller.measured_bytes.get(step_key, 0),
            ), False
        if (
            not getattr(self, "_elastic_restore_mode", False)
            and hasattr(self, "_elastic_admission_controller")
            and hasattr(self, "_elastic_admission_controller")
        ):
            physical_owner_set_hot = bool(physical_keys) and all(
                (entry := self._elastic_admission_controller.entries.get(key))
                is not None
                and entry.hot
                for key in physical_keys
            )
            physical_owner_set_cold = bool(physical_keys) and not (
                physical_owner_set_hot
            )
            if physical_owner_set_hot:
                # Physical residency, not the preceding logical step key, is
                # authoritative. Alternating K3 verification/acceptance FULL
                # shapes and retained PIECEWISE hits must not reopen their cold
                # catalog envelopes, remap KV, or enter capture accounting.
                # The graph entries are not the whole executable envelope:
                # replay can also materialize a shape-specific cuBLAS/runtime
                # workspace.  Maintenance publishes the entries before that
                # replay, so resident_bytes alone would underfund the first
                # real HOT hit.  A sealed exact row's HOT measurement is the
                # consumed-path price for this already-resident owner set.
                hot_envelope = self._elastic_hot_replay_envelope(step_key)
                first_replay_workspace = (
                    self._elastic_admission_controller.cublas_workspace_bytes
                    if self._elastic_admission_controller.last_maintenance_step_key
                    == step_key
                    else 0
                )
                return max(
                    self._elastic_admission_controller.resident_bytes
                    + first_replay_workspace,
                    hot_envelope,
                ), False
        if (
            step_key == current_residency_step_key
            and pending_same_key_grant
            and (
                step_key not in self._elastic_admission_controller.measured_bytes
                or self._elastic_admission_controller.recapture_pending_key == step_key
            )
        ):
            return pending_same_key_grant, False
        if step_key == current_residency_step_key and getattr(
            self, "_elastic_restore_mode", False
        ):
            # The cold epoch may have expanded a known row.  Its immediate HOT
            # replay must inherit that just-measured maximum, not the smaller
            # hot value from an earlier epoch.  After seal, the independently
            # stabilized hot envelope remains authoritative.
            row = getattr(self, "_elastic_graph_catalog", {}).get(step_key)
            cold_peak = 0 if row is None else int(row.get("cold_peak_bytes", 0))
            if cold_peak:
                return cold_peak, False
        if (
            step_key == current_residency_step_key
            and step_key in self._elastic_admission_controller.measured_bytes
            and not physical_owner_set_cold
        ):
            return self._elastic_admission_controller.measured_bytes[step_key], False

        prior_capture = self._elastic_capture_envelope(step_key)
        if (
            step_key == current_residency_step_key
            and prior_capture is not None
            and not physical_owner_set_cold
        ):
            # The first replay after cold calibration still uses the cold
            # envelope; its settlement records the smaller HOT high-water.
            return prior_capture[0], False
        measured_external = self._elastic_admission_controller.measured_bytes.get(
            step_key
        )
        if prior_capture is not None:
            desired_external, _floor_delta, _sampling_delta = (
                self._adjust_elastic_capture_envelope(
                    prior_capture,
                    0,
                    0,
                )
            )
            # This is the independent destination endpoint, including its own
            # historical floor. C + D - S already carries current residency and
            # source teardown leftovers; adding the source transition floor to
            # D would charge it twice. Reclaim reduces C only by physical proof.
        elif measured_external is not None:
            desired_external = max(
                measured_external, self._elastic_admission_controller.floor_bytes
            )
        else:
            # Zero is an explicit UNKNOWN sentinel here, not a claim that a
            # capture is free. _can_fund/_plan replace it with the exact
            # prospective physical tail returned by the coordinator.
            desired_external = 0
        desired_external = max(
            desired_external,
            self._elastic_piecewise_physical_floor_bytes(step_key),
        )
        return desired_external, True

    def _elastic_catalog_prices_are_sealed(self) -> bool:
        """Serving consumes isolated prices; live union receipts are not prices.

        Activation validates this source digest before accepting any rows.
        Explicit calibration may still collect new isolated witnesses, while
        unsealed diagnostic schedulers retain their measurement lifecycle.
        """
        return bool(
            not getattr(self, "_elastic_restore_mode", False)
            and getattr(self, "_elastic_graph_catalog_coverage", {}).get(
                "_catalog_source_sha256"
            )
        )

    def _elastic_hot_replay_envelope(self, step_key: tuple[int, ...]) -> int:
        """Resolve an aggregate HOT price for the identical physical owner set.

        Logical prefill M may change while compiled target/prefill paths own
        no graph and the sole FULL draft owner stays identical. A missing
        logical row is not missing physical evidence. Do not sum owner prices
        or substitute a cold peak: shared replay workspace belongs to the
        independently measured aggregate HOT endpoint.
        """
        measured = self._elastic_admission_controller.measured_bytes
        if step_key in measured:
            return measured[step_key]
        requested = frozenset(self._resolve_elastic_step_physical_keys(step_key))
        if not requested:
            return 0
        return max(
            (
                int(row.get("hot_peak_bytes", 0))
                for catalog_key, row in getattr(
                    self, "_elastic_graph_catalog", {}
                ).items()
                if frozenset(self._resolve_elastic_step_physical_keys(catalog_key))
                == requested
            ),
            default=0,
        )

    def _can_fund_elastic_graph_step(
        self,
        step_key: tuple[int, ...] | None,
        *,
        minimum_free_primary_blocks: int,
        minimum_attention_blocks: int = 0,
        gdn_blocks: int | None = None,
        allow_maintenance: bool = False,
        mm_activation_loan_bytes: int = 0,
        preview_only: bool = False,
    ) -> tuple[bool, int, int]:
        """Check prospective graph/KV coexistence.

        ``preview_only`` is a strict read-only query used to compare FCFS
        prefixes.  In that mode the first result means that the candidate is
        executable either immediately or after one feasible maintenance
        transaction.  No transaction id, pending plan, defer counter or
        diagnostic state is changed.
        """
        if mm_activation_loan_bytes < 0:
            raise ValueError("MM activation loan cannot be negative")
        if not getattr(self, "elastic_on_demand_graphs", False):
            return True, 0, 0
        residency_step_key = step_key
        coordinator = self.kv_cache_manager.coordinator
        _destination_current_keys, _successor_keys, physical_keys = (
            self._elastic_step_residency_intent(residency_step_key)
        )
        pending_plan = self._elastic_admission_controller.pending_maintenance_plan
        if (
            pending_plan is not None
            and pending_plan.kind == ElasticPlanKind.MAINTENANCE
            and (
                self._elastic_admission_controller.pending_maintenance_step_key
                != residency_step_key
                or pending_plan.physical_keys != physical_keys
            )
        ):
            if preview_only:
                return False, 0, 0
            # Serving maintenance is only a pre-mutation proposal. Queue or
            # prefix drift invalidates it. Restore its tentative KV mapping
            # before taking any resource snapshot for the replacement wave.
            self._clear_pre_mutation_serving_maintenance(
                reason="stale_maintenance_shape_replanned"
            )
            pending_plan = None
        desired_external, capture_envelope_planned = (
            self._estimate_elastic_graph_step_bytes(residency_step_key)
        )
        # HOT replay has no destination capture endpoint or sharing deduction.
        # Keep exact-set catalog resolution off the steady decode path.
        prior_capture = None
        destination_envelope_key_bytes = None
        if capture_envelope_planned:
            prior_capture, _prior_envelope_keys = (
                self._elastic_capture_envelope_with_provenance(residency_step_key)
            )
            destination_envelope_key_bytes = self._elastic_capture_shared_evidence(
                residency_step_key,
                desired_external,
            )
        cold_unknown = (
            capture_envelope_planned
            and prior_capture is None
            and residency_step_key
            not in self._elastic_admission_controller.measured_bytes
        )
        unsettled_grant = self._elastic_admission_controller.max_pending_grant()
        available_external = coordinator.max_elastic_external_memory(
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            minimum_attention_blocks=minimum_attention_blocks,
            gdn_blocks=gdn_blocks,
        )
        if pending_plan is not None:
            # The exact immutable COUPLED_USER proposal is already the single
            # maintenance transition promised by preview semantics. Its
            # capture loan, not a partially populated catalog row, is the
            # authority for this pending transaction.
            if preview_only:
                return (
                    pending_plan.capture_loan_bytes <= available_external,
                    pending_plan.capture_loan_bytes,
                    available_external,
                )
            return False, pending_plan.capture_loan_bytes, available_external
        current_external = coordinator.elastic_external_memory_bytes
        # The worker measures a destination set independently, but exact HOT
        # destination owners may already be present in the current set. Their
        # receipt-backed resident bytes are a proved intersection and must be
        # deducted once. Retained successors qualify only when the selected
        # endpoint's receipt explicitly proves the same allocation identity.
        shared_resident_bytes = self._elastic_capture_shared_resident_bytes(
            physical_keys, destination_envelope_key_bytes
        )
        compiled_only_step = bool(
            step_key is not None
            and not physical_keys
            and step_key[0] == 0
            and step_key[1] == 0
            and step_key[3]
            in getattr(self, "_elastic_compiled_piecewise_sizes", frozenset())
        )
        cold_keys = tuple(
            key
            for key in physical_keys
            if not (
                (entry := self._elastic_admission_controller.entries.get(key))
                and entry.hot
            )
        )
        cold_mtp_decode = tuple(
            key for key in cold_keys if key.logical.owner == "mtp_decode"
        )
        if (
            cold_mtp_decode
            and not getattr(self, "_elastic_restore_mode", False)
            and (
                terminal_x := getattr(self, "_elastic_terminal_decode_carrier_x", None)
            )
            is not None
        ):
            serving_carrier = tuple(
                key
                for key in getattr(self, "_elastic_serving_carrier_keys", ())
                if key.logical.owner == "mtp_decode"
                and key.physical_num_reqs == terminal_x
                and (
                    (entry := self._elastic_admission_controller.entries.get(key))
                    is not None
                    and entry.hot
                )
            )
            if len(cold_mtp_decode) != 1 or len(serving_carrier) != 1:
                raise RuntimeError(
                    "serving MTP decode cold capture requires exactly one HOT "
                    "terminal recovery carrier and one current execution key: "
                    f"terminal_x={terminal_x} "
                    f"step_key={residency_step_key!r} "
                    f"cold={tuple(key.identity for key in cold_mtp_decode)!r} "
                    f"carrier={tuple(key.identity for key in serving_carrier)!r}"
                )
        # ``current_external`` already contains the source workspace and the
        # measured destination endpoint/peak already contains its workspace.
        # Their C + D - S composition therefore carries both lifetimes. A
        # third inferred workspace would double-count the destination cost;
        # only a separately measured transition allocation may populate this
        # typed term in a future catalog schema.
        retained_transition_overlap_bytes = 0
        calibration_cold_capture = bool(
            cold_keys and getattr(self, "_elastic_restore_mode", False)
        )
        priced_product_cold_capture = bool(
            cold_keys
            and not getattr(self, "_elastic_restore_mode", False)
            and not cold_unknown
        )
        if cold_keys and mm_activation_loan_bytes:
            # CUDA Graph capture and the MM encoder are distinct physical
            # phases.  Do not charge their peaks additively, and do not execute
            # either against an unproved future state.  A read-only preflight
            # may arm one graph-only transaction only when both the capture
            # phase and the later HOT+MM phase fit the same prospective KV wave.
            if pending_plan is not None:
                if preview_only:
                    return False, 0, available_external
                self._clear_pre_mutation_serving_maintenance(
                    reason="mm_cold_transition_replanned"
                )
                pending_plan = None
            assert residency_step_key is not None
            hot_endpoint_bytes = self._elastic_hot_replay_envelope(residency_step_key)
            if cold_unknown or hot_endpoint_bytes <= 0:
                if not preview_only:
                    self._elastic_last_defer_reason = "missing_hot_mm_replay_envelope"
                return False, 0, available_external
            post_shared_resident_bytes = min(shared_resident_bytes, hot_endpoint_bytes)
            if not allow_maintenance:
                hot_union_bytes = (
                    self._elastic_admission_controller.compose_destination_capture_loan(
                        current_residency_bytes=current_external,
                        destination_capture_endpoint_bytes=hot_endpoint_bytes,
                        shared_resident_bytes=post_shared_resident_bytes,
                    )
                )
                hot_mm_required = coordinator.normalize_elastic_external_memory(
                    hot_union_bytes + mm_activation_loan_bytes
                )
                if not preview_only:
                    self._elastic_last_defer_reason = (
                        "mm_cold_transition_requires_wave_preflight"
                    )
                return False, hot_mm_required, available_external
            plan = self._elastic_admission_controller.plan(
                (
                    "elastic-read-only-preview"
                    if preview_only
                    else self._next_elastic_transaction_id()
                ),
                physical_keys,
                request_bytes=current_external,
                available_bytes=available_external,
                destination_capture_endpoint_bytes=desired_external,
                retained_transition_overlap_bytes=(retained_transition_overlap_bytes),
                shared_resident_bytes=shared_resident_bytes,
                post_transition_endpoint_bytes=hot_endpoint_bytes,
                post_transition_shared_resident_bytes=post_shared_resident_bytes,
                post_transition_extra_bytes=mm_activation_loan_bytes,
                maintenance_execution=ElasticMaintenanceExecution.GRAPH_ONLY,
            )
            if plan.kind == ElasticPlanKind.MAINTENANCE:
                post_capture_current = max(0, current_external - plan.reclaim_bytes)
                hot_union_bytes = (
                    self._elastic_admission_controller.compose_destination_capture_loan(
                        current_residency_bytes=post_capture_current,
                        destination_capture_endpoint_bytes=hot_endpoint_bytes,
                        shared_resident_bytes=post_shared_resident_bytes,
                    )
                )
                hot_mm_required = coordinator.normalize_elastic_external_memory(
                    hot_union_bytes + mm_activation_loan_bytes
                )
                if hot_mm_required > available_external:
                    raise RuntimeError(
                        "graph-only MM plan did not fund its HOT consumer"
                    )
                if plan.capture_loan_bytes > available_external:
                    raise RuntimeError(
                        "graph-only MM preflight produced an over-budget capture"
                    )
                if preview_only:
                    return (
                        True,
                        max(plan.capture_loan_bytes, hot_mm_required),
                        available_external,
                    )
                self._elastic_admission_controller.arm_maintenance(
                    plan, residency_step_key
                )
                self._elastic_last_defer_reason = "mm_graph_capture_pending"
                return (
                    False,
                    max(plan.capture_loan_bytes, hot_mm_required),
                    available_external,
                )
            if plan.kind == ElasticPlanKind.DEFER:
                hot_union_bytes = (
                    self._elastic_admission_controller.compose_destination_capture_loan(
                        current_residency_bytes=current_external,
                        destination_capture_endpoint_bytes=hot_endpoint_bytes,
                        shared_resident_bytes=post_shared_resident_bytes,
                    )
                )
                hot_mm_required = coordinator.normalize_elastic_external_memory(
                    hot_union_bytes + mm_activation_loan_bytes
                )
                if not preview_only:
                    self._elastic_admission_controller.observe_defer(plan)
                    self._elastic_last_defer_reason = plan.defer_reason
                return (
                    False,
                    max(plan.capture_loan_bytes, hot_mm_required),
                    available_external,
                )
            raise RuntimeError("cold MM preflight unexpectedly produced a USER plan")
        if calibration_cold_capture or priced_product_cold_capture:
            # Discovery is safe before scheduler mutation. The immutable plan
            # is bound to this exact preflight key and, when admission commits,
            # capture and the first USER consumer execute in the same
            # transaction. This removes the request-free shape-drift window.
            if not allow_maintenance:
                if not preview_only:
                    self._elastic_last_defer_reason = "final_shape_maintenance_required"
                    self._elastic_last_defer_step_key = residency_step_key
                    self._elastic_last_defer_physical_keys = tuple(
                        key.identity for key in physical_keys
                    )
                required_capture_loan = (
                    self._elastic_admission_controller.compose_destination_capture_loan(
                        current_residency_bytes=current_external,
                        destination_capture_endpoint_bytes=desired_external,
                        retained_transition_overlap_bytes=(
                            retained_transition_overlap_bytes
                        ),
                        shared_resident_bytes=shared_resident_bytes,
                    )
                )
                return False, required_capture_loan, available_external
            planned_capture_loan = desired_external
            if self._elastic_admission_controller.pending_maintenance_plan is None:
                transaction_id = (
                    "elastic-read-only-preview"
                    if preview_only
                    else self._next_elastic_transaction_id()
                )
                capture_envelope_bytes = (
                    max(
                        0,
                        available_external - retained_transition_overlap_bytes,
                    )
                    if calibration_cold_capture
                    else desired_external
                )
                plan = self._elastic_admission_controller.plan(
                    transaction_id,
                    physical_keys,
                    request_bytes=current_external,
                    available_bytes=available_external,
                    destination_capture_endpoint_bytes=capture_envelope_bytes,
                    retained_transition_overlap_bytes=(
                        retained_transition_overlap_bytes
                    ),
                    # Calibration constructs an all-available endpoint that
                    # already contains the complete current state. Product
                    # rows are independent measured endpoints and may deduct
                    # only the worker-published shared-pool component.
                    shared_resident_bytes=(
                        current_external
                        if calibration_cold_capture
                        else shared_resident_bytes
                    ),
                    maintenance_execution=(
                        ElasticMaintenanceExecution.GRAPH_ONLY
                        if calibration_cold_capture
                        else ElasticMaintenanceExecution.COUPLED_USER
                    ),
                )
                planned_capture_loan = plan.capture_loan_bytes
                if plan.kind == ElasticPlanKind.MAINTENANCE:
                    if preview_only:
                        return True, plan.capture_loan_bytes, available_external
                    self._elastic_admission_controller.arm_maintenance(
                        plan, residency_step_key
                    )
                    self._elastic_last_defer_reason = "cold_promotion_pending"
                    return False, plan.capture_loan_bytes, available_external
                if not preview_only:
                    self._elastic_last_defer_reason = plan.defer_reason
                if plan.kind == ElasticPlanKind.DEFER and not preview_only:
                    self._elastic_admission_controller.observe_defer(plan)
            else:
                pending_plan = (
                    self._elastic_admission_controller.pending_maintenance_plan
                )
                assert pending_plan is not None
                planned_capture_loan = pending_plan.capture_loan_bytes
            return False, planned_capture_loan, available_external
        physical_resident_lower_bound = (
            self._elastic_admission_controller.resident_bytes
            if step_key is not None
            else 0
        )
        proven_required = coordinator.normalize_elastic_external_memory(
            max(desired_external, physical_resident_lower_bound)
            + mm_activation_loan_bytes
        )
        if step_key is not None and getattr(self, "_elastic_restore_mode", False):
            # Before seal, every logical witness is a measurement epoch. Two
            # mixed request distributions can share one graph descriptor yet
            # exercise different eager workspaces, so even a HOT descriptor
            # receives the exact prospective tail during calibration.
            desired_external = available_external
        elif cold_unknown:
            # Accepted product execution never turns "all remaining bytes"
            # into a capture-price proof.  The candidate remains uncommitted
            # until startup/calibration publishes an identity-compatible exact
            # price or conservative class envelope.
            if not preview_only:
                self._elastic_last_defer_reason = (
                    "missing_exact_price_or_class_envelope"
                )
            if allow_maintenance and physical_keys and not preview_only:
                deferred = self._elastic_admission_controller.plan(
                    self._next_elastic_transaction_id(),
                    physical_keys,
                    request_bytes=current_external,
                    available_bytes=available_external,
                )
                if deferred.kind != ElasticPlanKind.DEFER:
                    raise RuntimeError("unpriced graph unexpectedly produced a plan")
                self._elastic_admission_controller.observe_defer(deferred)
            return False, 0, available_external
        desired_external = coordinator.normalize_elastic_external_memory(
            desired_external + mm_activation_loan_bytes
        )
        transition_external = (
            current_external
            if residency_step_key is not None
            and self._elastic_admission_controller.step_key is not None
            and residency_step_key != self._elastic_admission_controller.step_key
            else 0
        )
        # The next worker step releases owners that its descriptor cannot use
        # before applying this loan. The current step's grant is therefore not
        # a permanent lower bound. Only older FIFO work and the proposed step
        # must coexist in scheduler state.
        required_external = max(
            unsettled_grant,
            desired_external,
            proven_required,
            transition_external,
            current_external if step_key is not None else 0,
        )
        return (
            (step_key is None or required_external > 0 or compiled_only_step)
            and proven_required <= available_external
            and required_external <= available_external,
            required_external,
            available_external,
        )

    def _elastic_idle_cleanup_outstanding(self) -> bool:
        """Return whether administrative X0 still has reclaimable work.

        Pinned HOT residency remains a valid steady-state cache and therefore
        does not keep calibration alive. Only an evictable executable or a
        measured physical teardown floor requires another X0 tick.
        """
        return bool(
            self._elastic_admission_controller.evictable_resident_bytes
            or self._elastic_admission_controller.floor_bytes
        )

    def _needs_elastic_idle_reclaim(self) -> bool:
        """Derive the sole request-free reclaim predicate from physical state."""
        return bool(
            self.elastic_on_demand_graphs
            # Startup calibration explicitly tears down each independent
            # measurement epoch. Ordinary serving retains its last HOT set;
            # replacement and pressure plans already carry explicit victims.
            # Treating every evictable serving graph as scheduler work keeps
            # EngineCore and all TP workers in an unbounded empty-step loop.
            and self._elastic_restore_mode
            and not self.running
            and not len(getattr(self, "waiting", ()))
            and not len(getattr(self, "skipped_waiting", ()))
            and not self._elastic_admission_controller.pending_loans
            and self._elastic_idle_cleanup_outstanding()
        )

    def _plan_elastic_graph_loan(
        self,
        step_key: tuple[int, ...] | None,
        *,
        minimum_free_primary_blocks: int,
        mm_activation_loan_bytes: int = 0,
    ) -> int:
        if not self.elastic_on_demand_graphs:
            return 0
        if mm_activation_loan_bytes < 0:
            raise ValueError("MM activation loan cannot be negative")
        if step_key is None and mm_activation_loan_bytes:
            raise RuntimeError("an MM activation loan requires an execution shape")
        residency_step_key = step_key
        coordinator = self.kv_cache_manager.coordinator
        if minimum_free_primary_blocks < 0:
            raise ValueError("elastic successor headroom cannot be negative")
        minimum_free = minimum_free_primary_blocks
        unsettled_grant = self._elastic_admission_controller.max_pending_grant()
        current_external = coordinator.elastic_external_memory_bytes
        pending_maintenance = (
            self._elastic_admission_controller.pending_maintenance_plan
        )
        if mm_activation_loan_bytes:
            physical_keys = self._elastic_step_residency_intent(residency_step_key)[2]
            if any(
                not (
                    (entry := self._elastic_admission_controller.entries.get(key))
                    and entry.hot
                )
                for key in physical_keys
            ):
                raise RuntimeError(
                    "MM activation cannot overlap a cold Graph transition"
                )
        if (
            pending_maintenance is not None
            and pending_maintenance.kind == ElasticPlanKind.MAINTENANCE
            and step_key
            == self._elastic_admission_controller.pending_maintenance_step_key
        ):
            step_grant = coordinator.normalize_elastic_external_memory(
                pending_maintenance.capture_loan_bytes
            )
            available_external = coordinator.max_elastic_external_memory(
                minimum_free_primary_blocks=minimum_free
            )
            if step_grant <= 0 or step_grant > available_external:
                raise RuntimeError(
                    "planned maintenance capture no longer fits the prospective "
                    "KV boundary: "
                    f"required_bytes={step_grant} "
                    f"available_bytes={available_external} step_key={step_key!r}"
                )
            if not coordinator.set_elastic_external_memory(
                step_grant,
                minimum_free_primary_blocks=minimum_free,
            ):
                raise RuntimeError(
                    "planned maintenance capture loan could not be committed"
                )
            self._elastic_admission_controller.reserve_loan(step_key, step_grant)
            self._elastic_admission_controller.mark_recapture(
                cast(tuple[int, ...], residency_step_key)
            )
            return step_grant
        if step_key is None and (
            pending_maintenance is None
            or pending_maintenance.kind
            not in {ElasticPlanKind.RECLAIM, ElasticPlanKind.PRESSURE_RECLAIM}
        ):
            # There is no next execution shape yet, so neither graph nor KV is
            # a useful consumer. Preserve the last settled HOT working set and
            # its loan. Only an admitted RECLAIM may evict before KV growth;
            # an idle-work predicate does not supply that worker transaction.
            step_grant = max(unsettled_grant, current_external)
            self._elastic_admission_controller.reserve_loan(None, step_grant)
            return step_grant
        sampling_workspace = 0
        owner_key = self._elastic_graph_owner_key(residency_step_key)
        desired_external, capture_envelope_planned = (
            self._estimate_elastic_graph_step_bytes(residency_step_key)
        )
        proven_required = coordinator.normalize_elastic_external_memory(
            desired_external
        )
        calibration_step = step_key is not None and getattr(
            self, "_elastic_restore_mode", False
        )
        prior_capture = None
        cold_unknown = False
        if capture_envelope_planned:
            available_external = coordinator.max_elastic_external_memory(
                minimum_free_primary_blocks=minimum_free
            )
            prior_capture = self._elastic_capture_envelope(residency_step_key)
            cold_unknown = (
                prior_capture is None
                and residency_step_key
                not in self._elastic_admission_controller.measured_bytes
            )
            if prior_capture is not None:
                prior_grant = prior_capture[0]
                desired_external, floor_delta, sampling_delta = (
                    self._adjust_elastic_capture_envelope(
                        prior_capture,
                        self._elastic_admission_controller.transition_floor_bytes,
                        sampling_workspace,
                    )
                )
                logger.debug(
                    "Elastic CUDA Graph measured envelope reused: "
                    "owner_key=%s step_key=%s prior_grant=%d "
                    "floor_delta=%d sampling_delta=%d desired_bytes=%d",
                    owner_key,
                    step_key,
                    prior_grant,
                    floor_delta,
                    sampling_delta,
                    desired_external,
                )
            elif cold_unknown and not calibration_step:
                raise RuntimeError(
                    "unpriced elastic CUDA Graph reached committed product "
                    "execution; candidate admission must defer before KV/request "
                    f"mutation: owner_key={owner_key!r} step_key={step_key!r}"
                )
            if getattr(self, "_elastic_restore_mode", False):
                # A measured row is not trusted until every declared witness
                # has observed a non-expanding replay. Discovery therefore
                # borrows the full rank-safe tail and settlement immediately
                # returns the excess to KV.
                if proven_required > available_external:
                    raise RuntimeError(
                        "scheduler admitted a calibration step below its proven "
                        "PIECEWISE physical floor: "
                        f"required_bytes={proven_required} "
                        f"available_bytes={available_external} "
                        f"step_key={step_key!r}"
                    )
                desired_external = available_external
            if desired_external <= 0 or desired_external > available_external:
                raise RuntimeError(
                    "current scheduler admission cannot fund the measured "
                    "CUDA Graph capture estimate: "
                    f"required_bytes={desired_external} "
                    f"available_bytes={available_external} "
                    f"step_key={step_key!r} owner_key={owner_key!r} "
                    f"current_step_key={self._elastic_admission_controller.step_key!r}"
                )
            self._elastic_admission_controller.mark_recapture(
                cast(tuple[int, ...], residency_step_key)
            )
        elif calibration_step:
            # PIECEWISE graph identity intentionally coalesces request-length
            # distributions. Their eager attention/sampling workspaces can
            # still have different high-water marks, so a new logical witness
            # must not execute against the smaller envelope measured by an
            # earlier witness of the same HOT descriptor.
            available_external = coordinator.max_elastic_external_memory(
                minimum_free_primary_blocks=minimum_free
            )
            if proven_required > available_external:
                raise RuntimeError(
                    "scheduler admitted a calibration witness below its proven "
                    "physical floor: "
                    f"required_bytes={proven_required} "
                    f"available_bytes={available_external} "
                    f"step_key={step_key!r}"
                )
            desired_external = available_external
        step_grant = coordinator.normalize_elastic_external_memory(desired_external)
        if residency_step_key is not None:
            # Settled worker residency can include quarantined wrappers that
            # are intentionally retained until X0. It is a lower bound for
            # every useful step even when execution keys share one carrier.
            # The coordinator contains the aggregate external-memory grant,
            # including any same-step MM activation loan.  Remove that typed
            # component before deriving the Graph lower bound; otherwise MM is
            # charged twice or, for a compiled-only carrier, live Graph bytes
            # are relabelled as MM.  Preserve a larger graph-only coordinator
            # envelope because it can be an unsettled capture high-water mark.
            resident_external = self._elastic_admission_controller.resident_bytes
            settled_external = (
                resident_external
                if self._elastic_admission_controller.pending_loans
                else max(
                    resident_external,
                    current_external - mm_activation_loan_bytes,
                )
            )
            step_grant = max(
                step_grant,
                settled_external,
            )
        if capture_envelope_planned and step_grant > available_external:
            raise RuntimeError(
                "current scheduler admission cannot carry changed-key CUDA "
                "Graph residency through capture: "
                f"required_bytes={step_grant} "
                f"available_bytes={available_external} "
                f"step_key={step_key!r} "
                f"current_step_key={self._elastic_admission_controller.step_key!r}"
            )
        if capture_envelope_planned and not self._elastic_catalog_prices_are_sealed():
            # Admission is already fail-closed above. Publish the normalized
            # provisional envelope immediately so async duplicates can share
            # it. Settlement must replace both its bytes and floor with worker
            # measurement; retaining a cold full-tail grant here would turn
            # discovery into a permanent fake reserve on every recapture.
            assert owner_key is not None
            self._record_elastic_capture_envelope(
                cast(tuple[int, ...], residency_step_key),
                (
                    step_grant,
                    self._elastic_admission_controller.floor_bytes,
                    sampling_workspace,
                ),
                publish_global=False,
            )
        graph_step_grant = step_grant
        if mm_activation_loan_bytes:
            step_grant = coordinator.normalize_elastic_external_memory(
                graph_step_grant + mm_activation_loan_bytes
            )
            available_external = coordinator.max_elastic_external_memory(
                minimum_free_primary_blocks=minimum_free
            )
            if step_grant > available_external:
                raise RuntimeError(
                    "same-step Graph+MM external loan exceeds the prospective KV "
                    f"boundary: required_bytes={step_grant} "
                    f"available_bytes={available_external} step_key={step_key!r}"
                )
        # KV admission must remain safe for older outstanding work. Changed-key
        # transition residency is already included in step_grant above; it is
        # deliberately excluded for same-key replay and X0 so it cannot become
        # a sticky reserve or require a redundant cleanup output.
        safe_external = max(unsettled_grant, step_grant)
        if not coordinator.set_elastic_external_memory(
            safe_external,
            minimum_free_primary_blocks=minimum_free,
        ):
            raise RuntimeError(
                "same-step CUDA Graph loan could not be committed after "
                "admission: "
                f"step_key={step_key!r} requested_bytes={safe_external} "
                f"current_bytes={current_external} "
                f"minimum_free_primary_blocks={minimum_free} "
                f"rejection={coordinator.last_elastic_rejection!r}"
            )
        # The coordinator holds the global FIFO-safe maximum, while step_grant
        # is the authoritative quantum-normalized value propagated to this
        # SchedulerOutput and observed by the worker.
        committed_external = coordinator.elastic_external_memory_bytes
        if committed_external < safe_external:
            raise RuntimeError(
                "elastic CUDA Graph loan commit returned less than requested: "
                f"requested_bytes={safe_external} "
                f"committed_bytes={committed_external}"
            )
        self._elastic_admission_controller.reserve_loan(step_key, step_grant)
        # This is the latest worker state requested in FIFO order. Do not roll it
        # back when an older asynchronous result settles.
        self._elastic_admission_controller.publish_step_key(step_key)
        capacity_known = prior_capture is not None or (
            residency_step_key in self._elastic_admission_controller.measured_bytes
            and self._elastic_admission_controller.recapture_pending_key
            != residency_step_key
        )
        if (
            step_key is not None
            and step_key[0] == 0
            and step_key[1] == 0
            and step_key[3]
            in getattr(self, "_elastic_compiled_piecewise_sizes", frozenset())
            and not self._resolve_elastic_step_physical_keys(step_key)
        ):
            capacity_known = True
        self._log_elastic_executable_capacity(
            step_key,
            step_grant,
            known=capacity_known,
        )
        return step_grant

    def _next_elastic_transaction_id(self) -> str:
        return self._elastic_admission_controller.next_transaction_id()

    def _log_elastic_executable_capacity(
        self,
        step_key: tuple[int, ...] | None,
        external_memory_bytes: int,
        *,
        known: bool,
    ) -> None:
        """Publish executable capacity without turning estimates into facts."""
        primary_blocks_per_request = getattr(
            self, "_elastic_primary_blocks_per_max_request", None
        )
        if primary_blocks_per_request is None:
            return
        if step_key is None:
            receipt: tuple[object, ...] = ("X0", external_memory_bytes)
            if self._elastic_admission_controller.capacity_receipt_changed(receipt):
                logger.debug(
                    "Elastic graph loan released: no executable capacity claim "
                    "changes until the next catalog-proven step; external_bytes=%d",
                    external_memory_bytes,
                )
                self._elastic_admission_controller.remember_capacity_receipt(receipt)
            return

        if not known:
            receipt = ("UNKNOWN", step_key, external_memory_bytes)
            if self._elastic_admission_controller.capacity_receipt_changed(receipt):
                logger.debug(
                    "Elastic executable KV calibration pending for cold step; "
                    "step_key=%s provisional_external_bytes=%d",
                    step_key,
                    external_memory_bytes,
                )
                self._elastic_admission_controller.remember_capacity_receipt(receipt)
            return

        coordinator = self.kv_cache_manager.coordinator
        max_requests = max(
            self.scheduler_config.max_num_seqs,
            self.kv_cache_config.num_blocks,
        )
        executable_x = coordinator.max_elastic_full_context_requests(
            primary_blocks_per_request,
            external_memory_bytes,
            max_requests,
        )
        receipt = (
            "KNOWN",
            step_key,
            external_memory_bytes,
            executable_x,
        )
        if self._elastic_admission_controller.capacity_receipt_changed(receipt):
            kv = coordinator.elastic_kv_authority_receipt(
                primary_blocks_per_request,
                external_memory_bytes,
            )
            logger.debug(
                "Elastic executable KV capacity: max_full_context_x=%d "
                "max_model_len=%d external_bytes=%d step_key=%s "
                "raw_attention_blocks_per_rank=%s "
                "raw_attention_bytes_per_rank=%s "
                "raw_attention_token_equivalent_per_rank=%s "
                "effective_attention_blocks_per_rank=%s "
                "effective_attention_bytes_per_rank=%s "
                "effective_attention_token_equivalent_per_rank=%s "
                "active_gdn_blocks=%d primary_blocks_per_max_request=%d",
                executable_x,
                self.max_model_len,
                external_memory_bytes,
                step_key,
                kv["raw_attention_blocks_per_rank"],
                kv["raw_attention_bytes_per_rank"],
                kv["raw_attention_token_equivalent_per_rank"],
                kv["effective_attention_blocks_per_rank"],
                kv["effective_attention_bytes_per_rank"],
                kv["effective_attention_token_equivalent_per_rank"],
                kv["active_gdn_blocks"],
                kv["primary_blocks_per_max_request"],
            )
            self._elastic_admission_controller.remember_capacity_receipt(receipt)

    def _publish_elastic_startup_capacity(self) -> None:
        """Publish distinct catalog-proven KV and execution guarantees."""
        coordinator = self.kv_cache_manager.coordinator
        primary_blocks = self._elastic_primary_blocks_per_max_request
        coverage = getattr(self, "_elastic_graph_catalog_coverage", {})
        product_k = int(getattr(self, "num_spec_tokens", 0))
        speculative_config = getattr(
            getattr(self, "vllm_config", None), "speculative_config", None
        )
        product_prefill_k = (
            0
            if speculative_config is not None
            and speculative_config.disable_speculation_on_non_decode
            else product_k
        )
        startup_kv = coordinator.elastic_kv_authority_receipt(
            primary_blocks,
            self._elastic_admission_controller.resident_bytes,
        )
        # Synthetic Scheduler unit fixtures predate the concrete coordinator
        # receipt and use an open Mock. The coordinator's dedicated tests own
        # this ABI; production always returns the validated dictionary.
        if isinstance(startup_kv, dict):
            logger.info(
                "Elastic KV authority before READY: max_model_len=%d "
                "raw_attention_blocks_per_rank=%s "
                "raw_attention_bytes_per_rank=%s "
                "raw_attention_token_equivalent_per_rank=%s "
                "effective_attention_blocks_per_rank=%s "
                "effective_attention_bytes_per_rank=%s "
                "effective_attention_token_equivalent_per_rank=%s "
                "graph_steady_bytes=%d graph_transition_floor_bytes=%d "
                "graph_workspace_unit_bytes=%d mm_activation_loan_bytes=%d "
                "active_gdn_blocks=%d primary_blocks_per_max_request=%d "
                "rank_budget_bytes=%s",
                startup_kv["max_model_len"],
                startup_kv["raw_attention_blocks_per_rank"],
                startup_kv["raw_attention_bytes_per_rank"],
                startup_kv["raw_attention_token_equivalent_per_rank"],
                startup_kv["effective_attention_blocks_per_rank"],
                startup_kv["effective_attention_bytes_per_rank"],
                startup_kv["effective_attention_token_equivalent_per_rank"],
                startup_kv["graph_external_bytes"],
                self._elastic_admission_controller.transition_floor_bytes,
                self._elastic_admission_controller.cublas_workspace_bytes,
                getattr(self, "_elastic_mm_activation_loan_bytes", 0),
                startup_kv["active_gdn_blocks"],
                startup_kv["primary_blocks_per_max_request"],
                startup_kv["rank_budget_bytes"],
            )
        sealed_decode_max = int(coverage.get("decode_max_x", 0))
        sealed_mixed_max = int(coverage.get("mixed_max_x", 0))
        sealed_full_context_max = int(coverage.get("full_context_max_x", 0))
        by_k: dict[int, dict[str, dict[int, int]]] = defaultdict(
            lambda: {"decode": {}, "mixed_b": {}, "full_context": {}}
        )
        complete_shapes = 0
        from vllm.v1.core.elastic_catalog import (
            elastic_graph_catalog_row_complete,
        )

        representation = coverage.get("representation", "pinned_full_family")
        for step_key, row in self._elastic_graph_catalog.items():
            # Every advertised boundary must coexist with the whole pinned
            # FULL family that is simultaneously resident after startup. A
            # per-shape cold observation from an earlier calibration epoch is
            # an alternative measurement, not the aggregate product floor.
            cold_peak = max(
                _elastic_catalog_cold_residency_envelope(row),
                self._elastic_admission_controller.pinned_resident_bytes,
            )
            complete = elastic_graph_catalog_row_complete(
                step_key,
                row
                if "allocation_profile" in row
                else {**row, "cold_peak_bytes": cold_peak},
                representation=representation,
            )
            if not complete:
                continue
            complete_shapes += 1
            x = step_key[2]
            decode_shape = bool(step_key[0])
            policy = getattr(self, "_elastic_graph_execution_policy", None)
            if not decode_shape and isinstance(policy, GraphExecutionPolicy):
                # FULL/PIECEWISE describes execution, not semantic phase.
                # Resolve the terminal verification shape through the same
                # policy consumed by dispatch, including token-major decode.
                k = step_key[1]
                decode_shape = (
                    step_key
                    == canonical_graph_step_key(
                        SemanticGraphStep(
                            k,
                            x,
                            x * (k + 1),
                            k + 1,
                            "decode",
                            ("target",)
                            + tuple(
                                owner.owner
                                for owner in policy.owners
                                if owner.owner != "target"
                                and (owner.activation == "always" or k > 0)
                            ),
                        ),
                        policy,
                        physical_num_reqs=x,
                        max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
                    )
                    if x * (k + 1) <= self.scheduler_config.max_num_batched_tokens
                    else False
                )
            if decode_shape:
                if (
                    step_key[1] == product_prefill_k
                    and sealed_decode_max
                    and x > sealed_decode_max
                ):
                    continue
                by_k[step_key[1]]["decode"][x] = max(
                    cold_peak,
                    by_k[step_key[1]]["decode"].get(x, 0),
                )
                # A short decode shape and simultaneous max-model-length KV
                # residency are different contracts.  Test the latter against
                # this exact shape without filtering it out of DecodeMaxX.
                full_context_x = coordinator.max_elastic_full_context_requests(
                    primary_blocks,
                    cold_peak,
                    x,
                )
                if full_context_x >= x:
                    by_k[step_key[1]]["full_context"][x] = max(
                        cold_peak,
                        by_k[step_key[1]]["full_context"].get(x, 0),
                    )
            if (
                not step_key[0]
                and step_key[3] == self.scheduler_config.max_num_batched_tokens
            ):
                if (
                    step_key[1] == product_k
                    and sealed_mixed_max
                    and x > sealed_mixed_max
                ):
                    continue
                by_k[step_key[1]]["mixed_b"][x] = max(
                    cold_peak,
                    by_k[step_key[1]]["mixed_b"].get(x, 0),
                )
        # A configured compiled-only max-B carrier deliberately has no CUDA
        # Graph row.  The sealed boundary still needs an exact phase-split
        # witness at MixedMaxX, while the intersecting decode row continues to
        # carry the physical Graph/KV envelope.  A zero here means "no distinct
        # Graph owner", not "execution is free".
        compiled_sizes: frozenset[int] = getattr(
            self, "_elastic_compiled_piecewise_sizes", frozenset()
        )
        max_b = self.scheduler_config.max_num_batched_tokens
        if max_b in compiled_sizes and sealed_mixed_max:
            by_k[product_prefill_k]["mixed_b"][sealed_mixed_max] = 0
        for k, classes in sorted(by_k.items()):
            decode_max = max(classes["decode"], default=0)
            guaranteed_max = max(
                classes["decode"].keys() & classes["mixed_b"].keys(),
                default=0,
            )
            full_context_max = max(classes["full_context"], default=0)
            if k == product_k and coverage:
                if decode_max != sealed_decode_max:
                    raise RuntimeError(
                        "sealed DecodeMaxX lacks an exact complete decode row: "
                        f"sealed={sealed_decode_max} catalog={decode_max} K={k}"
                    )
                if (
                    product_prefill_k == product_k
                    and guaranteed_max != sealed_mixed_max
                ):
                    raise RuntimeError(
                        "sealed MixedMaxX lacks an exact complete max-B row: "
                        f"sealed={sealed_mixed_max} catalog={guaranteed_max} K={k}"
                    )
                full_context_max = sealed_full_context_max
            decode_envelope = classes["decode"].get(decode_max, 0)
            max_b_envelope = max(
                classes["decode"].get(guaranteed_max, 0),
                classes["mixed_b"].get(guaranteed_max, 0),
            )
            full_context_envelope = classes["full_context"].get(full_context_max, 0)
            capacity = coordinator.elastic_full_context_capacity_receipt(
                primary_blocks,
                full_context_envelope,
                full_context_max,
            )
            if capacity["residual_attention_blocks"] < 0:
                raise RuntimeError(
                    "sealed elastic catalog published an impossible startup "
                    f"capacity: K={k} receipt={capacity}"
                )
            logger.info(
                "Elastic calibrated startup capacity: DecodeMaxX[K%d]=%d "
                "GuaranteedMaxX[K%d,B%d]=%d GuaranteedFullContextX[K%d]=%d "
                "decode_cold_envelope_bytes=%d max_b_cold_envelope_bytes=%d "
                "full_context_cold_envelope_bytes=%d "
                "attention_blocks=%d required_attention_blocks=%d "
                "residual_attention_blocks=%d gdn_blocks=%d "
                "complete_catalog_shapes=%d catalog_shapes=%d",
                k,
                decode_max,
                k,
                self.scheduler_config.max_num_batched_tokens,
                guaranteed_max,
                k,
                full_context_max,
                decode_envelope,
                max_b_envelope,
                full_context_envelope,
                capacity["attention_blocks"],
                capacity["required_attention_blocks"],
                capacity["residual_attention_blocks"],
                capacity["gdn_blocks"],
                complete_shapes,
                len(self._elastic_graph_catalog),
            )
        if product_prefill_k != product_k:
            decode_rows = by_k[product_k]["decode"]
            mixed_rows = by_k[product_prefill_k]["mixed_b"]
            combined_max = max(decode_rows.keys() & mixed_rows.keys(), default=0)
            if coverage and combined_max != sealed_mixed_max:
                raise RuntimeError(
                    "sealed phase-split MixedMaxX lacks exact complete decode "
                    "and max-B rows: "
                    f"sealed={sealed_mixed_max} catalog={combined_max} "
                    f"decode_k={product_k} prefill_k={product_prefill_k}"
                )
            logger.info(
                "Elastic calibrated phase-split capacity: "
                "GuaranteedMaxX[decodeK%d,prefillK%d,B%d]=%d "
                "decode_cold_envelope_bytes=%d "
                "max_b_cold_envelope_bytes=%d",
                product_k,
                product_prefill_k,
                self.scheduler_config.max_num_batched_tokens,
                combined_max,
                decode_rows.get(combined_max, 0),
                mixed_rows.get(combined_max, 0),
            )

    def _sync_elastic_residency_receipt(
        self,
        receipt: ElasticResidencyReceipt,
        expected_transaction_id: str | None = None,
    ) -> None:
        (
            self._elastic_admission_controller.pinned_resident_bytes,
            self._elastic_admission_controller.evictable_resident_bytes,
        ) = self._elastic_admission_controller.accept_residency_receipt(
            receipt,
            expected_transaction_id=expected_transaction_id,
        )

    def _settle_elastic_graph_loan(
        self,
        scheduler_output: SchedulerOutput,
        worker_resident_bytes: int,
        worker_floor_bytes: int,
        worker_peak_bytes: int = 0,
        worker_transition_floor_bytes: int | None = None,
        worker_receipt: ElasticResidencyReceipt | None = None,
    ) -> None:
        if not getattr(self, "elastic_on_demand_graphs", False):
            return
        coordinator = self.kv_cache_manager.coordinator
        transaction_id = scheduler_output.elastic_transaction_id
        plan = scheduler_output.elastic_step_plan
        controller = self._elastic_admission_controller
        prior_workspace_unit = controller.cublas_workspace_bytes
        worker_cublas_workspace_unit_bytes = (
            0 if worker_receipt is None else worker_receipt.cublas_workspace_bytes
        )
        active_retention_without_receipt = bool(
            getattr(self, "_elastic_restore_retention_id", None)
            and worker_receipt is None
        )
        preserved_noop = bool(
            scheduler_output.elastic_preserve_graph_residency
            and plan is None
            and scheduler_output.total_num_scheduled_tokens == 0
        )
        receiptless_preservation = bool(
            active_retention_without_receipt
            or (preserved_noop and worker_receipt is None)
        )
        if receiptless_preservation:
            # Some calibration output paths omit the optional HOT sidecar. An
            # active retention epoch has already validated and leased every
            # protected key on each worker before execution; a missing key is
            # rejected there. Thus an absent sidecar is not an authoritative
            # empty set until the retention epoch ends. The ordinary serving
            # path and every non-retained step remain strict below.
            worker_resident_bytes = max(
                worker_resident_bytes,
                self._elastic_admission_controller.resident_bytes,
            )
            worker_floor_bytes = max(
                worker_floor_bytes,
                self._elastic_admission_controller.floor_bytes,
            )
            if worker_transition_floor_bytes is None:
                worker_transition_floor_bytes = (
                    self._elastic_admission_controller.transition_floor_bytes
                )
            worker_peak_bytes = max(worker_peak_bytes, worker_resident_bytes)
            worker_cublas_workspace_unit_bytes = max(
                worker_cublas_workspace_unit_bytes,
                self._elastic_admission_controller.cublas_workspace_bytes,
            )
        if not receiptless_preservation:
            if worker_receipt is None:
                raise RuntimeError("elastic lifecycle step omitted residency receipt")
            required_hot_keys = (
                tuple(dict.fromkeys((*plan.physical_keys, *plan.protected_keys)))
                if plan is not None
                and plan.kind in {ElasticPlanKind.USER, ElasticPlanKind.MAINTENANCE}
                else ()
            )
            controller.validate_residency_publication(
                worker_receipt,
                expected_transaction_id=transaction_id,
                required_hot_keys=required_hot_keys,
                releasing_transaction_id=transaction_id,
            )
            if plan is not None and plan.kind == ElasticPlanKind.PRESSURE_RECLAIM:
                retained_keys = {entry.key for entry in worker_receipt.entries}
                unexpected = retained_keys.difference(plan.protected_keys)
                if unexpected:
                    raise RuntimeError(
                        "administrative idle reclaim retained an unprotected "
                        "Graph executable"
                    )
        if not receiptless_preservation:
            assert worker_receipt is not None
            worker_resident_bytes = worker_receipt.resident_bytes
            worker_floor_bytes = worker_receipt.floor_bytes
            worker_transition_floor_bytes = worker_receipt.transition_floor_bytes
            worker_peak_bytes = worker_receipt.peak_bytes
            worker_cublas_workspace_unit_bytes = worker_receipt.cublas_workspace_bytes
        worker_resident_key_bytes = (
            None
            if worker_receipt is None
            else tuple(
                (entry.key.identity, entry.resident_bytes)
                for entry in worker_receipt.entries
            )
        )
        minimum_free = scheduler_output.elastic_successor_primary_headroom
        if minimum_free < 0:
            raise RuntimeError(
                "scheduler output carries negative elastic successor headroom: "
                f"{minimum_free}"
            )
        if worker_transition_floor_bytes is None:
            worker_transition_floor_bytes = worker_floor_bytes
        if worker_floor_bytes < 0 or worker_floor_bytes > worker_resident_bytes:
            raise RuntimeError(
                "worker CUDA Graph physical floor is outside aggregate "
                f"external memory: floor_bytes={worker_floor_bytes} "
                f"resident_bytes={worker_resident_bytes}"
            )
        if not (
            worker_floor_bytes <= worker_transition_floor_bytes <= worker_resident_bytes
        ):
            raise RuntimeError(
                "worker CUDA Graph prospective transition floor is outside "
                "aggregate external memory: "
                f"floor_bytes={worker_floor_bytes} "
                f"transition_floor_bytes={worker_transition_floor_bytes} "
                f"resident_bytes={worker_resident_bytes}"
            )
        if worker_peak_bytes < worker_resident_bytes:
            worker_peak_bytes = worker_resident_bytes
        if worker_cublas_workspace_unit_bytes < 0:
            raise RuntimeError("worker returned a negative cuBLAS workspace unit")
        if (
            prior_workspace_unit
            and worker_cublas_workspace_unit_bytes
            and prior_workspace_unit != worker_cublas_workspace_unit_bytes
        ):
            raise RuntimeError(
                "worker cuBLAS workspace unit changed within one runtime: "
                f"prior={prior_workspace_unit} "
                f"actual={worker_cublas_workspace_unit_bytes}"
            )
        if not getattr(self, "elastic_on_demand_graphs", False):
            if not coordinator.set_elastic_external_memory(
                worker_resident_bytes,
                minimum_free_primary_blocks=minimum_free,
            ):
                raise RuntimeError(
                    "worker CUDA Graph residency could not be committed to the "
                    "scheduler KV layout: "
                    f"requested_bytes={worker_resident_bytes}"
                )
            return

        observed_step_key = self._canonical_elastic_graph_step_key(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.num_spec_tokens_to_schedule,
            scheduler_output.is_pure_decode_step,
        )
        scheduled_step_key = scheduler_output.elastic_graph_step_key
        if (
            scheduled_step_key is None
            and scheduler_output.total_num_scheduled_tokens > 0
        ):
            # Compatibility for synthetic SchedulerOutput fixtures. Production
            # outputs always carry the immutable commit-time key above.
            scheduled_step_key = observed_step_key
        pending_loans = controller.pending_loans
        if not pending_loans:
            raise RuntimeError(
                "worker returned CUDA Graph residency without a pending scheduler loan"
            )
        # Peek only.  Every fallible identity/accounting check below must pass
        # before the FIFO is consumed or any HOT/lease authority is changed.
        settled_loan = pending_loans[0]
        pending_key = settled_loan.step_key
        granted_bytes = settled_loan.grant_bytes
        if pending_key is not None and pending_key[3] <= 0:
            # Async completion can retain a finished request dictionary whose
            # physical step contained no work. It owns no executable shape.
            pending_key = None
        maintenance = (
            scheduler_output.elastic_step_plan is not None
            and scheduler_output.elastic_step_plan.kind == ElasticPlanKind.MAINTENANCE
        )
        step_key = pending_key if maintenance else scheduled_step_key
        if not maintenance and pending_key != scheduled_step_key:
            raise RuntimeError(
                "asynchronous CUDA Graph loan settled out of FIFO order: "
                f"expected={pending_key!r} actual={scheduled_step_key!r}"
            )
        residency_step_key = step_key
        compiled_only_step = bool(
            residency_step_key is not None
            and not self._resolve_elastic_step_physical_keys(residency_step_key)
        )
        if granted_bytes != scheduler_output.elastic_external_memory_bytes:
            raise RuntimeError(
                "scheduler output CUDA Graph grant identity changed in flight: "
                f"expected={granted_bytes} actual="
                f"{scheduler_output.elastic_external_memory_bytes}"
            )
        mm_activation_loan_bytes = scheduler_output.elastic_mm_activation_loan_bytes
        graph_granted_bytes = scheduler_output.elastic_graph_external_memory_bytes
        if graph_granted_bytes == 0 and mm_activation_loan_bytes == 0:
            # Compatibility for synthetic/unit outputs created before the
            # explicit breakdown; production always fills both fields.
            graph_granted_bytes = granted_bytes
        if graph_granted_bytes + mm_activation_loan_bytes != granted_bytes:
            raise RuntimeError(
                "scheduler external-loan breakdown changed in flight: "
                f"graph_bytes={graph_granted_bytes} "
                f"mm_bytes={mm_activation_loan_bytes} total_bytes={granted_bytes}"
            )
        idle_floor_retained = (
            step_key is None
            and (plan is None or plan.kind != ElasticPlanKind.PRESSURE_RECLAIM)
            and worker_floor_bytes == worker_resident_bytes
            and worker_resident_bytes > granted_bytes
        )
        explicit_zero_token_reclaim = bool(
            step_key is None
            and scheduler_output.total_num_scheduled_tokens == 0
            and plan is not None
            and plan.kind == ElasticPlanKind.RECLAIM
        )
        idle_pinned_residency_retained = (
            step_key is None
            and (plan is None or plan.kind != ElasticPlanKind.PRESSURE_RECLAIM)
            and (not self.running or explicit_zero_token_reclaim)
            and worker_resident_bytes > granted_bytes
            and worker_receipt is not None
            and bool(worker_receipt.entries)
            and all(entry.pinned for entry in worker_receipt.entries)
        )
        was_cold_capture = (
            residency_step_key is not None
            and self._elastic_admission_controller.recapture_pending_key
            == residency_step_key
        )
        if was_cold_capture and mm_activation_loan_bytes:
            raise RuntimeError("MM activation overlapped a cold Graph capture")
        self._validate_elastic_graph_mm_overlap(
            worker_peak_bytes=worker_peak_bytes,
            granted_bytes=granted_bytes,
            graph_granted_bytes=graph_granted_bytes,
            mm_activation_loan_bytes=mm_activation_loan_bytes,
        )
        if worker_resident_bytes > graph_granted_bytes and not (
            idle_floor_retained
            or idle_pinned_residency_retained
            or preserved_noop
            or receiptless_preservation
        ):
            raise RuntimeError(
                "worker CUDA Graph residency exceeds its Graph loan component: "
                f"resident_bytes={worker_resident_bytes} "
                f"graph_granted_bytes={graph_granted_bytes}"
            )
        maintenance_started = None
        if (
            plan is not None
            and plan.kind
            in {
                ElasticPlanKind.MAINTENANCE,
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }
            and transaction_id is not None
        ):
            maintenance_started = self._elastic_maintenance_started.get(transaction_id)
            if maintenance_started is None:
                raise RuntimeError(
                    "elastic Graph transaction settled without a start receipt: "
                    f"tx={transaction_id}"
                )

        # Commit external KV geometry first, using the staged post-pop FIFO.
        # Once this succeeds, the remaining controller/catalog publications are
        # deterministic operations whose validation has already completed.
        remaining_pending_grant = max(
            (loan.grant_bytes for loan in pending_loans[1:]), default=0
        )
        settled_external = coordinator.normalize_elastic_external_memory(
            max(worker_resident_bytes, remaining_pending_grant)
        )
        current_external = coordinator.elastic_external_memory_bytes
        if (
            settled_external != current_external
            and not coordinator.set_elastic_external_memory(
                settled_external,
                minimum_free_primary_blocks=minimum_free,
            )
        ):
            raise RuntimeError(
                "worker CUDA Graph residency could not be synchronized with "
                "the scheduler KV layout after settlement: "
                f"current_bytes={current_external} "
                f"settled_bytes={settled_external} "
                f"worker_resident_bytes={worker_resident_bytes}"
            )

        if transaction_id is not None:
            controller.release(transaction_id)
        if not receiptless_preservation:
            assert worker_receipt is not None
            self._sync_elastic_residency_receipt(
                worker_receipt,
                transaction_id,
            )
        committed_loan = controller.settle_next_loan()
        assert committed_loan == settled_loan
        graph_observation_peak_bytes = (
            worker_resident_bytes if mm_activation_loan_bytes else worker_peak_bytes
        )
        if (
            residency_step_key is not None
            and not compiled_only_step
            and not self._elastic_catalog_prices_are_sealed()
        ):
            owner_key = self._elastic_graph_owner_key(residency_step_key)
            assert owner_key is not None
            catalog = getattr(self, "_elastic_graph_catalog", None)
            if catalog is None:
                catalog = {}
                self._elastic_graph_catalog = catalog
            row = catalog.setdefault(
                residency_step_key,
                {
                    "cold_peak_bytes": 0,
                    "hot_peak_bytes": 0,
                    "resident_bytes": 0,
                    "floor_bytes": 0,
                    "resident_key_bytes": None,
                    "cold_observations": 0,
                    "cold_stable_replays": 0,
                    "hot_observations": 0,
                    "hot_stable_replays": 0,
                },
            )
            # A canonical physical class can have multiple logical witnesses
            # (for example balanced prefill and decode+prefill at the same
            # padded M/X).  Retain the observed envelope, never whichever
            # witness happened to run last.
            row["resident_bytes"] = max(row["resident_bytes"], worker_resident_bytes)
            row["floor_bytes"] = max(row["floor_bytes"], worker_floor_bytes)
            if was_cold_capture:
                physical_envelope = max(
                    graph_observation_peak_bytes,
                    worker_resident_bytes,
                    worker_floor_bytes,
                )
                # The catalog prices how many bytes must be borrowed from KV,
                # not total CUDA high-water.  A capture that published all
                # Graph owners inside its granted loan can transiently consume
                # additional non-KV allocator/driver slack; charging that same
                # physical delta to KV double-counts it.  Real residency/floor
                # overruns remain fail-closed and retain the full physical
                # envelope so calibration can downshift MaxX.
                measured_envelope = (
                    min(physical_envelope, granted_bytes)
                    if max(worker_resident_bytes, worker_floor_bytes) <= granted_bytes
                    else physical_envelope
                )
                measured_envelope = coordinator.normalize_elastic_external_memory(
                    measured_envelope
                )
                prior_cold_peak = row["cold_peak_bytes"]
                prior_resident_key_bytes = row.get("resident_key_bytes")
                if measured_envelope > prior_cold_peak:
                    row_resident_key_bytes = worker_resident_key_bytes
                elif measured_envelope < prior_cold_peak:
                    row_resident_key_bytes = prior_resident_key_bytes
                elif (
                    prior_resident_key_bytes is None
                    or worker_resident_key_bytes is None
                ):
                    row_resident_key_bytes = None
                else:
                    worker_resident_map = dict(worker_resident_key_bytes)
                    row_resident_key_bytes = tuple(
                        sorted(
                            (
                                identity,
                                min(value, worker_resident_map[identity]),
                            )
                            for identity, value in prior_resident_key_bytes
                            if identity in worker_resident_map
                        )
                    )
                row["resident_key_bytes"] = row_resident_key_bytes
                prior_capture = self._elastic_capture_envelope(residency_step_key)
                prior_sampling = 0 if prior_capture is None else prior_capture[2]
                self._record_elastic_capture_envelope(
                    residency_step_key,
                    (
                        max(row["cold_peak_bytes"], measured_envelope),
                        max(row["floor_bytes"], worker_floor_bytes),
                        prior_sampling,
                    ),
                    resident_key_bytes=row_resident_key_bytes,
                )
                self._elastic_admission_controller.record_measurement(
                    residency_step_key,
                    measured_envelope,
                    keep_max=True,
                )
                row["cold_observations"] = row.get("cold_observations", 0) + 1
                bounded_restore_envelope = (
                    max(prior_cold_peak, row.get("hot_peak_bytes", 0))
                    if getattr(self, "_elastic_restore_mode", False)
                    and self._elastic_graph_catalog_coverage.get("representation")
                    == "bounded_exact_hotset"
                    else prior_cold_peak
                )
                if prior_cold_peak and measured_envelope <= bounded_restore_envelope:
                    row["cold_stable_replays"] = row.get("cold_stable_replays", 0) + 1
                else:
                    row["cold_stable_replays"] = 0
                row["cold_peak_bytes"] = max(prior_cold_peak, measured_envelope)
                logger.debug(
                    "Elastic CUDA Graph cold envelope measured: "
                    "owner_key=%s step_key=%s "
                    "granted_bytes=%d resident_bytes=%d floor_bytes=%d "
                    "peak_bytes=%d measured_loan_envelope_bytes=%d "
                    "cold_observations=%d cold_stable_replays=%d",
                    owner_key,
                    step_key,
                    granted_bytes,
                    worker_resident_bytes,
                    worker_floor_bytes,
                    worker_peak_bytes,
                    measured_envelope,
                    row["cold_observations"],
                    row["cold_stable_replays"],
                )
            else:
                # HOT and COLD prices describe the KV loan, not total CUDA
                # high-water. A completed replay may also use allocator/driver
                # slack outside KV; residency and MM overrun guards above
                # remain authoritative before publishing this envelope.
                physical_envelope = max(
                    graph_observation_peak_bytes,
                    worker_resident_bytes,
                    worker_floor_bytes,
                )
                measured_envelope = (
                    min(physical_envelope, granted_bytes)
                    if max(worker_resident_bytes, worker_floor_bytes) <= granted_bytes
                    else physical_envelope
                )
                observed_hot_envelope = coordinator.normalize_elastic_external_memory(
                    measured_envelope
                )
                prior_hot_peak = row["hot_peak_bytes"]
                row["hot_observations"] = row.get("hot_observations", 0) + 1
                if prior_hot_peak and observed_hot_envelope <= prior_hot_peak:
                    row["hot_stable_replays"] = row.get("hot_stable_replays", 0) + 1
                else:
                    row["hot_stable_replays"] = 0
                hot_envelope = max(prior_hot_peak, observed_hot_envelope)
                cold_growth = hot_envelope - prior_hot_peak if prior_hot_peak else 0
                if cold_growth:
                    # A coalesced PIECEWISE descriptor can encounter a larger
                    # eager-workspace witness without recapturing its compiled
                    # segments. Compose that measured HOT delta with the prior
                    # cold envelope so a later cold replay is funded by a
                    # mathematical upper bound, not the smaller witness that
                    # happened to capture first.
                    row["cold_peak_bytes"] = (
                        coordinator.normalize_elastic_external_memory(
                            row["cold_peak_bytes"] + cold_growth
                        )
                    )
                    prior_resident_key_bytes = row.get("resident_key_bytes")
                    worker_resident_map = dict(worker_resident_key_bytes or ())
                    row["resident_key_bytes"] = (
                        None
                        if (
                            prior_resident_key_bytes is None
                            or worker_resident_key_bytes is None
                        )
                        else tuple(
                            sorted(
                                (
                                    identity,
                                    min(value, worker_resident_map[identity]),
                                )
                                for identity, value in prior_resident_key_bytes
                                if identity in worker_resident_map
                            )
                        )
                    )
                    prior_capture = self._elastic_capture_envelope(residency_step_key)
                    capture_grant, capture_floor, capture_sampling = (
                        prior_capture
                        if prior_capture is not None
                        else (row["cold_peak_bytes"] - cold_growth, 0, 0)
                    )
                    self._record_elastic_capture_envelope(
                        residency_step_key,
                        (
                            max(
                                row["cold_peak_bytes"],
                                coordinator.normalize_elastic_external_memory(
                                    capture_grant + cold_growth
                                ),
                            ),
                            max(capture_floor, worker_floor_bytes),
                            capture_sampling,
                        ),
                        resident_key_bytes=row.get("resident_key_bytes"),
                    )
                self._elastic_admission_controller.record_measurement(
                    residency_step_key,
                    hot_envelope,
                )
                row["hot_peak_bytes"] = hot_envelope
                logger.debug(
                    "Elastic CUDA Graph hot envelope measured: step_key=%s "
                    "resident_bytes=%d floor_bytes=%d peak_bytes=%d "
                    "hot_envelope_bytes=%d cold_growth_bytes=%d "
                    "cold_envelope_bytes=%d hot_observations=%d "
                    "hot_stable_replays=%d",
                    step_key,
                    worker_resident_bytes,
                    worker_floor_bytes,
                    worker_peak_bytes,
                    hot_envelope,
                    cold_growth,
                    row["cold_peak_bytes"],
                    row["hot_observations"],
                    row["hot_stable_replays"],
                )
        # Capture settlement is a runtime lifecycle event even when its
        # cumulative receipt must not rewrite a sealed isolated price.
        if was_cold_capture:
            assert residency_step_key is not None
            self._elastic_admission_controller.finish_recapture(residency_step_key)
        self._elastic_admission_controller.publish_physical_accounting(
            resident_bytes=worker_resident_bytes,
            floor_bytes=worker_floor_bytes,
            transition_floor_bytes=worker_transition_floor_bytes,
            cublas_workspace_bytes=worker_cublas_workspace_unit_bytes,
            maintenance_step_key=(residency_step_key if maintenance else None),
        )

        if (
            plan is not None
            and plan.kind
            in {
                ElasticPlanKind.MAINTENANCE,
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }
            and transaction_id is not None
        ):
            started = self._elastic_maintenance_started.pop(transaction_id)
            assert maintenance_started is not None and started == maintenance_started
            started_at, stats_before, external_before = started
            stats_after = self._elastic_admission_controller.stats
            wall_ms = (time.monotonic() - started_at) * 1000.0
            self._elastic_maintenance_wall_ms_total = (
                getattr(self, "_elastic_maintenance_wall_ms_total", 0.0) + wall_ms
            )
            self._elastic_maintenance_transactions_total = (
                getattr(self, "_elastic_maintenance_transactions_total", 0) + 1
            )
            key_outcomes = getattr(self, "_elastic_graph_key_outcomes", None)
            if key_outcomes is None:
                key_outcomes = self._elastic_graph_key_outcomes = defaultdict(int)
            for physical_key in plan.physical_keys:
                logical = physical_key.logical
                outcome_key = (
                    f"{logical.owner}|{logical.mode}|{logical.token_bucket}|HOT"
                )
                key_outcomes[outcome_key] += 1
            logger.info(
                "Elastic Graph transaction end: tx=%s kind=%s final_key=%s "
                "wall_ms=%.3f captures=%d evictions=%d evicted_bytes=%d "
                "external_before=%d external_after=%d floor=%d peak=%d outcome=HOT",
                transaction_id,
                plan.kind.value,
                step_key,
                wall_ms,
                stats_after.promotions - stats_before.promotions,
                stats_after.evictions - stats_before.evictions,
                stats_after.evicted_bytes - stats_before.evicted_bytes,
                external_before,
                settled_external,
                worker_floor_bytes,
                worker_peak_bytes,
            )
        elif (
            plan is not None
            and plan.kind == ElasticPlanKind.USER
            and transaction_id is not None
        ):
            started_at = getattr(self, "_elastic_useful_started", {}).pop(
                transaction_id, None
            )
            if started_at is not None:
                self._elastic_useful_wall_ms_total = (
                    getattr(self, "_elastic_useful_wall_ms_total", 0.0)
                    + (time.monotonic() - started_at) * 1000.0
                )
                self._elastic_useful_transactions_total = (
                    getattr(self, "_elastic_useful_transactions_total", 0) + 1
                )
            key_outcomes = getattr(self, "_elastic_graph_key_outcomes", None)
            if key_outcomes is None:
                key_outcomes = self._elastic_graph_key_outcomes = defaultdict(int)
            for physical_key in plan.physical_keys:
                logical = physical_key.logical
                outcome_key = (
                    f"{logical.owner}|{logical.mode}|{logical.token_bucket}|HIT"
                )
                key_outcomes[outcome_key] += 1

        # Settlement returns only unused provisional capacity. The next call
        # to _plan_elastic_graph_loan remains the sole owner of the next
        # physical target; the worker releases incompatible Graph owners
        # before applying that target.
        no_pending_loans = not self._elastic_admission_controller.pending_loans
        # Async scheduling can return a zero-token output whose request-key
        # dictionary still reflects an older in-flight batch. Scheduler
        # ownership is the authoritative idle state here, not that stale key.
        idle_physical_floor = not self.running and worker_floor_bytes > 0
        cleanup_required = (
            no_pending_loans
            and idle_physical_floor
            and self._needs_elastic_idle_reclaim()
        )
        if cleanup_required:
            now = time.monotonic()
            if self._elastic_admission_controller.idle_cleanup_expired(
                required=True,
                now=now,
                timeout_s=getattr(self, "_elastic_graph_idle_cleanup_timeout_s", 5.0),
            ):
                raise RuntimeError(
                    "idle CUDA Graph physical floor did not return to KV "
                    "within the cleanup deadline: "
                    f"floor_bytes={worker_floor_bytes} "
                    f"resident_bytes={worker_resident_bytes} "
                    f"granted_bytes={granted_bytes}"
                )
        else:
            self._elastic_admission_controller.idle_cleanup_expired(
                required=False,
                now=0.0,
                timeout_s=0.0,
            )

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None

        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        # Freeze the exact scheduler token stream that the worker will receive
        # on resumption.  This is intentionally captured before resetting
        # computed state and is not derived from the original prompt length.
        request.execution_prefill_len = len(request._all_token_ids)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.execution_prefill_len = session.num_prompt_tokens
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
        encoder_wave_overlay: EncoderWaveOverlay | None = None,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, [], []
        encoder_inputs_to_schedule: list[int] = []
        cached_encoder_inputs: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = (
            set()
            if encoder_wave_overlay is None
            else set(encoder_wave_overlay.scheduled_identifiers)
        )
        num_embeds_to_schedule = 0
        # Planning must replay the same cache/LRU transitions as commit, but it
        # must not mutate the live manager before Graph and KV admission.  A
        # request-local clone is sufficient on the normal scheduler path
        # because accepted requests are committed sequentially; an elastic
        # cohort passes one shared overlay across all candidate requests.
        cache_manager = (
            self.encoder_cache_manager.clone_for_preview()
            if encoder_wave_overlay is None
            else encoder_wave_overlay.cache_manager
        )

        encoder_window_end = (
            num_computed_tokens + num_new_tokens + shift_computed_tokens
        )
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    if not cache_manager.check_and_update_cache(request, i):
                        raise RuntimeError(
                            "encoder preview lost a scheduled cohort item"
                        )
                    cached_encoder_inputs.append(i)
                    continue

                if cache_manager.contains(request, i):
                    cached_encoder_inputs.append(i)
                    if not cache_manager.check_and_update_cache(request, i):
                        raise RuntimeError(
                            "encoder preview cache membership changed internally"
                        )
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not cache_manager.can_allocate(
                request,
                i,
                encoder_compute_budget,
                num_embeds_to_schedule,
                evict=True,
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                cache_manager.allocate(request, i)
                continue

            cache_manager.allocate(request, i)
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        if encoder_wave_overlay is not None:
            encoder_wave_overlay.scheduled_identifiers = mm_hashes_to_schedule
        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
            cached_encoder_inputs,
        )

    def _commit_encoder_cache_plan(
        self,
        request: Request,
        cached_input_ids: list[int],
        computed_input_ids: list[int] | None,
        external_input_ids: list[int],
    ) -> None:
        """Commit the ordered encoder plan after Graph and KV admission."""
        cached_ids = set(cached_input_ids)
        computed_ids = set(computed_input_ids or ())
        external_ids = set(external_input_ids)
        if cached_ids & (computed_ids | external_ids) or computed_ids & external_ids:
            raise RuntimeError("encoder cache plan contains overlapping operations")

        # Input ids are the multimodal feature order used by preflight.  Do not
        # group claims before allocations: claiming a freeable cached item can
        # consume the exact capacity an earlier/later allocation planned to
        # reclaim.
        for input_id in sorted(cached_ids | computed_ids | external_ids):
            if input_id in cached_ids:
                if not self.encoder_cache_manager.check_and_update_cache(
                    request, input_id
                ):
                    raise RuntimeError(
                        "encoder cache membership changed after admission preflight"
                    )
                continue
            num_embeds = request.get_num_encoder_embeds(input_id)
            if not self.encoder_cache_manager.can_allocate(
                request,
                input_id,
                num_embeds,
                0,
            ):
                raise RuntimeError(
                    "encoder cache capacity changed after admission preflight"
                )
            self.encoder_cache_manager.allocate(request, input_id)
            if self.ec_connector is not None:
                self.ec_connector.update_state_after_alloc(request, input_id)

    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()

        for req_id, input_ids in scheduled_encoder_inputs.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            for input_id in input_ids:
                mm_feature = request.mm_features[input_id]
                stats.num_inputs += 1
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()

        return stats if stats.num_inputs else None

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        ec_connector_output = model_runner_output.ec_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        coordinator = self._gdn_checkpoint_coordinator
        if (
            coordinator is not None
            and model_runner_output.gdn_checkpoint_keys is not None
        ):
            # Reaching update_from_output proves the worker forward completed.
            # Mirror the actual worker membership instead of maintaining an
            # independently inferred LRU that can diverge on waiting/admission.
            coordinator.sync_gdn_checkpoints(model_runner_output.gdn_checkpoint_keys)

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        if (
            model_runner_output.elastic_mm_activation_loan_bytes
            != scheduler_output.elastic_mm_activation_loan_bytes
        ):
            raise RuntimeError(
                "worker MM activation loan echo differs from scheduler output: "
                f"scheduler={scheduler_output.elastic_mm_activation_loan_bytes} "
                f"worker={model_runner_output.elastic_mm_activation_loan_bytes}"
            )
        self._settle_elastic_graph_loan(
            scheduler_output,
            model_runner_output.elastic_external_memory_bytes,
            model_runner_output.elastic_external_memory_floor_bytes,
            model_runner_output.elastic_external_memory_peak_bytes,
            model_runner_output.elastic_external_memory_transition_floor_bytes,
            model_runner_output.elastic_residency_receipt,
        )
        completed_plan = scheduler_output.elastic_step_plan
        self._elastic_accepted_decode_consensus_epoch = (
            completed_plan.execution_epoch_fingerprint
            if completed_plan is not None
            and completed_plan.reusable_decode_consensus_epoch
            else None
        )
        if scheduler_output.total_num_scheduled_tokens > 0:
            execution_step_key = scheduler_output.elastic_graph_step_key
            if execution_step_key is not None:
                execution_shape = (
                    len(scheduler_output.num_scheduled_tokens),
                    execution_step_key,
                    tuple(
                        (
                            dispatch.invocation.owner,
                            dispatch.invocation.physical_num_reqs,
                            dispatch.invocation.physical_num_tokens,
                            dispatch.invocation.uniform_query_len,
                            dispatch.representation.value,
                        )
                        for dispatch in completed_plan.current_dispatch
                    )
                    if completed_plan is not None
                    else (),
                )
                if execution_shape != getattr(
                    self, "_elastic_last_execution_shape", None
                ):
                    logger.info(
                        "AG2 elastic execution transition: semantic_x=%d "
                        "physical_x=%d step_key=%s retained_carrier=%s owners=%s",
                        execution_shape[0],
                        execution_step_key[2],
                        execution_step_key,
                        self._elastic_graph_carrier_step_key,
                        execution_shape[2],
                    )
                    self._elastic_last_execution_shape = execution_shape
            self._commit_elastic_graph_carrier_step_key(execution_step_key)

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                if request.spec_decode_metrics is not None:
                    # Exclude grammar-invalidated drafts from the proposed
                    # count, mirroring make_spec_decoding_stats; the accepted
                    # bucket (j) is unaffected.
                    adj_draft_tokens = num_draft_tokens
                    if scheduler_output.num_invalid_spec_tokens:
                        adj_draft_tokens -= (
                            scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                        )
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=adj_draft_tokens,
                        num_accepted=num_accepted,
                        detailed=self.spec_decode_metrics_level == "detailed",
                    )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_sampling_mask = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_mm_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if self.return_sampling_mask:
                sampling_masks = model_runner_output.sampling_masks
                if new_token_ids and sampling_masks is not None:
                    new_sampling_mask = sampling_masks.slice_request(
                        req_index, len(new_token_ids)
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_sampling_mask=new_sampling_mask,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        spec_decode_metrics=(
                            request.spec_decode_metrics
                            if finish_reason is not None
                            else None
                        ),
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # Loan settlement runs before terminal outputs are applied. If this
        # result removed the final running request, preserve one scheduler tick
        # so the next explicit X0 step can evict the last Graph and return its
        # loan before the engine sleeps.

        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)
        if self.ec_connector is not None:
            # An encoder input the connector can no longer obtain. Failing is
            # retryable: re-issuing the request re-runs the encode.
            error_req_ids.update(self.ec_connector.take_unavailable_requests())

        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # EC Connector: update state from worker-side EC connector output.
        if self.ec_connector is not None and ec_connector_output:
            self.ec_connector.update_connector_output(ec_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.take_finished_request_ids()
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _ec_transfer_pending(self, request: Request, num_computed_tokens: int) -> bool:
        """Whether an encoder input this request needs is still in transit."""
        return (
            self.ec_connector is not None
            and bool(request.mm_features)
            and not self.ec_connector.ensure_cache_available(
                request, num_computed_tokens
            )
        )

    def take_finished_request_ids(self) -> dict[int, set[str]]:
        """Consume client-facing terminal notifications exactly once."""
        pending = self.finished_req_ids_dict
        if not pending:
            return {}
        consumed = {
            client_index: set(request_ids)
            for client_index, request_ids in pending.items()
            if request_ids
        }
        pending.clear()
        return consumed

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _elastic_restore_admission_diagnostic(
        self,
        stop_reason: str | None,
        token_budget: int,
        scheduled_tokens: dict[str, int],
    ) -> dict[str, object]:
        """Snapshot a failed maintenance commit without mutating admission."""
        coordinator = self.kv_cache_manager.coordinator
        primary = coordinator.block_pool
        gdn = coordinator.mamba_block_pool

        def pool_state(pool):
            if pool is None:
                return None
            return {
                "active": pool.active_num_gpu_blocks,
                "free": pool.get_num_free_blocks(),
                "referenced_ids": tuple(
                    block.block_id for block in pool.blocks if block.ref_cnt
                ),
            }

        controller = self._elastic_admission_controller
        return {
            "stop_reason": stop_reason,
            "remaining_token_budget": token_budget,
            "cap": self.max_num_running_reqs,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "skipped_waiting": len(self.skipped_waiting),
            "scheduled_tokens": dict(scheduled_tokens),
            "preemptions": tuple(
                (request_id, request.num_preemptions)
                for request_id, request in self.requests.items()
                if request.num_preemptions
            ),
            "primary": pool_state(primary),
            "gdn": pool_state(gdn),
            "watermark": self.kv_cache_manager.watermark_blocks,
            "external_bytes": coordinator.elastic_external_memory_bytes,
            "resident_bytes": controller.resident_bytes,
            "pending_grant": controller.max_pending_grant(),
            "graph_rejection": self._elastic_last_graph_admission_rejection,
            "allocation_rejection": self.kv_cache_manager.last_allocation_rejection,
            "capacity_rejection": coordinator.last_elastic_rejection,
            "defer_reason": getattr(self, "_elastic_last_defer_reason", None),
            "deferred_frees": len(self.deferred_frees),
            "step_seq": (self.sched_step_seq, self.processed_step_seq),
        }

    def _has_prepared_elastic_waiting_admission(
        self,
        preplanned_step_key: tuple[int, ...] | None,
        calibration_wave_target: int,
    ) -> bool:
        """Whether the complete waiting-wave layout already has one owner."""
        return preplanned_step_key is not None or bool(
            self._elastic_restore_mode and calibration_wave_target
        )

    def _elastic_irreducible_external_bytes(self) -> int:
        catalog_pinned = int(
            getattr(self, "_elastic_graph_catalog_coverage", {}).get(
                "pinned_full_bytes", 0
            )
        )
        serving_carrier = self._elastic_serving_carrier_bytes()
        return (
            max(
                self._elastic_admission_controller.pinned_resident_bytes,
                catalog_pinned,
                serving_carrier,
            )
            + self._elastic_admission_controller.floor_bytes
        )

    def _elastic_serving_carrier_bytes(self) -> int:
        return sum(
            entry.price.resident_bytes
            for key in getattr(self, "_elastic_serving_carrier_keys", ())
            if (entry := self._elastic_admission_controller.entries.get(key))
            is not None
            and entry.hot
            and entry.price is not None
        )

    def _elastic_pressure_floor_external_bytes(self) -> int:
        """Bytes that even an administrative pressure reclaim cannot return.

        Legacy pinned captures may be drained at physical quiescence.  The
        sealed serving carrier is different: it is the only legal product
        replay endpoint, so admitting KV by evicting it would merely defer a
        forbidden cold capture to the next model step.
        """
        return self._elastic_serving_carrier_bytes() + (
            self._elastic_admission_controller.floor_bytes
        )

    def _prepare_elastic_waiting_deficit_reclaim(
        self,
        primary_requirements: tuple[int, ...],
        *,
        declared_wave_size: int | None = None,
        physical_quiescent: bool = True,
    ) -> bool:
        """Arm X0 only when pressure reclaim increases waiting-wave MaxX."""
        if (
            not primary_requirements
            or not self._elastic_admission_controller.resident_bytes
            or self._elastic_admission_controller.pending_maintenance_plan is not None
            or getattr(self, "_elastic_deferred_mm_wave", None) is not None
        ):
            return False
        if declared_wave_size is not None and declared_wave_size > len(
            primary_requirements
        ):
            # Frontend tokenization can expose a concurrent product burst to
            # EngineCore over several scheduler ticks.  Before the first idle
            # prefix commits, probe only its next possible member.  This
            # detects a saturated physical prefix without reserving the unseen
            # remainder or turning the declared boundary into a logical cap.
            primary_requirements = primary_requirements + (max(primary_requirements),)
        coordinator = self.kv_cache_manager.coordinator
        current = coordinator.plan_elastic_admission_wave(primary_requirements)
        post_reclaim = coordinator.plan_elastic_admission_wave(
            primary_requirements,
            external_memory_bytes=self._elastic_irreducible_external_bytes(),
        )
        current_max = current.max_requests if current is not None else 0
        post_reclaim_max = post_reclaim.max_requests if post_reclaim is not None else 0
        target = len(primary_requirements)
        pressure_floor = self._elastic_pressure_floor_external_bytes()
        post_pressure_reclaim = coordinator.plan_elastic_admission_wave(
            primary_requirements,
            external_memory_bytes=pressure_floor,
        )
        post_pressure_reclaim_max = (
            post_pressure_reclaim.max_requests
            if post_pressure_reclaim is not None
            else 0
        )
        needs_pressure_reclaim = bool(
            current_max < target
            and post_reclaim_max < target
            and post_pressure_reclaim_max >= target
        )
        if not needs_pressure_reclaim and post_reclaim_max <= current_max:
            return False
        if not physical_quiescent:
            if not needs_pressure_reclaim:
                return False
            reason = "pressure_reclaim_worker_step_inflight"
            self._elastic_admission_controller.defer_admission(
                self._next_elastic_transaction_id(),
                request_bytes=self._elastic_admission_controller.resident_bytes,
                available_bytes=self._elastic_irreducible_external_bytes(),
                reason=reason,
            )
            self._elastic_last_defer_reason = reason
            return True
        if self._elastic_admission_controller.pending_loans:
            if not needs_pressure_reclaim:
                return False
            reason = "pressure_reclaim_has_outstanding_loan"
            self._elastic_admission_controller.defer_admission(
                self._next_elastic_transaction_id(),
                request_bytes=self._elastic_admission_controller.resident_bytes,
                available_bytes=self._elastic_irreducible_external_bytes(),
                reason=reason,
            )
            self._elastic_last_defer_reason = reason
            return True
        plan_reclaim = (
            self._elastic_admission_controller.plan_pressure_reclaim_all
            if needs_pressure_reclaim
            else self._elastic_admission_controller.plan_reclaim_all
        )
        reclaim = plan_reclaim(
            self._next_elastic_transaction_id(),
            request_bytes=self._elastic_admission_controller.resident_bytes,
            available_bytes=0,
            protected_keys=getattr(self, "_elastic_serving_carrier_keys", ()),
        )
        if reclaim.kind in {
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            self._elastic_admission_controller.arm_maintenance(reclaim, None)
            return True
        if reclaim.kind == ElasticPlanKind.DEFER:
            self._elastic_admission_controller.observe_defer(reclaim)
            self._elastic_last_defer_reason = reclaim.defer_reason
            return needs_pressure_reclaim
        return False

    def _apply_elastic_waiting_candidate(
        self,
        request: Request,
        *,
        token_budget: int,
        waiting_count: int,
        physical_quiescent: bool = True,
    ) -> int:
        """Map the feasible waiting cohort before per-request allocation.

        A one-request lookahead can accept a smaller layout against retained
        Graph bytes and then serialize the whole queue without ever reaching
        a pressure reclaim boundary. Compare the current and post-reclaim MaxX
        using the coordinator's existing wave planner; reclaim only when it
        strictly increases the admissible cohort.
        """
        if (
            token_budget <= 0
            or not self.kv_cache_manager.kv_cache_config.elastic_mapping_quantum
        ):
            return 0
        occupied_slots = len(self.running) + self.num_waiting_for_streaming_input
        available_slots = max(1, self.max_num_running_reqs - occupied_slots)
        waiting_snapshot = self._elastic_schedulable_waiting_snapshot()
        candidates = (
            waiting_snapshot[:available_slots]
            if waiting_snapshot
            and any(candidate is request for candidate in waiting_snapshot)
            else (request,)
        )
        primary_requirements = [
            self.kv_cache_manager.estimate_uncached_full_sequence_requirements(
                candidate
            ).primary
            for candidate in candidates
        ]
        if self.running:
            primary_requirements[0] += self.kv_cache_manager.watermark_blocks

        if self._prepare_elastic_waiting_deficit_reclaim(
            tuple(primary_requirements),
            physical_quiescent=physical_quiescent,
        ):
            return 0
        admitted = self.kv_cache_manager.coordinator.apply_elastic_admission_wave(
            tuple(primary_requirements)
        )
        if (
            admitted == 0
            and self.elastic_on_demand_graphs
            and self._elastic_admission_controller.pending_maintenance_plan is None
        ):
            reclaim = self._elastic_admission_controller.plan_reclaim_all(
                self._next_elastic_transaction_id(),
                request_bytes=self._elastic_admission_controller.resident_bytes,
                available_bytes=0,
                protected_keys=getattr(self, "_elastic_serving_carrier_keys", ()),
            )
            if reclaim.kind == ElasticPlanKind.RECLAIM:
                self._elastic_admission_controller.arm_maintenance(reclaim, None)
            elif reclaim.kind == ElasticPlanKind.DEFER:
                self._elastic_admission_controller.observe_defer(reclaim)
        return admitted

    def prepare_elastic_restore_idle_reclaim(self) -> bool:
        """Prepare a shape-less reclaim before a calibration wave exists.

        Reclaim must precede ``add_request``.  Merely requiring an empty
        ``running`` list is insufficient because WAITING requests are already
        across the scheduler commit boundary and may be selected by the same
        step that executes maintenance.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic idle reclaim is calibration-only")
        if (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or self.num_waiting_for_streaming_input
        ):
            raise RuntimeError("elastic idle reclaim must precede every request commit")
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            raise RuntimeError("elastic idle reclaim found pending maintenance")
        if not self._elastic_admission_controller.resident_bytes:
            return False
        # Calibration rows are independent measurements, not a cumulative
        # serving hotset. Retaining earlier pinned FULL forms makes admission
        # depend on sweep order and can falsely contract a later product X.
        # The controller keeps zero-proof entries pinned fail-closed.
        self._elastic_admission_controller.unpin_idle()
        reclaim = self._elastic_admission_controller.plan_reclaim_all(
            self._next_elastic_transaction_id(),
            request_bytes=self._elastic_admission_controller.resident_bytes,
            available_bytes=0,
            protected_keys=getattr(self, "_elastic_serving_carrier_keys", ()),
        )
        if reclaim.kind != ElasticPlanKind.RECLAIM:
            hot_entries = tuple(
                entry
                for entry in self._elastic_admission_controller.entries.values()
                if entry.hot
            )
            protected = frozenset(getattr(self, "_elastic_serving_carrier_keys", ()))
            if (
                reclaim.kind == ElasticPlanKind.DEFER
                and reclaim.defer_reason == "no_reclaimable_piecewise_graphs"
                and hot_entries
                and {entry.key for entry in hot_entries} <= protected
            ):
                # The preceding restore cohort has already transferred its
                # complete HOT DAG into the serving carrier.  There is no
                # administrative residue to reclaim before extending that
                # carrier with the next cohort.
                return False
            if (
                reclaim.kind == ElasticPlanKind.DEFER
                and hot_entries
                and all(entry.pinned for entry in hot_entries)
                and not self._elastic_admission_controller.floor_bytes
            ):
                return False
            raise RuntimeError(
                "idle elastic restore could not reclaim the preceding "
                f"HOT set: reason={reclaim.defer_reason}"
            )
        self._elastic_admission_controller.arm_maintenance(reclaim, None)
        self._elastic_restore_wave_target = 0
        self._elastic_restore_wave_step_key = None
        return True

    def has_pending_elastic_maintenance(self) -> bool:
        """Whether ``schedule`` owes one explicit request-free transaction."""
        return self._elastic_admission_controller.pending_maintenance_plan is not None

    def prepare_elastic_restore_capture(
        self,
        step_key: tuple[int, ...],
    ) -> bool:
        """Prepare one COLD calibration owner set before KV admission.

        Returns whether the next scheduler output must execute request-free
        maintenance.  A HOT or compiled-only owner set needs no transaction.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic capture preparation is calibration-only")
        if self.running or self.num_waiting_for_streaming_input:
            raise RuntimeError("elastic restore capture requires no committed requests")
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            raise RuntimeError("elastic restore already has pending maintenance")

        fits, required_external, available_external = self._can_fund_elastic_graph_step(
            step_key,
            minimum_free_primary_blocks=0,
            allow_maintenance=True,
        )
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            return True
        if fits:
            return False
        raise RuntimeError(
            "elastic restore owner set could not be prepared before KV "
            "admission: "
            f"step_key={step_key!r} required_bytes={required_external} "
            f"available_bytes={available_external} "
            f"reason={getattr(self, '_elastic_last_defer_reason', None)!r}"
        )

    def assert_elastic_restore_captures_hot(
        self,
        step_keys: Sequence[tuple[int, ...]],
    ) -> None:
        """Require the complete declared physical sequence before admission."""
        missing = tuple(
            physical_key
            for step_key in step_keys
            for physical_key in self._resolve_elastic_step_physical_keys(step_key)
            if not (
                (entry := self._elastic_admission_controller.entries.get(physical_key))
                and entry.hot
            )
        )
        if missing:
            raise RuntimeError(
                "elastic restore physical sequence did not remain HOT "
                "through request-free preparation: "
                f"missing={tuple(key.identity for key in missing)}"
            )

    def retain_elastic_restore_captures(
        self,
        step_keys: Sequence[tuple[int, ...]],
    ) -> str:
        """Retain one declared physical DAG across multiple user steps."""
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic retention is calibration-only")
        physical_keys = tuple(
            dict.fromkeys(
                physical_key
                for step_key in step_keys
                for physical_key in self._resolve_elastic_step_physical_keys(step_key)
            )
        )
        if getattr(self, "_elastic_restore_retention_id", None) is not None:
            raise RuntimeError("elastic restore retention is already active")
        transaction_id = self._next_elastic_transaction_id()
        self._elastic_admission_controller.retain_hot(transaction_id, physical_keys)
        self._elastic_restore_retention_id = transaction_id
        self._elastic_restore_retained_physical_keys = physical_keys
        return transaction_id

    def release_elastic_restore_retention(self, transaction_id: str) -> None:
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic retention release is calibration-only")
        if getattr(self, "_elastic_restore_retention_id", None) != transaction_id:
            raise RuntimeError("elastic restore retention identity changed")
        self._elastic_admission_controller.release(transaction_id)
        self._elastic_restore_retention_id = None
        self._elastic_restore_retained_physical_keys = ()

    def promote_elastic_restore_retention_to_serving(
        self,
        transaction_id: str,
        physical_keys: Sequence[PhysicalReplayKey],
    ) -> None:
        """Protect selected shared owners before releasing a restore lease.

        Request-free drain may run immediately after the lease is released.  The
        serving-carrier set therefore has to become authoritative first; publishing
        it later in the engine bootstrap leaves a reclaim window between restore
        cohorts.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic carrier promotion is calibration-only")
        if getattr(self, "_elastic_restore_retention_id", None) != transaction_id:
            raise RuntimeError("elastic restore retention identity changed")
        retained = tuple(getattr(self, "_elastic_restore_retained_physical_keys", ()))
        serving = tuple(dict.fromkeys(physical_keys))
        if (
            not serving
            or not set(serving) <= set(retained)
            or any(
                not (
                    (entry := self._elastic_admission_controller.entries.get(key))
                    and entry.hot
                )
                for key in serving
            )
        ):
            raise RuntimeError("elastic serving promotion requires a HOT restore DAG")
        self._elastic_serving_carrier_keys = tuple(
            dict.fromkeys((*self._elastic_serving_carrier_keys, *serving))
        )
        self._elastic_serving_carrier_resident_bytes = (
            self._elastic_serving_carrier_bytes()
        )
        self._elastic_admission_controller.release(transaction_id)
        self._elastic_restore_retention_id = None
        self._elastic_restore_retained_physical_keys = ()

    def prepare_elastic_restore_execution(self, step_key: tuple[int, ...]) -> None:
        """Bind one retained final shape across incremental RUNNING assembly.

        Calibration constructs some product waves from an already-running
        decode cohort plus a new prefill.  The normal product preflight sees
        the complete cohort before the RUNNING loop, while calibration has
        already captured and retained that final owner set.  Bind that same
        final identity here so incremental X1..Xn prefixes do not request
        unrelated COLD graphs.  KV allocation and waiting admission remain on
        their ordinary paths; only Graph prefix discovery is bypassed.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic execution binding is calibration-only")
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            raise RuntimeError("elastic execution binding found pending maintenance")
        if getattr(self, "_elastic_restore_execution_step_key", None) is not None:
            raise RuntimeError("elastic restore execution is already bound")
        retained = frozenset(
            getattr(self, "_elastic_restore_retained_physical_keys", ())
        )
        physical_keys = self._resolve_elastic_step_physical_keys(step_key)
        if not physical_keys or not set(physical_keys) <= retained:
            raise RuntimeError(
                "elastic restore execution is outside its retained epoch"
            )
        if any(
            not (
                (entry := self._elastic_admission_controller.entries.get(physical_key))
                and entry.hot
            )
            for physical_key in physical_keys
        ):
            raise RuntimeError("elastic restore execution lost a HOT owner")
        self._elastic_restore_execution_step_key = step_key

    def prepare_elastic_restore_admission(
        self,
        request_ids: Sequence[str],
        *,
        retain_hot_graphs: bool = False,
    ) -> int:
        """Map one declared calibration wave before its first allocation.

        Normal serving deliberately maps only the current waiting request and
        one lookahead. A calibration probe instead asks how many members of
        one exact finite wave can coexist. Preparing that layout here keeps
        the serving lease policy intact and prevents a one-candidate tick from
        being mistaken for the physical MaxX boundary.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic wave preparation is calibration-only")
        if self.running or self.num_waiting_for_streaming_input:
            raise RuntimeError(
                "elastic restore wave preparation requires an idle scheduler"
            )
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError("elastic restore wave requires unique requests")
        requests = []
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None or request.status != RequestStatus.WAITING:
                raise RuntimeError(
                    "elastic restore wave contains a non-waiting request: "
                    f"request_id={request_id!r}"
                )
            requests.append(request)
        graph_resident_bytes = int(self._elastic_admission_controller.resident_bytes)
        if graph_resident_bytes and not retain_hot_graphs:
            raise RuntimeError(
                "elastic restore admission retained a preceding HOT set; "
                "idle reclaim must complete before requests are added"
            )
        if retain_hot_graphs and not graph_resident_bytes:
            raise RuntimeError(
                "retained-HOT calibration admission requires an existing "
                "graph working set"
            )
        # Price the final prefill and its immediate decode successor in one
        # finite-wave fixed point. Successor headroom is a block-boundary
        # delta, not one whole block per request.
        requirements = []
        for request in requests:
            estimate = (
                self.kv_cache_manager.estimate_uncached_full_sequence_requirements(
                    request
                )
            )
            primary = estimate.primary
            # The immediate decode token normally reuses the writable tail of
            # the block allocated by prefill. Charge an additional primary
            # block only when the declared sequence ends exactly on the real
            # scheduler block boundary. The old unconditional +1 reproduced
            # v9's false X27 ceiling: X40 required 82 blocks although its
            # two-token prompts consume only 42 including null+watermark.
            current_blocks = (
                request.num_tokens + self.block_size - 1
            ) // self.block_size
            successor_blocks = (
                request.num_tokens + 1 + self.block_size - 1
            ) // self.block_size
            requirements.append(primary + successor_blocks - current_blocks)
        # ``allocate_slots`` preserves the block manager watermark once these
        # requests become RUNNING.  The finite-wave preflight starts from an
        # idle scheduler, so no ordinary running-request admission call has
        # priced that shared (not per-request) attention tail yet.  Reserve it
        # exactly once here.  Without it X23 consumed null + 2*X == 47 blocks,
        # leaving no free watermark block when the measured graph envelope
        # settled one attention block below the provisional layout.
        requirements[0] += self.kv_cache_manager.watermark_blocks
        prospective_tokens = {
            request_id: request.num_tokens
            for request_id, request in zip(request_ids, requests, strict=True)
        }
        # Use the same request-aware K selector as the eventual commit.  The
        # calibration inventory deliberately includes a forced-K0 control in
        # a globally K3 runtime; using ``self.num_spec_tokens`` here prepared a
        # K3 owner set while schedule() correctly committed those requests as
        # K0, turning the resulting physical-key mismatch into a misleading
        # atomic-hotset budget failure.
        prospective_k = self._num_spec_tokens_for_step(
            prospective_tokens,
            is_pure_decode_step=False,
        )
        step_key = self._canonical_elastic_graph_step_key(
            prospective_tokens,
            prospective_k,
            False,
        )
        capture_envelope = self._elastic_capture_envelope(step_key)
        capture_envelope_bytes = (
            0 if capture_envelope is None else int(capture_envelope[0])
        )
        # Use the same complete physical-loan estimator as commit.  A catalog
        # row can describe only the resident graph subset (125.8 MiB for the
        # observed K3/X43/B4096 carrier), while the already measured logical
        # witness also owns eager/cuBLAS runtime workspace (528.5 MiB).  Pricing
        # only ``_elastic_capture_envelope`` admitted KV that left no room for
        # the proven commit floor.  The estimator is side-effect free and folds
        # the exact HOT owner set, measured witness bytes and carrier closure.
        prospective_graph_bytes, _capture_planned = (
            self._estimate_elastic_graph_step_bytes(step_key)
        )
        # Rebuilding request KV around an intentionally retained executable
        # must coexist with the current physical Graph charge.  A prefill
        # envelope can be smaller than that already-live decode owner set.
        external_memory_bytes = max(
            graph_resident_bytes if retain_hot_graphs else 0,
            capture_envelope_bytes,
            prospective_graph_bytes,
        )
        admitted = self.kv_cache_manager.coordinator.apply_elastic_admission_wave(
            tuple(requirements),
            external_memory_bytes=external_memory_bytes,
        )
        self._elastic_restore_wave_target = admitted
        self._elastic_restore_wave_step_key = step_key
        return admitted

    def cancel_unexecuted_elastic_restore_step(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Release graph transaction state when a synthetic wave is not run."""
        if not self._elastic_restore_mode:
            raise RuntimeError("unexecuted-step cancellation is calibration-only")
        latest_loan = self._elastic_admission_controller.latest_loan
        if latest_loan is None:
            raise RuntimeError("unexecuted calibration step lost its graph loan")
        pending_step_key = latest_loan.step_key
        pending_grant = latest_loan.grant_bytes
        plan = scheduler_output.elastic_step_plan
        transaction_id = scheduler_output.elastic_transaction_id
        if plan is not None:
            if plan.kind == ElasticPlanKind.USER:
                assert transaction_id is not None
                self._elastic_admission_controller.cancel(transaction_id)
                getattr(self, "_elastic_useful_started", {}).pop(transaction_id, None)
            elif plan.kind == ElasticPlanKind.MAINTENANCE:
                self._elastic_admission_controller.fail_maintenance(plan)
                started = self._elastic_maintenance_started.pop(
                    plan.transaction_id, None
                )
                if started is not None:
                    started_at, stats_before, external_before = started
                    stats_after = self._elastic_admission_controller.stats
                    wall_ms = (time.monotonic() - started_at) * 1000.0
                    self._elastic_maintenance_wall_ms_total = (
                        getattr(self, "_elastic_maintenance_wall_ms_total", 0.0)
                        + wall_ms
                    )
                    self._elastic_maintenance_transactions_total = (
                        getattr(self, "_elastic_maintenance_transactions_total", 0) + 1
                    )
                    key_outcomes = getattr(self, "_elastic_graph_key_outcomes", None)
                    if key_outcomes is None:
                        key_outcomes = self._elastic_graph_key_outcomes = defaultdict(
                            int
                        )
                    for physical_key in plan.physical_keys:
                        logical = physical_key.logical
                        outcome_key = (
                            f"{logical.owner}|{logical.mode}|"
                            f"{logical.token_bucket}|CANCELLED"
                        )
                        key_outcomes[outcome_key] += 1
                    logger.info(
                        "Elastic Graph transaction end: tx=%s kind=%s "
                        "final_key=%s wall_ms=%.3f captures=%d evictions=%d "
                        "evicted_bytes=%d external_before=%d external_after=%d "
                        "floor=%d peak=%d outcome=CANCELLED",
                        plan.transaction_id,
                        plan.kind.value,
                        pending_step_key,
                        wall_ms,
                        stats_after.promotions - stats_before.promotions,
                        stats_after.evictions - stats_before.evictions,
                        stats_after.evicted_bytes - stats_before.evicted_bytes,
                        external_before,
                        self._elastic_admission_controller.resident_bytes,
                        self._elastic_admission_controller.floor_bytes,
                        0,
                    )
            elif plan.kind in {
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }:
                raise RuntimeError(
                    "cannot cancel an unexecuted physical reclaim transaction"
                )
        cancelled_loan = self._elastic_admission_controller.cancel_latest_loan()
        if (
            cancelled_loan.step_key != pending_step_key
            or cancelled_loan.grant_bytes != pending_grant
        ):
            raise RuntimeError("unexecuted calibration step FIFO tail changed")
        if pending_grant != scheduler_output.elastic_external_memory_bytes:
            raise RuntimeError(
                "unexecuted calibration step graph loan is not the FIFO tail"
            )
        latest_loan = self._elastic_admission_controller.latest_loan
        self._elastic_admission_controller.publish_step_key(
            None if latest_loan is None else latest_loan.step_key
        )

    def recover_elastic_execution_plan_mismatch(
        self, scheduler_output: SchedulerOutput
    ) -> list[Request]:
        """Abort one pre-mutation execution while preserving the engine epoch."""
        self._elastic_accepted_decode_consensus_epoch = None
        plan = scheduler_output.elastic_step_plan
        if plan is None:
            raise RuntimeError("execution-plan recovery requires an immutable plan")
        has_user_tokens = scheduler_output.total_num_scheduled_tokens > 0
        if has_user_tokens and plan.execution_manifest is None:
            raise RuntimeError("USER execution recovery requires an immutable manifest")
        if not has_user_tokens and plan.kind not in {
            ElasticPlanKind.MAINTENANCE,
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            raise RuntimeError("request-free recovery requires a maintenance plan")
        transaction_id = scheduler_output.elastic_transaction_id
        if transaction_id != plan.transaction_id:
            raise RuntimeError("execution-plan recovery transaction changed")
        if scheduler_output.elastic_plan_fingerprint != plan.fingerprint:
            raise RuntimeError("execution-plan recovery fingerprint changed")
        pending_loans = self._elastic_admission_controller.pending_loans
        if not pending_loans:
            raise RuntimeError("execution-plan recovery lost its scheduler loan")
        if pending_loans[0].step_key != scheduler_output.elastic_graph_step_key:
            raise RuntimeError(
                "execution-plan recovery encountered an out-of-order loan"
            )
        if not (
            pending_loans[0].grant_bytes
            == scheduler_output.elastic_external_memory_bytes
            == plan.capture_loan_bytes
        ):
            raise RuntimeError("execution-plan recovery grant identity changed")

        deferred_mm_wave = self._elastic_deferred_mm_wave
        has_exact_deferred_binding = bool(
            not has_user_tokens
            and plan.kind == ElasticPlanKind.MAINTENANCE
            and plan.maintenance_execution == ElasticMaintenanceExecution.GRAPH_ONLY
            and deferred_mm_wave is not None
            and deferred_mm_wave.step_key == scheduler_output.elastic_graph_step_key
        )
        bound_request_ids = (
            ()
            if not has_exact_deferred_binding
            else (
                # The bool proof above makes this non-optional.
                # Keep the expression local so stale waves can never select
                # unrelated request IDs for terminal recovery.
                *cast(DeferredMMWave, deferred_mm_wave).running_request_ids,
                *cast(DeferredMMWave, deferred_mm_wave).waiting_request_ids,
            )
        )

        self._elastic_admission_controller.rollback_pre_mutation(plan)
        loan = self._elastic_admission_controller.settle_next_loan()
        assert loan == pending_loans[0]
        if (
            self._elastic_admission_controller.recapture_pending_key
            == scheduler_output.elastic_graph_step_key
        ):
            self._elastic_admission_controller.recapture_pending_key = None
        getattr(self, "_elastic_useful_started", {}).pop(transaction_id, None)
        self._elastic_maintenance_started.pop(transaction_id, None)

        # A rejected request-free GRAPH_ONLY capture has no physical result for
        # its frozen USER binding. Terminate that exact cohort after rollback;
        # retrying the same deterministic rank mismatch would otherwise spin at
        # 100% CPU without ever producing output.
        if not has_user_tokens:
            self._elastic_deferred_mm_wave = None
            self._elastic_graph_carrier_step_key = None
            self._elastic_preflight_joint_waiting_request_ids = ()
            self._elastic_preflight_waiting_ignore_prefix_request_ids = ()

        if self.defer_block_free and has_user_tokens:
            self.processed_step_seq += 1
            self._drain_deferred_frees()
        failed = (
            self.finish_requests(
                scheduler_output.num_scheduled_tokens,
                RequestStatus.FINISHED_ERROR,
            )
            if has_user_tokens
            else self.finish_requests(bound_request_ids, RequestStatus.FINISHED_ERROR)
        )

        coordinator = self.kv_cache_manager.coordinator
        coordinator.rebalance_elastic_capacity()
        retained_external = self._elastic_admission_controller.resident_bytes
        if not coordinator.set_elastic_external_memory(retained_external):
            raise RuntimeError(
                "execution-plan recovery could not restore settled Graph/KV memory"
            )
        if not has_user_tokens and not bound_request_ids:
            raise RuntimeError(
                "request-free administrative execution-plan mismatch recovered "
                "physical state but has no bounded cohort; stopping engine epoch"
            )
        logger.error(
            "ELASTIC_EXECUTION_PLAN_MISMATCH recovered before worker mutation: "
            "transaction=%s step_key=%r requests=%s retained_external=%d",
            transaction_id,
            scheduler_output.elastic_graph_step_key,
            tuple(request.request_id for request in failed),
            retained_external,
        )
        return failed

    def reconcile_elastic_restore_rollback(self) -> None:
        """Restore the idle KV/Graph fixed point after calibration rollback.

        ``prepare_elastic_restore_admission`` is a physical mutation: it can
        resize both KV arenas and publish a larger external-memory charge even
        when it returns only a feasible prefix.  If the caller rejects that
        prefix, no scheduler output exists to consume and settle the prepared
        wave.  A later pre-execution gate can also cancel an already-built
        scheduler output, while post-execution rejection has already settled
        its loan. Normalize every rollback state here before any administrative
        drain is allowed to run so callers cannot omit physical cleanup.
        """
        if not self._elastic_restore_mode:
            raise RuntimeError("elastic restore rollback is calibration-only")
        has_wave_target = bool(self._elastic_restore_wave_target)
        has_wave_key = self._elastic_restore_wave_step_key is not None
        if has_wave_target != has_wave_key:
            raise RuntimeError(
                "elastic restore rollback found an incomplete wave identity"
            )
        self._elastic_restore_wave_target = 0
        self._elastic_restore_wave_step_key = None
        self._elastic_restore_execution_step_key = None
        coordinator = self.kv_cache_manager.coordinator
        coordinator.rebalance_elastic_capacity()
        required_external = max(
            [
                self._elastic_admission_controller.resident_bytes,
                *(
                    loan.grant_bytes
                    for loan in self._elastic_admission_controller.pending_loans
                ),
            ]
        )
        if not coordinator.set_elastic_external_memory(required_external):
            raise RuntimeError(
                "elastic restore rollback could not restore KV/graph "
                f"layout: external_bytes={required_external}"
            )

    def finish_elastic_restore_epoch(self) -> None:
        """Drop synthetic logical affinity without evicting restored HOT state."""
        if self.running or self.waiting or self.skipped_waiting:
            raise RuntimeError(
                "elastic restore epoch ended with a live synthetic request cohort"
            )
        if self._elastic_admission_controller.pending_maintenance_plan is not None:
            raise RuntimeError("elastic restore epoch ended with pending maintenance")
        if self._elastic_admission_controller.pending_loans:
            raise RuntimeError("elastic restore epoch ended with pending loans")
        if getattr(self, "_elastic_restore_retention_id", None) is not None:
            raise RuntimeError("elastic restore epoch ended with active retention")
        self._elastic_graph_carrier_step_key = None

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's read-ahead has also passed it,
        # mirroring the shift the encoder scheduling path applies.
        spec_lookahead = self.num_prefill_lookahead

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self._free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self._free_encoder_input(request, input_id)

    def _free_encoder_input(self, request: Request, input_id: int) -> None:
        self.encoder_cache_manager.free_encoder_input(request, input_id)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_free(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return self.kv_cache_manager.usage

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.spec_decode_metrics_level != "none":
                request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                    self.num_spec_tokens
                )
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        logger.info("setting pause state to %s", pause_state.name)
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or self._needs_elastic_idle_reclaim()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.to_dict() if kv_connector_stats else None
        )
        num_kv_tail_deferrals = self.num_kv_tail_deferrals_since_last_stats
        self.num_kv_tail_deferrals_since_last_stats = 0
        num_canonical_prefill_admission_deferrals = (
            self.num_canonical_prefill_deferrals_since_last_stats
        )
        self.num_canonical_prefill_deferrals_since_last_stats = 0
        elastic_stats = self._elastic_admission_controller.stats
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            num_kv_tail_deferrals=num_kv_tail_deferrals,
            num_canonical_prefill_admission_deferrals=(
                num_canonical_prefill_admission_deferrals
            ),
            elastic_graph_stats={
                "hot_hits": elastic_stats.hot_hits,
                "cold_misses": elastic_stats.cold_misses,
                "promotions": elastic_stats.promotions,
                "evictions": elastic_stats.evictions,
                "evicted_bytes": elastic_stats.evicted_bytes,
                "deferrals": elastic_stats.deferrals,
                "defer_reasons": dict(elastic_stats.defer_reasons),
                "pinned_bytes": (
                    self._elastic_admission_controller.pinned_resident_bytes
                ),
                "evictable_bytes": (
                    self._elastic_admission_controller.evictable_resident_bytes
                ),
                "external_bytes": self._elastic_admission_controller.resident_bytes,
                "external_floor_bytes": self._elastic_admission_controller.floor_bytes,
                "maintenance_wall_ms_total": getattr(
                    self, "_elastic_maintenance_wall_ms_total", 0.0
                ),
                "maintenance_transactions_total": getattr(
                    self, "_elastic_maintenance_transactions_total", 0
                ),
                "useful_wall_ms_total": getattr(
                    self, "_elastic_useful_wall_ms_total", 0.0
                ),
                "useful_transactions_total": getattr(
                    self, "_elastic_useful_transactions_total", 0
                ),
                "rate_limited_logs_total": getattr(
                    self, "_elastic_rate_limited_logs_total", 0
                ),
                "key_totals": dict(getattr(self, "_elastic_graph_key_outcomes", {})),
            },
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        finished_partial_tails: list[tuple[int, int, int]] = []
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.is_kv_producer:
            finished_partial_tails = (
                self.kv_cache_manager.finalize_partial_tail_offloads(request)
            )

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )
        partial_tail_delay = False
        if finished_partial_tails:
            partial_tail_delay = self.connector.register_finished_partial_tail(
                request,
                block_ids,
                finished_partial_tails,
            )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            delay_free, kv_xfer_params = self.connector.request_finished(
                request, block_ids[0]
            )
        else:
            delay_free, kv_xfer_params = self.connector.request_finished_all_groups(
                request, block_ids
            )
        return delay_free or partial_tail_delay, kv_xfer_params

    def _request_remaining_blocks(
        self, request: Request
    ) -> KVCacheBlockPoolRequirements:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        computed_tokens = request.num_computed_tokens
        blocks = self.kv_cache_manager.empty_kv_cache_blocks
        bound = getattr(self, "_elastic_prefix_hits", {}).get(request.request_id)
        if bound is not None and computed_tokens == 0:
            blocks, computed_tokens, _boundary, _diverged = bound[1]
        return self.kv_cache_manager.coordinator.get_block_pool_requirements(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=computed_tokens,
            num_local_computed_tokens=computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(
        self, *, exclude: Request | None = None
    ) -> KVCacheBlockPoolRequirements:
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        reserved = KVCacheBlockPoolRequirements()
        for request in self._inflight_prefills:
            if request is exclude:
                continue
            reserved += self._request_remaining_blocks(request)
        return reserved

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not structured_output_req or structured_output_req.grammar is None:
                return False
            if isinstance(structured_output_req.grammar, Exception):
                self.grammar_compile_error_reqs.add(request.request_id)
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
