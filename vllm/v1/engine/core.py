# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast, get_args

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.pooler import POOLER_CONFIG_LOG_FIELDS
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    stateless_destroy_torch_distributed_process_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalCacheMissError
from vllm.sampling_params import SamplingParams
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils import numa_utils
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.elastic_graph import ElasticPlanKind
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
    update_kv_cache_capacity,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.elastic_bootstrap import (
    complete_elastic_startup,
    prepare_elastic_runtime,
    resolve_elastic_graph_execution_policy,
    synchronize_elastic_runtime_generation,
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_physical_gpu_ids_for_local_dp_rank,
)
from vllm.v1.executor import Executor
from vllm.v1.executor.worker_failure import WorkerFailureCode, WorkerRemoteError
from vllm.v1.fault_tolerance.engine_core_sentinel import (
    FT_UTILITY_METHOD,
    EngineCoreSentinel,
    fault_tolerant_wrapper,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    get_kv_cache_spec_kind,
    is_full_attention_spec,
)
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class _ElasticRestorePartialWave(RuntimeError):
    def __init__(self, admitted_x: int, requested_x: int) -> None:
        super().__init__(
            "elastic restore refused a partial active cohort before model "
            f"execution: requested_x={requested_x} admitted_x={admitted_x}"
        )
        self.admitted_x = admitted_x


HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats
        # Opaque weight version supplied by the caller.
        self._weight_version = "default"

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        self._pooler_config_logged = False
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        prepared_runtime = prepare_elastic_runtime(
            vllm_config=vllm_config,
            initialize_kv_cache=lambda: self._initialize_kv_caches(vllm_config),
            collective_rpc=self.collective_rpc,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
        )
        self.structured_output_manager = prepared_runtime.structured_output_manager
        self.scheduler = prepared_runtime.scheduler
        self._initialize_effective_attention_block_size()
        self.elastic_runtime_generation_receipt = prepared_runtime.generation_receipt
        hash_block_size = prepared_runtime.hash_block_size
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.check_for_draft_tokens = (
            self.use_spec_decode or vllm_config.model_config.is_diffusion
        )
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
        if self.scheduler.ec_connector is not None:  # type: ignore
            self.model_executor.init_ec_output_aggregator()

        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {(pp_rank, tp_rank): metadata}
                # Merge all worker dicts into a single dict
                content: dict[tuple[int, int], Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata_pp_aware(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_consumer = (
            vllm_config.ec_transfer_config is None
            or vllm_config.ec_transfer_config.is_ec_consumer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # Bounded research observer for attributing the host-side gap between
        # otherwise identical saturated decode iterations. It is inert unless
        # an output path is explicitly configured, keeps records in memory,
        # and performs one atomic write only after the requested steady window.
        self._ag2_step_trace_path = os.environ.get("AG2_VLLM_ENGINE_STEP_TRACE", "")
        self._ag2_step_trace_min_running = int(
            os.environ.get("AG2_VLLM_ENGINE_STEP_TRACE_MIN_RUNNING", "0")
        )
        self._ag2_step_trace_limit = int(
            os.environ.get("AG2_VLLM_ENGINE_STEP_TRACE_STEPS", "64")
        )
        self._ag2_step_trace_include_context = (
            os.environ.get("AG2_VLLM_ENGINE_STEP_TRACE_INCLUDE_CONTEXT", "0") == "1"
        )
        self._ag2_step_trace_require_context = (
            os.environ.get("AG2_VLLM_ENGINE_STEP_TRACE_REQUIRE_CONTEXT", "0") == "1"
        )
        self._ag2_step_trace_saw_context = False
        self._ag2_step_trace_records: list[dict[str, int | float]] = []
        self._ag2_step_trace_previous_start_ns: int | None = None
        self._ag2_step_trace_complete = False
        if self._ag2_step_trace_path:
            if self._ag2_step_trace_min_running < 1:
                raise ValueError("engine step trace requires MIN_RUNNING >= 1")
            if self._ag2_step_trace_limit < 1:
                raise ValueError("engine step trace requires STEPS >= 1")
            if (
                self._ag2_step_trace_require_context
                and not self._ag2_step_trace_include_context
            ):
                raise ValueError(
                    "engine step trace REQUIRE_CONTEXT requires INCLUDE_CONTEXT"
                )
            logger.info(
                "AG2 engine-step trace initialized: path=%s min_running=%d "
                "steps=%d include_context=%s require_context=%s",
                self._ag2_step_trace_path,
                self._ag2_step_trace_min_running,
                self._ag2_step_trace_limit,
                self._ag2_step_trace_include_context,
                self._ag2_step_trace_require_context,
            )

        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # This completes the same production-owned bootstrap used by the
        # standalone full-DAG producer before either route can publish READY.
        complete_elastic_startup(self)

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    def _resolve_elastic_graph_execution_policy(
        self, vllm_config: VllmConfig, kv_cache_config: Any
    ) -> None:
        resolve_elastic_graph_execution_policy(
            vllm_config, kv_cache_config, self.collective_rpc
        )

    def _synchronize_elastic_runtime_generation(self) -> None:
        """Make the post-KV scheduler generation authoritative on workers."""
        synchronize_elastic_runtime_generation(self.scheduler, self.collective_rpc)

    def _restore_elastic_pinned_full_family(self) -> None:
        """Recapture the sealed FULL family before serving becomes healthy.

        CUDA graph executables are not disk-serializable.  The catalog and
        recipe make a warm restart bounded, but every process epoch must still
        recreate and pin the executable/address-stable state.
        """
        scheduler = cast(Any, self.scheduler)
        coverage = scheduler._elastic_graph_catalog_coverage
        max_x = int(coverage["mixed_max_x"])
        configured_k = int(scheduler.num_spec_tokens)
        query_lens = (1, 1 + configured_k) if configured_k else (1,)
        required = {
            tuple(int(value) for value in key)
            for key in coverage["required_step_keys"]
            if isinstance(key, list) and len(key) == 5 and key[0] == 1
        }
        expected = {
            (1, configured_k, x, x * query_len, query_len)
            for query_len in query_lens
            for x in range(1, max_x + 1)
        }
        if not expected.issubset(required):
            missing = sorted(expected - required)
            raise RuntimeError(
                "sealed catalog omitted required pinned FULL classes: "
                f"missing={len(missing)} first={missing[:8]}"
            )

        started = time.monotonic()
        previous_restore_mode = getattr(scheduler, "_elastic_restore_mode", False)
        scheduler._elastic_restore_mode = True
        scheduler.max_num_running_reqs = max_x
        serial = 0
        try:
            # Restore the exact accepted layer order. Aggregate prices and KV
            # coexistence are capture-context dependent; family-major replay
            # would materialize an unmeasured q4 suffix before q1/X1.
            for x in range(1, max_x + 1):
                for query_len in reversed(query_lens):
                    serial += 1
                    measured_key, admitted_x = self._run_elastic_full_restore_wave(
                        k=configured_k,
                        x=x,
                        query_len=query_len,
                        serial=serial,
                    )
                    wanted = (1, configured_k, x, x * query_len, query_len)
                    if measured_key != wanted or admitted_x != x:
                        raise RuntimeError(
                            "sealed pinned FULL boundary failed warm restore: "
                            f"expected={wanted} actual={measured_key} "
                            f"admitted_x={admitted_x}"
                        )
                    elapsed = time.monotonic() - started
                    if elapsed > 120:
                        raise RuntimeError(
                            "sealed pinned FULL warm restore exceeded 120 s: "
                            f"completed={serial}/{len(expected)} elapsed={elapsed:.3f}"
                        )
        finally:
            scheduler._elastic_restore_mode = previous_restore_mode
        logger.warning(
            "Elastic sealed pinned FULL family restored before READY: "
            "logical_shapes=%d max_x=%d wall_seconds=%.3f",
            len(expected),
            max_x,
            time.monotonic() - started,
        )

    def _restore_elastic_bounded_hotset(self) -> None:
        """Restore only the sealed minimal carrier set before READY."""
        from vllm.v1.core.elastic_catalog import (
            validate_elastic_catalog_key_inventory,
        )

        scheduler = cast(Any, self.scheduler)
        coverage = scheduler._elastic_graph_catalog_coverage
        expected_owner = "target" if scheduler.num_spec_tokens == 0 else "mtp_decode"
        expected_contract = (
            "retained-terminal-target-no-cold-serving-v1"
            if scheduler.num_spec_tokens == 0
            else "retained-terminal-mtp-no-cold-serving-v1"
        )
        if coverage.get("serving_carrier_contract") != expected_contract:
            raise RuntimeError(
                "bounded elastic serving carrier contract is unsupported"
            )
        if coverage.get("serving_carrier_owner") != expected_owner:
            raise RuntimeError("bounded elastic serving carrier owner is unsupported")
        _required, restore_inventory = validate_elastic_catalog_key_inventory(
            coverage.get("required_step_keys"),
            coverage.get("serving_carrier_step_keys", []),
            label="bounded elastic serving carrier",
            require_restore=True,
        )
        restore = set(restore_inventory)
        terminal_x = coverage.get("decode_max_x")
        if (
            type(terminal_x) is not int
            or terminal_x < 1
            or terminal_x > scheduler.max_num_running_reqs
        ):
            raise RuntimeError(
                "bounded elastic startup has an invalid terminal decode cohort"
            )
        k = scheduler.num_spec_tokens
        terminal_query_len = scheduler.num_spec_tokens + 1

        started = time.monotonic()
        previous_restore_mode = getattr(scheduler, "_elastic_restore_mode", False)
        scheduler._elastic_restore_mode = True
        try:
            q1_key = self._elastic_restore_decode_key(k=k, x=terminal_x, query_len=1)
            terminal_key = self._elastic_restore_decode_key(
                k=k,
                x=terminal_x,
                query_len=terminal_query_len,
            )
            q1_execution = self._elastic_restore_execution_step_keys(
                k=k,
                x=terminal_x,
                query_len=1,
            )
            verification_execution = self._elastic_restore_execution_step_keys(
                k=k,
                x=terminal_x,
                query_len=terminal_query_len,
            )
            expected_restore = {
                key
                for key in (*q1_execution, *verification_execution)
                if scheduler._resolve_elastic_step_physical_keys(key)
            }
            if restore != expected_restore:
                raise RuntimeError(
                    "bounded elastic serving carrier differs from the exact "
                    "prefill/q1/q(K+1) identity: "
                    f"declared={sorted(restore)!r} "
                    f"expected={sorted(expected_restore)!r}"
                )
            q1_measured_key, q1_admitted_x = (
                self._run_elastic_full_restore_wave_in_epoch(
                    k=q1_key[1],
                    x=q1_key[2],
                    query_len=1,
                    serial=1,
                    preserve_hotset=True,
                )
            )
            if q1_measured_key != q1_key or q1_admitted_x != q1_key[2]:
                raise RuntimeError(
                    "bounded elastic restore did not reproduce its q1 carrier: "
                    f"expected={q1_key!r} actual={q1_measured_key!r} "
                    f"admitted_x={q1_admitted_x}"
                )
            # The q1 wave deliberately leaves restore mode to prevent its
            # terminal owners from being reclaimed while requests drain.  The
            # second independent product-shaped wave re-enters calibration
            # mode only for its pre-READY capture transaction.
            scheduler._elastic_restore_mode = True
            measured_key, admitted_x = self._run_elastic_full_restore_wave_in_epoch(
                k=q1_key[1],
                x=q1_key[2],
                query_len=terminal_query_len,
                serial=2,
                preserve_hotset=True,
            )
            if measured_key != terminal_key or admitted_x != q1_key[2]:
                raise RuntimeError(
                    "bounded elastic restore did not reproduce its terminal carrier: "
                    f"expected={terminal_key!r} actual={measured_key!r} "
                    f"admitted_x={admitted_x}"
                )
            carrier_keys = scheduler.resolve_elastic_serving_carrier_physical_keys(
                (q1_key, terminal_key)
            )
            scheduler._elastic_serving_carrier_keys = carrier_keys
            scheduler._elastic_serving_carrier_resident_bytes = (
                scheduler._elastic_serving_carrier_bytes()
            )
            hotset_xs = tuple(getattr(scheduler, "_elastic_serving_hotset_xs", ()))
            hotset_steps: list[tuple[int, ...]] = []
            for semantic_x in hotset_xs:
                # Mixed/prefill target and MTP-prefill owners are compiled-only;
                # this shape retains the exact-X MTP decode owner used while a
                # long prompt is in flight.
                hotset_steps.append((0, k, semantic_x, semantic_x, 0))
                physical_x = scheduler._elastic_short_decode_physical_x(semantic_x)
                hotset_steps.append(
                    (
                        0,
                        k,
                        physical_x,
                        physical_x * terminal_query_len,
                        terminal_query_len,
                    )
                )
            hotset_steps = [
                key
                for key in hotset_steps
                if scheduler._resolve_elastic_step_physical_keys(key)
            ]
            hotset_steps = list(dict.fromkeys(hotset_steps))
            declared_hotset_steps = [
                tuple(key) for key in coverage.get("serving_hotset_step_keys", ())
            ]
            if hotset_xs and (
                coverage.get("serving_hotset_contract")
                != "pre-ready-hot-no-runtime-maintenance-v1"
                or declared_hotset_steps != hotset_steps
            ):
                raise RuntimeError(
                    "configured serving hotset differs from the sealed catalog: "
                    f"declared={declared_hotset_steps!r} "
                    f"configured={hotset_steps!r}"
                )
            required_steps = {
                tuple(key) for key in coverage.get("required_step_keys", ())
            }
            missing_hotset_steps = tuple(
                key for key in hotset_steps if key not in required_steps
            )
            if missing_hotset_steps:
                raise RuntimeError(
                    "configured serving hotset is absent from the sealed catalog: "
                    f"missing={missing_hotset_steps!r}"
                )
            for step_key in hotset_steps:
                scheduler._elastic_restore_mode = True
                self._prepare_elastic_restore_capture(step_key)
                scheduler.assert_elastic_restore_captures_hot((step_key,))
                physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
                if not physical_keys:
                    continue
                retention = scheduler.retain_elastic_restore_captures((step_key,))
                try:
                    scheduler.promote_elastic_restore_retention_to_serving(
                        retention, physical_keys
                    )
                finally:
                    if scheduler._elastic_restore_retention_id == retention:
                        scheduler.release_elastic_restore_retention(retention)
            scheduler._elastic_serving_carrier_resident_bytes = (
                scheduler._elastic_serving_carrier_bytes()
            )
            if not carrier_keys or any(
                not (
                    (entry := scheduler._elastic_admission_controller.entries.get(key))
                    and entry.hot
                )
                for key in carrier_keys
            ):
                raise RuntimeError("terminal serving carrier was not retained HOT")
            if time.monotonic() - started > 120:
                raise RuntimeError("bounded elastic hotset restore exceeded 120 s")
        finally:
            scheduler._elastic_restore_mode = previous_restore_mode
        scheduler.max_num_running_reqs = int(coverage["mixed_max_x"])
        logger.warning(
            "Elastic bounded terminal carrier restored before READY: "
            "restore_shapes=%d hotset_shapes=%d max_x=%d resident_bytes=%d "
            "wall_seconds=%.3f",
            len(restore),
            len(hotset_steps),
            coverage["mixed_max_x"],
            scheduler._elastic_serving_carrier_resident_bytes,
            time.monotonic() - started,
        )

    def _synchronize_elastic_startup_residency(self) -> None:
        """Publish restored HOT state and its replay workspace before READY."""
        receipts: list[Any] = self.collective_rpc("get_elastic_graph_residency_receipt")
        if not receipts:
            raise RuntimeError("elastic startup returned no worker residency receipt")

        scheduler = cast(Any, self.scheduler)
        expected_generation = scheduler._elastic_admission_controller.generation.value
        receipt_generations = [receipt.generation.value for receipt in receipts]
        workspace_units = [receipt.cublas_workspace_bytes for receipt in receipts]
        if any(generation != expected_generation for generation in receipt_generations):
            raise RuntimeError(
                "elastic startup workspace generation differs from scheduler: "
                f"expected={expected_generation!r} "
                f"receipts={receipts!r}"
            )
        if any(unit <= 0 for unit in workspace_units):
            raise RuntimeError(
                "elastic startup workspace unit must be positive on every rank: "
                f"receipts={receipts!r}"
            )
        rank_safe_workspace_unit = max(workspace_units)

        def rank_safe_projection(receipt: Any) -> tuple[object, ...]:
            # local_pool_bytes is rank-local; all other fields are consensus.
            entries = tuple(
                (
                    entry.key,
                    entry.pinned,
                    entry.resident_bytes,
                    entry.reclaimable_bytes,
                    entry.lease_ids,
                )
                for entry in receipt.entries
            )
            return (
                receipt.generation,
                receipt.transaction_id,
                receipt.resident_bytes,
                receipt.floor_bytes,
                receipt.transition_floor_bytes,
                receipt.peak_bytes,
                receipt.complete,
                receipt.schema,
                receipt.schema_fingerprint,
                entries,
            )

        reference = receipts[0]
        projected = rank_safe_projection(reference)
        if any(rank_safe_projection(receipt) != projected for receipt in receipts):
            raise RuntimeError(
                "elastic startup HOT residency differs across worker ranks"
            )

        # Mutate scheduler state only after both receipts validate. A stale or
        # malformed workspace epoch must preserve the previously accepted HOT
        # snapshot and unit rather than partially publishing startup state.
        scheduler._sync_elastic_residency_receipt(reference)
        scheduler._elastic_cublas_workspace_unit_bytes = rank_safe_workspace_unit
        serving_carrier = tuple(getattr(scheduler, "_elastic_serving_carrier_keys", ()))
        if serving_carrier:
            missing = tuple(
                key
                for key in serving_carrier
                if not (
                    (entry := scheduler._elastic_admission_controller.entries.get(key))
                    and entry.hot
                )
            )
            if missing:
                raise RuntimeError(
                    "startup residency receipt dropped retained serving carrier: "
                    f"missing={tuple(key.identity for key in missing)!r}"
                )
            scheduler._elastic_serving_carrier_resident_bytes = (
                scheduler._elastic_serving_carrier_bytes()
            )
        # The scheduler constructor can only publish the catalog projection;
        # this second receipt is the first authoritative post-restore physical
        # KV/Graph state and therefore must precede READY.
        scheduler._publish_elastic_startup_capacity()
        logger.warning(
            "Elastic startup HOT residency synchronized: entries=%d pinned=%d "
            "resident_bytes=%d reclaimable_bytes=%d workspace_unit_bytes=%d "
            "generation=%s",
            len(reference.entries),
            sum(entry.pinned for entry in reference.entries),
            sum(entry.resident_bytes for entry in reference.entries),
            sum(entry.reclaimable_bytes for entry in reference.entries),
            rank_safe_workspace_unit,
            scheduler._elastic_admission_controller.generation.value,
        )

    def _add_elastic_restore_request(
        self,
        *,
        request_id: str,
        prompt_len: int,
        k: int,
        max_tokens: int,
    ) -> None:
        if prompt_len <= 0:
            raise ValueError("elastic restore prompt length must be positive")
        configured_k = int(getattr(self.scheduler, "num_spec_tokens", 0))
        if k not in {0, configured_k}:
            raise ValueError(f"elastic restore K must be 0 or {configured_k}, got {k}")
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,
            ignore_eos=True,
            extra_args=({"ag2_force_non_speculative": 1} if k == 0 else None),
        )
        # Cache salt, request id and token body are all unique. Calibration
        # must exercise cold physical owners, not accidentally turn itself
        # into a prefix-cache benchmark.
        token = 100 + (abs(hash(request_id)) % 1000)
        core_request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[token] * prompt_len,
            mm_features=None,
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=request_id,
            data_parallel_rank=None,
        )
        request, request_wave = self.preprocess_add_request(core_request)
        self.add_request(request, request_wave)

    def _run_elastic_restore_step(self) -> SchedulerOutput:
        outputs, model_executed = self.step()
        self.post_step(model_executed)
        scheduler_output = getattr(self, "_last_scheduler_output", None)
        if scheduler_output is None:
            raise RuntimeError("elastic restore step produced no scheduler output")
        return scheduler_output

    def _prepare_elastic_restore_admission(
        self,
        request_ids: list[str],
        *,
        retain_hot_graphs: bool = False,
    ) -> int:
        scheduler = cast(Any, self.scheduler)
        return int(
            scheduler.prepare_elastic_restore_admission(
                request_ids,
                retain_hot_graphs=retain_hot_graphs,
            )
        )

    def _prepare_elastic_restore_capture(
        self,
        step_key: tuple[int, ...],
    ) -> None:
        scheduler = cast(Any, self.scheduler)
        if not scheduler.prepare_elastic_restore_capture(step_key):
            return
        maintenance = self._run_elastic_restore_step()
        expected_physical_keys = scheduler._elastic_step_residency_intent(step_key)[2]
        actual_physical_keys = (
            ()
            if maintenance.elastic_step_plan is None
            else maintenance.elastic_step_plan.physical_keys
        )
        if (
            maintenance.total_num_scheduled_tokens
            or maintenance.elastic_step_plan is None
            or maintenance.elastic_step_plan.kind != ElasticPlanKind.MAINTENANCE
            or actual_physical_keys != expected_physical_keys
        ):
            raise RuntimeError(
                "elastic restore COLD capture crossed a request commit "
                "boundary or changed physical owner identity: "
                f"step_key={step_key!r} expected={expected_physical_keys!r} "
                f"actual={actual_physical_keys!r}"
            )

    def _reclaim_elastic_restore_hotset_before_wave(self) -> None:
        scheduler = cast(Any, self.scheduler)
        if not scheduler.prepare_elastic_restore_idle_reclaim():
            return
        reclaim_output = self._run_elastic_restore_step()
        if (
            reclaim_output.total_num_scheduled_tokens
            or reclaim_output.elastic_step_plan is None
            or reclaim_output.elastic_step_plan.kind != ElasticPlanKind.RECLAIM
        ):
            raise RuntimeError("idle elastic reclaim crossed a request commit boundary")

    def _begin_elastic_restore_physical_epoch(
        self,
        step_keys: tuple[tuple[int, ...], ...],
    ) -> None:
        declared_step_keys = tuple(dict.fromkeys(step_keys))
        self._reclaim_elastic_restore_hotset_before_wave()
        for step_key in declared_step_keys:
            self._prepare_elastic_restore_capture(step_key)
        scheduler = cast(Any, self.scheduler)
        scheduler.assert_elastic_restore_captures_hot(declared_step_keys)

    def _rollback_elastic_restore_physical_epoch(
        self,
        *,
        request_ids: list[str],
        step_keys: tuple[tuple[int, ...], ...],
    ) -> None:
        """Return a rejected calibration cohort and its graphs to X0.

        Calibration retries are transactions over both request/KV state and
        the physical graph owner set.  Retiring only the requests would leave
        pinned graphs from a rejected larger cohort resident and make the next
        prefix pay for stale state from the failed epoch.
        """
        scheduler = cast(Any, self.scheduler)
        retention_id = getattr(scheduler, "_elastic_restore_retention_id", None)
        if retention_id is not None:
            scheduler.release_elastic_restore_retention(retention_id)
        self.abort_requests(request_ids)
        scheduler.reconcile_elastic_restore_rollback()
        self._drain_elastic_restore()
        self._reclaim_elastic_restore_hotset_before_wave()

    def _drain_elastic_restore(self, *, max_steps: int = 32) -> None:
        steps = 0
        while self._has_scheduler_step_work():
            scheduler = cast(Any, self.scheduler)
            reclaim_ready = bool(
                scheduler._needs_elastic_idle_reclaim()
                and not scheduler.has_unfinished_requests()
                and not scheduler.has_finished_requests()
            )
            if reclaim_ready:
                self._reclaim_elastic_restore_hotset_before_wave()
            else:
                self._run_elastic_restore_step()
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    "elastic startup calibration did not reach X0 within "
                    f"{max_steps} steps"
                )
        scheduler = cast(Any, self.scheduler)
        scheduler.finish_elastic_restore_epoch()

    def _assert_elastic_restore_key(
        self,
        scheduler_output: SchedulerOutput,
        expected: tuple[int, ...],
    ) -> None:
        scheduler = cast(Any, self.scheduler)
        actual = scheduler._canonical_elastic_graph_step_key(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.num_spec_tokens_to_schedule,
            scheduler_output.is_pure_decode_step,
        )
        if actual != expected:
            raise RuntimeError(
                "elastic startup calibration shape drifted: "
                f"expected={expected!r} actual={actual!r} "
                f"tokens={scheduler_output.num_scheduled_tokens!r}"
            )

    def _elastic_restore_decode_key(
        self, *, k: int, x: int, query_len: int
    ) -> tuple[int, int, int, int, int]:
        """Derive decode identity through the scheduler's active policy."""
        scheduler = cast(Any, self.scheduler)
        key = scheduler._canonical_elastic_graph_step_key(
            {f"_elastic_restore_decode_shape_{index}": query_len for index in range(x)},
            k,
            True,
        )
        if key is None:
            raise RuntimeError("elastic restore decode has no graph identity")
        return key

    def _elastic_restore_wave_step_keys(
        self, *, k: int, x: int, query_len: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Derive the exact prefill/decode pair before physical mutation."""
        if x <= 0:
            raise ValueError("elastic restore wave requires a positive X")
        scheduler = cast(Any, self.scheduler)
        prompt_len = self._elastic_restore_prefill_prompt_len()
        prospective_prefill_tokens = {
            f"_elastic_restore_shape_{index}": prompt_len for index in range(x)
        }
        prefill_key = scheduler._canonical_elastic_graph_step_key(
            prospective_prefill_tokens,
            self._elastic_restore_prefill_k(k),
            False,
        )
        if prefill_key is None:
            raise RuntimeError("elastic restore prefill has no graph identity")
        decode_key = self._elastic_restore_decode_key(
            k=k,
            x=x,
            query_len=query_len,
        )
        return prefill_key, decode_key

    @staticmethod
    def _elastic_restore_prefill_prompt_len() -> int:
        """Single source of truth for the restore cohort's live prefill M."""
        from vllm.v1.core.elastic_catalog import ELASTIC_RESTORE_PREFILL_PROMPT_LEN

        return ELASTIC_RESTORE_PREFILL_PROMPT_LEN

    def _elastic_restore_execution_step_keys(
        self, *, k: int, x: int, query_len: int
    ) -> tuple[tuple[int, ...], ...]:
        """Return every physical step needed to reach the target decode.

        When the effective MTP policy disables speculation on non-decode work,
        prefill produces no drafts.  A q(1 + K) target then has a real q1
        bridge that produces the drafts consumed by the target step.  The
        bridge belongs to the same predeclared physical epoch because
        calibration cannot capture a new graph while requests own KV state.
        """
        prefill_key, target_key = self._elastic_restore_wave_step_keys(
            k=k,
            x=x,
            query_len=query_len,
        )
        if not self._elastic_restore_needs_decode_bridge(k=k, query_len=query_len):
            return prefill_key, target_key
        bridge_key = self._elastic_restore_decode_key(k=k, x=x, query_len=1)
        return tuple(dict.fromkeys((prefill_key, bridge_key, target_key)))

    def _elastic_restore_needs_decode_bridge(self, *, k: int, query_len: int) -> bool:
        return k > 0 and query_len > 1 and self._elastic_restore_prefill_k(k) == 0

    def _elastic_restore_prefill_k(self, k: int) -> int:
        """Mirror the scheduler's non-decode speculation phase policy."""
        scheduler = cast(Any, self.scheduler)
        speculative_config = getattr(
            getattr(scheduler, "vllm_config", None), "speculative_config", None
        )
        if (
            speculative_config is not None
            and speculative_config.disable_speculation_on_non_decode
        ):
            return 0
        return k

    def _run_elastic_full_restore_wave(
        self,
        *,
        k: int,
        x: int,
        query_len: int,
        serial: int,
    ) -> tuple[tuple[int, ...] | None, int]:
        # The prefill used to construct a decode cohort is an administrative
        # witness, not part of the pinned FULL family. The restore helper
        # explicitly reclaims incompatible evictable carriers before capture;
        # pinned FULL publications survive that cleanup.
        return self._run_elastic_full_restore_wave_in_epoch(
            k=k,
            x=x,
            query_len=query_len,
            serial=serial,
        )

    def _run_elastic_full_restore_wave_in_epoch(
        self,
        *,
        k: int,
        x: int,
        query_len: int,
        serial: int,
        preserve_hotset: bool = False,
    ) -> tuple[tuple[int, ...] | None, int]:
        req_ids = [
            f"_elastic_restore_full_{serial}_{k}_{query_len}_{x}_{index}"
            for index in range(x)
        ]
        scheduler = cast(Any, self.scheduler)
        declared_step_keys = self._elastic_restore_execution_step_keys(
            k=k,
            x=x,
            query_len=query_len,
        )
        self._begin_elastic_restore_physical_epoch(declared_step_keys)
        retention_id = scheduler.retain_elastic_restore_captures(declared_step_keys)
        prompt_len = self._elastic_restore_prefill_prompt_len()
        for req_id in req_ids:
            self._add_elastic_restore_request(
                request_id=req_id,
                prompt_len=prompt_len,
                k=k,
                max_tokens=16,
            )

        # Both known physical owner sets are now settled.  Apply request/KV
        # admission against their aggregate residency before executing either
        # shape; a smaller prefix retries in a new untouched epoch.
        retain_hot_graphs = bool(scheduler._elastic_admission_controller.resident_bytes)
        prepared_x = self._prepare_elastic_restore_admission(
            req_ids,
            retain_hot_graphs=retain_hot_graphs,
        )
        if prepared_x == 0:
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            raise RuntimeError("elastic wave preflight admitted no requests")
        if prepared_x != x:
            if not 0 < prepared_x < x:
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic restore admission returned an invalid prefix: "
                    f"requested_x={x} prepared_x={prepared_x}"
                )
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            logger.warning(
                "Elastic pre-READY restore wave downshifted before model "
                "execution: requested_x=%d prepared_x=%d K=%d query_len=%d",
                x,
                prepared_x,
                k,
                query_len,
            )
            return None, prepared_x

        # Real prefill establishes request and sampler/MTP state only after
        # every declared COLD owner set and the complete KV wave are committed.
        prefill = self._run_elastic_restore_step()
        admitted_x = len(prefill.num_scheduled_tokens)
        if admitted_x > prepared_x:
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            raise RuntimeError(
                "elastic MaxX scheduler exceeded its exact wave preflight: "
                f"requested_x={x} prepared_x={prepared_x} "
                f"admitted_x={admitted_x}"
            )
        if admitted_x != x:
            if not 0 < admitted_x < x:
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic MaxX probe made no monotone admission progress: "
                    f"requested_x={x} admitted_x={admitted_x}"
                )
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            logger.warning(
                "Elastic pre-READY MaxX probe downshifted by admission: "
                "requested_x=%d admitted_x=%d K=%d query_len=%d",
                x,
                admitted_x,
                k,
                query_len,
            )
            return None, admitted_x
        needs_decode_bridge = self._elastic_restore_needs_decode_bridge(
            k=k,
            query_len=query_len,
        )
        if k > 0 and (query_len == 1 or needs_decode_bridge):
            for req_id in req_ids:
                scheduler.requests[req_id].spec_token_ids.clear()
        if needs_decode_bridge:
            bridge_expected = self._elastic_restore_decode_key(
                k=k,
                x=x,
                query_len=1,
            )
            scheduler.prepare_elastic_restore_execution(bridge_expected)
            self._elastic_restore_expected_decode_ids = frozenset(req_ids)
            try:
                bridge = self._run_elastic_restore_step()
            except _ElasticRestorePartialWave as contraction:
                self._elastic_restore_expected_decode_ids = frozenset()
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                logger.warning(
                    "Elastic MTP bridge downshifted before target decode: "
                    "requested_x=%d admitted_x=%d K=%d query_len=%d",
                    x,
                    contraction.admitted_x,
                    k,
                    query_len,
                )
                return None, contraction.admitted_x
            finally:
                self._elastic_restore_expected_decode_ids = frozenset()
            bridge_admitted_x = len(bridge.num_scheduled_tokens)
            if bridge_admitted_x != x:
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic MTP bridge made no exact decode progress: "
                    f"requested_x={x} admitted_x={bridge_admitted_x}"
                )
            self._assert_elastic_restore_key(bridge, bridge_expected)
            draft_counts = {
                req_id: len(scheduler.requests[req_id].spec_token_ids)
                for req_id in req_ids
            }
            if any(count != k for count in draft_counts.values()):
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic MTP bridge did not produce the target draft width: "
                    f"expected_k={k} draft_counts={draft_counts!r}"
                )
        elif k > 0 and query_len > 1:
            draft_counts = {
                req_id: len(scheduler.requests[req_id].spec_token_ids)
                for req_id in req_ids
            }
            if any(count != k for count in draft_counts.values()):
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic MTP prefill did not produce the target draft width: "
                    f"expected_k={k} draft_counts={draft_counts!r}"
                )
        expected = self._elastic_restore_decode_key(k=k, x=x, query_len=query_len)
        scheduler.prepare_elastic_restore_execution(expected)
        self._elastic_restore_expected_decode_ids = frozenset(req_ids)
        try:
            cold = self._run_elastic_restore_step()
        except _ElasticRestorePartialWave as contraction:
            self._elastic_restore_expected_decode_ids = frozenset()
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            logger.warning(
                "Elastic pinned-family pre-execution gate downshifted FULL "
                "decode: requested_x=%d admitted_x=%d K=%d query_len=%d",
                x,
                contraction.admitted_x,
                k,
                query_len,
            )
            return None, contraction.admitted_x
        finally:
            self._elastic_restore_expected_decode_ids = frozenset()
        cold_admitted_x = len(cold.num_scheduled_tokens)
        if cold_admitted_x != x:
            if not 0 < cold_admitted_x < x:
                self._rollback_elastic_restore_physical_epoch(
                    request_ids=req_ids,
                    step_keys=declared_step_keys,
                )
                raise RuntimeError(
                    "elastic FULL probe made no monotone decode admission "
                    f"progress: requested_x={x} admitted_x={cold_admitted_x}"
                )
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            logger.warning(
                "Elastic pinned-family fixed point downshifted FULL decode: "
                "requested_x=%d admitted_x=%d K=%d query_len=%d",
                x,
                cold_admitted_x,
                k,
                query_len,
            )
            return None, cold_admitted_x
        self._assert_elastic_restore_key(cold, expected)
        if k > 0 and query_len == 1:
            for req_id in req_ids:
                request = scheduler.requests.get(req_id)
                if request is not None:
                    request.spec_token_ids.clear()
        scheduler.prepare_elastic_restore_execution(expected)
        self._elastic_restore_expected_decode_ids = frozenset(req_ids)
        try:
            hot = self._run_elastic_restore_step()
        except _ElasticRestorePartialWave as contraction:
            self._elastic_restore_expected_decode_ids = frozenset()
            self._rollback_elastic_restore_physical_epoch(
                request_ids=req_ids,
                step_keys=declared_step_keys,
            )
            return None, contraction.admitted_x
        finally:
            self._elastic_restore_expected_decode_ids = frozenset()
        self._assert_elastic_restore_key(hot, expected)
        if preserve_hotset:
            carrier_keys = scheduler.resolve_elastic_serving_carrier_physical_keys(
                (expected,)
            )
            scheduler.promote_elastic_restore_retention_to_serving(
                retention_id, carrier_keys
            )
        else:
            scheduler.release_elastic_restore_retention(retention_id)
        self.abort_requests(req_ids)
        if preserve_hotset:
            scheduler._elastic_restore_mode = False
        self._drain_elastic_restore()
        return expected, x

    def _shutdown_failed_elastic_startup(self) -> None:
        """Release workers when construction fails before assignment."""
        try:
            self.shutdown()
        except Exception:
            logger.exception(
                "ELASTIC_STARTUP_SHUTDOWN_FAILED: explicit pre-READY worker "
                "teardown raised; process exit remains mandatory"
            )

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        start = time.time()

        # register all kvcache specs in enginecore process.
        register_all_kvcache_specs(vllm_config)

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Some layers (e.g. Prefix LM attention) run non-causally and tag their
        # KV cache spec with ``non_causal=True``. The specs are collected here in
        # the engine-core process (the same process that builds the scheduler),
        # so this is the multiproc-safe place to translate that layer-level
        # signal into a scheduling policy: chunked prefill and prefix caching
        # both assume causal attention and would corrupt non-causal prefill.
        if any(
            getattr(spec, "non_causal", False)
            for worker_specs in kv_cache_specs
            for spec in worker_specs.values()
        ):
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.info(
                    "Disabling chunked prefill: model has non-causal attention layers."
                )
                vllm_config.scheduler_config.enable_chunked_prefill = False
            if vllm_config.cache_config.enable_prefix_caching:
                logger.info(
                    "Disabling prefix caching: model has non-causal attention layers."
                )
                vllm_config.cache_config.enable_prefix_caching = False

        # Resolve the KV cache layout before memory profiling: workers that
        # capture full cudagraphs initialize a minimal KV cache during it.
        # Attention-free models resolve the default so layout reads never precede
        # resolution.
        layout = resolve_kv_cache_layout(
            vllm_config,
            self.model_executor.get_supported_kv_cache_layouts(),
            [s for specs in kv_cache_specs for s in specs.values()],
        )
        self.model_executor.set_kv_cache_layout(layout.name)

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        for kv_cache_config in kv_cache_configs:
            kv_cache_config.kv_cache_layout = vllm_config.cache_config.kv_cache_layout

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
            kv_cache_configs,
            configured_max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            enable_auto_resident_cap=(
                os.environ.get("AG2_VLLM_ELASTIC_AUTO_RESIDENT_CAP", "0") == "1"
            ),
        )
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            # Exclude groups that opt out of prefix caching (e.g. GLM-5.3-Flash
            # kpool tail, a 1-block/req scratch buffer with block_size=kpool):
            # their small block_size would otherwise drag the global block_size
            # below the real allocator block size and desync it from mamba.
            participating = [
                g.kv_cache_spec.block_size
                for g in kv_cache_groups
                if g.kv_cache_spec.prefix_cacheable
            ]
            vllm_config.cache_config.block_size = min(
                participating
                if participating
                else [g.kv_cache_spec.block_size for g in kv_cache_groups]
            )
            update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)

        vllm_config.validate_block_size()

        self.model_executor.initialize_from_config(kv_cache_configs)
        if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.model_executor.compile_or_warm_up_model()

        elapsed = time.time() - start
        compile_time = vllm_config.compilation_config.compilation_time
        encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
        if encoder_compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,
                compile_time + encoder_compile_time,
                compile_time,
                encoder_compile_time,
            )
        elif compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,
                compile_time,
            )
        else:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,
            )
        return scheduler_kv_cache_config

    def _initialize_effective_attention_block_size(self) -> None:
        cache_config = self.vllm_config.cache_config
        cache_config.effective_attention_block_size = None
        cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        if cache_manager is None:
            return
        block_sizes = {
            manager.block_size
            for manager in cache_manager.coordinator.single_type_managers
            if is_full_attention_spec(manager.kv_cache_spec)
        }
        cache_config.effective_attention_block_size = (
            block_sizes.pop() if len(block_sizes) == 1 else None
        )

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        supported_tasks = self.model_executor.supported_tasks
        self._log_pooler_config(supported_tasks)
        return supported_tasks

    def _log_pooler_config(self, supported_tasks: tuple[SupportedTask, ...]) -> None:
        if self._pooler_config_logged:
            return

        model_config = self.vllm_config.model_config
        pooler_config = model_config.pooler_config
        if (
            self.vllm_config.parallel_config.data_parallel_rank_local
            or model_config.runner_type != "pooling"
            or pooler_config is None
        ):
            return

        supported_pooling_tasks = tuple(
            sorted(set(supported_tasks) & set(POOLING_TASKS))
        )
        if not supported_pooling_tasks:
            return

        self._pooler_config_logged = True
        task_set = set(supported_pooling_tasks)
        use_activation = pooler_config.use_activation
        if use_activation is None:
            use_activation = True
        sources = getattr(model_config, "_pooler_config_sources", {})
        pooling_type_field = (
            "seq_pooling_type"
            if task_set & {"embed", "classify"}
            else "tok_pooling_type"
        )

        def log_field(name: str, field: str) -> str:
            value = (
                use_activation
                if field == "use_activation"
                else getattr(pooler_config, field)
            )
            source = sources.get(field, "unknown")
            return f"{name}={value}(source={source})"

        log_items = [("pooling_type", pooling_type_field)]
        log_items.extend(
            (field, field)
            for field in POOLER_CONFIG_LOG_FIELDS
            if field != pooling_type_field
        )
        config_fields = ", ".join(log_field(name, field) for name, field in log_items)

        logger.info_once(
            "Resolved pooling config: %s, supported_tasks=%s",
            config_fields,
            supported_pooling_tasks,
        )

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        """Return serializable KV cache metadata for external event consumers
        (e.g., Dynamo). This function has no call sites within vLLM but needs to
        be kept.
        """
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        kv_cache_manager = cast(Any, self.scheduler).kv_cache_manager
        managers = kv_cache_manager.coordinator.single_type_managers
        metadata: list[dict[str, int | str | None]] = []
        for group_idx, (group, manager) in enumerate(
            zip(kv_cache_config.kv_cache_groups, managers, strict=True)
        ):
            spec = group.kv_cache_spec
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": get_kv_cache_spec_kind(spec).value,
                    "block_size": manager.block_size,
                    "sliding_window": getattr(spec, "sliding_window", None),
                }
            )
        return metadata

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        if (
            request.ec_transfer_params is not None
            and self.scheduler.get_ec_connector() is None
        ):
            logger.warning(
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        self.scheduler.add_request(request)
        if request.abort_immediately:
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            self.abort_requests([request.request_id])

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def capture_iteration_details(
        self, scheduler_output: SchedulerOutput | None
    ) -> Generator[SchedulerIterationDetails | None, None, None]:
        enable_details = (
            self.vllm_config.observability_config.enable_logging_iteration_details
        )
        if not self.log_stats or not enable_details:
            yield None
            return
        # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
        if (
            scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens == 0
        ):
            yield None
            return

        iteration_index = getattr(self, "_iteration_index", 0)
        # scheduler_output=None marks a DP dummy iteration.
        if scheduler_output is None:
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=0,
                num_ctx_tokens=0,
                num_generation_requests=0,
                num_generation_tokens=0,
                elapsed_ms=0.0,
                is_dummy=True,
            )
        else:
            details = compute_iteration_details(scheduler_output)
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=details.num_ctx_requests,
                num_ctx_tokens=details.num_ctx_tokens,
                num_generation_requests=details.num_generation_requests,
                num_generation_tokens=details.num_generation_tokens,
                elapsed_ms=0.0,
                num_encoder_inputs=details.num_encoder_inputs,
                num_encoder_output_tokens=details.num_encoder_output_tokens,
            )

        start_time = time.monotonic()
        yield iteration_details
        iteration_details.elapsed_ms = (time.monotonic() - start_time) * 1000
        self._iteration_index = iteration_index + 1

    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        stats = self.scheduler.make_stats() or SchedulerStats()
        stats.iteration_details = iteration_details
        return stats

    def _attach_iteration_details(
        self,
        outputs: dict[int, EngineCoreOutputs],
        iteration_details: SchedulerIterationDetails | None,
    ) -> None:
        if iteration_details is None:
            return

        if (eco := next(iter(outputs.values()), None)) is None:
            outputs[0] = eco = EngineCoreOutputs()
        if eco.scheduler_stats is None:
            eco.scheduler_stats = self._make_iteration_details_stats(iteration_details)
        else:
            eco.scheduler_stats.iteration_details = iteration_details

    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False

    def _has_scheduler_step_work(self) -> bool:
        return bool(
            self.scheduler.has_requests()
            or cast(Any, self.scheduler).has_pending_elastic_maintenance()
        )

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self._has_scheduler_step_work():
            return {}, False
        trace_enabled = bool(
            self._ag2_step_trace_path and not self._ag2_step_trace_complete
        )
        step_start_ns = time.perf_counter_ns() if trace_enabled else 0
        scheduler_output = self.scheduler.schedule(
            self._should_throttle_prefills(), physical_quiescent=True
        )
        self._last_scheduler_output = scheduler_output
        expected_calibration_ids: frozenset[str] = getattr(
            self, "_elastic_restore_expected_decode_ids", frozenset()
        )
        if expected_calibration_ids and scheduler_output.is_pure_decode_step:
            scheduled_ids = frozenset(scheduler_output.num_scheduled_tokens)
            if scheduled_ids != expected_calibration_ids:
                cast(Any, self.scheduler).cancel_unexecuted_elastic_restore_step(
                    scheduler_output
                )
                raise _ElasticRestorePartialWave(
                    len(scheduled_ids), len(expected_calibration_ids)
                )
        schedule_end_ns = time.perf_counter_ns() if trace_enabled else 0
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        submit_end_ns = time.perf_counter_ns() if trace_enabled else 0
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            try:
                model_output = future.result()
                if model_output is None:
                    model_output = self.model_executor.sample_tokens(grammar_output)
            except WorkerRemoteError as error:
                if (
                    error.failure.code
                    != WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH
                ):
                    raise
                # Match the ordinary result path: aborts accepted while the
                # worker future was running take effect before its output.
                self._process_aborts_queue()
                failed = cast(
                    Any, self.scheduler
                ).recover_elastic_execution_plan_mismatch(scheduler_output)
                recovered_outputs = {
                    client_index: EngineCoreOutputs(finished_requests=set(request_ids))
                    for client_index, request_ids in (
                        cast(Any, self.scheduler).take_finished_request_ids().items()
                    )
                }
                for request in failed:
                    client_output = recovered_outputs.setdefault(
                        request.client_index,
                        EngineCoreOutputs(finished_requests=set()),
                    )
                    assert client_output.finished_requests is not None
                    client_output.finished_requests.add(request.request_id)
                    client_output.outputs.append(
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                            events=request.take_events(),
                            trace_headers=request.trace_headers,
                        )
                    )
                self._attach_iteration_details(recovered_outputs, iteration_details)
                return recovered_outputs, False
        future_end_ns = time.perf_counter_ns() if trace_enabled else 0

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        abort_end_ns = time.perf_counter_ns() if trace_enabled else 0
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        update_end_ns = time.perf_counter_ns() if trace_enabled else 0
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        if trace_enabled:
            details = compute_iteration_details(scheduler_output)
            if (
                self._ag2_step_trace_include_context and details.num_ctx_tokens > 0
            ) or details.num_generation_requests >= self._ag2_step_trace_min_running:
                self._ag2_step_trace_saw_context |= details.num_ctx_tokens > 0
                previous_start_ns = self._ag2_step_trace_previous_start_ns
                self._ag2_step_trace_records.append(
                    {
                        "context_requests": details.num_ctx_requests,
                        "context_tokens": details.num_ctx_tokens,
                        "generation_requests": details.num_generation_requests,
                        "generation_tokens": details.num_generation_tokens,
                        "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                        "step_start_ns": step_start_ns,
                        "start_spacing_ms": (
                            (step_start_ns - previous_start_ns) / 1e6
                            if previous_start_ns is not None
                            else -1.0
                        ),
                        "schedule_ms": (schedule_end_ns - step_start_ns) / 1e6,
                        "execute_submit_ms": (submit_end_ns - schedule_end_ns) / 1e6,
                        "future_wait_ms": (future_end_ns - submit_end_ns) / 1e6,
                        "abort_ms": (abort_end_ns - future_end_ns) / 1e6,
                        "update_ms": (update_end_ns - abort_end_ns) / 1e6,
                        "step_total_ms": (update_end_ns - step_start_ns) / 1e6,
                    }
                )
                self._ag2_step_trace_previous_start_ns = step_start_ns
                if len(self._ag2_step_trace_records) >= self._ag2_step_trace_limit and (
                    not self._ag2_step_trace_require_context
                    or self._ag2_step_trace_saw_context
                ):
                    payload = {
                        "schema": "ag2-engine-step-trace-v2",
                        "min_running": self._ag2_step_trace_min_running,
                        "include_context": self._ag2_step_trace_include_context,
                        "require_context": self._ag2_step_trace_require_context,
                        "records": self._ag2_step_trace_records,
                    }
                    path = self._ag2_step_trace_path
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    temporary = f"{path}.tmp.{os.getpid()}"
                    with open(temporary, "wb") as stream:
                        stream.write(msgspec.json.encode(payload))
                    os.replace(temporary, path)
                    self._ag2_step_trace_complete = True
                    logger.info("AG2 engine-step trace complete: %s", path)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        trace_enabled = bool(
            self._ag2_step_trace_path and not self._ag2_step_trace_complete
        )
        step_start_ns = time.perf_counter_ns() if trace_enabled else 0
        schedule_end_ns = step_start_ns
        submit_end_ns = step_start_ns

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(
                self._should_throttle_prefills(),
                physical_quiescent=not batch_queue,
            )
            schedule_end_ns = time.perf_counter_ns() if trace_enabled else 0
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output
            submit_end_ns = time.perf_counter_ns() if trace_enabled else 0

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        wait_start_ns = time.perf_counter_ns() if trace_enabled else 0
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
        future_end_ns = time.perf_counter_ns() if trace_enabled else 0

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        abort_end_ns = time.perf_counter_ns() if trace_enabled else 0
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        update_end_ns = time.perf_counter_ns() if trace_enabled else 0
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # When draft tokens are used with structured output, validate them
            # before computing the grammar bitmask for the deferred request.
            if self.check_for_draft_tokens:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    # Update the draft token ids in the scheduler output to
                    # filter out the invalid spec tokens, which will be padded
                    # with -1 and skipped by the grammar bitmask computation.
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        deferred_end_ns = time.perf_counter_ns() if trace_enabled else 0
        if trace_enabled:
            details = compute_iteration_details(scheduler_output)
            if (
                self._ag2_step_trace_include_context and details.num_ctx_tokens > 0
            ) or details.num_generation_requests >= self._ag2_step_trace_min_running:
                self._ag2_step_trace_saw_context |= details.num_ctx_tokens > 0
                previous_start_ns = self._ag2_step_trace_previous_start_ns
                self._ag2_step_trace_records.append(
                    {
                        "context_requests": details.num_ctx_requests,
                        "context_tokens": details.num_ctx_tokens,
                        "generation_requests": details.num_generation_requests,
                        "generation_tokens": details.num_generation_tokens,
                        "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                        "step_start_ns": step_start_ns,
                        "start_spacing_ms": (
                            (step_start_ns - previous_start_ns) / 1e6
                            if previous_start_ns is not None
                            else -1.0
                        ),
                        "schedule_ms": (schedule_end_ns - step_start_ns) / 1e6,
                        "execute_and_sample_submit_ms": (
                            submit_end_ns - schedule_end_ns
                        )
                        / 1e6,
                        "queue_pre_wait_ms": (wait_start_ns - submit_end_ns) / 1e6,
                        "future_wait_ms": (future_end_ns - wait_start_ns) / 1e6,
                        "abort_ms": (abort_end_ns - future_end_ns) / 1e6,
                        "update_ms": (update_end_ns - abort_end_ns) / 1e6,
                        "deferred_sample_ms": (deferred_end_ns - update_end_ns) / 1e6,
                        "step_total_ms": (deferred_end_ns - step_start_ns) / 1e6,
                    }
                )
                self._ag2_step_trace_previous_start_ns = step_start_ns
                if len(self._ag2_step_trace_records) >= self._ag2_step_trace_limit and (
                    not self._ag2_step_trace_require_context
                    or self._ag2_step_trace_saw_context
                ):
                    payload = {
                        "schema": "ag2-engine-step-trace-v2",
                        "min_running": self._ag2_step_trace_min_running,
                        "include_context": self._ag2_step_trace_include_context,
                        "require_context": self._ag2_step_trace_require_context,
                        "records": self._ag2_step_trace_records,
                    }
                    path = self._ag2_step_trace_path
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    temporary = f"{path}.tmp.{os.getpid()}"
                    with open(temporary, "wb") as stream:
                        stream.write(msgspec.json.encode(payload))
                    os.replace(temporary, path)
                    self._ag2_step_trace_complete = True
                    logger.info("AG2 engine-step trace complete: %s", path)

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        gc.unfreeze()
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        cleanup_dist_env_and_memory()
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(
        self,
        reset_running_requests: bool = True,
        reset_connector: bool = True,
    ) -> None:
        # reset_connector=True so external connectors clear alongside
        # local caches, matching the pause_generation(clear_cache=True)
        # contract. No-op when no connector is configured.
        self.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def _finish_pause(self, clear_cache: bool) -> None:
        # A completed pause promises an idle device: nothing else waits on
        # the last dummy batch an idle DP rank launches.
        self.model_executor.collective_rpc("synchronize_device")
        if clear_cache:
            self._reset_caches()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in get_args(PauseMode):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        self._finish_pause(clear_cache)

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.

        """
        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.

        Returns:
            Whether all executor memory is resident again (fully awake).

        """
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Partial wakes intentionally keep the remaining allocations asleep.
        # Resume scheduling only once all executor memory is resident again.
        fully_awake = not self.model_executor.is_sleeping
        if fully_awake:
            self.resume_scheduler()
        return fully_awake

    def release_kv_cache_memory(self) -> None:
        """Discard KV cache physical memory. Requires a completed pause
        and all executor memory to be resident. Kept requests are recomputed
        after wake-up.
        """
        if not (
            self.is_scheduler_paused()
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            raise RuntimeError(
                "release_kv_cache_memory() requires a completed pause first"
            )
        if self.model_executor.is_sleeping:
            raise RuntimeError(
                "release_kv_cache_memory() requires all executor memory to be resident"
            )
        self._reset_caches()
        self.model_executor.discard(("kv_cache",))

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def set_weight_version(self, weight_version: str) -> None:
        self._weight_version = weight_version

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self._weight_version

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self, notification_type: EEPNotificationType
    ):
        raise NotImplementedError


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing
            self.last_counts = (0, 0)

            self.addresses = addresses
            self.process_input_queue_block = True
            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Initialize fault tolerance settings.
            self.enable_fault_tolerance = (
                vllm_config.parallel_config.enable_fault_tolerance
            )
            if self.enable_fault_tolerance:
                self.ft_sentinel = EngineCoreSentinel(
                    engine=self,
                    parallel_config=vllm_config.parallel_config,
                )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            # We only need to handshake with one party.
            with handshake as addresses:
                yield addresses
        else:
            # We need to handshake with rank 0 front-end and our colocated frontend.
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # Include config hash for DP configuration validation
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""
        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        clean_shutdown = False
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.reconfigure_for_independent_dp_rank()
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit as e:
            logger.info_once("[shutdown] EngineCore: exiting busy loop")
            clean_shutdown = (
                e.code in (None, 0)
                and engine_core is not None
                and engine_core.shutdown_state == EngineShutdownState.SHUTTING_DOWN
                and not engine_core.has_work()
                and engine_core.vllm_config.shutdown_timeout == 0
            )
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()
            if clean_shutdown:
                from vllm.platforms import current_platform

                if current_platform.is_rocm():
                    # Cleanup above already unfreezes and collects the heap.
                    # Freeze the surviving graph to skip another slow cyclic-GC
                    # scan during finalization; process exit reclaims it.
                    gc.freeze()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            self._maybe_publish_request_counts()

        raise SystemExit

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, kv_cache_usage=self.scheduler.get_kv_cache_usage()
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout
            mode = "abort" if shutdown_timeout == 0 else "drain"

            logger.info(
                "[shutdown] EngineCore: start mode=%s timeout=%ds",
                mode,
                shutdown_timeout,
            )

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: aborting in-flight requests count=%d",
                        num_requests,
                    )
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: draining in-flight requests "
                        "count=%d timeout=%ds",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""
        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.debug(
            "[shutdown] EngineCore: rejecting new request request_id=%s",
            request.request_id,
        )
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning(
            "[shutdown] EngineCore: rejecting utility call method=%s",
            method_name,
        )
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""
        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def _make_ready_response(self) -> EngineCoreReadyResponse:
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        return EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            block_size=self.vllm_config.cache_config.block_size,
            mamba_block_size=self.vllm_config.cache_config.mamba_block_size,
            effective_attention_block_size=(
                self.vllm_config.cache_config.effective_attention_block_size
            ),
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
            vllm_version=VLLM_VERSION,
            world_size=self.vllm_config.parallel_config.world_size,
            data_parallel_size=parallel_config.data_parallel_size,
            kv_cache_size_tokens=self.vllm_config.cache_config.kv_cache_size_tokens,
            kv_cache_max_concurrency=(
                self.vllm_config.cache_config.kv_cache_max_concurrency
            ),
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            data_parallel_rank=self.engine_index,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            instance_id=self.vllm_config.instance_id,
            supports_lora=self.vllm_config.lora_config is not None,
            max_loras=(
                self.vllm_config.lora_config.max_loras
                if self.vllm_config.lora_config is not None
                else 0
            ),
            kv_events_config=self.scheduler.get_kv_event_publisher_config(),
            weight_transfer_backend=(
                self.vllm_config.weight_transfer_config.backend
                if self.vllm_config.weight_transfer_config is not None
                else None
            ),
            enable_sleep_mode=self.vllm_config.model_config.enable_sleep_mode,
            supports_draft_weight_updates=(
                self.model_executor.supports_draft_weight_updates()
                if self.vllm_config.weight_transfer_config is not None
                else False
            ),
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""
        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = self._make_ready_response()
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except MultiModalCacheMissError as e:
                            # P0/P1 shadow drift -- return a retryable signal (P0
                            # drops the stale entry, client resends with data).
                            self._handle_mm_cache_miss(req, e)
                            continue
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    elif request_type == EngineCoreRequestType.UTILITY:
                        request = generic_decoder.decode(data_frames)
                        client_idx, call_id, method, args = request
                        if method == FT_UTILITY_METHOD:
                            self.ft_sentinel.handle_command(
                                client_idx, call_id, args[0]
                            )
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""
        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Payload buffers that can't be reused yet because zmq may still be
        # sending them.
        # Buffers of the zero-copy tensor/ndarray frames don't need tracking
        # here: zmq itself holds a reference to each until it's done with it.
        pending = deque[tuple[zmq.MessageTracker, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reclaimed = pending.pop()[1]
                    if len(reuse_buffers) < max_reuse_bufs:
                        reuse_buffers.append(reclaimed)

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = self._send_msg_tracking_payload(
                    sockets[client_index], buffers
                )
                if not tracker.done:
                    pending.appendleft((tracker, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_mm_cache_miss(
        self, request: EngineCoreRequest, err: MultiModalCacheMissError
    ) -> None:
        """Return a retryable response for a P0/P1 cache-drift miss.

        Surfaces every drifted hash via ``EngineCoreOutput.mm_cache_miss_hashes`` so
        the frontend drops them from its sender cache and the client resends with
        data (see ``MultiModalCacheMissError``). Logged at warning, not exception,
        because it is expected and self-healing.
        """
        logger.warning(
            "Multi-modal cache miss for request %s (mm_hashes=%s): P0/P1 cache "
            "drift; returning a retryable response so the items are resent with data.",
            request.request_id,
            err.mm_hashes,
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                            mm_cache_miss_hashes=err.mm_hashes,
                        )
                    ],
                ),
            )
        )

    @staticmethod
    def _send_msg_tracking_payload(
        socket: zmq.Socket, buffers: Sequence[bytestr]
    ) -> zmq.MessageTracker:
        """Send `buffers` as a zero-copy multipart message, returning a tracker
        for the *first* frame.

        Used instead of `Socket.send_multipart()` because we reuse the buffer
        passed to `MsgpackEncoder.encode_into()`: `send_multipart()` returns a
        tracker for the last frame only.
        """
        more_flag = zmq.SNDMORE if len(buffers) > 1 else 0
        tracker = socket.send(buffers[0], more_flag, copy=False, track=True)
        if more_flag:
            socket.send_multipart(buffers[1:], copy=False)
        return tracker

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in get_args(PauseMode):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            engine._finish_pause(clear_cache)
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)

        if self._pause_complete():
            self._finish_pause(clear_cache)
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        return not self.has_work()

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[Request]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for request in aborted_reqs:
                by_client[request.client_index].add(request.request_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        scheduler_config = vllm_config.scheduler_config
        self.prefill_schedule_interval = scheduler_config.prefill_schedule_interval
        self.dp_sync_interval = vllm_config.parallel_config.dp_sync_interval

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0

        # Two-phase pause protocol state. When pending_pause is True, the
        # engine keeps stepping (dummy batches) while waiting for all DP
        # ranks to also set pending_pause. Once all ranks agree via
        # all-reduce, ignore_start_dp_wave is set so that stale
        # START_DP_WAVE messages cannot re-wake the engines.
        self.pending_pause = False
        self.ignore_start_dp_wave = False

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def _pause_complete(self) -> bool:
        """Two-phase DP-aware pause.

        Phase 1: Set local pause state and ``pending_pause`` flag. If the
        engines are idle, kick-start them by setting ``engines_running`` to
        True so ranks enter the stepping loop and reach the all-reduce
        consensus checkpoint in ``_has_global_unfinished_reqs``.

        Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
        confirms that **all** ranks have ``pending_pause`` set, collectively
        stop stepping and set ``ignore_start_dp_wave`` so that stale
        ``START_DP_WAVE`` messages cannot re-wake any engine.
        """
        self.pending_pause = True
        self.engines_running = True

        return False

    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
            raise RuntimeError(
                "resume_scheduler called while pause is still in "
                "flight. Wait for the pause future to resolve before "
                "resuming."
            )
        if self.engines_running:
            logger.debug("Resume called while engines are not paused, ignoring.")
            return

        super().resume_scheduler()
        self.ignore_start_dp_wave = False

        # Barrier: wait for all DP ranks to have resumed (and cleared
        # ignore_start_dp_wave) before any rank starts stepping. Uses
        # the existing all-reduce which is safe because engines are
        # stopped.
        has_global_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, self.scheduler.has_unfinished_requests()
        )

        if has_global_unfinished:
            self.engines_running = True

    def barrier(self):
        """Blocking barrier on the DP process group (test-only utility)."""
        import torch.distributed as dist

        dist.barrier(group=self.dp_group)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            if self.ignore_start_dp_wave:
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed), stamped with the
        # lockstep-synchronized step counter and wave number.
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts,
                kv_cache_usage=self.scheduler.get_kv_cache_usage(),
                step_counter=self.step_counter,
                current_wave=self.current_wave,
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def _should_throttle_prefills(self) -> bool:
        # Throttle new prefills to cadence-aligned steps for DP balancing.
        # step_counter is identical across DP ranks. On a fresh wave the
        # counter is 0, so prefills are admitted immediately after idle.
        return (
            self.prefill_schedule_interval > 1
            and self.step_counter % self.prefill_schedule_interval != 0
        )

    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""
        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            was_running = self.engines_running
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()

            if self.eep_scaling_state is not None:
                state = self.eep_scaling_state
                if state.commit_requested or not state.is_ready_for_switch():
                    state.progress()
                if state.is_complete():
                    if state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None
                elif not state.commit_requested and state.is_ready_for_switch():
                    self.process_input_queue_block = True

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # Execute a dummy pass when no ready requests ran, unless the
                # engine is sleeping.
                elif not self.model_executor.is_sleeping:
                    with self.capture_iteration_details(None) as iteration_details:
                        self.execute_dummy_batch()
                    if iteration_details is not None and not self.has_coordinator:
                        stats = self._make_iteration_details_stats(iteration_details)
                        self.output_queue.put_nowait(
                            (0, EngineCoreOutputs(scheduler_stats=stats))
                        )

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0
            elif (
                not was_running
                and self.has_coordinator
                and self.dp_rank == 0
                and not self.pending_pause
            ):
                # Mirror of the wave_complete notification above: the
                # coordinator must observe this edge too rather than assume
                # that a START_DP_WAVE it sent was acted upon, since a paused
                # engine discards it.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Sync step 1 too: an idle pause needs one dummy batch, not a full interval.
        self.step_counter += 1
        if self.step_counter != 1 and self.step_counter % self.dp_sync_interval != 0:
            return True

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,
            has_unfinished=local_unfinished,
            pending_pause=self.pending_pause,
        )

        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

        return has_unfinished

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> str:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        if self.eep_scaling_state is not None:
            raise RuntimeError("Elastic EP reconfiguration is already active")

        state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.eep_scaling_state = state

        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )
        return state.ready_key

    def commit_prepared_elastic_ep(self) -> None:
        state = self.eep_scaling_state
        if state is None or state.commit_requested or not state.is_ready_for_switch():
            raise RuntimeError("No prepared Elastic EP reconfiguration is ready")
        state.commit_requested = True
        self.process_input_queue_block = False
        logger.info("[Elastic EP] Committing prepared reconfiguration")

    def _eep_send_engine_core_notification(
        self, notification_type: EEPNotificationType
    ):
        """Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        2) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.ignore_start_dp_wave = True
        state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        if self.eep_scaling_state is not None:
            raise RuntimeError("Elastic EP reconfiguration is already active")
        self.eep_scaling_state = state
        state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """Ray actor for running EngineCore in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        self._set_nixl_side_channel_host()

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    @staticmethod
    def _set_nixl_side_channel_host():
        import ray

        # The driver-side value is excluded from Ray actor env propagation.
        # Fill in an actor-local default while preserving explicit overrides.
        os.environ.setdefault(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", ray.util.get_node_ip_address()
        )

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        device_control_env_var = current_platform.device_control_env_var
        self._set_assigned_physical_gpu_ids(
            vllm_config, local_dp_rank, device_control_env_var
        )

    def _set_assigned_physical_gpu_ids(
        self,
        vllm_config: VllmConfig,
        local_dp_rank: int,
        device_control_env_var: str,
    ):
        world_size = vllm_config.parallel_config.world_size
        try:
            physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                device_control_env_var,
                local_dp_rank,
                world_size,
                user_assigned_gpu_ids=(
                    vllm_config.parallel_config.assigned_physical_gpu_ids
                ),
            )
            vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
        except IndexError as e:
            raise Exception(
                f"Error computing assigned_physical_gpu_ids: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """Run the engine core busy loop."""
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.reconfigure_for_independent_dp_rank()
        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
