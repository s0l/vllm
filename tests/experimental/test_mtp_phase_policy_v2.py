from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm.distributed.parallel_state as parallel_state
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.parallel_state import (
    _is_tp3_sd_phase_reduce,
    _should_use_tp3_mtp_device_ce,
)
from vllm.forward_context import BatchDescriptor, override_forward_context
from vllm.model_executor.models.qwen3_5 import (
    _ag2_layer0_trace_match_row,
)
from vllm.model_executor.models.qwen3_5_mtp import (
    _mtp_fc_padded_output_size,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import BatchDCPPrefillWrapper
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.kv_cache_utils import (
    _elastic_gdn_blocks_per_seq,
    _elastic_gdn_pool_blocks,
    _shrink_kv_cache_tensor_blocks,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheTensor, MambaSpec
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.elastic_gdn import (
    ElasticKVController,
    V2GDNCheckpointManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    _resolve_prefill_cudagraph_mode,
)
from vllm.v1.worker.gpu.warmup import (
    _make_warmup_block_allocator,
    _set_elastic_warmup_transition,
)
from vllm.v1.worker.mamba_utils import GDNPrefixCheckpointStore


class _Owner:
    def __init__(self, committed: int, quantum: int = 4, reserved: int = 64):
        self.committed = committed
        self.quantum = quantum
        self.reserved = reserved
        self.resizes: list[int] = []
        self.transfers: list[tuple[_Owner, int]] = []

    @property
    def info(self):
        return SimpleNamespace(
            committed=self.committed,
            quantum=self.quantum,
            reserved=self.reserved,
        )

    def resize(self, committed: int, event) -> None:
        self.committed = committed
        self.resizes.append(committed)

    def transfer_to(self, destination: _Owner, bytes_: int, event) -> None:
        if bytes_ % self.quantum:
            raise ValueError("unaligned transfer")
        self.committed -= bytes_
        destination.committed += bytes_
        self.transfers.append((destination, bytes_))


class _CapacityOwner(_Owner):
    def __init__(self, committed: int, max_committed: int):
        super().__init__(committed)
        self.max_committed = max_committed

    def resize(self, committed: int, event) -> None:
        if committed > self.max_committed:
            raise RuntimeError("out of memory")
        super().resize(committed, event)


class _Event:
    def record(self, stream) -> None:
        pass


class ElasticGDNTopologyTests(unittest.TestCase):
    def test_capacity_counts_all_mamba_groups_and_mtp_scratch(self):
        config = SimpleNamespace(
            num_speculative_tokens=2,
            additional_config={"elastic_gdn_max_seqs": 64},
        )
        self.assertEqual(_elastic_gdn_blocks_per_seq(config, 3), 9)
        self.assertEqual(_elastic_gdn_pool_blocks(config, 3), 577)

    def test_stale_per_group_override_fails_closed(self):
        config = SimpleNamespace(
            num_speculative_tokens=2,
            additional_config={"elastic_gdn_blocks_per_seq": 2},
        )
        with self.assertRaisesRegex(ValueError, "MTP topology requires"):
            _elastic_gdn_blocks_per_seq(config, 3)

    def test_replay_commit_removes_speculative_state_blocks(self):
        config = SimpleNamespace(
            num_speculative_tokens=3,
            additional_config={
                "gdn_mtp_replay_commit": True,
                "elastic_gdn_max_seqs": 64,
                # A stale pre-replay value must not retain hidden scratch cost.
                "elastic_gdn_blocks_per_seq": 4,
            },
        )
        self.assertEqual(_elastic_gdn_blocks_per_seq(config, 3), 3)
        self.assertEqual(_elastic_gdn_pool_blocks(config, 3), 193)


class _Store:
    def __init__(self):
        self.calls = []

    def zero_blocks(self, block_ids, *args):
        self.calls.append(("zero", block_ids))

    def restore_blocks(self, key, block_ids, *args):
        self.calls.append(("restore", key, block_ids))

    def save_blocks(self, key, block_ids, *args):
        self.calls.append(("save", key, block_ids))

    def snapshot_keys(self):
        return (b"visible",)


def _scheduler_output(
    *,
    new=(),
    cached=None,
    finished=(),
    scheduled=None,
    restores=None,
    saves=None,
):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    output.scheduled_cached_reqs = cached or CachedRequestData.make_empty()
    output.finished_req_ids = set(finished)
    output.num_scheduled_tokens = scheduled or {}
    output.gdn_checkpoint_restore = restores
    output.gdn_checkpoint_save = saves
    return output


class TestMTPPhasePolicyV2(unittest.TestCase):
    def test_mtp_device_ce_requires_explicit_owner_and_exact_geometry(self):
        base = {
            "explicit_mtp_lane": True,
            "tensor_dim": 2,
            "rows": 31,
            "hidden_size": 5120,
            "tp_world_size": 3,
        }
        self.assertTrue(_should_use_tp3_mtp_device_ce(**base))
        for field, value in (
            ("explicit_mtp_lane", False),
            ("tensor_dim", 1),
            ("rows", 0),
            ("hidden_size", 4096),
            ("tp_world_size", 2),
        ):
            candidate = dict(base)
            candidate[field] = value
            self.assertFalse(_should_use_tp3_mtp_device_ce(**candidate))

    def test_mtp_device_ce_dispatch_does_not_change_target_lane(self):
        tensor = torch.ones((3, 5120), dtype=torch.bfloat16)
        default = MagicMock(return_value=torch.full_like(tensor, 9))
        group = SimpleNamespace(
            world_size=3,
            _all_reduce_out_place=default,
        )
        candidate = torch.full_like(tensor, 7)
        env = {
            "AG2_VLLM_MTP_DEVICE_CE": "1",
            "AG2_VLLM_TP3_PIECEWISE_DEVICE_CE": "0",
            "VLLM_TP3_CE_REDUCE": "0",
            "VLLM_TP3_SD_PHASE_REDUCE": "0",
            "VLLM_TP3_SD_DETERMINISTIC_REDUCE": "0",
            "VLLM_TP3_SD_CANONICAL_REDUCE": "0",
        }
        with (
            patch.dict(parallel_state._groups, {"ag2-test": lambda: group}, clear=True),
            patch.dict("os.environ", env, clear=False),
            patch.object(
                parallel_state,
                "_tp3_device_ce_reduce",
                return_value=candidate,
            ) as reduce,
            override_forward_context(
                SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    num_tokens_unpadded=3,
                    tp3_ce_reduce=False,
                    tp3_mtp_device_ce=True,
                    tp3_sd_phase_reduce=False,
                )
            ),
        ):
            output = parallel_state.all_reduce(tensor, "ag2-test")
        self.assertIs(output, candidate)
        reduce.assert_called_once_with(tensor, group)
        default.assert_not_called()

        with (
            patch.dict(parallel_state._groups, {"ag2-test": lambda: group}, clear=True),
            patch.dict("os.environ", env, clear=False),
            patch.object(parallel_state, "_tp3_device_ce_reduce") as reduce,
            override_forward_context(
                SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    num_tokens_unpadded=3,
                    tp3_ce_reduce=False,
                    tp3_mtp_device_ce=False,
                    tp3_sd_phase_reduce=False,
                )
            ),
        ):
            output = parallel_state.all_reduce(tensor, "ag2-test")
        torch.testing.assert_close(output, torch.full_like(tensor, 9))
        reduce.assert_not_called()
        default.assert_called_once_with(tensor)

    def test_phase_reduce_requires_explicit_forward_lane(self):
        with override_forward_context(SimpleNamespace(tp3_sd_phase_reduce=True)):
            self.assertTrue(_is_tp3_sd_phase_reduce())
        with override_forward_context(SimpleNamespace(tp3_sd_phase_reduce=False)):
            self.assertFalse(_is_tp3_sd_phase_reduce())
        with override_forward_context(None):
            self.assertFalse(_is_tp3_sd_phase_reduce())

    def test_phase_reduce_is_part_of_piecewise_graph_identity(self):
        prefill = BatchDescriptor(num_tokens=3, tp3_sd_phase_reduce=False)
        verify = BatchDescriptor(num_tokens=3, tp3_sd_phase_reduce=True)
        self.assertNotEqual(prefill, verify)
        self.assertNotEqual(hash(prefill), hash(verify))

    def test_phase_reduce_dispatches_only_on_explicit_lane(self):
        tensor = torch.ones((1, 5120), dtype=torch.bfloat16)
        communicator = SimpleNamespace(
            all_gather=lambda value, dim: torch.cat(
                (value, value * 2, value * 3), dim=dim
            )
        )
        default = MagicMock(return_value=torch.full_like(tensor, 9))
        group = SimpleNamespace(
            world_size=3,
            device_communicator=communicator,
            _all_reduce_out_place=default,
        )
        env = {
            "VLLM_TP3_CE_REDUCE": "0",
            "VLLM_TP3_SD_PHASE_REDUCE": "1",
            "VLLM_TP3_SD_DETERMINISTIC_REDUCE": "0",
            "VLLM_TP3_SD_CANONICAL_REDUCE": "0",
        }
        with (
            patch.dict(parallel_state._groups, {"ag2-test": lambda: group}, clear=True),
            patch.dict("os.environ", env, clear=False),
            override_forward_context(
                SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    num_tokens_unpadded=1,
                    tp3_ce_reduce=False,
                    tp3_mtp_device_ce=False,
                    tp3_sd_phase_reduce=True,
                )
            ),
        ):
            reduced = parallel_state.all_reduce(tensor, "ag2-test")
        torch.testing.assert_close(reduced, torch.full_like(tensor, 6))
        default.assert_not_called()

        with (
            patch.dict(parallel_state._groups, {"ag2-test": lambda: group}, clear=True),
            patch.dict("os.environ", env, clear=False),
            override_forward_context(
                SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    num_tokens_unpadded=1,
                    tp3_ce_reduce=False,
                    tp3_mtp_device_ce=False,
                    tp3_sd_phase_reduce=False,
                )
            ),
        ):
            reduced = parallel_state.all_reduce(tensor, "ag2-test")
        torch.testing.assert_close(reduced, torch.full_like(tensor, 9))
        default.assert_called_once_with(tensor)

    def test_layer0_trace_filter_is_exact_unique_and_one_shot(self):
        values = {
            "output": "/tmp/trace",
            "saved_query_lens": set(),
            "query_len": 3,
            "expected_query_lens": {1, 3},
            "token_ids": [9, 15, 21],
            "expected_token_id": 15,
            "positions": [41, 42, 43],
            "expected_position": 42,
        }
        self.assertEqual(_ag2_layer0_trace_match_row(**values), 1)
        for name, replacement in (
            ("output", ""),
            ("saved_query_lens", {3}),
            ("query_len", 2),
            ("token_ids", [9, 19, 21]),
            ("positions", [41, 43, 44]),
        ):
            candidate = dict(values)
            candidate[name] = replacement
            self.assertIsNone(_ag2_layer0_trace_match_row(**candidate))

        duplicate = dict(values)
        duplicate["token_ids"] = [15, 15, 21]
        duplicate["positions"] = [42, 42, 43]
        self.assertIsNone(_ag2_layer0_trace_match_row(**duplicate))

    def test_elastic_tensor_shrink_preserves_quantized_vmm_reservation(self):
        tensor = KVCacheTensor(
            size=12,
            shared_by=["attention"],
            committed_size=12,
            mapping_quantum=4,
            num_blocks=5,
            logical_block_size=2,
        )
        _shrink_kv_cache_tensor_blocks(tensor, old_num_blocks=5, new_num_blocks=4)
        self.assertEqual(tensor.size, 8)
        self.assertEqual(tensor.committed_size, 8)
        self.assertEqual(tensor.num_blocks, 4)

    def test_dense_tensor_shrink_remains_proportional(self):
        tensor = KVCacheTensor(size=50, shared_by=["attention"])
        _shrink_kv_cache_tensor_blocks(tensor, old_num_blocks=5, new_num_blocks=4)
        self.assertEqual(tensor.size, 40)

    def test_mtp_prefill_uses_piecewise_when_full_attention_is_decode_only(self):
        self.assertEqual(
            _resolve_prefill_cudagraph_mode(
                CUDAGraphMode.FULL_AND_PIECEWISE,
                AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
                query_len=3,
            ),
            CUDAGraphMode.PIECEWISE,
        )
        self.assertEqual(
            _resolve_prefill_cudagraph_mode(
                CUDAGraphMode.FULL_DECODE_ONLY,
                AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
                query_len=3,
            ),
            CUDAGraphMode.NONE,
        )
        self.assertEqual(
            _resolve_prefill_cudagraph_mode(
                CUDAGraphMode.FULL_AND_PIECEWISE,
                AttentionCGSupport.UNIFORM_BATCH,
                query_len=3,
            ),
            CUDAGraphMode.FULL_AND_PIECEWISE,
        )
        self.assertEqual(
            _resolve_prefill_cudagraph_mode(
                CUDAGraphMode.FULL_AND_PIECEWISE,
                AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
                query_len=1,
            ),
            CUDAGraphMode.FULL_AND_PIECEWISE,
        )

    def test_elastic_on_demand_defers_but_preserves_draft_graph_modes(self):
        config = SimpleNamespace(
            additional_config={"elastic_gdn_backing": True},
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4]),
        )
        speculator = SimpleNamespace(
            vllm_config=config,
            num_speculative_steps=3,
            attn_cg_support=SimpleNamespace(
                min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
                min_cg_attn_backend="flashinfer",
            ),
            device=torch.device("cpu"),
            kv_cache_config=SimpleNamespace(effective_max_resident_seqs=8),
            max_num_reqs=32,
            _ag2_mtp_layer_capture=None,
        )
        manager = MagicMock()
        with patch(
            "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator."
            "SpeculatorCudaGraphManager",
            return_value=manager,
        ) as constructor:
            AutoRegressiveSpeculator.init_cudagraph_manager(
                speculator, CUDAGraphMode.FULL_AND_PIECEWISE
            )

        assert constructor.call_count == 2
        assert (
            constructor.call_args_list[0].args[2]
            == CUDAGraphMode.FULL_AND_PIECEWISE
        )
        assert (
            constructor.call_args_list[1].args[2]
            == CUDAGraphMode.FULL_DECODE_ONLY
        )

    def test_dcp_prefill_reuses_stable_head_index_tensor_during_capture(self):
        wrapper = object.__new__(BatchDCPPrefillWrapper)
        wrapper._local_kv_head_index_tensors = {}
        device = torch.device("cpu")
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            first = wrapper._get_local_kv_head_index_tensor([0, 2], device)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(torch, "tensor", side_effect=AssertionError("reallocated")),
        ):
            second = wrapper._get_local_kv_head_index_tensor([0, 2], device)
        self.assertIs(first, second)

    def test_qwen35_mtp_fc_uses_aligned_tp_padding(self):
        self.assertEqual(_mtp_fc_padded_output_size(5120, 3), 5184)
        self.assertEqual(_mtp_fc_padded_output_size(5120, 2), 5120)
        self.assertEqual(_mtp_fc_padded_output_size(5120, 3, 1), 5121)

    def test_pure_decode_rejects_prefill_replay_and_shape_mismatch(self):
        scheduler = object.__new__(Scheduler)
        scheduler.num_sampled_tokens_per_step = 1
        scheduler.requests = {
            "a": SimpleNamespace(
                num_computed_tokens=10,
                num_prompt_tokens=10,
                use_structured_output=False,
            ),
            "b": SimpleNamespace(
                num_computed_tokens=20,
                num_prompt_tokens=20,
                use_structured_output=False,
            ),
        }

        scheduled = {"a": 1, "b": 3}
        drafts = {"b": [11, 12]}
        self.assertTrue(scheduler._is_pure_decode_step(scheduled, drafts))

        scheduler.requests["b"].num_computed_tokens = 19
        self.assertFalse(scheduler._is_pure_decode_step(scheduled, drafts))
        scheduler.requests["b"].num_computed_tokens = 20

        scheduled["b"] = 2
        self.assertFalse(scheduler._is_pure_decode_step(scheduled, drafts))
        self.assertFalse(scheduler._is_pure_decode_step({}, {}))

    def test_dynamic_full_graph_has_distinct_k0_and_k2_lanes(self):
        compilation_config = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=list(range(1, 193)),
        )
        compilation_config.max_cudagraph_capture_size = 192
        compilation_config.post_init_cudagraph_sizes()
        config = MagicMock(spec=VllmConfig)
        config.compilation_config = compilation_config
        config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=64)
        config.parallel_config = ParallelConfig()
        config.num_speculative_tokens = 2
        spec = MagicMock()
        spec.uses_dynamic_speculative_decoding.return_value = True
        spec.num_speculative_tokens_per_batch_size = [(1, 8, 2), (9, 64, 0)]
        config.speculative_config = spec

        with (
            patch.object(
                cudagraph_utils,
                "get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch.object(
                cudagraph_utils.current_platform,
                "get_global_graph_pool",
                return_value=object(),
            ),
        ):
            manager = cudagraph_utils.ModelCudaGraphManager(
                vllm_config=config,
                device=torch.device("cpu"),
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                decode_query_len=3,
                full_decode_query_lens={1},
                tp3_sd_phase_reduce=True,
            )
        manager._graphs_captured = True
        self.assertTrue(manager.tp3_sd_phase_reduce)

        for num_reqs, query_len in ((9, 1), (40, 1), (64, 1)):
            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_reqs * query_len,
                uniform_token_count=query_len,
                num_active_loras=0,
            )
            self.assertEqual(desc.cg_mode, CUDAGraphMode.FULL)
            self.assertEqual(desc.uniform_token_count, query_len)

        for num_reqs in (1, 8):
            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_reqs * 3,
                uniform_token_count=3,
                num_active_loras=0,
            )
            self.assertEqual(desc.cg_mode, CUDAGraphMode.PIECEWISE)
            self.assertIsNone(desc.uniform_token_count)

        mixed = manager.dispatch(
            num_reqs=2,
            num_tokens=4,
            uniform_token_count=None,
            num_active_loras=0,
        )
        self.assertEqual(mixed.cg_mode, CUDAGraphMode.PIECEWISE)

    def test_k2_only_schedule_still_captures_phase_policy_k0_full_lane(self):
        compilation_config = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
        )
        compilation_config.max_cudagraph_capture_size = 64
        compilation_config.post_init_cudagraph_sizes()
        config = MagicMock(spec=VllmConfig)
        config.compilation_config = compilation_config
        config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=64)
        config.parallel_config = ParallelConfig()
        config.num_speculative_tokens = 2
        spec = MagicMock()
        spec.uses_dynamic_speculative_decoding.return_value = True
        spec.num_speculative_tokens_per_batch_size = [(1, 64, 2)]
        config.speculative_config = spec

        with (
            patch.object(
                cudagraph_utils,
                "get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch.object(
                cudagraph_utils.current_platform,
                "get_global_graph_pool",
                return_value=object(),
            ),
        ):
            manager = cudagraph_utils.ModelCudaGraphManager(
                vllm_config=config,
                device=torch.device("cpu"),
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                decode_query_len=3,
                full_decode_query_lens={1},
            )
        manager._graphs_captured = True

        k0 = manager.dispatch(
            num_reqs=21,
            num_tokens=21,
            uniform_token_count=1,
            num_active_loras=0,
        )
        self.assertEqual(k0.cg_mode, CUDAGraphMode.FULL)

        k2 = manager.dispatch(
            num_reqs=21,
            num_tokens=63,
            uniform_token_count=3,
            num_active_loras=0,
        )
        self.assertEqual(k2.cg_mode, CUDAGraphMode.PIECEWISE)

    def test_drafter_graph_shapes_do_not_expand_target_dynamic_k(self):
        compilation_config = CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
        )
        compilation_config.max_cudagraph_capture_size = 64
        compilation_config.post_init_cudagraph_sizes()
        config = MagicMock(spec=VllmConfig)
        config.compilation_config = compilation_config
        config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=64)
        config.parallel_config = ParallelConfig()
        config.num_speculative_tokens = 2
        spec = MagicMock()
        spec.uses_dynamic_speculative_decoding.return_value = True
        spec.num_speculative_tokens_per_batch_size = [(1, 8, 2), (9, 64, 0)]
        config.speculative_config = spec

        with (
            patch.object(
                cudagraph_utils,
                "get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch.object(
                cudagraph_utils.current_platform,
                "get_global_graph_pool",
                return_value=object(),
            ),
        ):
            manager = cudagraph_utils.CudaGraphManager(
                vllm_config=config,
                device=torch.device("cpu"),
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                decode_query_len=1,
                expand_dynamic_decode_query_lens=False,
            )

        descs = manager._capture_descs[CUDAGraphMode.FULL]
        self.assertTrue(descs)
        self.assertTrue(
            all(
                desc.uniform_token_count == 1
                and desc.num_reqs is not None
                and 0 < desc.num_reqs <= desc.num_tokens
                for desc in descs
            )
        )

    def test_dynamic_graph_working_set_retains_hot_entries_when_x_drops(self):
        compilation_config = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=[4],
        )
        compilation_config.max_cudagraph_capture_size = 4
        compilation_config.post_init_cudagraph_sizes()
        config = MagicMock(spec=VllmConfig)
        config.compilation_config = compilation_config
        config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
        config.parallel_config = ParallelConfig()
        config.speculative_config = None
        config.num_speculative_tokens = 0
        config.additional_config = {
            "dynamic_cudagraph_full_capture_sizes": [3, 5],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_guard_mb": 0,
            "dynamic_cudagraph_min_hits": 1,
        }

        with (
            patch.object(
                cudagraph_utils,
                "get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch.object(
                cudagraph_utils.current_platform,
                "get_global_graph_pool",
                return_value=object(),
            ),
        ):
            manager = cudagraph_utils.ModelCudaGraphManager(
                vllm_config=config,
                device=torch.device("cpu"),
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                decode_query_len=1,
                owner="target",
            )
        manager._graphs_captured = True
        entries = {
            entry.descriptor.num_tokens: entry
            for entry in manager._dynamic_graph_entries.values()
            if entry.descriptor.cg_mode == CUDAGraphMode.FULL
        }
        old_entry = entries[3]
        next_entry = entries[5]
        old_entry.state = cudagraph_utils.DynamicGraphResidency.HOT
        old_entry.charged_bytes = 16
        manager._dynamic_resident_bytes = 16
        next_entry.state = cudagraph_utils.DynamicGraphResidency.QUEUED
        next_entry.estimated_bytes = 32
        manager._dynamic_pending = next_entry.descriptor

        with patch.object(manager, "_evict_dynamic_entry") as evict:
            self.assertFalse(manager.prepare_pending_dynamic_capture(31))
            self.assertEqual(manager._dynamic_pending, next_entry.descriptor)
            self.assertTrue(manager.prepare_pending_dynamic_capture(32))
            self.assertEqual(manager.dynamic_resident_bytes, 16)
            evict.assert_not_called()

            next_entry.state = cudagraph_utils.DynamicGraphResidency.HOT
            next_entry.charged_bytes = 20
            manager._dynamic_resident_bytes = 36
            manager._dynamic_pending = None
            manager.begin_dynamic_step()
            self.assertEqual(manager.dispatch(5, 5, 1, 0), next_entry.descriptor)
            self.assertEqual(manager.finish_dynamic_step(), 36)

            manager.begin_dynamic_step()
            manager.release_unused_dynamic_residency(1, 1, 1, 0)
            self.assertEqual(manager.dynamic_resident_bytes, 36)
            evict.assert_not_called()

        next_entry.state = cudagraph_utils.DynamicGraphResidency.QUEUED
        manager._dynamic_pending = next_entry.descriptor
        with (
            patch.object(manager, "_consensus_dynamic_candidate", return_value=True),
            patch.object(manager, "_admit_dynamic_capture", return_value=True),
            patch.object(
                manager, "capture", side_effect=RuntimeError("capture failed")
            ),
            patch.object(manager, "_destroy_dynamic_graphs", return_value=0) as destroy,
            patch.object(
                cudagraph_utils.current_platform,
                "graph_pool_handle",
                return_value=object(),
            ),
            patch.object(cudagraph_utils.gc, "collect"),
            patch.object(torch.accelerator, "empty_cache") as empty_cache,
            patch.object(torch.accelerator, "get_memory_info", return_value=(64, 128)),
            patch.object(torch.cuda, "synchronize"),
            patch(
                "vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances",
                [],
            ),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            manager.capture_next_dynamic(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                None,
                MagicMock(),
                [],
                MagicMock(),
            )
        destroy.assert_called_once_with(next_entry)
        empty_cache.assert_called()
        self.assertIsNone(manager._dynamic_pending)
        self.assertEqual(
            next_entry.state, cudagraph_utils.DynamicGraphResidency.COOLDOWN
        )

    def test_warmup_uses_separate_pool_ids_and_elastic_transitions(self):
        primary_spec = SimpleNamespace()
        mamba_spec = MagicMock(spec=MambaSpec)
        mamba_spec.separate_pool = True
        alloc, next_ids = _make_warmup_block_allocator([primary_spec, mamba_spec])
        self.assertEqual(alloc(3, primary_spec), [1, 2, 3])
        self.assertEqual(alloc(2, mamba_spec), [1, 2])
        self.assertEqual(next_ids, {"primary": 4, "mamba": 3})

        config = SimpleNamespace(
            elastic_mapping_quantum=4,
            elastic_gdn_stride=4,
            elastic_budget_bytes=64,
            elastic_gdn_initial_blocks=2,
            num_blocks=8,
            kv_cache_tensors=[
                SimpleNamespace(
                    backing_id="elastic-attention-0",
                    logical_block_size=4,
                )
            ],
        )
        runner = SimpleNamespace(kv_cache_config=config)
        output = SchedulerOutput.make_empty()
        _set_elastic_warmup_transition(output, runner, next_ids)
        self.assertEqual(output.elastic_kv_transition, (8, 3))
        self.assertTrue(output.is_synthetic_warmup)

        cleanup = SchedulerOutput.make_empty()
        _set_elastic_warmup_transition(cleanup, runner, next_ids, restore_initial=True)
        self.assertEqual(cleanup.elastic_kv_transition, (8, 2))
        self.assertTrue(cleanup.is_synthetic_warmup)

        no_elastic = SchedulerOutput.make_empty()
        no_elastic_runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(elastic_mapping_quantum=0)
        )
        _set_elastic_warmup_transition(no_elastic, no_elastic_runner, next_ids)
        self.assertTrue(no_elastic.is_synthetic_warmup)
        self.assertIsNone(no_elastic.elastic_kv_transition)

    def test_separate_pool_mtp_exposes_scratch_but_checkpoints_only_base(self):
        spec = MambaSpec(
            block_size=4,
            shapes=((2,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
            separate_pool=True,
            separate_pool_num_blocks=7,
        )
        block_table = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)
        selected = mamba_get_block_table_tensor(
            block_table,
            torch.tensor([4], dtype=torch.int32),
            spec,
            "align",
        )
        torch.testing.assert_close(
            selected, torch.tensor([[3, 4, 5]], dtype=torch.int32)
        )

        state = torch.arange(14, dtype=torch.float32).view(7, 2)
        config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["gdn"])]
        )
        forward_context = {"gdn": SimpleNamespace(kv_cache=[state])}
        block_ids = ([3, 4, 5],)
        store = GDNPrefixCheckpointStore(limit=2, advertised_limit=1)
        expected_base = state[3].clone()
        store.save_blocks(b"base", block_ids, config, forward_context)

        state[3:6].fill_(99)
        store.restore_blocks(b"base", block_ids, config, forward_context)
        torch.testing.assert_close(state[3], expected_base)
        torch.testing.assert_close(state[4:6], torch.full((2, 2), 99.0))

        store.zero_blocks(block_ids, config, forward_context)
        torch.testing.assert_close(state[3:6], torch.zeros((3, 2)))

    def test_v2_gdn_block_lifecycle_and_checkpoint_commands(self):
        manager = V2GDNCheckpointManager()
        store = _Store()
        manager.store = store

        new = SimpleNamespace(
            req_id="req",
            block_ids=([1, 2], [7]),
        )
        manager.update_request_blocks(_scheduler_output(new=[new]))

        cached = CachedRequestData(
            req_ids=["req"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[([3], [])],
            num_computed_tokens=[0],
            num_output_tokens=[0],
        )
        manager.update_request_blocks(_scheduler_output(cached=cached))
        self.assertEqual(manager._block_ids["req"], ([1, 2, 3], [7]))

        output = _scheduler_output(
            scheduled={"req": 1},
            restores={"req": b"hit"},
            saves={"req": b"save"},
        )
        manager.prepare(output, object(), {"req": 0}, {})
        self.assertEqual(
            store.calls,
            [("restore", b"hit", ([1, 2, 3], [7]))],
        )
        self.assertEqual(manager.save(output, object(), {}), (b"visible",))
        self.assertEqual(
            store.calls[-1],
            ("save", b"save", ([1, 2, 3], [7])),
        )

        manager.update_request_blocks(_scheduler_output(finished=["req"]))
        self.assertNotIn("req", manager._block_ids)

    def test_v2_gdn_ignores_kernel_warmup_requests(self):
        manager = V2GDNCheckpointManager()
        store = _Store()
        manager.store = store
        warmup = SimpleNamespace(
            req_id="_warmup_0_",
            block_ids=([1], [0, 2]),
        )
        output = _scheduler_output(
            new=[warmup],
            scheduled={"_warmup_0_": 3},
        )
        manager.update_request_blocks(output)
        manager.prepare(output, object(), {"_warmup_0_": 0}, {})
        self.assertEqual(manager._block_ids, {})
        self.assertEqual(store.calls, [])

    def test_v2_elastic_vote_uses_tp_group_and_rolls_back(self):
        controller = ElasticKVController(torch.device("cpu"))
        controller.backings = {
            "elastic-attention-0": _Owner(16),
            "elastic-gdn": _Owner(8),
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        tp_group = object()
        calls = 0

        def reject(vote, *, op, group):
            nonlocal calls
            self.assertIs(group, tp_group)
            calls += 1
            # Preflight MIN and any-change MAX pass. Reject only the
            # post-transfer vote, then allow the rollback vote to pass.
            if calls == 3:
                vote.zero_()

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce", side_effect=reject),
            patch(
                "vllm.v1.worker.gpu.elastic_gdn.get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
            self.assertRaisesRegex(RuntimeError, "aborted"),
        ):
            controller.apply((2, 4))

        self.assertEqual(controller.backings["elastic-attention-0"].committed, 16)
        self.assertEqual(controller.backings["elastic-gdn"].committed, 8)
        self.assertIsNone(controller._logical_transition)
        self.assertEqual(calls, 4)

    def test_v2_elastic_preflight_rejects_before_any_rank_mutates(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _Owner(16)
        gdn = _Owner(8)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        controller.configure_physical_budget(24, 4, (4, 2))
        tp_group = object()

        def reject_preflight(vote, *, op, group):
            self.assertIs(group, tp_group)
            vote.zero_()

        with (
            patch.object(torch.cuda, "Event") as event,
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=reject_preflight,
            ),
            patch(
                "vllm.v1.worker.gpu.elastic_gdn.get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
            self.assertRaisesRegex(RuntimeError, "rank-synchronous preflight"),
        ):
            controller.apply((4, 4))

        event.assert_not_called()
        self.assertEqual(attention.committed, 16)
        self.assertEqual(gdn.committed, 8)
        self.assertEqual(attention.transfers, [])
        self.assertEqual(gdn.transfers, [])

    def test_v2_elastic_constant_budget_moves_handles_without_resize(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _Owner(16)
        gdn = _Owner(8)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "mem_get_info", return_value=(24, 24)),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            controller.apply((2, 4))

        self.assertEqual(attention.committed, 8)
        self.assertEqual(gdn.committed, 16)
        self.assertEqual(attention.resizes, [])
        self.assertEqual(gdn.resizes, [])
        self.assertEqual(attention.transfers, [(gdn, 8)])

    def test_v2_elastic_retains_slack_for_lower_total_target(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _Owner(16)
        gdn = _Owner(8)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            controller.apply((2, 3))

        self.assertGreaterEqual(attention.committed, 8)
        self.assertGreaterEqual(gdn.committed, 12)
        self.assertEqual(attention.committed + gdn.committed, 24)
        self.assertEqual(attention.resizes, [])
        self.assertEqual(gdn.resizes, [])

    def test_v2_elastic_uses_configured_planner_budget(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _Owner(16)
        gdn = _Owner(8)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        controller.configure_physical_budget(30, 4, (4, 2))

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            controller.apply((2, 3))
            controller.apply((3, 4))

        self.assertEqual(attention.committed + gdn.committed, 28)
        self.assertEqual(controller._physical_budget_bytes, 28)

    def test_v2_elastic_external_owner_returns_pages_to_current_kv_layout(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _Owner(16)
        gdn = _Owner(8)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        controller.configure_physical_budget(24, 4, (4, 2))

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "mem_get_info", return_value=(24, 24)),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            controller.apply((2, 2), external_memory_bytes=8)
            self.assertEqual(attention.committed + gdn.committed, 16)
            self.assertEqual(controller.reconcile_external_memory(0), 0)

        self.assertEqual(attention.committed + gdn.committed, 24)
        self.assertEqual(controller._external_memory_bytes, 0)

    def test_v2_elastic_external_return_preflights_and_retries_physical_floor(self):
        controller = ElasticKVController(torch.device("cpu"))
        attention = _CapacityOwner(12, max_committed=12)
        gdn = _Owner(4)
        controller.backings = {
            "elastic-attention-0": attention,
            "elastic-gdn": gdn,
        }
        controller.geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        controller.configure_physical_budget(24, 4, (3, 1))

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(
                torch.cuda,
                "mem_get_info",
                side_effect=((4, 24), (8, 24)),
            ),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            controller.apply((2, 2), external_memory_bytes=8)
            self.assertEqual(controller.reconcile_external_memory(0), 4)
            self.assertEqual(attention.committed + gdn.committed, 20)
            self.assertEqual(controller._external_memory_bytes, 4)

            # The floor is observational, not a permanent reserve.  Once the
            # lower layer can map the pages, the next step returns all of it.
            attention.max_committed = 16
            self.assertEqual(controller.reconcile_external_memory(0), 0)

        self.assertEqual(attention.committed + gdn.committed, 24)
        self.assertEqual(controller._external_memory_bytes, 0)

    def test_v2_scheduler_step_reconciles_only_external_loan(self):
        controller = object.__new__(ElasticKVController)
        with (
            patch.object(
                controller,
                "reconcile_external_memory",
                return_value=12,
            ) as reconcile,
            patch.object(controller, "apply") as apply,
        ):
            self.assertEqual(controller.apply_scheduler_step(None, 8), 12)
            reconcile.assert_called_once_with(8)
            apply.assert_not_called()

            self.assertEqual(controller.apply_scheduler_step((7, 5), 16), 16)
            apply.assert_called_once_with((7, 5), 16)
            self.assertEqual(reconcile.call_count, 1)


if __name__ == "__main__":
    unittest.main()
