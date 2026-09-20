# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.block_table import BlockTables


@pytest.mark.parametrize("rank", range(3))
@pytest.mark.parametrize("x", (1, 2, 4, 8, 16))
@torch.inference_mode()
def test_dummy_slot_source_is_masked_and_graph_accepts_live_rows(rank, x):
    if torch.accelerator.device_count() <= rank:
        pytest.skip("three-device DCP control")
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    tables = BlockTables(
        [192, 192, 8],
        64,
        64,
        [8, 8, 8],
        device,
        [192, 192, 8],
        cp_size=3,
        cp_rank=rank,
        slot_mapping_enabled=[True, True, False],
        cp_replicated=[False, True, False],
    )
    for source in tables.block_tables:
        source.gpu.fill_(1 << 20)
    tables.get_dummy_block_tables(x)
    tables.get_dummy_slot_mappings(x)
    indices = torch.arange(x, dtype=torch.int64, device=device)
    starts = torch.arange(x + 1, dtype=torch.int32, device=device)
    positions = torch.zeros(x, dtype=torch.int64, device=device)
    padding = torch.ones(64, dtype=torch.bool, device=device)

    # This is the old producer, without dereferencing its invalid KV addresses.
    old = tables.compute_slot_mappings(indices, starts, positions, x)
    assert torch.all(old[1] >= (1 << 20) * 192)
    for table in tables.input_block_tables:
        assert not table.any()

    def produce():
        return tables.compute_slot_mappings(
            indices, starts, positions, x, is_padding=padding
        )

    assert torch.all(produce() == -1)
    assert all(torch.all(source.gpu == 1 << 20) for source in tables.block_tables)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        produce()
    torch.cuda.current_stream(device).wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        produce()
    graph.replay()
    assert torch.all(tables.slot_mappings == -1)

    source_values = torch.arange(64 * 8, dtype=torch.int32).reshape(64, 8) + 1
    for source in tables.block_tables:
        source.gpu.copy_(source_values)
    base_positions = [0, 1, 2, 191, 192, 575, 576, 577]
    for live, delta in ((x, 0), (max(1, x - 1), 1), (x, 3), (x, 0)):
        order = list(reversed(range(x)))
        pos = [base_positions[i % len(base_positions)] + delta for i in range(x)]
        indices.copy_(torch.tensor(order, device=device))
        positions.copy_(torch.tensor(pos, device=device))
        starts.copy_(torch.arange(x + 1, device=device).clamp_max(live))
        padding.fill_(True)
        padding[:live] = False
        expected = torch.full((3, 64), -1, dtype=torch.int64)
        for row in range(live):
            for group in (0, 1):
                position = pos[row]
                if group == 0:
                    if position % 3 != rank:
                        continue
                    local_position = position // 3
                else:
                    local_position = position
                block, offset = divmod(local_position, 192)
                expected[group, row] = source_values[order[row], block] * 192 + offset
        graph.replay()
        torch.testing.assert_close(tables.slot_mappings.cpu(), expected, rtol=0, atol=0)
        # Same graph must also reject stale padded source indices/positions.
        padding.fill_(True)
        indices.fill_(1 << 20)
        positions.fill_(1 << 40)
        graph.replay()
        assert torch.all(tables.slot_mappings == -1)
        assert all(
            torch.equal(source.gpu.cpu(), source_values)
            for source in tables.block_tables
        )
    torch.accelerator.synchronize(device)
    graph.reset()
