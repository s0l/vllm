# SPDX-License-Identifier: Apache-2.0
"""Deterministic control for elastic attention CoW with a rounded VMM tail."""

from __future__ import annotations

import torch

from vllm.v1.worker.utils import (
    KVCacheBlockCopyRegion,
    copy_kv_cache_blocks_inplace,
)


def main() -> None:
    num_blocks = 5
    stride = 7
    padding = 11
    storage = torch.full(
        (num_blocks * stride + padding,),
        0xEE,
        dtype=torch.uint8,
    )
    logical = storage[: num_blocks * stride].view(num_blocks, stride)
    for block_id in range(num_blocks):
        logical[block_id].fill_(block_id + 10)
    padding_before = storage[num_blocks * stride :].clone()

    region = KVCacheBlockCopyRegion(
        tensor=logical,
        num_blocks=num_blocks,
        block_stride_bytes=stride,
    )
    copy_kv_cache_blocks_inplace(
        [region],
        num_blocks,
        [(1, 2), (1, 3)],
    )
    assert logical[1].eq(11).all()
    assert logical[2].eq(11).all()
    assert logical[3].eq(11).all()
    assert logical[0].eq(10).all()
    assert logical[4].eq(14).all()
    assert torch.equal(storage[num_blocks * stride :], padding_before)

    # The source side must be snapshotted before overlapping destinations are
    # written: (1 -> 2, 2 -> 4) copies the original block 2 into block 4.
    logical[1].fill_(21)
    logical[2].fill_(22)
    logical[4].fill_(24)
    copy_kv_cache_blocks_inplace(
        [region],
        num_blocks,
        [(1, 2), (2, 4)],
    )
    assert logical[2].eq(21).all()
    assert logical[4].eq(22).all()

    try:
        copy_kv_cache_blocks_inplace([region], num_blocks - 1, [(0, 1)])
    except ValueError as exc:
        assert "disagrees with worker block count" in str(exc)
    else:
        raise AssertionError("mismatched block geometry was accepted")

    print("ELASTIC_ATTENTION_COW_REGION_CONTROL_OK")


if __name__ == "__main__":
    main()
