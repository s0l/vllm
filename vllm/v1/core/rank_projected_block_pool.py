# SPDX-License-Identifier: Apache-2.0
"""Rank-local physical IDs for logically global KV-cache blocks."""

from collections import Counter
from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.rank_projected_owner import ElasticPageOwnerPolicy


class RankProjectedPhysicalBlockPool:
    """Transactional physical-page allocator for a central logical block pool."""

    def __init__(self, capacities: Sequence[int]) -> None:
        if len(capacities) <= 1 or any(capacity <= 0 for capacity in capacities):
            raise ValueError("rank-projected capacities must contain multiple ranks")
        self.capacities = tuple(int(capacity) for capacity in capacities)
        self.world_size = len(self.capacities)
        self._free_ids = [
            list(range(capacity - 1, -1, -1)) for capacity in self.capacities
        ]
        self._generations = [[0] * capacity for capacity in self.capacities]

    def get_num_free_blocks(self) -> tuple[int, ...]:
        return tuple(len(ids) for ids in self._free_ids)

    def get_usage(self) -> tuple[float, ...]:
        return tuple(
            1 - len(self._free_ids[rank]) / capacity
            for rank, capacity in enumerate(self.capacities)
        )

    def allocate(
        self,
        blocks: Sequence[KVCacheBlock],
        owners: Sequence[int],
    ) -> None:
        if len(blocks) != len(owners):
            raise ValueError("physical owner count must match logical blocks")
        if len({id(block) for block in blocks}) != len(blocks):
            raise ValueError("logical block appears more than once in allocation")
        if any(block.is_rank_projected for block in blocks):
            raise ValueError("logical block already has a physical projection")
        if any(not 0 <= owner < self.world_size for owner in owners):
            raise ValueError("physical owner outside rank world")

        required = Counter(owners)
        for rank, count in required.items():
            if count > len(self._free_ids[rank]):
                raise ValueError(f"rank {rank} physical block pool exhausted")

        # Capacity and every block invariant are validated before the first
        # mutation. The commit below contains no remaining failure branch.
        for block, owner in zip(blocks, owners, strict=True):
            physical_id = self._free_ids[owner].pop()
            block.set_rank_block_id(
                owner_rank=owner,
                physical_block_id=physical_id,
                world_size=self.world_size,
                generation=self._generations[owner][physical_id],
            )

    def allocate_request_pages(
        self,
        blocks: Sequence[KVCacheBlock],
        *,
        start_page: int,
        policy: ElasticPageOwnerPolicy,
    ) -> tuple[int, ...]:
        if self.world_size != 3:
            raise ValueError("Exp11 elastic owner policy requires DCP world size 3")
        owners = policy.owners(start_page, len(blocks))
        self.allocate(blocks, owners)
        return owners

    def free(self, blocks: Sequence[KVCacheBlock]) -> None:
        seen: set[tuple[int, int]] = set()
        resolved: list[tuple[KVCacheBlock, int, int]] = []
        for block in blocks:
            ids = block.rank_block_ids
            if ids is None or len(ids) != self.world_size:
                raise ValueError("block is not owned by this rank-projected world")
            owned = [
                (rank, value)
                for rank, value in enumerate(ids)
                if value is not None
            ]
            if len(owned) != 1:
                raise ValueError("rank-projected block must have exactly one owner")
            owner, physical_id = owned[0]
            identity = (owner, physical_id)
            if identity in seen:
                raise ValueError("physical block appears more than once in free")
            if not 0 <= physical_id < self.capacities[owner]:
                raise ValueError("physical block ID outside owner capacity")
            if block.ref_cnt != 0:
                raise ValueError("cannot free referenced physical block")
            seen.add(identity)
            resolved.append((block, owner, physical_id))

        for block, owner, physical_id in resolved:
            self._generations[owner][physical_id] += 1
            block.reset_rank_block_id()
            self._free_ids[owner].append(physical_id)
