import copy
import unittest

from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    RankProjectedKVCacheUpdate,
    SchedulerOutput,
)


def make_output() -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id="new",
            prompt_token_ids=[],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([10], [20]),
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["cached", "unchanged"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([30], [40]), None],
        num_computed_tokens=[16, 32],
        num_output_tokens=[0, 0],
    )
    return output


def make_update() -> RankProjectedKVCacheUpdate:
    return RankProjectedKVCacheUpdate(
        policy_version="exp11-884-776-equal-v1",
        world_size=3,
        request_block_ids={
            "new": (([101], [201]), ([102], [202]), ([103], [203])),
            "cached": (([301], [401]), ([302], [402]), ([303], [403])),
        },
        new_block_ids_to_zero=([101, 301], [102, 302], [103, 303]),
        kv_cache_block_copies=(None, [(1, 2)], None),
    )


class RankProjectedKVCacheUpdateTest(unittest.TestCase):
    def test_complete_projection_selects_one_atomic_rank_view(self) -> None:
        output = make_output()
        update = make_update()
        output.rank_projected_kv_update = update
        update.validate(output)
        self.assertEqual(([103], [203]), update.block_ids_for("new", 2))
        self.assertEqual(([301], [401]), update.block_ids_for("cached", 0))
        self.assertEqual([102, 302], update.zero_ids_for(1))
        self.assertEqual([(1, 2)], update.copies_for(1))

    def test_incomplete_or_ambiguous_projection_fails_closed(self) -> None:
        mutations = (
            (
                lambda update, output: setattr(update, "policy_version", "stale"),
                "policy",
            ),
            (lambda update, output: setattr(update, "world_size", 2), "world size"),
            (lambda update, output: update.request_block_ids.pop("cached"), "coverage"),
            (
                lambda update, output: setattr(output, "new_block_ids_to_zero", [7]),
                "scalar zeroing",
            ),
            (
                lambda update, output: setattr(
                    output, "kv_cache_block_copies", [(1, 2)]
                ),
                "scalar CoW",
            ),
        )
        for mutation, match in mutations:
            with self.subTest(match=match):
                output = make_output()
                update = make_update()
                mutation(update, output)
                with self.assertRaisesRegex(ValueError, match):
                    update.validate(output)

    def test_validation_and_selection_do_not_mutate_broadcast_payload(self) -> None:
        output = make_output()
        update = make_update()
        before = copy.deepcopy(update)
        update.validate(output)
        for rank in range(3):
            update.block_ids_for("new", rank)
            update.zero_ids_for(rank)
            update.copies_for(rank)
        self.assertEqual(before, update)


if __name__ == "__main__":
    unittest.main()
