# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import unittest
from dataclasses import replace

from vllm.distributed.device_communicators.ag2_runtime_plan import (
    BackendCapabilities,
    CalibrationSurface,
    GpuEndpoint,
    MeasuredPlan,
    ModelGeometry,
    PairLink,
    PhysicalTopology,
    PlanError,
    compile_runtime_plan,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def model(hidden_size: int = 5120, rms_root_size: int = 1024) -> ModelGeometry:
    return ModelGeometry(
        model_identity=SHA_A,
        hidden_size=hidden_size,
        quant_group_size=16,
        rms_root_size=rms_root_size,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        speculative_tokens=3,
        tensor_parallel_size=3,
    )


def backend(*, root_local: bool = True) -> BackendCapabilities:
    return BackendCapabilities(
        backend_identity=SHA_B,
        owner_alignment=256,
        preserves_root_local_math=root_local,
    )


def topology(*, fast_pair: tuple[str, str] = ("gpu-a", "gpu-b")) -> PhysicalTopology:
    ids = ("gpu-a", "gpu-b", "gpu-c")
    links = []
    for left, right in ((ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2])):
        is_fast = {left, right} == set(fast_pair)
        links.append(PairLink(left, right, 24.0 if is_fast else 12.0, 8.0))
    return PhysicalTopology(
        topology_identity=SHA_C,
        gpus=(
            GpuEndpoint("gpu-a", "0000:01:00.0", 1.0),
            GpuEndpoint("gpu-b", "0000:02:00.0", 1.0),
            GpuEndpoint("gpu-c", "0000:81:00.0", 1.0),
        ),
        links=tuple(links),
    )


class RuntimePlanTest(unittest.TestCase):
    def test_exact_safe_fallback_derives_884_from_five_rms_roots(self) -> None:
        plan = compile_runtime_plan(
            model=model(), backend=backend(), topology=topology(), rows=64
        )
        self.assertEqual(plan.owner_widths, (2048, 2048, 1024))
        self.assertEqual(plan.owner_offsets, (0, 2048, 4096))
        self.assertEqual(plan.logical_to_physical, ("gpu-a", "gpu-b", "gpu-c"))
        self.assertEqual(plan.exact_sum_order, ((0, 1), 2))
        self.assertEqual(plan.source, "derived-exact-safe-fallback")

    def test_model_geometry_mutation_changes_plan_without_model_constants(self) -> None:
        plan = compile_runtime_plan(
            model=model(hidden_size=6144),
            backend=backend(),
            topology=topology(),
            rows=64,
        )
        self.assertEqual(plan.owner_widths, (2048, 2048, 2048))
        self.assertEqual(sum(plan.owner_widths), 6144)

    def test_fast_pair_permutation_changes_mapping_not_exact_sum_order(self) -> None:
        plan = compile_runtime_plan(
            model=model(),
            backend=backend(),
            topology=topology(fast_pair=("gpu-a", "gpu-c")),
            rows=64,
        )
        self.assertEqual(plan.logical_to_physical[:2], ("gpu-a", "gpu-c"))
        self.assertEqual(plan.logical_to_physical[2], "gpu-b")
        self.assertEqual(plan.exact_sum_order, ((0, 1), 2))

    def test_complete_measured_surface_selects_shape_specific_geometry(self) -> None:
        topo = topology()
        common = {
            "logical_to_physical": ("gpu-a", "gpu-b", "gpu-c"),
            "exact": True,
        }
        surface = CalibrationSurface(
            model_identity=SHA_A,
            backend_identity=SHA_B,
            topology_identity=SHA_C,
            transport_identity=SHA_D,
            complete=True,
            entries=(
                MeasuredPlan(32, (1792, 1792, 1536), critical_ms=18.3, **common),
                MeasuredPlan(32, (2048, 2048, 1024), critical_ms=18.8, **common),
                MeasuredPlan(160, (2304, 2304, 512), critical_ms=72.4, **common),
                MeasuredPlan(160, (2048, 2048, 1024), critical_ms=75.0, **common),
            ),
        )
        transport_backend = backend(root_local=False)
        m32 = compile_runtime_plan(
            model=model(),
            backend=transport_backend,
            topology=topo,
            rows=32,
            calibration=surface,
        )
        m160 = compile_runtime_plan(
            model=model(),
            backend=transport_backend,
            topology=topo,
            rows=160,
            calibration=surface,
        )
        self.assertEqual(m32.owner_widths, (1792, 1792, 1536))
        self.assertEqual(m160.owner_widths, (2304, 2304, 512))
        self.assertEqual(m32.source, m160.source)
        self.assertEqual(m32.source, "measured-complete-surface")

    def test_incomplete_surface_falls_back_instead_of_partial_tuning(self) -> None:
        topo = topology()
        surface = CalibrationSurface(
            model_identity=SHA_A,
            backend_identity=SHA_B,
            topology_identity=SHA_C,
            transport_identity=SHA_D,
            complete=False,
            entries=(
                MeasuredPlan(
                    32,
                    (1792, 1792, 1536),
                    ("gpu-a", "gpu-b", "gpu-c"),
                    1.0,
                    True,
                ),
            ),
        )
        plan = compile_runtime_plan(
            model=model(),
            backend=backend(root_local=False),
            topology=topo,
            rows=32,
            calibration=surface,
        )
        self.assertEqual(plan.source, "derived-exact-safe-fallback")
        self.assertEqual(plan.owner_widths, (1792, 1792, 1536))

    def test_stale_model_or_topology_calibration_fails_closed(self) -> None:
        topo = topology()
        surface = CalibrationSurface(
            model_identity=SHA_A,
            backend_identity=SHA_B,
            topology_identity=SHA_C,
            transport_identity=SHA_D,
            complete=True,
            entries=(),
        )
        with self.assertRaisesRegex(PlanError, "stale"):
            compile_runtime_plan(
                model=replace(model(), model_identity="e" * 64),
                backend=backend(),
                topology=topo,
                rows=32,
                calibration=surface,
            )
        with self.assertRaisesRegex(PlanError, "stale"):
            compile_runtime_plan(
                model=model(),
                backend=backend(),
                topology=replace(topo, topology_identity="f" * 64),
                rows=32,
                calibration=surface,
            )

    def test_root_local_consumer_rejects_transport_only_776(self) -> None:
        surface = CalibrationSurface(
            model_identity=SHA_A,
            backend_identity=SHA_B,
            topology_identity=SHA_C,
            transport_identity=SHA_D,
            complete=True,
            entries=(
                MeasuredPlan(
                    32,
                    (1792, 1792, 1536),
                    ("gpu-a", "gpu-b", "gpu-c"),
                    1.0,
                    True,
                ),
            ),
        )
        with self.assertRaisesRegex(PlanError, "aligned to 1024"):
            compile_runtime_plan(
                model=model(),
                backend=backend(root_local=True),
                topology=topology(),
                rows=32,
                calibration=surface,
            )


if __name__ == "__main__":
    unittest.main()
