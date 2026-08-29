# SPDX-License-Identifier: Apache-2.0
"""Compiler-visible fixed-buffer writes for off-by-default diagnostics."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _projection_capture_copy_impl(
    source: torch.Tensor,
    destination: torch.Tensor,
) -> None:
    if source.shape != destination.shape:
        raise RuntimeError(
            "projection capture source/destination shape mismatch: "
            f"{tuple(source.shape)} != {tuple(destination.shape)}"
        )
    if source.dtype != destination.dtype or source.device != destination.device:
        raise RuntimeError(
            "projection capture source/destination type mismatch: "
            f"{source.dtype}/{source.device} != "
            f"{destination.dtype}/{destination.device}"
        )
    destination.copy_(source)


def _projection_capture_copy_fake(
    source: torch.Tensor,
    destination: torch.Tensor,
) -> None:
    del source, destination


direct_register_custom_op(
    op_name="ag2_projection_capture_copy",
    op_func=_projection_capture_copy_impl,
    mutates_args=["destination"],
    fake_impl=_projection_capture_copy_fake,
)


def projection_capture_copy(
    source: torch.Tensor,
    destination: torch.Tensor,
) -> None:
    """Persist ``source`` even when the destination is not a model output."""
    torch.ops.vllm.ag2_projection_capture_copy(source, destination)
