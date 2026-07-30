"""CUDA contract control for releasing packed Qwen GDN prefill tensors."""

from __future__ import annotations

import gc
import json
import weakref

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fused_post_conv_prep,
)


TOKENS = 8192
NUM_K_HEADS = 6
NUM_V_HEADS = 18
HEAD_DIM = 128
DECODE_TOKENS = 3


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    conv_dim = (2 * NUM_K_HEADS + NUM_V_HEADS) * HEAD_DIM
    conv_output = torch.randn(TOKENS, conv_dim, device=device, dtype=dtype)
    decode_reference = conv_output[:DECODE_TOKENS].clone()
    decode_copy = conv_output[:DECODE_TOKENS].clone()
    a = torch.randn(TOKENS, NUM_V_HEADS, device=device, dtype=torch.float32)
    b = torch.randn_like(a)
    A_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    dt_bias = torch.randn_like(A_log)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_output[DECODE_TOKENS:],
        a=a[DECODE_TOKENS:],
        b=b[DECODE_TOKENS:],
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=NUM_K_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        apply_l2norm=True,
        output_g_exp=False,
    )
    torch.cuda.synchronize()
    packed_storage = conv_output.untyped_storage().data_ptr()
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert tensor.untyped_storage().data_ptr() != packed_storage, name

    packed_ref = weakref.ref(conv_output)
    del conv_output
    gc.collect()
    assert packed_ref() is None
    torch.testing.assert_close(decode_copy, decode_reference, rtol=0, atol=0)
    assert all(tensor.isfinite().all() for tensor in (q, k, v, g, beta))
    print(
        json.dumps(
            {
                "exact_decode_prefix": True,
                "packed_bytes": TOKENS * conv_dim * torch.tensor([], dtype=dtype).element_size(),
                "outputs_independent": True,
                "packed_reference_released": True,
            }
        )
    )


if __name__ == "__main__":
    main()
