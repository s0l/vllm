import torch

from vllm import envs
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


def test_gemma_deterministic_rms_matches_reference_and_repeats(
    monkeypatch,
) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setattr(envs, "AG2_VLLM_GEMMA_DETERMINISTIC_RMS", True)
    torch.manual_seed(17)
    layer = GemmaRMSNorm(5120, eps=1e-6).cuda().bfloat16()
    x = torch.randn(513, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    first, first_residual = layer.forward_native(x, residual)
    second, second_residual = layer.forward_native(x.clone(), residual.clone())
    reference_sum = x.float() + residual.float()
    reference_residual = reference_sum.to(torch.bfloat16)
    weight = layer.weight.float() + 1.0
    reference = (
        reference_sum
        * torch.rsqrt(
            reference_sum.square().mean(dim=-1, keepdim=True)
            + layer.variance_epsilon
        )
        * weight
    ).to(torch.bfloat16)

    assert torch.equal(first_residual, reference_residual)
    assert torch.equal(first_residual, second_residual)
    assert torch.equal(first, second)
    mismatch = torch.count_nonzero(first != reference).item()
    assert mismatch / first.numel() <= 1e-5
    assert (first.float() - reference.float()).abs().max().item() <= 0.03125

    # Guard the exact semantic bug that made the first deterministic POC
    # stable but changed product PPL: normalization must not consume the
    # BF16-rounded residual output.
    rounded_reference = (
        reference_residual.float()
        * torch.rsqrt(
            reference_residual.float().square().mean(dim=-1, keepdim=True)
            + layer.variance_epsilon
        )
        * weight
    ).to(torch.bfloat16)
    assert torch.count_nonzero(first != reference) < torch.count_nonzero(
        first != rounded_reference
    )

    cuda, cuda_residual = layer.forward_cuda(x.clone(), residual.clone())
    assert torch.equal(cuda_residual, first_residual)
    assert torch.equal(cuda, first)
