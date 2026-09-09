# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.utils import AutoWeightsLoader


@pytest.mark.parametrize(
    ("language_model_only", "expected_prefixes"),
    [
        (False, ["mtp."]),
        (True, ["mtp.", "visual."]),
    ],
)
def test_conditional_generation_uses_current_weight_loader_prefix_contract(
    monkeypatch: pytest.MonkeyPatch,
    language_model_only: bool,
    expected_prefixes: list[str],
) -> None:
    captured: dict[str, object] = {}

    class Loader:
        def __init__(self, module: object, **kwargs: object) -> None:
            captured["module"] = module
            captured["kwargs"] = kwargs

        def load_weights(self, weights: object, *, mapper: object) -> set[str]:
            captured["weights"] = weights
            captured["mapper"] = mapper
            return {"loaded"}

    monkeypatch.setattr(qwen3_5, "AutoWeightsLoader", Loader)
    model = SimpleNamespace(
        multimodal_config=SimpleNamespace(language_model_only=language_model_only),
        hf_to_vllm_mapper=object(),
    )
    weights = [("model.weight", object())]

    loaded = qwen3_5.Qwen3_5ForConditionalGeneration.load_weights(model, weights)

    assert loaded == {"loaded"}
    assert captured == {
        "module": model,
        "kwargs": {"ignore_unexpected_prefixes": expected_prefixes},
        "weights": weights,
        "mapper": model.hf_to_vllm_mapper,
    }


def test_current_weight_loader_ignores_only_declared_prefixes() -> None:
    module = nn.Linear(1, 1, bias=False)
    loader = AutoWeightsLoader(module, ignore_unexpected_prefixes=["mtp."])
    loaded = loader.load_weights(
        [("mtp.extra", torch.ones(1)), ("weight", torch.ones(1, 1))]
    )

    assert loaded == {"weight"}
    with pytest.raises(ValueError, match="There is no module or parameter"):
        AutoWeightsLoader(module, ignore_unexpected_prefixes=["mtp."]).load_weights(
            [("unexpected.extra", torch.ones(1))]
        )
