# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from vllm.model_executor.layers.quantization.utils import ag2_nvfp4_arc


def _sidecar() -> ag2_nvfp4_arc._Sidecar:
    return ag2_nvfp4_arc._Sidecar(
        manifest={"sidecar_sha256": "a" * 64},
        by_suffix={"layer.0": {}, "layer.1": {}},
        tensors={},
    )


def test_arc_application_logging_is_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    info = Mock()
    monkeypatch.setattr(ag2_nvfp4_arc.logger, "info", info)
    sidecar = _sidecar()

    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=2)
    info.assert_not_called()

    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.1", rank=2)
    info.assert_called_once_with(
        "AG2 ARC application complete rank=%d records=%d sha256=%s",
        2,
        2,
        "a" * 64,
    )


def test_arc_application_rejects_duplicate_prefix() -> None:
    sidecar = _sidecar()
    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=0)

    with pytest.raises(RuntimeError, match="applied twice"):
        ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=0)
