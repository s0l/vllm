# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from vllm.v1.worker.gpu import model_runner


class _StopBeforeModelConstruction(Exception):
    pass


class TestMRV2OffloaderLifecycle(TestCase):
    def test_initializes_offloader_before_model_construction(self):
        config = SimpleNamespace(offload_config=object())
        runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
        runner.vllm_config = config
        runner.load_config = SimpleNamespace(load_format="auto")
        runner.eplb = Mock()
        runner.eplb.prepare_load.side_effect = _StopBeforeModelConstruction

        offloader = object()
        create = Mock(return_value=offloader)
        install = Mock()
        with (
            patch.object(model_runner, "create_offloader", create),
            patch.object(model_runner, "set_offloader", install),
            self.assertRaises(_StopBeforeModelConstruction),
        ):
            runner.load_model()

        create.assert_called_once_with(config.offload_config)
        install.assert_called_once_with(offloader)


if __name__ == "__main__":
    main()
