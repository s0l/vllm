"""CPU-only frontend/worker observer lifecycle; no model or CUDA context."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM


@pytest.mark.parametrize("frontend", [False, True])
@pytest.mark.parametrize("injected", [False, True])
def test_profile_lifecycle_without_model(monkeypatch, frontend, injected):
    """Ignoring frontend tracing must still start/stop the worker observer."""
    from vllm.v1.engine import async_llm as module

    for name in (
        "maybe_register_config_serialize_by_value",
        "renderer_from_config",
        "InputProcessor",
        "OutputProcessor",
        "TorchProfilerWrapper",
    ):
        monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(module, "load_stat_logger_plugin_factories", lambda: [])
    core = MagicMock()
    core.profile_async = AsyncMock()
    monkeypatch.setattr(
        module.EngineCoreClient, "make_async_mp_client", MagicMock(return_value=core)
    )
    config = MagicMock()
    config.observability_config.otlp_traces_endpoint = None
    config.profiler_config.profiler = "torch"
    config.profiler_config.ignore_frontend = not frontend
    supplied = MagicMock() if injected else None
    engine = AsyncLLM(config, MagicMock(), log_stats=False, profiler=supplied)
    expected = module.TorchProfilerWrapper.return_value if frontend else supplied
    assert engine.profiler is expected

    async def exercise():
        await engine.start_profile("bounded")
        await engine.stop_profile()
        core.profile_async.side_effect = RuntimeError("observer unavailable")
        with pytest.raises(RuntimeError, match="observer unavailable"):
            await engine.start_profile()
        core.profile_async.side_effect = None
        await engine.stop_profile()

    asyncio.run(exercise())
    assert core.profile_async.await_count == 4
    assert core.profile_async.call_args_list[0].args == (True, "bounded")
    assert core.profile_async.call_args_list[-1].args == (False,)
    if expected is not None:
        assert expected.start.call_count == 2
        assert expected.stop.call_count == 2
    engine.shutdown()
