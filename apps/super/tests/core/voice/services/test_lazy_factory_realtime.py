import importlib
import sys
import types
from pathlib import Path


def _install_lazy_factory_stubs(monkeypatch, captured):
    monkeypatch.setitem(sys.modules, "pipecat", types.ModuleType("pipecat"))
    monkeypatch.setitem(sys.modules, "pipecat.services", types.ModuleType("pipecat.services"))
    monkeypatch.setitem(
        sys.modules,
        "pipecat.services.llm_service",
        types.SimpleNamespace(LLMService=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "pipecat.processors",
        types.ModuleType("pipecat.processors"),
    )
    monkeypatch.setitem(
        sys.modules,
        "pipecat.processors.aggregators",
        types.ModuleType("pipecat.processors.aggregators"),
    )
    monkeypatch.setitem(
        sys.modules,
        "pipecat.processors.aggregators.llm_context",
        types.SimpleNamespace(LLMContext=object),
    )

    class DummyAggregatorPair:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pipecat.processors.aggregators.llm_response_universal",
        types.SimpleNamespace(LLMContextAggregatorPair=DummyAggregatorPair),
    )

    class DummyServiceFactory:
        def __init__(
            self,
            config,
            logger=None,
            room_name=None,
            tool_calling=False,
            use_realtime=False,
            get_docs_callback=None,
        ):
            captured["use_realtime"] = use_realtime
            captured["config"] = config
            self.mixed_realtime_mode = config.get("mixed_realtime_mode", False)

        def _create_stt_service(self):
            return object()

        def _create_llm_service(self):
            return object()

        async def _create_tts_service_with_retry(self):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "super.core.voice.pipecat.services",
        types.SimpleNamespace(ServiceFactory=DummyServiceFactory),
    )


def test_lazy_factory_passes_use_realtime_to_service_factory(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    captured = {}
    _install_lazy_factory_stubs(monkeypatch, captured)

    for key in list(sys.modules.keys()):
        if key == "super.core.voice.services.lazy_factory":
            sys.modules.pop(key, None)

    module = importlib.import_module("super.core.voice.services.lazy_factory")
    lazy_factory = module.LazyServiceFactory(
        config={"use_realtime": True, "mixed_realtime_mode": False}
    )
    lazy_factory._get_service_factory()

    assert captured.get("use_realtime") is True
