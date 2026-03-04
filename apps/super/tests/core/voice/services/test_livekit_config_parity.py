# tests/core/voice/services/test_livekit_config_parity.py
"""
Integration tests to verify LiveKit services config parity with Pipecat.

These tests ensure that configuration options available in Pipecat
are also available in LiveKit services for seamless migration.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

MOCKED_DEPENDENCY_MODULES = (
    "pipecat",
    "pipecat.transports",
    "pipecat.transports.livekit",
    "pipecat.transports.livekit.transport",
    "pipecat.audio",
    "pipecat.audio.vad",
    "pipecat.audio.vad.silero",
    "pipecat.audio.vad.vad_analyzer",
    "pipecat.services",
    "pipecat.services.openai",
    "pipecat.services.deepgram",
    "pipecat.services.cartesia",
    "pipecat.services.elevenlabs",
    "pipecat.services.playht",
    "pipecat.services.google",
    "pipecat.services.lmnt",
    "pipecat.services.azure",
    "pipecat.services.anthropic",
    "pipecat.services.xtts",
    "pipecat.processors",
    "pipecat.processors.aggregators.llm_response",
    "pipecat.processors.aggregators",
    "pipecat.processors.filters",
    "pipecat.frames",
    "pipecat.frames.frames",
    "pipecat.pipeline",
    "pipecat.pipeline.pipeline",
    "pipecat.pipeline.task",
    "pipecat.pipeline.runner",
    "pipecat.clocks",
    "pipecat.clocks.system_clock",
    "pipecat.sync",
    "pipecat.sync.base_notifier",
    "pipecat.sync.event_notifier",
    "pipecat.transcriptions",
    "pipecat.transcriptions.language",
    "bson",
    "bson.objectid",
    "livekit",
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.pipeline",
    "livekit.agents.voice_assistant",
    "livekit.agents.voice",
    "livekit.agents.multimodal",
    "livekit.agents.tokenize",
    "livekit.agents.stt",
    "livekit.agents.tts",
    "livekit.plugins",
    "livekit.plugins.openai",
    "livekit.plugins.deepgram",
    "livekit.plugins.cartesia",
    "livekit.plugins.silero",
    "livekit.plugins.turn_detector",
    "livekit.plugins.anthropic",
    "livekit.plugins.google",
    "livekit.plugins.groq",
    "livekit.plugins.aws",
    "livekit.plugins.playai",
    "livekit.plugins.elevenlabs",
    "livekit.plugins.eleven_labs",
    "livekit.plugins.sarvam",
    "livekit.plugins.lmnt",
    "livekit.plugins.xai",
    "livekit.rtc",
    "livekit.rtc._proto",
)


@pytest.fixture(autouse=True)
def _mock_runtime_dependencies(monkeypatch):
    # Keep dependency mocking test-scoped so imports do not leak into other test packages.
    for module_name in MOCKED_DEPENDENCY_MODULES:
        monkeypatch.setitem(sys.modules, module_name, MagicMock())

    # Import this module fresh under mocked dependencies for each test.
    sys.modules.pop("super.core.voice.services.livekit_services", None)
    yield
    sys.modules.pop("super.core.voice.services.livekit_services", None)


class TestLiveKitConfigParity:
    """Integration tests for config parity between Pipecat and LiveKit."""

    def test_all_llm_providers_support_temperature(self):
        """All LLM providers should support temperature config."""
        providers = ["openai", "groq", "anthropic", "azure", "ollama", "cerebras"]

        for provider in providers:
            config = {
                "llm_provider": provider,
                "llm_model": "test-model",
                "temperature": 0.4,
            }

            from super.core.voice.services.livekit_services import (
                LiveKitServiceFactory,
            )

            factory = LiveKitServiceFactory(config)

            # Verify the factory accepts temperature config
            temp = factory._get_voice_optimized_temperature()
            assert temp == 0.4, f"{provider} should support temperature config"

    def test_all_tts_providers_support_language(self):
        """All TTS providers should support language config."""
        providers_with_language = [
            "elevenlabs",
            "cartesia",
            "google",
            "lmnt",
            "sarvam",
        ]

        for provider in providers_with_language:
            config = {
                "tts_provider": provider,
                "tts_voice": "test-voice",
                "language": "hi",
            }

            from super.core.voice.services.livekit_services import (
                LiveKitServiceFactory,
            )

            factory = LiveKitServiceFactory(config)

            # Verify language is in config
            assert factory.config.get("language") == "hi"

    def test_deepgram_stt_supports_interim_results_control(self):
        """Deepgram STT should support use_final_transcriptions_only."""
        config = {
            "stt_provider": "deepgram",
            "stt_model": "nova-3",
            "use_final_transcriptions_only": True,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("use_final_transcriptions_only") is True

    def test_openai_tts_supports_voice_instructions(self):
        """OpenAI TTS should support voice_instructions for modern models."""
        config = {
            "tts_provider": "openai",
            "tts_model": "gpt-4o-mini-tts",
            "tts_voice": "alloy",
            "voice_instructions": "Speak calmly and professionally.",
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert (
            factory.config.get("voice_instructions")
            == "Speak calmly and professionally."
        )

    @pytest.mark.parametrize(
        "provider,model,expected_temp",
        [
            ("openai", "gpt-4o-mini", 0.5),
            ("groq", "llama-3.1-70b", 0.3),
            ("anthropic", "claude-sonnet-4-20250514", 0.4),
            ("google", "gemini-2.0-flash", 0.6),
        ],
    )
    def test_llm_temperature_configurable_per_provider(
        self, provider: str, model: str, expected_temp: float
    ):
        """LLM temperature should be configurable for each provider."""
        config = {
            "llm_provider": provider,
            "llm_model": model,
            "temperature": expected_temp,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)
        actual_temp = factory._get_voice_optimized_temperature()

        assert actual_temp == expected_temp

    @pytest.mark.parametrize(
        "provider,config_key,config_value",
        [
            ("elevenlabs", "stability", 0.5),
            ("elevenlabs", "similarity", 0.6),
            ("elevenlabs", "style", 0.5),
            ("elevenlabs", "tts_speed", 0.95),
            ("sarvam", "pitch", 0.3),
            ("sarvam", "pace", 1.0),
            ("sarvam", "loudness", 1),
            ("cartesia", "speed", 1.0),
        ],
    )
    def test_tts_provider_specific_config(
        self, provider: str, config_key: str, config_value: float
    ):
        """TTS providers should support their specific configuration options."""
        config = {
            "tts_provider": provider,
            "tts_voice": "test-voice",
            config_key: config_value,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get(config_key) == config_value

    def test_deepgram_stt_config_options(self):
        """Deepgram STT should support various configuration options."""
        config = {
            "stt_provider": "deepgram",
            "stt_model": "nova-3",
            "stt_smart_format": True,
            "stt_filler_words": False,
            "stt_endpointing_ms": 25,
            "stt_profanity_filter": False,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("stt_smart_format") is True
        assert factory.config.get("stt_filler_words") is False
        assert factory.config.get("stt_endpointing_ms") == 25
        assert factory.config.get("stt_profanity_filter") is False

    def test_max_tokens_configurable(self):
        """max_tokens should be configurable for LLM providers."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "max_tokens": 800,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)
        max_tokens = factory._get_voice_optimized_max_tokens()

        assert max_tokens == 800

    def test_speaking_plan_config(self):
        """Speaking plan settings should be configurable."""
        speaking_plan = {
            "min_interruption_duration": 0.5,
            "min_interruption_words": 3,
            "min_silence_duration": 0.3,
            "vad_activation_threshold": 0.4,
            "eagerness": "high",
        }
        config = {
            "llm_provider": "openai",
            "speaking_plan": speaking_plan,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.speaking_plan == speaking_plan
        assert factory.speaking_plan.get("min_interruption_duration") == 0.5
        assert factory.speaking_plan.get("eagerness") == "high"

    @pytest.mark.asyncio
    async def test_session_defaults_disable_preemptive_generation(self, monkeypatch):
        """Voice sessions should not preemptively generate by default (prevents duplicate/garbled speech)."""
        import super.core.voice.services.livekit_services as livekit_services_mod

        captured_kwargs = {}

        class FakeAgentSession:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(livekit_services_mod, "AgentSession", FakeAgentSession)

        factory = livekit_services_mod.LiveKitServiceFactory(
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "stt_provider": "deepgram",
                "stt_model": "nova-3",
                "tts_provider": "cartesia",
                "tts_model": "sonic-3",
            }
        )

        monkeypatch.setattr(factory, "_create_vad", lambda: object())
        monkeypatch.setattr(factory, "create_stt_with_retry", AsyncMock(return_value=object()))
        monkeypatch.setattr(factory, "create_llm_with_retry", AsyncMock(return_value=object()))
        monkeypatch.setattr(factory, "create_tts_with_retry", AsyncMock(return_value=object()))

        await factory.create_session(mode=livekit_services_mod.LiveKitServiceMode.INFERENCE)

        assert captured_kwargs["preemptive_generation"] is False

    def test_reasoning_effort_config_for_o1_models(self):
        """reasoning_effort should be configurable for o1/o3 models."""
        config = {
            "llm_provider": "openai",
            "llm_model": "o1-preview",
            "reasoning_effort": "medium",
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("reasoning_effort") == "medium"

    def test_background_audio_config(self):
        """Background audio settings should be configurable."""
        background_audio = {
            "enabled": True,
            "ambient_sound": "OFFICE_AMBIENCE",
            "ambient_volume": 0.8,
            "thinking_sound": ["KEYBOARD_TYPING", "KEYBOARD_TYPING2"],
            "thinking_volume": 0.7,
        }
        config = {
            "llm_provider": "openai",
            "background_audio": background_audio,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("background_audio") == background_audio

    def test_google_tts_gender_config(self):
        """Google TTS should support gender configuration."""
        config = {
            "tts_provider": "google",
            "tts_voice": "en-US-Wavenet-F",
            "language": "en",
            "gender": "female",
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("gender") == "female"

    def test_retry_config_options(self):
        """Retry settings should be configurable for all service types."""
        config = {
            "llm_provider": "openai",
            "stt_retry_attempts": 3,
            "stt_retry_backoff": 0.5,
            "llm_retry_attempts": 2,
            "llm_retry_backoff": 0.5,
            "tts_retry_attempts": 2,
            "tts_retry_backoff": 0.5,
        }

        from super.core.voice.services.livekit_services import LiveKitServiceFactory

        factory = LiveKitServiceFactory(config)

        assert factory.config.get("stt_retry_attempts") == 3
        assert factory.config.get("llm_retry_attempts") == 2
        assert factory.config.get("tts_retry_attempts") == 2
