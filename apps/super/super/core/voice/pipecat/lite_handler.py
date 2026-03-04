"""
Lite Voice Handler - Minimal, plugin-based voice processing.

A lightweight alternative to PipecatVoiceHandler that reduces startup time
from 2-3 seconds to <500ms through lazy loading and plugin architecture.

Key differences from PipecatVoiceHandler:
- Services created on-demand, not at startup
- KB preloading happens in background after first response
- Optional features loaded as plugins
- Simpler linear pipeline (no complex ParallelPipeline by default)
"""

import asyncio
import json
import logging
import os
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipecat.frames.frames import (
    EndFrame,
    LLMMessagesAppendFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, NOT_GIVEN
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport

from super.core.callback.base import BaseCallback
from super.core.configuration import BaseModelConfig
from super.core.context.schema import Event, Message, Role, User
from super.core.handler.config import (
    ExecutionNature,
    HandlerConfiguration,
    RoleConfiguration,
)
from super.core.logging import logging as app_logging
from super.core.plugin.base import PluginLocation, PluginStorageFormat
from super.core.voice.base import BaseVoiceHandler
from super.core.voice.shared_mixin import SharedVoiceMixin
from super.core.voice.plugins import PluginRegistry
from super.core.voice.plugins.base import PluginPriority
from super.core.voice.managers.prompt_manager import PromptManager
from super.core.voice.schema import (
    AgentConfig,
    CallSession,
    Modality,
    TransportType,
    UserState,
)
from super.core.voice.services.lazy_factory import LazyServiceFactory
from super.core.voice.services.service_common import (
    estimate_tokens,
    get_model_context_limit,
    is_realtime_model,
)
from super.core.voice.managers.transport_manager import TransportManager
from super.core.voice.pipecat.utils import UpPipelineRunner, create_usage
from super.core.voice.livekit.state_observer import (
    LiveKitStateObserver,
    create_livekit_state_observer,
)

# Conversation state import
try:
    from super.core.voice.livekit.conversation_state import (
        DynamicConversationState,
        build_conversation_state,
    )

    CONVERSATION_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONVERSATION_INTELLIGENCE_AVAILABLE = False
    DynamicConversationState = None
    build_conversation_state = None


class LiteVoiceHandler(SharedVoiceMixin, BaseVoiceHandler, ABC):
    """
    Lightweight voice handler with plugin-based architecture.

    Optimized for fast startup (<500ms) while maintaining full functionality
    through lazy loading and modular plugins.

    Features:
    - Lazy service initialization (STT/LLM/TTS created on-demand)
    - Plugin system for optional features (RAG, streaming, silence trimming)
    - Background KB preloading (doesn't block first response)
    - Simplified pipeline construction
    """

    default_configuration = HandlerConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="super.core.voice.lite_handler.LiteVoiceHandler",
        ),
        role=RoleConfiguration(
            name="lite_voice_handler",
            role="A lightweight handler for voice conversations.",
            cycle_count=0,
            max_task_cycle_count=3,
        ),
        execution_nature=ExecutionNature.AUTO,
    )

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_state: Optional[UserState] = None,
        callback: Optional[BaseCallback] = None,
        model_config: Optional[BaseModelConfig] = None,
        configuration: HandlerConfiguration = default_configuration,
        observer: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            session_id=session_id,
            callback=callback,
            configuration=configuration,
            logger=logger or app_logging.get_logger("lite.voice.handler"),
        )

        self._session_id = session_id or (
            str(user_state.thread_id) if user_state else None
        )
        self._callback = callback
        self._logger = logger or app_logging.get_logger("lite.voice.handler")
        self._configuration = configuration

        # User state and config
        self.user_state = user_state
        self.config = model_config or (user_state.model_config if user_state else {})

        # Realtime mode configuration
        # use_realtime: Enable OpenAI Realtime API or Gemini Live
        # mixed_realtime_mode: Use Realtime LLM but separate TTS (text output)
        self.use_realtime = False
        self.mixed_realtime_mode = False

        # Transport
        self._transport_type = os.getenv("HANDLER_TRANSPORT", TransportType.LIVEKIT)
        self._transport: Optional[BaseTransport] = None
        self._room_name: Optional[str] = None

        # Observer for metrics
        self.observer = observer

        # Agent config
        self.agent_config = AgentConfig(
            agent_name=session_id or "LiteVoiceAgent",
            model_config=model_config,
        )

        # Plugin registry - manages optional features
        self.plugins = PluginRegistry(logger=self._logger.getChild("plugins"))

        # Lazy service factory - creates services on-demand
        self.services: Optional[LazyServiceFactory] = None

        # Prompt manager - initialized lazily
        self._prompt_manager: Optional[PromptManager] = None

        # Transport manager
        self._transport_manager: Optional[TransportManager] = None

        # Pipeline components
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[UpPipelineRunner] = None
        self._context_aggregator: Optional[Any] = None

        # Session tracking
        self.active_sessions: Dict[str, CallSession] = {}

        # State flags
        self._is_shutting_down = False
        self._services_initialized = False
        self._first_response_sent = asyncio.Event()
        self._kb_ready = asyncio.Event()

        # Agent/user references for callbacks
        self.agent: Optional[User] = None

        # Event bridge for LiveKit transcript publishing
        self._event_bridge: Optional[Any] = None

        # LiveKit state observer for agent state tracking
        self._livekit_state_observer: Optional[LiveKitStateObserver] = None

        # Conversation state for conversation intelligence tools
        self._conv_state: Optional[DynamicConversationState] = None

        # Token-aware context management
        self._context_limit: int = 0
        self._effective_context_limit: int = 0
        self._max_context_turns: int = 6

        # KB tool usage tracking for auto-inject fallback
        self._kb_tool_calls: int = 0

        # Initialize shared mixin (idle timeout, eval, cleanup, KB, etc.)
        self._init_shared_mixin()

        self._logger.info(f"LiteVoiceHandler created (session={self._session_id})")

    @property
    def prompt_manager(self) -> PromptManager:
        """Lazy-initialized prompt manager."""
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                config=self.config or {},
                agent_config=self.agent_config,
                session_id=self._session_id,
                tool_calling=self.config.get("tool_calling", True),
                logger=self._logger.getChild("prompt"),
            )
            if self.user_state:
                self._prompt_manager.user_state = self.user_state
        return self._prompt_manager

    @property
    def transport_manager(self) -> TransportManager:
        """Lazy-initialized transport manager."""
        if self._transport_manager is None:
            self._transport_manager = TransportManager(
                logger=self._logger.getChild("transport"),
                config=self.config,
                user_state=self.user_state,
                room_name=self._room_name,
                transport_type=self._transport_type,
                task=self.task,
                active_sessions=self.active_sessions,
                end_call_callback=self.end_call,
            )
        return self._transport_manager

    def _add_users(self) -> None:
        """Create agent and user references."""
        import uuid

        self.agent = User.add_user(
            name=self.config.get("agent_id", ""),
            role=Role.ASSISTANT,
        )

        if self.user_state:
            # Extract user_id from dict, handling both 'id' and 'user_id' keys
            # and converting to string since the value may be an int
            user_id = str(uuid.uuid4())
            if isinstance(self.user_state.user, dict):
                raw_id = self.user_state.user.get(
                    "id", self.user_state.user.get("user_id")
                )
                if raw_id is not None:
                    user_id = str(raw_id)

            self.user_state.user = User.add_user(
                name=self.user_state.user_name,
                role=Role.USER,
                _id=user_id,
            )

    def set_realtime_fields(self) -> None:
        """
        Determine realtime mode using priority chain:
        1. Database flag (llm_realtime) - source of truth
        2. Model name detection (LLM_REALTIME_MODELS list)

        Env var MIXED_REALTIME_MODE only affects mixed mode, not realtime detection.
        """
        # Priority 1: Database flag (source of truth)
        if self.config.get("llm_realtime"):
            self.use_realtime = True
            self._logger.info(
                "Realtime mode enabled via database flag (llm_realtime=True)"
            )
        # Priority 2: Model name detection
        elif is_realtime_model(self.config.get("llm_model", "")):
            self.use_realtime = True
            self._logger.info(
                f"Realtime mode enabled via model detection: {self.config.get('llm_model')}"
            )
        else:
            self.use_realtime = False

        # Mixed mode from env only (for text output with separate TTS)
        if self.use_realtime:
            self.mixed_realtime_mode = self.config.get(
                "mixed_realtime_mode",
                os.getenv("MIXED_REALTIME_MODE", "false").lower() == "true",
            )
        else:
            self.mixed_realtime_mode = False

        self._logger.info(
            f"Realtime mode: use_realtime={self.use_realtime}, "
            f"mixed_realtime_mode={self.mixed_realtime_mode}"
        )

    @property
    def _is_text_only(self) -> bool:
        """Check if this session is TEXT-only mode (no STT/TTS)."""
        if not self.user_state:
            return False
        modality = getattr(self.user_state, "modality", Modality.TEXT_AUDIO)
        modality_str = str(modality).lower().strip()
        return modality_str == "text" or modality == Modality.TEXT

    async def preload_agent(
        self,
        user_state: UserState,
        observer: Optional[Any] = None,
    ) -> bool:
        """
        Minimal preload - only essential initialization.

        Target: <200ms (vs 2-3s for PipecatVoiceHandler)

        Args:
            user_state: User state for the session
            observer: Optional observer for metrics

        Returns:
            True if preload successful
        """
        import time
        from super.core.voice.common.common import add_perf_log

        start_time = time.time()
        self._logger.info("Starting lite preload...")

        try:
            # Update state
            self.user_state = user_state
            self.config = user_state.model_config or self.config
            self._room_name = user_state.room_name
            self.observer = observer

            # Determine realtime mode from DB flag or model name
            self.set_realtime_fields()

            # Token-aware context management
            llm_model = self.config.get("llm_model", "") if self.config else ""
            self._context_limit = get_model_context_limit(llm_model)
            max_tokens = (
                self.config.get("llm_max_tokens", 350) if self.config else 350
            )
            self._effective_context_limit = (
                self._context_limit - max_tokens - 200
            )
            self._max_context_turns = (
                self.config.get("max_context_turns", 6) if self.config else 6
            )

            # Build conversation state for conversation intelligence
            if CONVERSATION_INTELLIGENCE_AVAILABLE and build_conversation_state:
                system_prompt = self.prompt_manager._create_assistant_prompt()
                self._conv_state = build_conversation_state(
                    config=self.config,
                    system_prompt=system_prompt,
                )
                self._logger.info("Conversation intelligence state initialized")

            # Create users
            self._add_users()

            # Initialize lazy service factory with realtime mode settings
            _factory_start = time.time()
            service_config = self.config.copy() if self.config else {}
            service_config["use_realtime"] = self.use_realtime
            service_config["mixed_realtime_mode"] = self.mixed_realtime_mode

            # For realtime mode, create assistant prompt for LLM initialization
            assistant_prompt = None
            if self.use_realtime:
                llm_provider = self.config.get("llm_provider", "openai").lower()
                if llm_provider in ["openai", "google"]:
                    assistant_prompt = self.prompt_manager._create_assistant_prompt()
                    self._logger.info(
                        f"Created assistant prompt for {llm_provider.upper()} "
                        "Realtime API"
                    )

            self.services = LazyServiceFactory(
                config=service_config,
                logger=self._logger.getChild("services"),
                get_docs_callback=self.get_docs,
                assistant_prompt=assistant_prompt,
            )
            add_perf_log(
                user_state,
                "service_factory_init",
                (time.time() - _factory_start) * 1000,
            )

            # Get transport (required before pipeline)
            _transport_start = time.time()
            self._transport = await self.get_transport()
            add_perf_log(
                user_state, "transport_init", (time.time() - _transport_start) * 1000
            )

            # Start STT preload in background (only for standard mode)
            # Skip STT preload for realtime or TEXT-only modes
            if self._is_text_only:
                self._logger.info(
                    "Skipping STT preload (TEXT-only mode)"
                )
            elif not self.use_realtime:
                _stt_start = time.time()
                await self.services.preload_stt()
                add_perf_log(
                    user_state, "stt_preload", (time.time() - _stt_start) * 1000
                )
            else:
                self._logger.info(
                    "Skipping STT preload (realtime mode handles audio input)"
                )

            # Activate plugins from config
            _plugins_start = time.time()
            await self.plugins.activate_from_config(self, self.config)
            add_perf_log(
                user_state, "plugins_activation", (time.time() - _plugins_start) * 1000
            )

            # Start background KB warmup (non-blocking)
            if self._kb_enabled():
                asyncio.create_task(
                    self._background_kb_warmup(),
                    name=f"{self._session_id}-kb-warmup",
                )

            self._services_initialized = True
            elapsed = (time.time() - start_time) * 1000
            add_perf_log(user_state, "lite_preload_total", elapsed)
            self._logger.info(f"Lite preload complete in {elapsed:.0f}ms")
            return True

        except Exception as e:
            self._logger.error(f"Preload failed: {e}")
            return False

    # _kb_enabled inherited from SharedVoiceMixin

    async def _background_kb_warmup(self) -> None:
        """Delegates to shared mixin KB warmup (non-immediate for Pipecat)."""
        await self._shared_background_kb_warmup(immediate=False)

    async def execute(self, *args, **kwargs) -> Any:
        """Execute the voice handler pipeline."""
        try:
            self._logger.info(f"Starting lite pipeline for {self.user_state.user_name}")

            if not self._transport:
                self._transport = await self.get_transport()

            # Create pipeline
            pipeline, context_aggregator = await self._create_pipeline(self._transport)
            self._context_aggregator = context_aggregator

            # Create observers
            observers = []
            if self.observer:
                observers.append(self.observer(user_state=self.user_state))

            # Add LiveKit state observer for agent state tracking
            # emit_transcripts=False because transcripts are handled via event bridge
            self._livekit_state_observer = create_livekit_state_observer(
                user_state=self.user_state,
                transport=self._transport,
                logger=self._logger.getChild("livekit.state"),
                emit_transcripts=False,
            )
            observers.append(self._livekit_state_observer)

            # Create pipeline task
            self.task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    enable_metrics=True,
                    enable_usage_metrics=True,
                    audio_in_sample_rate=self.config.get("audio_in_sample_rate", 16000),
                    audio_out_sample_rate=self.config.get(
                        "audio_out_sample_rate", 24000
                    ),
                ),
                observers=observers,
            )

            # Update transport manager with task
            self.transport_manager.set_task(self.task)

            # Set up transport event handlers
            await self._setup_transport_events()

            # Run pipeline
            self.runner = UpPipelineRunner()
            await self.runner.run(self.task)

        except Exception as e:
            self._logger.error(f"Pipeline error: {e}")
            raise

    async def _create_pipeline(self, transport: BaseTransport) -> tuple:
        """
        Create pipeline with plugin processors and parallel transcript tracking.

        Supports four modes:
        - Standard mode: STT → LLM → TTS pipeline
        - Full Realtime mode: Realtime LLM with integrated audio (no STT/TTS)
        - Mixed Realtime mode: Realtime LLM (text output) → separate TTS
        - TEXT-only mode: No STT/TTS, data channel for input/output

        Returns:
            Tuple of (pipeline, context_aggregator)
        """
        # Check TEXT-only mode first
        is_text_only = self._is_text_only

        # Determine pipeline mode
        is_realtime_mode = self.use_realtime
        is_mixed_mode = self.mixed_realtime_mode

        # Log pipeline mode
        if is_text_only:
            self._logger.info(
                "TEXT-only Pipeline: data channel I/O (no STT/TTS)"
            )
        elif is_realtime_mode and is_mixed_mode:
            self._logger.info(
                "Mixed Realtime Pipeline: Realtime LLM (text) -> Separate TTS"
            )
        elif is_realtime_mode:
            self._logger.info(
                "Full Realtime Pipeline: Realtime LLM (integrated audio)"
            )
        else:
            self._logger.info("Standard Pipeline: STT -> LLM -> TTS")

        # Get services based on mode
        stt = None
        tts = None
        llm = await self.services.get_llm()

        # Sync mixed_realtime_mode from ServiceFactory after LLM creation.
        # The ServiceFactory auto-detects whether TTS uses the same realtime
        # model as the LLM (e.g. both Gemini native-audio) and updates its
        # mixed_realtime_mode accordingly. We must pick up that change here
        # so the pipeline doesn't create a separate TTS that will fail with
        # "streaming not supported" errors.
        factory = getattr(self.services, "_service_factory", None)
        if factory and hasattr(factory, "mixed_realtime_mode"):
            if factory.mixed_realtime_mode != is_mixed_mode:
                self._logger.info(
                    f"ServiceFactory updated mixed_realtime_mode: "
                    f"{is_mixed_mode} -> {factory.mixed_realtime_mode}"
                )
                is_mixed_mode = factory.mixed_realtime_mode
                self.mixed_realtime_mode = is_mixed_mode

        # Register tool handlers on LLM service
        if llm and hasattr(llm, "register_function"):
            self._register_tool_handlers(llm)

        if is_text_only:
            # TEXT-only: Only LLM (data channel handles I/O)
            self._logger.info("TEXT-only: Using only LLM (data channel I/O)")
        elif is_realtime_mode and not is_mixed_mode:
            # Full realtime: Only LLM (integrated STT/TTS)
            self._logger.info("Full realtime: Using only LLM (integrated audio)")
        elif is_realtime_mode and is_mixed_mode:
            # Mixed realtime: LLM + TTS (Realtime handles STT)
            tts = await self.services.get_tts()
            self._logger.info("Mixed realtime: Using LLM + TTS (no STT)")
        else:
            # Standard mode: All three services
            stt = await self.services.get_stt()
            tts = await self.services.get_tts()
            self._logger.info("Standard mode: Using STT + LLM + TTS")

        # Validate required services based on mode
        if not llm:
            raise Exception(
                "LLM service could not be created - missing API keys or unavailable"
            )

        if is_text_only:
            self._logger.info("TEXT-only mode: LLM service validated")
        elif is_realtime_mode and not is_mixed_mode:
            self._logger.info(
                "Full realtime mode: LLM service validated (STT/TTS integrated)"
            )
        elif is_realtime_mode and is_mixed_mode:
            if not tts:
                raise Exception(
                    "Mixed realtime mode: TTS service required but could not be created"
                )
            self._logger.info("Mixed realtime mode: LLM + TTS services validated")
        else:
            if not stt:
                raise Exception(
                    "Standard mode: STT service required but could not be created"
                )
            if not tts:
                raise Exception(
                    "Standard mode: TTS service required but could not be created"
                )
            self._logger.info("Standard mode: All services (STT/LLM/TTS) validated")

        # Create context with new universal LLMContext
        context = self._create_context()
        context_aggregator = LLMContextAggregatorPair(context)

        # Set up transcript handling via aggregator events
        self._setup_aggregator_transcript_events(context_aggregator)

        # Build pipeline stages based on mode
        stages: List[Any] = []

        # Input from transport (always needed)
        stages.append(transport.input())

        if is_text_only:
            # TEXT-only Pipeline: transport.input → context_aggregator → output
            # No STT/TTS; data channel handler drives LLM generation
            stages.append(context_aggregator.user())
            stages.append(
                ParallelPipeline(
                    [transport.output(), context_aggregator.assistant()],
                )
            )
        elif is_realtime_mode and not is_mixed_mode:
            # Full Realtime Pipeline: transport.input → realtime_llm → output
            # The realtime LLM handles audio input/output directly
            stages.append(
                ParallelPipeline(
                    [context_aggregator.user(), llm],
                )
            )
            stages.append(
                ParallelPipeline(
                    [transport.output(), context_aggregator.assistant()],
                )
            )
        elif is_realtime_mode and is_mixed_mode:
            # Mixed Realtime Pipeline: realtime_llm → tts → output
            # Realtime LLM outputs text, separate TTS generates audio

            # PRE_LLM plugins
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.PRE_LLM)
            )

            # User processing
            stages.append(
                ParallelPipeline(
                    [context_aggregator.user(), llm],
                )
            )

            # POST_LLM plugins
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.POST_LLM)
            )

            # PRE_TTS plugins
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.PRE_TTS)
            )

            # TTS
            stages.append(tts)

            # POST_TTS plugins
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.POST_TTS)
            )

            # Output
            stages.append(
                ParallelPipeline(
                    [transport.output(), context_aggregator.assistant()],
                )
            )
        else:
            # Standard Pipeline: STT → LLM → TTS
            # STT (Speech-to-Text)
            stages.append(stt)

            # PRE_LLM plugins (e.g., RAG context enrichment)
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.PRE_LLM)
            )

            # User processing
            stages.append(
                ParallelPipeline(
                    [context_aggregator.user(), llm],
                )
            )

            # POST_LLM plugins (e.g., streaming parser for early TTS)
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.POST_LLM)
            )

            # PRE_TTS plugins
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.PRE_TTS)
            )

            # TTS (Text-to-Speech)
            stages.append(tts)

            # POST_TTS plugins (e.g., silence trimmer)
            stages.extend(
                self.plugins.get_processors_at_priority(PluginPriority.POST_TTS)
            )

            # Output processing
            stages.append(
                ParallelPipeline(
                    [transport.output(), context_aggregator.assistant()],
                )
            )

        pipeline = Pipeline(stages)

        # Log final pipeline configuration
        if is_text_only:
            mode_name = "TEXT-only"
        elif is_realtime_mode and is_mixed_mode:
            mode_name = "Mixed Realtime"
        elif is_realtime_mode:
            mode_name = "Full Realtime"
        else:
            mode_name = "Standard"

        self._logger.info(
            f"✅ Pipeline created with {len(stages)} stages ({mode_name} mode)"
        )

        return pipeline, context_aggregator

    def _create_context(self) -> LLMContext:
        """Create LLM context with system prompt and tool schemas."""
        system_prompt = self.prompt_manager._create_assistant_prompt()

        messages = [{"role": "system", "content": system_prompt}]

        # Add tools if enabled
        tools = []
        tool_calling = self.config.get("tool_calling", True)

        if tool_calling and self._kb_enabled():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_knowledge_base_info",
                        "description": (
                            "Search the knowledge base for relevant information. "
                            "ALWAYS call this tool if answer is not in context "
                            "before answering domain-specific, factual, or "
                            "product/service-related questions from the user."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "Query to get relevant documents"
                                    ),
                                },
                                "kb_name": {
                                    "type": "string",
                                    "description": "Knowledge Base Name",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        if tool_calling:
            # Core tools matching LiveKit agent parity
            tools.extend(self._get_tool_schemas())

        tools_schema = NOT_GIVEN
        if tools:
            tools_schema = ToolsSchema(
                standard_tools=[],
                custom_tools={AdapterType.SHIM: tools},
            )
        return LLMContext(messages=messages, tools=tools_schema)

    def _get_tool_schemas(self) -> list:
        """Return OpenAI function schemas for all LLM-invocable tools."""
        schemas: list = []

        # voicemail_detector
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "voicemail_detector",
                    "description": (
                        "Call this tool if you have detected a voicemail "
                        "system, AFTER hearing the voicemail greeting. "
                        "Use when you detect an automated answering "
                        "machine or voicemail greeting."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "greeting_message": {
                                "type": "string",
                                "description": (
                                    "The voicemail greeting message detected"
                                ),
                            }
                        },
                        "required": [],
                    },
                },
            }
        )

        # end_call
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": (
                        "End the current call and disconnect gracefully. "
                        "Use this tool ONLY when: "
                        "- User explicitly says goodbye "
                        "- User confirms they want to end "
                        "- User says they're done and confirms "
                        "- User thanks you and indicates completion. "
                        "Do NOT use when: "
                        "- User says 'okay' mid-conversation "
                        "- Conversation is still ongoing "
                        "- User is asking a question"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Brief reason why the call is ending"
                                ),
                            }
                        },
                        "required": [],
                    },
                },
            }
        )

        # get_past_calls
        if self.config.get("enable_memory"):
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_past_calls",
                        "description": (
                            "Retrieve the user's recent past call records. "
                            "Returns transcripts and metadata of the last "
                            "few calls. Use when you need to reference "
                            "previous conversations."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "Optional search query to filter "
                                        "past calls"
                                    ),
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )

        # create_followup_or_callback
        if self.config.get("follow_up_enabled"):
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "create_followup_or_callback",
                        "description": (
                            "Schedule a follow-up or callback request. "
                            "Use when the user asks to be called later "
                            "or requests a callback."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "preferred_time": {
                                    "type": "string",
                                    "description": (
                                        "User's preferred callback time"
                                    ),
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )

        # Conversation intelligence tools (when state is available)
        if self._conv_state:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "record_user_info",
                        "description": (
                            "Record information provided by the user. "
                            "Use when user shares: name, preferred time, "
                            "contact method, etc."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": (
                                        "Info type (e.g., 'name', "
                                        "'visit_date', 'contact_method')"
                                    ),
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The value to record",
                                },
                            },
                            "required": ["key", "value"],
                        },
                    },
                }
            )

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "mark_topic_covered",
                        "description": (
                            "Mark a topic as covered after discussing it. "
                            "Use when you've explained a topic from the "
                            "checklist."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic_id": {
                                    "type": "string",
                                    "description": (
                                        "The topic ID from checklist "
                                        "(e.g., 'pricing', 'location')"
                                    ),
                                }
                            },
                            "required": ["topic_id"],
                        },
                    },
                }
            )

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "report_breakdown",
                        "description": (
                            "Report a conversation breakdown when you "
                            "detect confusion. Use when: "
                            "- User asks to repeat (non_understanding) "
                            "- User says you misunderstood "
                            "(misunderstanding) "
                            "- User seems confused"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "breakdown_type": {
                                    "type": "string",
                                    "description": (
                                        "Type: 'non_understanding' or "
                                        "'misunderstanding'"
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "Brief reason for breakdown"
                                    ),
                                },
                            },
                            "required": ["breakdown_type"],
                        },
                    },
                }
            )

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "mark_objective_achieved",
                        "description": (
                            "Mark the conversation objective as achieved. "
                            "Use when user accepts the CTA (e.g., agrees "
                            "to book visit, confirms interest)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "outcome": {
                                    "type": "string",
                                    "description": (
                                        "Outcome: 'primary_success', "
                                        "'fallback_success', or 'declined'"
                                    ),
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )

        return schemas

    def _setup_aggregator_transcript_events(
        self, context_aggregator: LLMContextAggregatorPair,
    ) -> None:
        """Set up transcript handling via aggregator turn events.

        Replaces the deprecated TranscriptProcessor with event handlers
        on the user and assistant aggregators (Pipecat >=0.0.102).
        """

        @context_aggregator.user().event_handler("on_user_turn_stopped")
        async def on_user_turn(aggregator, strategy, message):
            content = message.content
            if not content:
                return

            # Store transcript
            self.user_state.transcript.append({
                "role": "user",
                "content": content,
                "timestamp": str(datetime.now()),
            })

            # Send callback
            msg = Message.create(
                content,
                user=self.user_state.user,
                event=Event.USER_MESSAGE,
            )
            self._send_callback(msg, thread_id=str(self.user_state.thread_id))

            # Reset idle timer and record eval
            self._reset_idle_timer()
            self._record_eval_user_message(content)

            # Detect breakdown and increment turn
            if self._conv_state:
                self._conv_state.increment_turn()
                self._detect_breakdown(content)

            # Notify plugins
            await self.plugins.broadcast_event("on_user_speech", content)

            # Publish to LiveKit
            await self._publish_livekit_transcript(
                role="user", content=content, is_final=True,
            )
            if self._event_bridge:
                await self._publish_transcript("user", content)

        @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn(aggregator, message):
            content = message.content
            if not content:
                return

            # Store transcript
            self.user_state.transcript.append({
                "role": "assistant",
                "content": content,
                "timestamp": str(datetime.now()),
            })

            # Send callback
            msg = Message.create(
                content,
                user=self.agent,
                event=Event.AGENT_MESSAGE,
            )
            self._send_callback(msg, thread_id=str(self.user_state.thread_id))

            # Reset idle timer, record eval, quality tracking
            self._reset_idle_timer()
            await self._record_agent_response_for_quality(content)
            response_llm_latency = self._consume_eval_llm_latency()
            self._record_eval_agent_response(
                content, llm_latency=response_llm_latency,
            )

            # Signal first response sent
            if not self._first_response_sent.is_set():
                self._first_response_sent.set()

            # Notify plugins
            await self.plugins.broadcast_event("on_agent_speech", content)

            # Publish to LiveKit
            await self._publish_livekit_transcript(
                role="assistant", content=content, is_final=True,
            )
            if self._event_bridge:
                await self._publish_transcript("assistant", content)

    async def _setup_transport_events(self) -> None:
        """Set up transport event handlers."""
        is_text_only = self._is_text_only

        # Set up data channel handler (for both TEXT and VOICE modes)
        if self._event_bridge:
            self._setup_data_channel_handler(is_text_only=is_text_only)

        @self._transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant_id):
            self._logger.info(f"Participant {participant_id} joined")
            self.user_state.start_time = datetime.now()

            # Wait for thread_id to become available (resolved async)
            if not self.user_state.thread_id:
                for _ in range(10):  # 10 × 50ms = 500ms max
                    await asyncio.sleep(0.05)
                    if self.user_state.thread_id:
                        break
                if not self.user_state.thread_id:
                    self._logger.warning(
                        "thread_id still not available after 500ms"
                    )

            # Notify plugins
            await self.plugins.broadcast_event("on_call_start")

            # Send first message (TTS for voice, data channel for text)
            first_message = self.config.get(
                "first_message",
                "Hello! How can I help you today?",
            )
            first_message = self.prompt_manager._replace_template_params(first_message)

            if is_text_only:
                # TEXT-only: Send via data channel instead of TTS
                await self._send_data_response(first_message)
            else:
                await self.task.queue_frame(TTSSpeakFrame(first_message))

            # Load previous chat context if available
            await self._load_chat_context()

            # Start idle timeout monitor
            await self._start_idle_monitor()

            # Send start callback
            msg = Message.create(
                "CALL STARTED",
                role="system",
                event=Event.TASK_START,
            )
            self._send_callback(msg, thread_id=str(self.user_state.thread_id))

        @self._transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant_id):
            self._logger.info(f"Participant {participant_id} disconnected")

            # CRITICAL: Shutdown state observer IMMEDIATELY to prevent Rust SDK panics
            if self._livekit_state_observer:
                self._livekit_state_observer.shutdown()

            self.user_state.end_time = datetime.now()
            self.user_state.end_reason = "Customer Ended The Call"

            # Persist quality metrics and trigger post-call
            await self._persist_session_quality_metrics(
                reason="participant_disconnect"
            )
            await self._ensure_post_call_triggered(
                reason="participant_disconnect"
            )

            # Notify plugins before shared cleanup clears plugin resources
            await self.plugins.broadcast_event("on_call_end")

            # Cleanup shared resources
            await self._cleanup_runtime_resources()

            # Send end callback
            msg = Message.create(
                "CALL ENDED",
                role="system",
                event=Event.TASK_END,
            )
            self._send_callback(msg, thread_id=str(self.user_state.thread_id))

            # Stop pipeline
            await self.task.queue_frame(EndFrame())

    async def _load_chat_context(self) -> None:
        """
        Load previous chat context into the LLM context if available.

        Extracts chat context from user_state.extra_data or config and appends
        it to the LLM context as a system message with previous conversation history.
        """
        try:
            chat_context_data = None

            # Try to get chat context from user_state.extra_data
            if self.user_state and self.user_state.extra_data:
                if isinstance(self.user_state.extra_data, dict):
                    chat_context_data = (
                        self.user_state.extra_data.get("data", {})
                        .get("pre_call_result", {})
                        .get("chat_context")
                    )
                elif isinstance(self.user_state.extra_data, str):
                    chat_context_data = self.user_state.extra_data

            # Fallback to config if not found in extra_data
            if not chat_context_data and self.config:
                chat_context_data = self.config.get("chat_context")

            if not chat_context_data:
                self._logger.debug("No previous chat context to load")
                return

            self._logger.info("Loading previous chat context into LLM context")

            # Append chat context as a system message
            system_message = {
                "role": "system",
                "content": f"\n[previous conversations with user]\n{chat_context_data}",
            }

            if self.task:
                await self.task.queue_frame(
                    LLMMessagesAppendFrame([system_message], run_llm=False)
                )
                self._logger.info("Previous chat context loaded successfully")

        except Exception as e:
            self._logger.warning(f"Failed to load previous chat context: {e}")

    async def get_transport(self) -> BaseTransport:
        """Get transport instance."""
        return await self.transport_manager.get_transport()

    async def get_docs(self, query: Any, params: Any = None) -> str:
        """
        Get documents from knowledge base.

        Args:
            query: Search query (string or dict)
            params: Optional function call params

        Returns:
            JSON string with results
        """
        # Handle dict or string query
        if isinstance(query, dict):
            query_str = query.get("query", "")
        else:
            query_str = str(query)

        if not query_str:
            return json.dumps({"error": "Empty query"})

        # Wait for KB to be ready (with timeout)
        try:
            await asyncio.wait_for(self._kb_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._logger.warning("KB not ready, attempting query anyway")

        # Get KB manager
        kb_manager = getattr(self, "_kb_manager", None)
        if kb_manager is None:
            return json.dumps({"error": "Knowledge base not initialized"})

        try:
            result = await kb_manager.get_docs(
                params=params,
                query=query_str,
                user_state=self.user_state,
            )
            return result
        except Exception as e:
            self._logger.error(f"KB query failed: {e}")
            return json.dumps({"error": str(e)})

    async def end_call(self, reason: str = "completed") -> str:
        """End the current call gracefully."""
        self._logger.info(f"Ending call: {reason}")
        self._is_shutting_down = True

        try:
            # Shutdown state observer FIRST to prevent Rust SDK panics
            if self._livekit_state_observer:
                self._livekit_state_observer.shutdown()

            # Persist quality metrics before cleanup
            await self._persist_session_quality_metrics(reason=reason)

            # Trigger post-call workflow
            await self._ensure_post_call_triggered(reason=reason)

            # Notify plugins before shared cleanup clears plugin resources
            await self.plugins.broadcast_event("on_call_end")

            # Shared cleanup (idle monitor, KB, plugins) + handler-specific
            await self._cleanup_runtime_resources()

            # Stop pipeline
            if self.task:
                await self.task.queue_frame(EndFrame())
                await asyncio.sleep(0.5)

            # Send end callback
            from super.core.context.schema import Event, Message

            self._send_callback(
                Message.create(
                    "CALL ENDED", role="system", event=Event.TASK_END
                ),
                thread_id=str(self.user_state.thread_id),
            )

            self._logger.info("Call ended successfully")
            return f"Call ended: {reason}"

        except Exception as e:
            self._logger.error(f"Error ending call: {e}")
            return f"Error: {e}"

    async def end_ongoing_agent(self) -> None:
        """Stop the agent (compatibility method)."""
        await self.end_call("agent_stopped")

    # ─── SharedVoiceMixin hook implementations ────────────────────────

    async def _send_speech(self, text: str) -> None:
        """Pipecat implementation: push TTSSpeakFrame or send via data channel."""
        if self._is_text_only:
            # TEXT-only: send via data channel
            await self._send_data_response(text)
        elif self.task:
            await self.task.queue_frame(TTSSpeakFrame(text))

    async def _disconnect_participant(self) -> None:
        """Pipecat implementation: end pipeline and disconnect transport."""
        if self.task:
            await self.task.queue_frame(EndFrame())
            await asyncio.sleep(0.5)
        if self._transport and hasattr(self._transport, "disconnect"):
            await self._transport.disconnect()

    def _get_conversation_messages(self) -> list:
        """Pipecat implementation: get messages from OpenAI context."""
        if self._context_aggregator:
            ctx = getattr(self._context_aggregator, "_context", None)
            if ctx and hasattr(ctx, "get_messages"):
                return ctx.get_messages()
        return []

    async def _append_context_message(
        self, role: str, content: str, run_llm: bool = False
    ) -> None:
        """Pipecat implementation: push LLMMessagesAppendFrame."""
        if self.task:
            message = {"role": role, "content": content}
            await self.task.queue_frame(
                LLMMessagesAppendFrame([message], run_llm=run_llm)
            )

    def _is_agent_speaking(self) -> bool:
        """Pipecat implementation: check whether agent is actively responding."""
        observer = getattr(self, "_livekit_state_observer", None)
        if observer and hasattr(observer, "get_current_state"):
            try:
                observed_state = observer.get_current_state()
                if observed_state:
                    self._agent_state = observed_state
            except Exception:
                # Fall back to local state cache when observer read fails
                pass

        return self._agent_state in ("speaking", "thinking")

    async def _handler_specific_cleanup(self) -> None:
        """Pipecat-specific cleanup: services and transport."""
        if self.services:
            await self.services.cleanup()
        if self._transport and hasattr(self._transport, "disconnect"):
            await self._transport.disconnect()
            self._transport = None

    # ─── Tool registration and handlers ────────────────────────────

    def _register_tool_handlers(self, llm: Any) -> None:
        """Register all tool handler functions on the LLM service."""
        tool_calling = self.config.get("tool_calling", True)
        if not tool_calling:
            return

        # Always register core tools
        llm.register_function(
            "voicemail_detector", self._tool_voicemail_detector
        )
        llm.register_function("end_call", self._tool_end_call)

        if self.config.get("enable_memory"):
            llm.register_function("get_past_calls", self._tool_get_past_calls)

        if self.config.get("follow_up_enabled"):
            llm.register_function(
                "create_followup_or_callback",
                self._tool_create_followup,
            )

        # Conversation intelligence tools
        if self._conv_state:
            llm.register_function(
                "record_user_info", self._tool_record_user_info
            )
            llm.register_function(
                "mark_topic_covered", self._tool_mark_topic_covered
            )
            llm.register_function(
                "report_breakdown", self._tool_report_breakdown
            )
            llm.register_function(
                "mark_objective_achieved",
                self._tool_mark_objective_achieved,
            )

        self._logger.info(
            "[TOOLS] Registered tool handlers on LLM service"
        )

    def _record_tool_call(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """Store tool call telemetry for post-call eval."""
        try:
            records = self._ensure_eval_records()
            if not records:
                return
            records["tool_calls"].append(
                {
                    "sequence_id": self._next_eval_sequence(),
                    "tool_name": tool_name,
                    "args": args or {},
                    "result": str(result)[:500] if result else None,
                    "status": status,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            self._logger.debug(f"Failed to record tool call: {e}")

    async def _tool_voicemail_detector(
        self, params: FunctionCallParams,
    ) -> None:
        """Handle voicemail detection - trigger post-call and end call."""
        greeting = params.arguments.get("greeting_message", "")
        self._logger.info(
            f"[TOOL] voicemail_detector called: {greeting[:100]}"
        )

        # Say voicemail message via TTS
        await self._send_speech(
            "I'll call back later. Thank you."
        )

        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}

        if self.user_state.extra_data.get("call_type") != "sdk":
            self.user_state.call_status = CallStatusEnum.VOICEMAIL
            self.user_state.end_time = datetime.utcnow()

            res = await build_call_result(self.user_state)
            await trigger_post_call(
                user_state=self.user_state, res=res
            )
            self.user_state.extra_data["post_call_triggered"] = True

        self._record_tool_call(
            "voicemail_detector",
            {"greeting_message": greeting},
            {"status": "voicemail_detected"},
        )

        await asyncio.sleep(3)
        await self._handle_tool_end_call("voicemail_detected")

        await params.result_callback(
            json.dumps({"status": "voicemail_detected"})
        )

    async def _tool_end_call(
        self, params: FunctionCallParams,
    ) -> None:
        """Handle LLM-invoked end call."""
        reason = params.arguments.get("reason", "user_goodbye")
        self._logger.info(f"[TOOL] end_call called: {reason}")

        # Capture conversation quality metrics
        call_summary = None
        if self._conv_state:
            self._conv_state.task_id = (
                self.user_state.task_id or ""
            )
            self._conv_state.session_id = self._session_id or ""
            self._conv_state.agent_handle = self.config.get(
                "agent_handle", ""
            )
            self._logger.info(self._conv_state.get_quality_log_line())
            call_summary = self._conv_state.get_call_summary_dict(
                end_reason=reason
            )
            self._conv_state.persist_quality_metrics(reason=reason)
            self._logger.info(
                f"\n{self._conv_state.build_call_summary(end_reason=reason)}"
            )

        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}

        if call_summary:
            self.user_state.extra_data["conversation_summary"] = (
                call_summary
            )

        if self.user_state.extra_data.get("call_type") != "sdk":
            self.user_state.call_status = CallStatusEnum.COMPLETED
            self.user_state.end_time = datetime.utcnow()

            res = await build_call_result(self.user_state)
            await trigger_post_call(
                user_state=self.user_state, res=res
            )
            self.user_state.extra_data["post_call_triggered"] = True
            self._logger.info(
                f"{'=' * 80}\n[END_CALL_TOOL] Post call triggered\n"
                f"{'=' * 80}"
            )

        self._record_tool_call(
            "end_call", {"reason": reason}, {"status": "ending_call"}
        )

        await self._handle_tool_end_call(reason)

        await params.result_callback(
            json.dumps(
                {
                    "status": "ending_call",
                    "message": (
                        "Call ending gracefully. "
                        "Say a brief goodbye to the user."
                    ),
                }
            )
        )

    async def _tool_get_past_calls(
        self, params: FunctionCallParams,
    ) -> None:
        """Retrieve past call records for the user."""
        query = params.arguments.get("query", "")
        self._logger.info(f"[TOOL] get_past_calls called: {query}")

        chat_context_data = None
        if self.user_state and self.user_state.extra_data:
            if isinstance(self.user_state.extra_data, dict):
                chat_context_data = (
                    self.user_state.extra_data.get("data", {})
                    .get("pre_call_result", {})
                    .get("chat_context")
                )
            elif isinstance(self.user_state.extra_data, str):
                chat_context_data = self.user_state.extra_data

        self._record_tool_call(
            "get_past_calls",
            {"query": query},
            {"call_details": bool(chat_context_data)},
        )

        if chat_context_data:
            await params.result_callback(
                json.dumps({"call_details": chat_context_data})
            )
        else:
            await params.result_callback(
                json.dumps(
                    {"call_details": "unable to get recent calls data"}
                )
            )

    async def _tool_create_followup(
        self, params: FunctionCallParams,
    ) -> None:
        """Schedule a follow-up or callback request."""
        preferred_time = params.arguments.get("preferred_time", "")
        self._logger.info(
            f"[TOOL] create_followup_or_callback: {preferred_time}"
        )

        # Inject callback guidelines into context
        if self.task:
            callback_msg = {
                "role": "system",
                "content": (
                    "\n[callback and followup guidelines]\n"
                    "1. Ask user about the time when they will be "
                    "available to initiate a call.\n"
                    "2. If user has already provided a time, just say "
                    "ok I will call you back at the given time."
                ),
            }
            await self.task.queue_frame(
                LLMMessagesAppendFrame([callback_msg], run_llm=False)
            )

        self._record_tool_call(
            "create_followup_or_callback",
            {"preferred_time": preferred_time},
            {"status": "callback_initiated"},
        )

        await params.result_callback(
            json.dumps(
                {
                    "instructions": (
                        "Ask user about the time when they will be "
                        "available to initiate a call"
                    ),
                    "response": (
                        "If user has already provided a time, say ok "
                        "I will call you back at the given time"
                    ),
                }
            )
        )

    async def _tool_record_user_info(
        self, params: FunctionCallParams,
    ) -> None:
        """Record user-provided information."""
        key = params.arguments.get("key", "")
        value = params.arguments.get("value", "")
        self._logger.info(f"[TOOL] record_user_info: {key}={value}")

        if self._conv_state and key and value:
            self._conv_state.record_user_info_item(key, value)
            self._record_tool_call(
                "record_user_info",
                {"key": key, "value": value},
                {"status": "recorded"},
            )
            await params.result_callback(
                json.dumps(
                    {"status": "recorded", "key": key, "value": value}
                )
            )
        else:
            self._record_tool_call(
                "record_user_info",
                {"key": key, "value": value},
                {"status": "no_state"},
            )
            await params.result_callback(
                json.dumps({"status": "no_state"})
            )

    async def _tool_mark_topic_covered(
        self, params: FunctionCallParams,
    ) -> None:
        """Mark a checklist topic as covered."""
        topic_id = params.arguments.get("topic_id", "")
        self._logger.info(f"[TOOL] mark_topic_covered: {topic_id}")

        if self._conv_state and topic_id:
            self._conv_state.mark_block_delivered(topic_id)
            progress = self._conv_state.get_delivery_progress()
            self._logger.info(
                f"[TOPIC] Marked covered: {topic_id} "
                f"({progress['delivered']}/{progress['total']})"
            )
            self._record_tool_call(
                "mark_topic_covered",
                {"topic_id": topic_id},
                {"status": "marked", "progress": progress},
            )
            await params.result_callback(
                json.dumps(
                    {
                        "status": "marked",
                        "topic": topic_id,
                        "progress": progress,
                    }
                )
            )
        else:
            self._record_tool_call(
                "mark_topic_covered",
                {"topic_id": topic_id},
                {"status": "no_state"},
            )
            await params.result_callback(
                json.dumps({"status": "no_state"})
            )

    async def _tool_report_breakdown(
        self, params: FunctionCallParams,
    ) -> None:
        """Report a conversation breakdown for repair handling."""
        breakdown_type = params.arguments.get(
            "breakdown_type", "non_understanding"
        )
        reason = params.arguments.get("reason", "")
        self._logger.info(
            f"[TOOL] report_breakdown: {breakdown_type} ({reason})"
        )

        if not self._conv_state:
            self._record_tool_call(
                "report_breakdown",
                {"breakdown_type": breakdown_type, "reason": reason},
                {"status": "no_state"},
            )
            await params.result_callback(
                json.dumps({"status": "no_state"})
            )
            return

        valid_types = ["non_understanding", "misunderstanding"]
        if breakdown_type not in valid_types:
            breakdown_type = "non_understanding"

        if self._conv_state.in_repair:
            self._conv_state.increment_repair_retry()
        else:
            self._conv_state.enter_repair(breakdown_type)

        repair_block = self._conv_state.get_repair_block()
        repair_strategy = (
            repair_block.content if repair_block else None
        )

        self._logger.info(
            f"[BREAKDOWN] {breakdown_type}, "
            f"retry={self._conv_state.repair_retry_count}"
        )

        self._record_tool_call(
            "report_breakdown",
            {"breakdown_type": breakdown_type, "reason": reason},
            {"status": "repair_mode"},
        )

        await params.result_callback(
            json.dumps(
                {
                    "status": "repair_mode",
                    "breakdown_type": breakdown_type,
                    "retry_count": self._conv_state.repair_retry_count,
                    "suggested_strategy": repair_strategy,
                    "instruction": (
                        "Use the suggested strategy to repair "
                        "the conversation"
                    ),
                }
            )
        )

    async def _tool_mark_objective_achieved(
        self, params: FunctionCallParams,
    ) -> None:
        """Mark conversation objective as achieved."""
        outcome = params.arguments.get("outcome", "primary_success")
        self._logger.info(f"[TOOL] mark_objective_achieved: {outcome}")

        if not self._conv_state:
            self._record_tool_call(
                "mark_objective_achieved",
                {"outcome": outcome},
                {"status": "no_state"},
            )
            await params.result_callback(
                json.dumps({"status": "no_state"})
            )
            return

        self._conv_state.objective_achieved = outcome in [
            "primary_success",
            "fallback_success",
        ]
        self._conv_state.objective_outcome = outcome

        self._logger.info(f"[OBJECTIVE] Marked: {outcome}")

        self._record_tool_call(
            "mark_objective_achieved",
            {"outcome": outcome},
            {
                "status": "recorded",
                "objective_achieved": (
                    self._conv_state.objective_achieved
                ),
            },
        )

        await params.result_callback(
            json.dumps(
                {
                    "status": "recorded",
                    "objective_achieved": (
                        self._conv_state.objective_achieved
                    ),
                    "outcome": outcome,
                }
            )
        )

    # ─── Conversation state accessor override ───────────────────────

    def _get_conversation_state(self) -> Optional[Any]:
        """Override: return handler-level _conv_state for Pipecat."""
        return self._conv_state

    # ─── Breakdown detection from user messages ──────────────────────

    # Non-understanding indicators
    _BREAKDOWN_NON_UNDERSTANDING = [
        "what", "huh", "sorry", "repeat", "again",
        "samajh nahi", "kya bola", "pardon",
        "didn't catch", "can't hear",
    ]

    # Misunderstanding indicators
    _BREAKDOWN_MISUNDERSTANDING = [
        "no i meant", "that's not what", "i said",
        "maine kaha", "galat", "wrong", "not that",
        "actually i want",
    ]

    def _detect_breakdown(self, user_text: str) -> None:
        """Detect conversation breakdown from user message text."""
        if not self._conv_state:
            return

        text_lower = user_text.lower()

        if any(p in text_lower for p in self._BREAKDOWN_NON_UNDERSTANDING):
            if self._conv_state.in_repair:
                self._conv_state.increment_repair_retry()
            else:
                self._conv_state.enter_repair("non_understanding")
            self._logger.info(
                f"[BREAKDOWN] non_understanding detected, "
                f"retry={self._conv_state.repair_retry_count}"
            )
            return

        if any(p in text_lower for p in self._BREAKDOWN_MISUNDERSTANDING):
            if self._conv_state.in_repair:
                self._conv_state.increment_repair_retry()
            else:
                self._conv_state.enter_repair("misunderstanding")
            self._logger.info(
                f"[BREAKDOWN] misunderstanding detected, "
                f"retry={self._conv_state.repair_retry_count}"
            )
            return

        # If in repair and user gives substantive response, exit
        if self._conv_state.in_repair and len(text_lower.split()) > 3:
            self._conv_state.exit_repair()
            self._logger.info(
                "[BREAKDOWN] repair successful, resuming normal flow"
            )

    # ─── Token-aware context truncation ──────────────────────────────

    def _truncate_context_to_limit(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit within context limit.

        Preserves system messages + most recent conversation turns.
        """
        if not messages or not self._effective_context_limit:
            return messages

        sys_msgs = [
            m for m in messages if m.get("role") == "system"
        ]
        conv_msgs = [
            m
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]

        def msg_tokens(m: Dict[str, str]) -> int:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            return estimate_tokens(str(content))

        sys_tokens = sum(msg_tokens(m) for m in sys_msgs)
        conv_tokens = sum(msg_tokens(m) for m in conv_msgs)
        total_tokens = sys_tokens + conv_tokens

        # Within limit: apply turn-based pruning as secondary limit
        if total_tokens <= self._effective_context_limit:
            max_msgs = self._max_context_turns * 2
            if len(conv_msgs) > max_msgs:
                conv_msgs = conv_msgs[-max_msgs:]
            return (sys_msgs[-1:] if sys_msgs else []) + conv_msgs

        self._logger.warning(
            f"Context overflow: {total_tokens} tokens > "
            f"{self._effective_context_limit} limit"
        )

        # Keep only latest system message
        sys_msgs = sys_msgs[-1:] if sys_msgs else []
        sys_tokens = sum(msg_tokens(m) for m in sys_msgs)

        # Budget for conversation
        conv_budget = (
            self._effective_context_limit - sys_tokens - 100
        )

        # Remove oldest messages until within budget
        while (
            conv_msgs
            and sum(msg_tokens(m) for m in conv_msgs) > conv_budget
        ):
            conv_msgs.pop(0)

        # Apply max_turns limit
        max_msgs = self._max_context_turns * 2
        if len(conv_msgs) > max_msgs:
            conv_msgs = conv_msgs[-max_msgs:]

        final_tokens = sys_tokens + sum(
            msg_tokens(m) for m in conv_msgs
        )
        self._logger.info(
            f"Context truncated: {total_tokens} -> {final_tokens} "
            f"tokens ({len(conv_msgs)} conversation messages)"
        )

        return sys_msgs + conv_msgs

    # ─── Data channel protocol ─────────────────────────────────────

    def _get_room(self) -> Any:
        """Get LiveKit room from job context (Pipecat runs on LiveKit transport)."""
        try:
            from livekit.agents import get_job_context

            ctx = get_job_context()
            if ctx and ctx.room:
                return ctx.room
        except Exception:
            pass
        return None

    async def _publish_data_to_client(
        self,
        data: Dict[str, Any],
        topic: str,
        reliable: bool = True,
    ) -> bool:
        """Publish data to client using room publish_data first, event bridge fallback."""
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

        room = self._get_room()
        if room and getattr(room, "local_participant", None):
            try:
                await room.local_participant.publish_data(
                    payload,
                    reliable=reliable,
                    topic=topic,
                )
                return True
            except Exception as e:
                self._logger.warning(
                    f"[DATA_CHANNEL] Room publish_data failed (topic={topic}): {e}"
                )

        if self._event_bridge:
            try:
                published = await self._event_bridge.publish_data(
                    data=data,
                    topic=topic,
                    reliable=reliable,
                )
                return bool(published)
            except Exception as e:
                self._logger.error(
                    f"[DATA_CHANNEL] Event bridge publish failed (topic={topic}): {e}"
                )

        self._logger.error(
            f"[DATA_CHANNEL] No publish path available for topic={topic}"
        )
        return False

    async def _send_streaming_text(
        self,
        text_chunk: str,
        message_id: str | None = None,
        chunk_index: int = 0,
        block_data: dict | None = None,
    ) -> None:
        """Send streaming text chunk via TOPIC_LK_STREAM topic."""
        import uuid as _uuid

        try:
            if not text_chunk:
                return

            pilot = "multi-ai"
            focus = "my_space"
            if block_data:
                pilot = block_data.get("pilot", pilot)
                data = block_data.get("data", {})
                inner_data = data.get("data", {})
                focus = inner_data.get("focus", focus)

            stream_block = {
                "event": "stream",
                "id": message_id or str(_uuid.uuid4())[:12],
                "pilot": pilot,
                "data": {
                    "content": text_chunk,
                    "chunk_index": chunk_index,
                    "is_final": False,
                    "focus": focus,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            success = await self._publish_data_to_client(
                data=stream_block,
                topic=self.TOPIC_LK_STREAM,
                reliable=True,
            )
            if not success:
                self._logger.error("[TEXT_MODE] Failed to publish streaming text")

        except Exception as e:
            self._logger.error(f"[TEXT_MODE] Failed to send streaming text: {e}")

    async def _send_data_response(
        self,
        response_text: str,
        block_data: dict | None = None,
        message_id: str | None = None,
        cards: dict | None = None,
    ) -> None:
        """Send final response as JSON block via data channel."""
        try:
            if not response_text:
                return

            response_block = self._build_response_block(
                response_text, block_data, message_id, cards
            )

            success = await self._publish_data_to_client(
                data=response_block,
                topic=self.TOPIC_LK_CHAT,
                reliable=True,
            )

            if success:
                self._logger.info(
                    f"[BLOCK_SENT] Published to {self.TOPIC_LK_CHAT} "
                    f"(id={message_id}): text={response_text[:80]}..."
                )
            else:
                self._logger.error("[TEXT_MODE] Failed to publish data response")

            self.user_state.transcript.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": str(datetime.now()),
                    "message_id": message_id,
                }
            )

            msg = Message.create(
                response_text,
                user=self.agent,
                event=Event.AGENT_MESSAGE,
            )
            self._send_callback(msg, thread_id=str(self.user_state.thread_id))

            await self.plugins.broadcast_event("on_agent_speech", response_text)

        except Exception as e:
            self._logger.error(f"[TEXT_MODE] Failed to send data response: {e}")

    async def _generate_and_send_text_response(
        self,
        user_message: str,
        block_data: dict | None = None,
        stream_response: bool = True,
        processed_files: list | None = None,
    ) -> None:
        """
        Generate LLM response with streaming text and final data block.

        Uses Pipecat's LLM service (from LazyServiceFactory) for generation.
        Streaming chunks sent via TOPIC_LK_STREAM, final via TOPIC_LK_CHAT.
        """
        import uuid as _uuid

        message_id = str(_uuid.uuid4())[:12]

        try:
            self._logger.info(
                f"[TEXT_MODE] Generating response for: {user_message[:100]}... "
                f"(id={message_id})"
            )

            if not self.services:
                self._logger.error("[TEXT_MODE] No services available")
                return

            llm = await self.services.get_llm()
            if not llm:
                self._logger.error("[TEXT_MODE] No LLM available")
                return

            # Build messages from current context + user message
            messages = self._get_conversation_messages()
            if not messages:
                system_prompt = self.prompt_manager._create_assistant_prompt()
                messages = [{"role": "system", "content": system_prompt}]

            # Add file references if present
            if processed_files:
                file_refs = []
                for f in processed_files:
                    file_refs.append(
                        f"[Attached: {f.get('name', 'file')} "
                        f"({f.get('media_type', 'unknown')})]"
                    )
                user_content = user_message + "\n" + "\n".join(file_refs)
            else:
                user_content = user_message

            messages.append({"role": "user", "content": user_content})

            # Apply token-aware context truncation
            if self._effective_context_limit:
                messages = self._truncate_context_to_limit(messages)

            # Use chat completion via Pipecat LLM
            ctx = LLMContext(messages=messages)
            response_text = ""
            chunk_index = 0

            stream = llm.chat(chat_ctx=ctx)
            try:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        chunk_content = chunk.delta.content
                        response_text += chunk_content

                        if stream_response and chunk_content:
                            await self._send_streaming_text(
                                chunk_content,
                                message_id=message_id,
                                chunk_index=chunk_index,
                                block_data=block_data,
                            )
                            chunk_index += 1
            finally:
                await stream.aclose()

            self._logger.info(
                f"[TEXT_MODE] Generated ({len(response_text)} chars, "
                f"{chunk_index} chunks): {response_text[:100]}..."
            )

            if response_text:
                await self._send_data_response(
                    response_text, block_data, message_id
                )

                # Update context with response
                await self._append_context_message(
                    "assistant", response_text, run_llm=False
                )

                # Record eval data
                await self._record_agent_response_for_quality(response_text)
                response_llm_latency = self._consume_eval_llm_latency()
                self._record_eval_agent_response(
                    response_text, llm_latency=response_llm_latency
                )

        except Exception as e:
            self._logger.error(f"[TEXT_MODE] Error generating text response: {e}")
            import traceback

            self._logger.error(traceback.format_exc())

    def _setup_data_channel_handler(self, is_text_only: bool = False) -> None:
        """
        Set up data channel and text stream handlers for custom protocols.

        Topics handled:
        - user_action: User interactions with UI cards
        - superkik.* topics: Custom SuperKik protocol messages
        - lk.chat: Text input in TEXT-ONLY mode (fallback for text streams)
        """
        self._logger.info(
            "[DATA_CHANNEL] Setting up topic listeners via event_bridge"
        )

        if not self._event_bridge:
            self._logger.warning(
                "[DATA_CHANNEL] No event_bridge, skipping topic registration"
            )
            return

        # Handler for user_action topic
        async def handle_user_action_event(event: Dict[str, Any]) -> None:
            try:
                data = event.get("data", {})
                self._logger.info(
                    f"[DATA_CHANNEL] user_action: {str(data)[:200]}..."
                )
                await self._handle_user_action(data)
            except Exception as e:
                self._logger.error(
                    f"[DATA_CHANNEL] Error handling user_action: {e}"
                )

        # Handler for lk.chat topic (TEXT-ONLY mode)
        async def handle_lk_chat_event(event: Dict[str, Any]) -> None:
            try:
                data = event.get("data", {})
                raw_data = event.get("raw_data", b"")

                if isinstance(raw_data, bytes) and raw_data:
                    raw_message = raw_data.decode("utf-8")
                elif data.get("raw"):
                    raw_message = data.get("text", "")
                else:
                    raw_message = json.dumps(data) if data else ""

                self._logger.info(
                    f"[TEXT_STREAM] lk.chat received: {raw_message[:200]}..."
                )

                if not raw_message or not raw_message.strip():
                    return

                message, block_data, files = self._parse_block_message(
                    raw_message
                )

                if not message or not message.strip():
                    self._logger.warning(
                        "[TEXT_STREAM] No content extracted, skipping"
                    )
                    return

                # Process files if present
                processed_files = None
                if files:
                    processed_files = await self._process_attached_files(files)
                    if processed_files:
                        self.user_state.files.extend(processed_files)

                self.user_state.transcript.append(
                    {
                        "role": "user",
                        "content": message,
                        "timestamp": str(datetime.now()),
                    }
                )

                msg = Message.create(
                    message,
                    user=self.user_state.user,
                    event=Event.USER_MESSAGE,
                )
                self._send_callback(
                    msg, thread_id=str(self.user_state.thread_id)
                )

                self._reset_idle_timer()
                self._record_eval_user_message(message)

                asyncio.create_task(
                    self.plugins.broadcast_event("on_user_speech", message)
                )

                asyncio.create_task(
                    self._generate_and_send_text_response(
                        message, block_data, processed_files=processed_files
                    )
                )

            except Exception as e:
                self._logger.error(
                    f"[TEXT_STREAM] Error handling lk.chat: {e}"
                )
                import traceback

                self._logger.error(traceback.format_exc())

        # Handler for superkik.* topics
        async def handle_all_data_event(event: Dict[str, Any]) -> None:
            try:
                topic = event.get("topic", "")
                data = event.get("data", {})

                if topic.startswith("superkik."):
                    self._logger.info(
                        f"[DATA_CHANNEL] superkik on '{topic}': "
                        f"{str(data)[:200]}..."
                    )
                    await self._handle_superkik_message(topic, data)
            except Exception as e:
                self._logger.error(
                    f"[DATA_CHANNEL] Error in data handler: {e}"
                )

        # Register topic-specific listener for user_action
        self._event_bridge.on_topic("user_action", handle_user_action_event)

        # Register global data listener for superkik.* topics
        self._event_bridge.on_data_received(handle_all_data_event)

        if is_text_only:
            self._event_bridge.on_topic(
                self.TOPIC_LK_CHAT, handle_lk_chat_event
            )
            self._event_bridge.register_text_stream_handlers(
                topics=[self.TOPIC_LK_CHAT, "user_action"]
            )
            self._logger.info(
                f"[DATA_CHANNEL] Registered '{self.TOPIC_LK_CHAT}' "
                "(TEXT-ONLY mode)"
            )
        else:
            # VOICE mode: Register for file attachments
            async def handle_voice_mode_attachments(
                event: Dict[str, Any],
            ) -> None:
                try:
                    data = event.get("data", {})
                    raw_data = event.get("raw_data", b"")

                    if isinstance(raw_data, bytes) and raw_data:
                        raw_message = raw_data.decode("utf-8")
                    elif data.get("raw"):
                        raw_message = data.get("text", "")
                    else:
                        raw_message = json.dumps(data) if data else ""

                    if not raw_message or not raw_message.strip():
                        return

                    message, block_data, files = self._parse_block_message(
                        raw_message
                    )

                    if files:
                        self._logger.info(
                            f"[VOICE_MODE] {len(files)} file attachment(s)"
                        )
                        processed_files = await self._process_attached_files(
                            files
                        )
                        if processed_files:
                            self.user_state.files.extend(processed_files)

                            # Inject file context into LLM
                            file_descs = []
                            for f in processed_files:
                                name = f.get("name", "unknown")
                                mtype = f.get("media_type", "unknown")
                                file_descs.append(f"- {name} ({mtype})")

                            file_ctx = (
                                f"[ATTACHED FILES]\n"
                                f"The user has attached:\n"
                                f"{chr(10).join(file_descs)}\n"
                                f"Use appropriate tools to process these "
                                f"files based on the user's request."
                            )
                            await self._append_context_message(
                                "system", file_ctx, run_llm=False
                            )

                except Exception as e:
                    self._logger.error(
                        f"[VOICE_MODE] Error handling attachment: {e}"
                    )

            self._event_bridge.on_topic(
                self.TOPIC_LK_CHAT, handle_voice_mode_attachments
            )
            self._event_bridge.register_text_stream_handlers(
                topics=[self.TOPIC_LK_CHAT, "user_action"]
            )
            self._logger.info(
                f"[DATA_CHANNEL] Registered file attachment listener "
                f"(VOICE mode)"
            )

        self._logger.info(
            "[DATA_CHANNEL] Topic listeners registered via event_bridge"
        )

    async def _handle_user_action(self, data: Dict[str, Any]) -> None:
        """Handle user action messages from frontend."""
        action = data.get("action", "")
        self._logger.info(f"[USER_ACTION] Received: {action}")

    async def _handle_superkik_message(
        self, topic: str, data: Dict[str, Any]
    ) -> None:
        """Handle SuperKik protocol messages."""
        self._logger.debug(
            f"[SUPERKIK] Received on {topic}: {str(data)[:100]}..."
        )

    def set_event_bridge(self, event_bridge: Any) -> None:
        """
        Set LiveKit event bridge for transcript publishing and state management.

        Args:
            event_bridge: LiveKitEventBridge instance
        """
        self._event_bridge = event_bridge
        self._logger.info("Event bridge set")

    async def set_agent_state(self, state: str) -> bool:
        """
        Set agent state via event bridge.

        Valid states: "initializing", "listening", "thinking", "speaking"

        Args:
            state: New agent state

        Returns:
            True if state was updated
        """
        self._agent_state = state
        if self._event_bridge:
            return await self._event_bridge.set_agent_state(state)
        return True

    async def listening(self) -> None:
        """Set agent state to listening."""
        await self.set_agent_state("listening")

    async def thinking(self) -> None:
        """Set agent state to thinking."""
        await self.set_agent_state("thinking")

    async def speaking(self) -> None:
        """Set agent state to speaking."""
        await self.set_agent_state("speaking")

    async def initializing(self) -> None:
        """Set agent state to initializing."""
        await self.set_agent_state("initializing")

    def set_streaming_parser_enabled(self, enabled: bool) -> bool:
        """Enable or disable the streaming text parser at runtime.

        The streaming parser enables TTS to start while LLM is still generating,
        reducing latency by 500-1500ms.

        Args:
            enabled: Whether to enable streaming parsing.

        Returns:
            True if the plugin was found and updated, False otherwise.
        """
        result = self.plugins.set_plugin_enabled("streaming", enabled)
        if result:
            self._logger.info(
                f"Streaming parser {'enabled' if enabled else 'disabled'}"
            )
        return result

    def get_streaming_parser_metrics(self) -> Optional[dict]:
        """Get metrics from the streaming text parser.

        Returns:
            Dict with metrics (chunk_count, ttfc, etc.) or None if not available.
        """
        plugin = self.plugins.get_plugin("streaming")
        if plugin:
            return plugin.get_metrics()
        return None

    async def _publish_transcript(
        self,
        role: str,
        content: str,
        is_final: bool = True,
    ) -> bool:
        """
        Publish transcript to LiveKit via event bridge.

        Args:
            role: Speaker role ("user" or "assistant")
            content: Transcript text
            is_final: Whether this is a final transcript

        Returns:
            True if published successfully
        """
        if not self._event_bridge or not content:
            return False

        try:
            result = await self._event_bridge.emit_transcript(
                role=role,
                content=content,
                is_final=is_final,
            )
            return result is not None
        except Exception as e:
            self._logger.error(f"Transcript publish failed: {e}")
            return False

    async def _publish_livekit_transcript(
        self,
        role: str,
        content: str,
        is_final: bool = True,
    ) -> bool:
        """
        Publish transcript via native LiveKit transcription API.

        Uses participant.publish_transcription() which the frontend
        useVoiceAssistant hook automatically handles.

        Args:
            role: Speaker role ("user" or "assistant")
            content: Transcript text
            is_final: Whether this is a final transcript

        Returns:
            True if published successfully
        """
        if not content:
            return False

        try:
            # Get participant from state observer or job context
            participant = None

            if self._livekit_state_observer:
                participant = self._livekit_state_observer._get_participant()

            if not participant:
                # Try job context directly
                try:
                    from livekit.agents import get_job_context

                    ctx = get_job_context()
                    if ctx and ctx.room:
                        participant = ctx.room.local_participant
                except Exception:
                    pass

            if not participant:
                self._logger.debug("No participant for LiveKit transcript publish")
                return False

            from livekit import rtc

            # Generate transcript ID
            transcript_id = getattr(self, "_lk_transcript_id", 0) + 1
            self._lk_transcript_id = transcript_id

            segment = rtc.TranscriptionSegment(
                id=f"{role}-{transcript_id}",
                text=content,
                start_time=0,
                end_time=0,
                language="en",
                final=is_final,
            )

            # Get audio track SID - required by Rust SDK (empty string causes panic)
            track_sid = ""
            for pub in participant.track_publications.values():
                if pub.source == rtc.TrackSource.SOURCE_MICROPHONE and pub.sid:
                    track_sid = pub.sid
                    break

            if not track_sid:
                self._logger.debug("No audio track SID available, skipping transcript publish")
                return False

            transcription = rtc.Transcription(
                participant_identity=participant.identity,
                track_sid=track_sid,
                segments=[segment],
            )

            await participant.publish_transcription(transcription)
            self._logger.debug(f"LiveKit transcript published: {role}")
            return True

        except Exception as e:
            self._logger.error(f"LiveKit transcript publish failed: {e}")
            return False

    def _send_callback(self, message: Message, thread_id: str) -> None:
        """Send message through callback."""
        if self._callback:
            self._callback.send(message, thread_id=thread_id)

    def message_callback(
        self,
        transcribed_text: str,
        role: str,
        user_state: UserState,
    ) -> None:
        """Handle message callbacks (compatibility method)."""
        thread_id = str(user_state.thread_id)
        if "Call Status" in transcribed_text and role == "system":
            msg = Message.add_notification(transcribed_text.replace("Call Status:", ""))
            self._send_callback(msg, thread_id=thread_id)
        elif "EOF" in transcribed_text and role == "system":
            user_state.end_time = datetime.now()
            usage = self.create_usage(user_state)
            msg = Message.add_task_end_message(
                "Voice Execution Completed",
                id=thread_id,
                data={
                    "start_time": (
                        user_state.start_time.isoformat()
                        if user_state.start_time
                        else None
                    ),
                    "end_time": user_state.end_time.isoformat()
                    if user_state.end_time
                    else None,
                    "usage": usage,
                    "recording_url": user_state.recording_url,
                    "transcript": user_state.transcript or [],
                },
            )
            self._send_callback(msg, thread_id=thread_id)

    def create_usage(self, user_state: UserState) -> Dict[str, Any]:
        """Create usage data from user state."""
        return create_usage(user_state)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics including plugin metrics."""
        metrics = {
            "handler": {
                "session_id": self._session_id,
                "services_initialized": self._services_initialized,
                "is_shutting_down": self._is_shutting_down,
                "plugins_active": len(self.plugins),
            },
            "plugins": self.plugins.get_all_metrics(),
        }
        return metrics

    def dump(self) -> dict:
        """Dump handler state."""
        return {
            "session_id": self._session_id,
            "agent_name": self.agent_config.agent_name if self.agent_config else None,
            "transport_type": str(self._transport_type),
            "services_initialized": self._services_initialized,
            "is_shutting_down": self._is_shutting_down,
            "active_plugins": list(self.plugins._activated),
        }

    def __repr__(self) -> str:
        return f"LiteVoiceHandler(session_id={self._session_id})"
