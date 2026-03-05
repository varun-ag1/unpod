"""
Transcript Plugin - Manages transcript storage and history.

Provides transcript history tracking for voice pipeline with callback support.
Transcript events are driven by aggregator turn events (Pipecat >=0.0.102).
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from super.core.voice.plugins.base import PipelinePlugin, PluginConfig, PluginPriority

if TYPE_CHECKING:
    from super.core.voice.pipecat.lite_handler import LiteVoiceHandler


class TranscriptPlugin(PipelinePlugin):
    """Transcript plugin for voice pipeline.

    Stores transcript history and provides callback integration.
    Transcript events come from aggregator turn handlers, not from
    a pipeline processor.

    Config options:
        store_transcript: Whether to store transcripts (default: True)
        max_history: Maximum transcript entries to keep (default: 1000)
    """

    name = "transcript"
    priority = PluginPriority.LAST

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger)
        self._transcript_history: List[Dict[str, Any]] = []
        self._on_user_message: Optional[Callable] = None
        self._on_assistant_message: Optional[Callable] = None

    async def initialize(self, handler: "LiteVoiceHandler") -> None:
        """Initialize transcript plugin."""
        await super().initialize(handler)

        self._store_transcript = self.options.get("store_transcript", True)
        self._max_history = self.options.get("max_history", 1000)

        self._logger.info("TranscriptPlugin initialized")

    async def on_user_speech(self, content: str) -> None:
        """Handle user speech event from aggregator."""
        if self._store_transcript:
            self._add_entry("user", content)
        if self._on_user_message:
            await self._on_user_message(content)

    async def on_agent_speech(self, content: str) -> None:
        """Handle agent speech event from aggregator."""
        if self._store_transcript:
            self._add_entry("assistant", content)
        if self._on_assistant_message:
            await self._on_assistant_message(content)

    def _add_entry(self, role: str, content: str) -> None:
        """Add a transcript entry and trim if needed."""
        self._transcript_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._transcript_history) > self._max_history:
            self._transcript_history = self._transcript_history[
                -self._max_history :
            ]

    def get_processors(self) -> List:
        """Return empty list - transcript handling uses aggregator events."""
        return []

    def register_user_callback(self, callback: Callable) -> None:
        """Register callback for user messages."""
        self._on_user_message = callback

    def register_assistant_callback(self, callback: Callable) -> None:
        """Register callback for assistant messages."""
        self._on_assistant_message = callback

    def get_transcript_history(self) -> List[Dict[str, Any]]:
        """Get full transcript history."""
        return self._transcript_history.copy()

    def get_last_n_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last N transcript messages."""
        return self._transcript_history[-n:]

    async def on_call_start(self) -> None:
        """Clear transcript on call start."""
        self._transcript_history.clear()
        self._logger.debug("Transcript history cleared")

    def get_metrics(self) -> Optional[dict]:
        """Get transcript metrics."""
        user_count = sum(
            1 for t in self._transcript_history if t.get("role") == "user"
        )
        assistant_count = sum(
            1 for t in self._transcript_history if t.get("role") == "assistant"
        )
        return {
            "total_messages": len(self._transcript_history),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "enabled": self._config.enabled,
        }

    async def cleanup(self) -> None:
        """Clean up transcript plugin resources."""
        self._transcript_history.clear()
        self._on_user_message = None
        self._on_assistant_message = None
        await super().cleanup()
