"""
SharedVoiceMixin - Provider-agnostic voice features shared by LiveKit and Pipecat handlers.

Extracts common business logic (idle timeout, eval_records, quality metrics,
post-call workflow, KB management, goodbye handling, cleanup orchestration)
into a mixin that both LiveKitLiteHandler and LiteVoiceHandler inherit.

Each handler implements a small set of abstract hooks for provider-specific
delivery (speech, text, disconnect, context access).
"""

import asyncio
import json
import logging
import os
import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from super.core.voice.common.common import build_call_result
from super.core.voice.common.prefect import trigger_post_call
from super.core.voice.common.services import save_execution_log
from super.core.voice.schema import CallStatusEnum

try:
    from super.core.voice.livekit.conversation_state import DynamicConversationState

    CONVERSATION_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONVERSATION_INTELLIGENCE_AVAILABLE = False
    DynamicConversationState = None

try:
    from super_services.db.services.models.task import TaskModel

    TASK_MODEL_AVAILABLE = True
except ImportError:
    TASK_MODEL_AVAILABLE = False
    TaskModel = None


class SharedVoiceMixin:
    """Provider-agnostic voice features for LiveKit and Pipecat handlers.

    Concrete methods use a small set of abstract hooks that each handler
    implements for provider-specific delivery.

    Attributes expected on the host class (set by handler __init__):
        user_state, config, _session_id, _logger, _is_shutting_down,
        _first_response_sent, _kb_ready, plugins, prompt_manager,
        _callback, _event_bridge, agent_config
    """

    # ─── Abstract hooks (handler-specific) ────────────────────────────

    @abstractmethod
    async def _send_speech(self, text: str) -> None:
        """Speak *text* via TTS.

        LiveKit: session.say() / session.generate_reply()
        Pipecat: TTSSpeakFrame pushed to pipeline
        """
        ...

    @abstractmethod
    async def _disconnect_participant(self) -> None:
        """Disconnect the call.

        LiveKit: room.disconnect() / delete room
        Pipecat: EndFrame + task.cancel()
        """
        ...

    @abstractmethod
    def _get_conversation_messages(self) -> List[Dict[str, str]]:
        """Return current chat messages as list of dicts.

        LiveKit: ChatContext messages
        Pipecat: OpenAILLMContext.get_messages()
        """
        ...

    @abstractmethod
    async def _append_context_message(
        self, role: str, content: str, run_llm: bool = False
    ) -> None:
        """Inject a message into the LLM context.

        LiveKit: ChatContext.append()
        Pipecat: LLMMessagesAppendFrame
        """
        ...

    @abstractmethod
    def _is_agent_speaking(self) -> bool:
        """Return True if agent is currently producing speech."""
        ...

    @abstractmethod
    async def _handler_specific_cleanup(self) -> None:
        """Release handler-specific runtime resources.

        Called by the shared cleanup orchestrator after common cleanup.
        """
        ...

    # ─── Shared init (call from handler __init__) ─────────────────────

    def _init_shared_mixin(self) -> None:
        """Initialize shared mixin state. Call from handler __init__."""
        # Idle timeout config
        idle_cfg = (
            self.config.get("idle_timeout_seconds", 30.0)
            if isinstance(self.config, dict)
            else 30.0
        )
        self._idle_timeout_seconds: float = float(idle_cfg)
        self._idle_warning_1_seconds: float = self._idle_timeout_seconds * 0.5
        self._idle_warning_2_seconds: float = self._idle_timeout_seconds * 0.83
        self._last_activity_time: float = time.time()
        self._idle_warning_count: int = 0
        self._idle_monitor_task: Optional[asyncio.Task] = None
        self._idle_monitor_running: bool = False
        self._sending_idle_warning: bool = False
        self._agent_state: str = "listening"

        # Post-call guard
        self._post_call_triggered: bool = False

        # Cleanup guard
        self._runtime_resources_cleaned: bool = False

        # KB state
        self._kb_warmup_task: Optional[asyncio.Task] = None
        self._kb_auto_injected: bool = False

        # Disconnect reason propagation for provider-specific fallback paths.
        self._disconnect_reason_override: Optional[str] = None

        # Latency queue for eval
        self._pending_eval_llm_latencies: List[float] = []

        # Seed ground truth when present in config/context
        self._seed_eval_ground_truth()

    # ─── Ground truth seeding ─────────────────────────────────────────

    def _seed_eval_ground_truth(self) -> None:
        """Seed ground truth from config into user_state.extra_data."""
        if not self.user_state:
            return
        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}
        if "eval_ground_truth" in self.user_state.extra_data:
            return

        qa_pairs: list = []
        if isinstance(self.config, dict):
            qa_pairs = (
                self.config.get("qa_pairs")
                or self.config.get("evaluation_qa_pairs")
                or []
            )
        if not qa_pairs and isinstance(
            self.user_state.extra_data.get("data"), dict
        ):
            qa_pairs = self.user_state.extra_data.get("data", {}).get(
                "qa_pairs", []
            )
        if isinstance(qa_pairs, list) and qa_pairs:
            self.user_state.extra_data["eval_ground_truth"] = qa_pairs

    # ─── Conversation intelligence property ───────────────────────────

    @property
    def use_conversation_intelligence(self) -> bool:
        """Check if dynamic conversation intelligence is enabled."""
        if not CONVERSATION_INTELLIGENCE_AVAILABLE:
            return False
        return self.config.get("use_conversation_intelligence", True)

    # ─── Idle timeout monitoring ──────────────────────────────────────

    def _reset_idle_timer(self) -> None:
        """Reset the idle timer on user or agent activity."""
        self._last_activity_time = time.time()
        self._idle_warning_count = 0
        self._logger.debug("[IDLE] Timer reset - activity detected")

    async def _start_idle_monitor(self) -> None:
        """Start the idle timeout monitoring task."""
        if self._idle_monitor_running:
            return
        self._idle_monitor_running = True
        self._last_activity_time = time.time()
        self._idle_warning_count = 0
        self._logger.info(
            f"[IDLE] Starting idle monitor "
            f"(timeout={self._idle_timeout_seconds}s)"
        )
        self._idle_monitor_task = asyncio.create_task(self._idle_monitor_loop())

    async def _stop_idle_monitor(self) -> None:
        """Stop the idle timeout monitoring task."""
        self._idle_monitor_running = False
        if self._idle_monitor_task:
            self._idle_monitor_task.cancel()
            try:
                await self._idle_monitor_task
            except asyncio.CancelledError:
                pass
            self._idle_monitor_task = None
        self._logger.debug("[IDLE] Idle monitor stopped")

    async def _idle_monitor_loop(self) -> None:
        """Background task to monitor user idle time."""
        try:
            while self._idle_monitor_running and not self._is_shutting_down:
                await asyncio.sleep(0.5)

                # Skip while agent is actively producing a response.
                agent_state = getattr(self, "_agent_state", "")
                if self._is_agent_speaking() or agent_state == "thinking":
                    continue

                idle_duration = time.time() - self._last_activity_time

                # First warning at 50%
                if (
                    idle_duration >= self._idle_warning_1_seconds
                    and self._idle_warning_count == 0
                ):
                    self._idle_warning_count = 1
                    await self._send_idle_warning(1)

                # Second warning at 83%
                elif (
                    idle_duration >= self._idle_warning_2_seconds
                    and self._idle_warning_count == 1
                ):
                    self._idle_warning_count = 2
                    await self._send_idle_warning(2)

                # Disconnect at full timeout
                elif (
                    idle_duration >= self._idle_timeout_seconds
                    and self._idle_warning_count >= 2
                ):
                    self._logger.info(
                        f"[IDLE] User idle for {idle_duration:.1f}s, "
                        "disconnecting"
                    )
                    await self._handle_idle_disconnect()
                    break

        except asyncio.CancelledError:
            self._logger.debug("[IDLE] Monitor task cancelled")
        except Exception as e:
            self._logger.error(f"[IDLE] Monitor error: {e}")

    async def _send_idle_warning(self, warning_number: int) -> None:
        """Send idle warning to user via agent speech."""
        user_language = (
            self.config.get("preferred_language", "en")
            if isinstance(self.config, dict)
            else "en"
        )
        param = f"idle_warning_{warning_number}"
        warning_msg = self.prompt_manager.get_message(user_language, param)
        if not warning_msg:
            warning_msg = self.prompt_manager.get_message("en", param)

        self._logger.info(
            f"[IDLE] Sending warning {warning_number}: {warning_msg}"
        )
        try:
            self._sending_idle_warning = True
            await self._send_speech(warning_msg)
        except Exception as e:
            self._logger.error(f"[IDLE] Failed to send warning: {e}")
        finally:
            self._sending_idle_warning = False

    async def _handle_idle_disconnect(self) -> None:
        """Handle disconnect due to user inactivity."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._logger.info("[IDLE] Disconnecting due to user inactivity")

        # Trigger post-call
        await self._ensure_post_call_triggered(reason="idle_timeout")

        # Say goodbye
        try:
            goodbye_msg = "No response detected. Goodbye! Have a great day."
            await self._send_speech(goodbye_msg)
            await asyncio.sleep(2.0)
        except Exception as e:
            self._logger.error(f"[IDLE] Failed to send goodbye: {e}")

        # Disconnect via handler hook
        self._disconnect_reason_override = "idle_timeout"
        try:
            await self._disconnect_participant()
        finally:
            self._disconnect_reason_override = None

    # ─── Goodbye / end-call intent ────────────────────────────────────

    GOODBYE_PATTERNS = [
        "bye", "goodbye", "good bye", "bye-bye", "bye bye",
        "ok bye", "okay bye", "see you", "take care",
        "hang up", "end call", "disconnect",
        "alvida", "phir milenge", "theek hai bye", "chalo bye",
        "thank you bye", "thanks bye", "thankyou bye",
    ]

    def _is_goodbye_intent(self, transcript: str) -> bool:
        """Check if transcript contains goodbye intent."""
        transcript_lower = transcript.lower().strip()
        transcript_clean = transcript_lower.rstrip("!.,?")

        for pattern in self.GOODBYE_PATTERNS:
            if transcript_lower == pattern or transcript_clean == pattern:
                return True
            if transcript_lower.startswith(f"{pattern} "):
                return True
            if transcript_lower.endswith(" bye") or transcript_clean.endswith(
                " bye"
            ):
                return True
        return False

    async def _handle_goodbye_intent(self) -> None:
        """Handle goodbye intent - say goodbye and disconnect."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._logger.info("[GOODBYE] Handling goodbye intent, ending session")

        await self._stop_idle_monitor()

        # Say goodbye
        try:
            goodbye_msg = (
                "Thank you for your time. Goodbye! Have a great day."
            )
            await self._send_speech(goodbye_msg)
            await asyncio.sleep(2.0)
        except Exception as e:
            self._logger.error(f"[GOODBYE] Failed to send goodbye: {e}")

        self._disconnect_reason_override = "user_goodbye"
        try:
            await self._disconnect_participant()
        finally:
            self._disconnect_reason_override = None

    async def _handle_tool_end_call(
        self, reason: str = "user_goodbye"
    ) -> None:
        """Handle end_call tool invocation - end session gracefully."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._logger.info(
            f"[END_CALL_TOOL] Ending session, reason: {reason}"
        )
        await self._stop_idle_monitor()
        self._disconnect_reason_override = reason
        try:
            await self._disconnect_participant()
        finally:
            self._disconnect_reason_override = None

    # ─── Post-call trigger ────────────────────────────────────────────

    async def _ensure_post_call_triggered(
        self, reason: str = "session_end"
    ) -> None:
        """Idempotent post-call workflow trigger."""
        if not self.user_state:
            self._logger.warning(
                "[POST_CALL] No user_state, skipping post-call trigger"
            )
            return

        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}

        if self.user_state.extra_data.get("post_call_triggered"):
            self._logger.info(
                f"[POST_CALL] Already triggered, skipping (reason={reason})"
            )
            return

        # Skip SDK calls
        if self.user_state.extra_data.get("call_type") == "sdk":
            self._logger.info(
                "[POST_CALL] SDK call, skipping post-call workflow"
            )
            return

        self.user_state.extra_data["post_call_triggered"] = True

        if not self.user_state.end_time:
            self.user_state.end_time = datetime.utcnow()

        if not self.user_state.call_status or self.user_state.call_status in (
            CallStatusEnum.InCall,
            CallStatusEnum.CONNECTED,
            None,
        ):
            self.user_state.call_status = CallStatusEnum.COMPLETED

        self._logger.info(
            f"[POST_CALL] Triggering post-call workflow (reason={reason}, "
            f"status={self.user_state.call_status})"
        )

        try:
            res = await build_call_result(self.user_state)
            await trigger_post_call(user_state=self.user_state, res=res)
            self._logger.info(
                f"{'=' * 80}\n[POST_CALL] Successfully triggered "
                f"(reason={reason})\n{'=' * 80}"
            )
        except Exception as e:
            self._logger.error(f"[POST_CALL] Failed: {e}")
            import traceback

            self._logger.error(traceback.format_exc())

    # ─── Quality metrics persistence ──────────────────────────────────

    async def _persist_session_quality_metrics(
        self, reason: str = "session_end"
    ) -> None:
        """Persist quality metrics from DynamicConversationState."""
        try:
            if not self.use_conversation_intelligence:
                return

            conv_state = self._get_conversation_state()
            if not conv_state:
                return

            # Update metadata
            if self.user_state:
                conv_state.task_id = (
                    getattr(self.user_state, "task_id", "") or ""
                )
            conv_state.session_id = self._session_id or ""
            conv_state.agent_handle = (
                self.config.get("agent_name", "") or ""
            )

            self._logger.info(conv_state.get_quality_log_line())

            metrics = conv_state.get_quality_metrics()
            progress = metrics.get("content_progress", {})
            self._logger.info(
                f"[SESSION_END] reason={reason} | "
                f"phase={metrics.get('current_phase', 'unknown')} | "
                f"content={progress.get('delivered', 0)}/"
                f"{progress.get('total', 0)} | "
                f"objective={'achieved' if metrics.get('objective_achieved') else metrics.get('objective_outcome', 'pending')} | "
                f"score={metrics['scores']['overall']}"
            )

            call_summary_text = conv_state.build_call_summary(
                end_reason=reason
            )
            self._logger.info(f"\n{call_summary_text}")

            call_summary_dict = conv_state.get_call_summary_dict(
                end_reason=reason
            )

            # Store in user_state
            if self.user_state:
                if not isinstance(self.user_state.extra_data, dict):
                    self.user_state.extra_data = {}
                self.user_state.extra_data["quality_metrics"] = metrics
                self.user_state.extra_data["conversation_state"] = (
                    conv_state.to_dict()
                )
                self.user_state.extra_data["call_summary"] = (
                    call_summary_dict
                )

            # Save to DB
            task_id = (
                self.user_state.task_id if self.user_state else None
            )
            if task_id:
                await save_execution_log(
                    task_id=task_id,
                    step="call_summary",
                    status="completed",
                    output=call_summary_dict,
                )
                self._logger.info(
                    f"[CALL_SUMMARY] Saved to execution log | "
                    f"task_id={task_id}"
                )

                if TASK_MODEL_AVAILABLE and TaskModel is not None:
                    try:
                        TaskModel._get_collection().update_one(
                            {"task_id": task_id},
                            {
                                "$set": {
                                    "output.call_summary": call_summary_dict
                                }
                            },
                        )
                    except Exception as db_err:
                        self._logger.error(
                            f"[CALL_SUMMARY] DB update failed: {db_err}"
                        )
            else:
                self._logger.warning(
                    "[CALL_SUMMARY] No task_id, skipping DB storage"
                )

            filepath = conv_state.persist_quality_metrics(reason=reason)
            if filepath:
                self._logger.info(
                    f"Quality metrics persisted to: {filepath}"
                )

        except Exception as e:
            self._logger.error(
                f"Error persisting session quality metrics: {e}"
            )

    def _get_conversation_state(self) -> Optional[Any]:
        """Get DynamicConversationState from agent. Override if needed."""
        agent = getattr(self, "_agent", None)
        if not agent:
            return None
        return getattr(agent, "_conv_state", None)

    async def _record_agent_response_for_quality(
        self, content: str
    ) -> None:
        """Record agent response for quality anti-pattern detection."""
        try:
            conv_state = self._get_conversation_state()
            if not conv_state:
                return

            result = conv_state.record_agent_response(
                response_text=content,
                phase="conversation",
                turn=conv_state.turn_count,
            )

            issues = result.get("issues", [])
            if issues:
                self._logger.warning(
                    f"[QUALITY_ANTIPATTERN] turn={conv_state.turn_count} "
                    f"repair={conv_state.in_repair} issues={issues}"
                )
        except Exception as e:
            self._logger.debug(
                f"Error recording agent response for quality: {e}"
            )

    # ─── Record tracking ──────────────────────────────────────────────

    def _ensure_eval_records(self) -> Dict[str, Any]:
        """Ensure user_state.extra_data has eval_records structure."""
        if not self.user_state:
            return {}

        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}

        records = self.user_state.extra_data.get("eval_records")
        if not isinstance(records, dict):
            records = {}
            self.user_state.extra_data["eval_records"] = records

        for key, default in [
            ("tool_calls", []),
            ("agent_responses", []),
            ("user_messages", []),
            ("sequence_counter", 0),
            ("llm_latency_samples", []),
        ]:
            if not isinstance(records.get(key), type(default)):
                records[key] = default

        return records

    def _next_eval_sequence(self) -> int:
        records = self._ensure_eval_records()
        if not records:
            return 0
        records["sequence_counter"] += 1
        return records["sequence_counter"]

    def _record_eval_user_message(self, content: str) -> None:
        """Capture user message for call-end QA alignment."""
        try:
            records = self._ensure_eval_records()
            if not records:
                return
            records["user_messages"].append(
                {
                    "sequence_id": self._next_eval_sequence(),
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            self._logger.debug(f"Failed to record user message: {e}")

    def _record_eval_llm_latency(
        self,
        llm_latency: Optional[float],
        source: str = "llm_metrics",
    ) -> None:
        """Store per-turn LLM latency for post-call analysis."""
        try:
            if llm_latency is None:
                return
            llm_latency = float(llm_latency)
            if llm_latency <= 0:
                return

            records = self._ensure_eval_records()
            if not records:
                return

            self._pending_eval_llm_latencies.append(llm_latency)
            records["llm_latency_samples"].append(
                {
                    "sequence_id": self._next_eval_sequence(),
                    "llm_latency": llm_latency,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            self._logger.debug(f"Failed to record llm latency: {e}")

    def _consume_eval_llm_latency(self) -> Optional[float]:
        """Consume next queued LLM latency for response-level alignment."""
        if not self._pending_eval_llm_latencies:
            return None
        return self._pending_eval_llm_latencies.pop(0)

    def _record_eval_agent_response(
        self,
        content: str,
        llm_latency: Optional[float] = None,
    ) -> None:
        """Capture assistant response for call-end QA tracking."""
        try:
            records = self._ensure_eval_records()
            if not records:
                return
            payload: Dict[str, Any] = {
                "sequence_id": self._next_eval_sequence(),
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if llm_latency is not None:
                payload["llm_latency"] = float(llm_latency)
            records["agent_responses"].append(payload)
        except Exception as e:
            self._logger.debug(
                f"Failed to record assistant response: {e}"
            )

    # ─── Latency chart ────────────────────────────────────────────────

    def add_latency_chart(
        self,
        turn_count: int,
        llm_latency: float,
        tts_latency: float,
        stt_latency: float,
        current_turn_total: float,
        avg_llm: float,
        avg_stt: float,
        avg_tts: float,
        avg_current_total: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record per-turn latency metrics into user_state.extra_data."""
        if not isinstance(self.user_state.extra_data, dict):
            self.user_state.extra_data = {}

        latency_data = self.user_state.extra_data.get(
            "latency_metrics", []
        )
        provider_data = self.user_state.extra_data.get("service_modes")

        latency_entry: Dict[str, Any] = {
            "turn_count": turn_count,
            "llm_latency": llm_latency,
            "tts_latency": tts_latency,
            "stt_latency": stt_latency,
            "total_latency": current_turn_total,
            "avg_llm_latency": avg_llm,
            "avg_stt_latency": avg_stt,
            "avg_tts_latency": avg_tts,
            "avg_total_latency": avg_current_total,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        if provider_data:
            try:
                latency_entry["stt_provider"] = provider_data.stt_provider
                latency_entry["tts_provider"] = provider_data.tts_provider
                latency_entry["llm_provider"] = provider_data.llm_provider
                latency_entry["llm_type"] = provider_data.llm_type
                latency_entry["stt_type"] = provider_data.stt_type
            except AttributeError:
                pass

        latency_data.append(latency_entry)
        self.user_state.extra_data["latency_metrics"] = latency_data

    # ─── KB management ────────────────────────────────────────────────

    def _kb_enabled(self) -> bool:
        """Check if knowledge base is configured."""
        if not self.config:
            return False
        return bool(self.config.get("knowledge_base"))

    async def _auto_inject_kb_context(self) -> None:
        """Proactively inject KB context when LLM hasn't called get_docs.

        Fires once after a few turns if the agent has a ready KB manager
        but the LLM never invoked get_docs.
        """
        if self._kb_auto_injected:
            return
        self._kb_auto_injected = True

        try:
            kb_manager = getattr(self, "_kb_manager", None)
            if not kb_manager:
                self._logger.warning(
                    "[KB_AUTO_INJECT] No KB manager available"
                )
                return

            last_user_msg = ""
            for entry in reversed(self.user_state.transcript):
                if entry.get("role") == "user":
                    last_user_msg = entry.get("content", "")
                    break

            if not last_user_msg:
                return

            self._logger.info(
                f"[KB_AUTO_INJECT] Proactively querying KB: "
                f"{last_user_msg[:80]}"
            )

            docs = await kb_manager.get_docs(
                query=last_user_msg,
                user_state=self.user_state,
            )

            if not docs or isinstance(docs, dict):
                self._logger.info("[KB_AUTO_INJECT] No documents found")
                return

            context_parts = [
                doc.content if hasattr(doc, "content") else str(doc)
                for doc in docs[:3]
            ]
            context = "\n".join(context_parts)

            if context.strip():
                kb_instruction = (
                    "Here is relevant information from the knowledge base "
                    "that you should reference:\n"
                    f"{context}\n\n"
                    "Use this context to enhance your response."
                )
                await self._append_context_message(
                    role="system",
                    content=kb_instruction,
                    run_llm=False,
                )
                self._logger.info(
                    "[KB_AUTO_INJECT] Injected KB context"
                )

        except Exception as e:
            self._logger.warning(f"[KB_AUTO_INJECT] Failed: {e}")

    async def _shared_background_kb_warmup(
        self, immediate: bool = True
    ) -> None:
        """Warm up KB in background.

        Args:
            immediate: If True, start immediately without waiting for
                first response.
        """
        try:
            if not self._kb_enabled():
                self._kb_ready.set()
                return

            if not immediate:
                await asyncio.wait_for(
                    self._first_response_sent.wait(),
                    timeout=30.0,
                )
                await asyncio.sleep(0.5)

            _start = asyncio.get_event_loop().time()

            from super.core.voice.managers.knowledge_base import (
                KnowledgeBaseManager,
            )

            kb_manager = KnowledgeBaseManager(
                logger=self._logger.getChild("kb"),
                session_id=self._session_id,
                user_state=self.user_state,
                config=self.config,
            )

            # Phase 1: local retrieval
            await kb_manager._init_context_retrieval()
            self._kb_manager = kb_manager
            self._kb_ready.set()
            self._logger.info(
                "[KB_WARMUP] Local KB retrieval initialized"
            )

            # Phase 2: remote preload
            await kb_manager._preload_knowledge_base_documents(
                self.user_state
            )

            elapsed_ms = (asyncio.get_event_loop().time() - _start) * 1000
            self._logger.info(
                f"Background KB warmup complete in {elapsed_ms:.0f}ms "
                f"(immediate={immediate})"
            )

        except asyncio.TimeoutError:
            self._logger.warning(
                "KB warmup timeout - first response not sent"
            )
            self._kb_ready.set()
        except Exception as e:
            self._logger.warning(f"KB warmup failed: {e}")
            self._kb_ready.set()

    # ─── Cleanup orchestration ────────────────────────────────────────

    async def _cleanup_runtime_resources(self) -> None:
        """Release runtime resources. Calls handler-specific cleanup."""
        if self._runtime_resources_cleaned:
            return
        self._runtime_resources_cleaned = True

        await self._stop_idle_monitor()

        # Cancel KB warmup task
        kb_task = self._kb_warmup_task
        self._kb_warmup_task = None
        if kb_task and not kb_task.done():
            kb_task.cancel()
            try:
                await kb_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._logger.debug(
                    f"KB warmup task cleanup warning: {e}"
                )

        # Plugin cleanup
        try:
            await self.plugins.cleanup_all()
        except Exception as e:
            self._logger.warning(
                f"Plugin cleanup failed during shutdown: {e}"
            )

        # Handler-specific cleanup
        await self._handler_specific_cleanup()

    # ─── Data channel constants ───────────────────────────────────────

    TOPIC_LK_CHAT = "lk.chat"
    TOPIC_LK_STREAM = "lk.stream"

    # ─── Block message parsing (provider-agnostic) ────────────────────

    def _parse_block_message(
        self, raw_message: str
    ) -> tuple:
        """
        Parse incoming message, extracting content and files from block JSON.

        Args:
            raw_message: Raw message text (may be JSON block or plain text)

        Returns:
            Tuple of (extracted_content, block_data or None, files list or None)
        """
        if not raw_message or not raw_message.strip():
            return "", None, None

        if raw_message.strip().startswith("{"):
            try:
                block_data = json.loads(raw_message)

                if isinstance(block_data, dict) and block_data.get("event") == "block":
                    raw_data = block_data.get("data")
                    if raw_data is None or not isinstance(raw_data, dict):
                        return raw_message, None, None
                    data = raw_data

                    inner_data = data.get("data") or {}
                    if not isinstance(inner_data, dict):
                        inner_data = {}

                    content = inner_data.get("content") or data.get("content") or ""
                    files = inner_data.get("files", [])

                    self._logger.info(
                        f"[TEXT_MODE] Parsed block message: "
                        f"block_type={data.get('block_type')}, "
                        f"content_type={data.get('content_type')}, "
                        f"content_length={len(content)}, "
                        f"files_count={len(files)}"
                    )

                    return content, block_data, files if files else None

            except json.JSONDecodeError:
                pass

        return raw_message, None, None

    # ─── Response block building (provider-agnostic) ──────────────────

    def _build_response_block(
        self,
        response_text: str,
        block_data: dict | None = None,
        message_id: str | None = None,
        cards: dict | None = None,
    ) -> dict:
        """
        Build response block dict matching BlockModelSchema structure.

        Args:
            response_text: The LLM response text (can be empty for card-only)
            block_data: Original block data from input (for context)
            message_id: Unique ID for correlating with streaming chunks
            cards: Optional card object

        Returns:
            Dict formatted as response block
        """
        import uuid

        pilot = "multi-ai"
        execution_type = "contact"
        focus = "my_space"

        if block_data:
            pilot = block_data.get("pilot", pilot)
            data = block_data.get("data", {})
            execution_type = data.get("execution_type", execution_type)
            inner_data = data.get("data", {})
            focus = inner_data.get("focus", focus)

        data_content: dict = {}
        if response_text:
            data_content["content"] = response_text
            data_content["focus"] = focus
        if cards:
            data_content["cards"] = cards

        block_type = "pilot_response" if response_text else "card_block"

        return {
            "event": "block_response",
            "id": message_id or str(uuid.uuid4())[:12],
            "pilot": pilot,
            "execution_type": execution_type,
            "block": "html",
            "block_type": block_type,
            "data": data_content,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ─── S3 file processing (provider-agnostic) ──────────────────────

    async def _download_file_from_s3(self, file_info: dict) -> str | None:
        """
        Download a file from S3 to a local temporary location.

        Args:
            file_info: File metadata dict with 'url' or 'media_url' keys

        Returns:
            Local file path if successful, None otherwise
        """
        try:
            from super_services.libs.core.s3 import s3_path_split, download_file
            import tempfile

            file_url = file_info.get("url") or file_info.get("media_url")
            if not file_url:
                self._logger.warning("No URL found in file info")
                return None

            bucket_name, file_key = s3_path_split(file_url)
            file_name = file_key.split("/")[-1]
            name, ext = os.path.splitext(file_name)

            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"voice_input_{name}{ext}")

            local_path = download_file(bucket_name, file_key, output_path)

            self._logger.info(
                f"[FILE_PROCESSING] Downloaded from S3: {file_name} -> {local_path}"
            )
            return local_path

        except Exception as e:
            self._logger.error(
                f"[FILE_PROCESSING] Failed to download from S3: {e}"
            )
            return None

    async def _process_attached_files(self, files: list) -> list:
        """
        Process attached files - download from S3 and prepare for LLM context.

        Args:
            files: List of file metadata dicts

        Returns:
            List of processed file info dicts with local paths
        """
        processed_files = []

        for file_info in files:
            media_type = file_info.get("media_type", "unknown")
            file_name = file_info.get("name", "unknown")

            self._logger.info(
                f"[FILE_PROCESSING] Processing: {file_name} ({media_type})"
            )

            local_path = await self._download_file_from_s3(file_info)
            if local_path:
                processed_files.append(
                    {
                        "name": file_name,
                        "media_type": media_type,
                        "local_path": local_path,
                        "original_url": file_info.get("url"),
                        "media_url": file_info.get("media_url"),
                    }
                )

        return processed_files
