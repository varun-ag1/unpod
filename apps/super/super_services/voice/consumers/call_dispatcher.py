"""
Lightweight Call Dispatcher — dispatch-only logic with minimal dependencies.

Mirrors the CallProvider + CallProviderFactory pattern but without importing
heavy voice/AI libraries (livekit.agents, pipecat, dspy, langchain, etc.).

Supports:
  - VAPI: HTTP POST to api.vapi.ai/call/phone (with channel concurrency check)
  - LiveKit: livekit.api agent dispatch (API client only, no agents/plugins)
  - Pipecat: Falls back to full execute_call (runs locally, needs heavy libs)

Dependencies (all lightweight):
  - requests, httpx (HTTP clients)
  - livekit.api (API client only ~42MB, no ML libs)
  - super_services.voice.models.config.ModelConfig (DB + Redis cache)
  - super_services.libs.core.timezone_utils (phone formatting)
  - super.app.call_agents (static dict)
"""

import asyncio
import json
import os
import random
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from super.core.voice.providers.base import CallProviderBase, CallResult


def _run_async(coro):
    """Run an async coroutine from sync code, handling existing event loops.

    When called from inside Prefect (which has a running event loop),
    asyncio.run() raises 'no running event loop' / 'cannot run nested'.
    This helper uses the existing loop when available.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop (Prefect context) — run in a new thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class CallDispatcher(CallProviderBase):
    """
    Lightweight call dispatcher — dispatch-only flows without heavy dependencies.

    Inherits from CallProviderBase (shared with CallProviderFactory):
      - get_provider_type(data)   — determine provider from data (static)
      - validate_data(data)       — validate required fields (static)

    Additional methods:
      - get_config(agent_id)      — fetch agent config (DB + Redis cache)
      - execute_pre_call(...)     — lightweight pre-call context lookup
      - execute_call(...)         — dispatch call to the appropriate provider

    Usage:
        dispatcher = CallDispatcher()
        response = dispatcher.execute_call(
            agent_id=agent_id, task_id=task_id, data=data, instructions=instructions
        )
        # response is a dict with 'status' and 'data' keys
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # ── Common Methods ───────────────────────────────────────────────────

    @staticmethod
    def _result_to_response(result: CallResult) -> dict:
        """Convert a CallResult to the standard response dict format.

        Matches the response format produced by call_execution.execute_call():
            {"status": ..., "data": {"call_id": ..., "customer": ..., ...}}
        """
        output = {
            "call_id": result.call_id,
            "customer": result.customer,
            "contact_number": result.contact_number,
            "call_end_reason": result.call_end_reason,
            "recording_url": result.recording_url,
            "transcript": result.transcript or [],
            "start_time": result.call_start,
            "end_time": result.call_end,
            "assistant_number": result.assistant_number,
            "call_summary": result.call_summary,
            "duration": result.duration,
            "cost": float(result.data.get("cost", 0)) if result.data else 0,
            "call_type": result.data.get("type", "") if result.data else "",
        }
        if result.error:
            output["error"] = result.error
        if result.notes:
            output["notes"] = result.notes
        if result.call_status:
            output["call_status"] = result.call_status
        if result.status_update:
            output["status_update"] = result.status_update
        # Keep raw provider data accessible
        if result.data:
            output["metadata"] = result.data

        return {"status": result.status, "data": output}

    def get_provider_name(self, data: dict) -> str:
        """Get the resolved provider name for the given data."""
        return self.get_provider_type(data)

    def get_config(self, agent_id: str) -> dict:
        """Fetch agent config via ModelConfig (DB + Redis, no heavy deps)."""
        from super_services.voice.models.config import ModelConfig

        return ModelConfig().get_config(agent_id)

    def execute_pre_call(self, task_id: str, data: dict, agent_id: str) -> dict:
        """Lightweight pre-call: fetch chat context from DB."""
        from super_services.db.services.models.task import TaskModel

        try:
            ref_id = data.get("document_id") or data.get("id")
            if not ref_id:
                task_obj = TaskModel.get(task_id=task_id)
                if task_obj:
                    ref_id = getattr(task_obj, "ref_id", None)

            # Check if memory is enabled for this agent
            from super_services.libs.core.db import executeQuery

            res = executeQuery(
                query="select enable_memory from core_components_pilot where handle=%(agent)s",
                params={"agent": agent_id},
            )
            if not res or not bool(res.get("enable_memory", 0)):
                return {}

            # Fetch past conversation context
            user_number = data.get("number", data.get("contact_number"))
            if ref_id:
                chats = list(
                    TaskModel._get_collection()
                    .find(
                        {"ref_id": ref_id, "assignee": agent_id},
                        {"output.transcript": 1},
                    )
                    .sort([("created", -1)])
                    .limit(5)
                )
            elif user_number:
                numbers = [user_number]
                if user_number.startswith("0"):
                    numbers.append(user_number[1:])
                chats = list(
                    TaskModel._get_collection()
                    .find(
                        {
                            "$or": [
                                {"input.contact_number": {"$in": numbers}},
                                {"output.contact_number": {"$in": numbers}},
                                {"input.number": {"$in": numbers}},
                                {"output.customer": {"$in": numbers}},
                            ],
                            "assignee": agent_id,
                        },
                        {"output.transcript": 1},
                    )
                    .sort([("created", -1)])
                    .limit(5)
                )
            else:
                return {}

            if chats:
                return {"chat_context": f"[Past Conversations]\n{chats}"}
            return {}
        except Exception as e:
            print(f"Pre-call lightweight error: {e}")
            return {}

    # ── Main Execute Call ────────────────────────────────────────────────

    def execute_call(
        self,
        agent_id: str,
        task_id: str,
        data: dict,
        instructions: str = None,
        call_type: str = "outbound",
    ) -> dict:
        """
        Execute a call via the appropriate provider.

        Mirrors CallProvider.execute_call() signature. Returns dict with
        'status' and 'data' keys, compatible with existing flow.
        """
        provider_name = self.get_provider_type(data)

        # Pre-call: fetch chat context (lightweight DB lookup)
        pre_call_result = self.execute_pre_call(task_id, data, agent_id)
        if pre_call_result:
            data["pre_call_result"] = pre_call_result

        # Get agent config (DB + Redis cache)
        config = self.get_config(agent_id)

        if instructions:
            data["objective"] = instructions
        elif not data.get("objective"):
            data["objective"] = "Say Hi This your voice assistant from Unpod"

        if provider_name == "vapi":
            return self._execute_vapi(data, task_id, agent_id, config)
        elif provider_name == "livekit":
            return self._execute_livekit(data, task_id, agent_id, config)
        else:
            # Pipecat runs locally — needs full provider stack
            return self._execute_pipecat(data, task_id, agent_id, instructions)

    # ── VAPI ─────────────────────────────────────────────────────────────

    def _execute_vapi(
        self, data: dict, task_id: str, agent_id: str, config: dict
    ) -> dict:
        """Execute a VAPI call via HTTP POST."""
        import requests
        from super_services.libs.core.timezone_utils import normalize_phone_number
        from super.app.call_agents import agent_ids

        contact_number = data.get("contact_number")
        contact_name = data.get("contact_name") or data.get("name") or "User"

        if not contact_number:
            return self._result_to_response(CallResult(
                status="failed", error="No Contact Number Found",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        try:
            formatted_number = normalize_phone_number(contact_number)
        except (ValueError, Exception):
            return self._result_to_response(CallResult(
                status="failed", error="Invalid phone number format.",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        vapi_agent_id = data.get("vapi_agent_id", agent_ids.get(agent_id))
        auth_token = os.getenv("VAPI_AUTH_TOKEN")

        if not vapi_agent_id:
            return self._execute_vapi_general(
                data, agent_id, formatted_number, contact_name, auth_token
            )

        # Get telephony config from model config
        telephony_list = config.get("telephony", []) if config else []
        if not telephony_list:
            return self._result_to_response(CallResult(
                status="failed", error="No telephony numbers configured.",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        tel_number = random.choice(telephony_list)
        vapi_phone_number_id = tel_number.get("association", {}).get(
            "phone_number_id", os.getenv("VAPI_PHONE_NUMBER_ID")
        )

        if not auth_token or not vapi_phone_number_id:
            return self._result_to_response(CallResult(
                status="failed", error="Missing VAPI credentials.",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        # Check VAPI channel concurrency before dispatching
        concurrency_ok = self._check_vapi_concurrency(auth_token, task_id)
        if not concurrency_ok:
            return self._result_to_response(CallResult(
                status="failed",
                error="VAPI channel concurrency limit reached, retry later.",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        # Build variable values (same logic as start_vapi_call)
        internal_keys = {
            "contact_number", "contact_name", "number_id", "vapi_agent_id",
            "token", "document_id", "thread_id", "user_id", "call_type",
            "pre_call_result",
        }
        mailing_address = data.get("address", "New Delhi")
        variable_values = {
            "name": contact_name,
            "customer_name": contact_name,
            "mailing_address": mailing_address,
            "location": (
                mailing_address.split(",")[1].strip()
                if "," in mailing_address
                else "New Delhi"
            ),
            "agent_name": agent_id,
            "source": data.get("source", "Website"),
            "context": data.get("context", data.get("objective", "")),
            "company_name": data.get("company_name", "Unpod AI"),
            "alternate_slots": data.get(
                "alternate_slots", ["Tuesday at 10 AM", "Thursday at 2 PM"]
            ),
        }
        for key, value in data.items():
            if key not in internal_keys:
                variable_values[key] = value

        json_data = {
            "assistantId": vapi_agent_id,
            "phoneNumberId": vapi_phone_number_id,
            "customer": {"name": contact_name, "number": formatted_number},
            "assistantOverrides": {
                "artifactPlan": {
                    "recordingEnabled": True,
                    "recordingFormat": "wav;l16",
                    "recordingUseCustomStorageEnabled": True,
                },
                "variableValues": variable_values,
            },
        }

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        print(
            f"[VAPI] Dispatching call for task {task_id} | "
            f"agent: {agent_id} | vapi_agent: {vapi_agent_id} | "
            f"number: {formatted_number} | phone_id: {vapi_phone_number_id}"
        )

        # Retry loop (same as start_vapi_call)
        for i in range(2):
            response = requests.post(
                "https://api.vapi.ai/call/phone", headers=headers, json=json_data,
            )
            if response.status_code in [200, 201]:
                call_data = response.json()
                call_id = call_data.get("id", "")
                print(
                    f"[VAPI] Call created for task {task_id} | "
                    f"call_id: {call_id} | status: in_progress"
                )
                return self._result_to_response(CallResult(
                    status="in_progress",
                    call_id=call_id,
                    customer=contact_name,
                    contact_number=contact_number,
                    data={
                        "call_id": call_id,
                        "call_type": "outbound",
                        "cost": 0,
                        "type": "outboundPhoneCall",
                    },
                ))
            else:
                print(
                    f"[VAPI] Call failed for task {task_id} | "
                    f"attempt: {i+1}/2 | status: {response.status_code} | "
                    f"response: {response.text[:200]}"
                )
                # Retry with default phone number on failure
                json_data["phoneNumberId"] = os.getenv("VAPI_PHONE_NUMBER_ID")

        return self._result_to_response(CallResult(
            status="failed",
            error=f"Failed to start VAPI call: {response.text}",
            data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
        ))

    def _check_vapi_concurrency(self, auth_token: str, task_id: str) -> bool:
        """Check VAPI active call concurrency — same as VapiProvider.check_current_calls."""
        import httpx
        from super.app.call_utils import fetch_with_manual_retry

        max_concurrent = int(os.getenv("VAPI_MAX_CONCURRENT_CALLS", 10))
        wait_interval = int(os.getenv("VAPI_CHANNEL_WAIT_INTERVAL", 10))
        max_wait_cycles = int(os.getenv("VAPI_MAX_WAIT_CYCLES", 30))

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        for cycle in range(max_wait_cycles):
            try:
                now = datetime.utcnow() - timedelta(minutes=5)
                analytics_query = {
                    "queries": [
                        {
                            "table": "subscription",
                            "name": "active_calls_concurrency",
                            "operations": [
                                {"operation": "history", "column": "concurrency"}
                            ],
                            "timeRange": {
                                "step": "second",
                                "start": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                                "end": (now + timedelta(minutes=5)).strftime(
                                    "%Y-%m-%dT%H:%M:%S.000Z"
                                ),
                            },
                        }
                    ]
                }

                client = httpx.Client()
                resp = fetch_with_manual_retry(
                    client, "POST", "https://api.vapi.ai/analytics",
                    headers=headers, json=analytics_query, timeout=10,
                    max_retries=5, backoff_factor=2,
                )
                client.close()

                if resp.status_code == 201:
                    result_data = resp.json()
                    result_list = result_data[0].get("result", []) if result_data else []
                    current_calls = (
                        result_list[-1].get("historyConcurrency", 0)
                        if result_list
                        else 0
                    )
                    if current_calls < max_concurrent:
                        return True
                    print(
                        f"VAPI channel wait ({current_calls}/{max_concurrent}) "
                        f"for task {task_id}, cycle {cycle + 1}"
                    )
                else:
                    print(
                        f"VAPI analytics fetch failed: {resp.status_code}, "
                        f"retrying for task {task_id}"
                    )
            except Exception as ex:
                print(f"Error checking VAPI concurrency: {ex}")

            time.sleep(wait_interval)

        return False

    def _execute_vapi_general(
        self,
        data: dict,
        agent_id: str,
        formatted_number: str,
        contact_name: str,
        auth_token: str,
    ) -> dict:
        """Execute a general VAPI call (no agent ID, inline script)."""
        import requests
        from super.core.config.constants import AGENTS_SEARCH_API

        vapi_phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID")
        if not auth_token or not vapi_phone_number_id:
            return self._result_to_response(CallResult(
                status="failed", error="Missing VAPI credentials.",
                data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
            ))

        objective = data.get("objective", "")

        # Fetch agent persona
        try:
            resp = requests.post(
                AGENTS_SEARCH_API, json={"query": "Agent", "handle": [agent_id]}
            )
            resp.raise_for_status()
            agent_docs = resp.json().get("data", [])
            persona = agent_docs[0]["metadata"]["persona"] if agent_docs else "Assistant"
        except Exception:
            persona = "Assistant"

        prompt = (
            f"[Role] You are an AI assistant named {persona} who communicates in English. "
            f"Your primary task is to execute the call with given objective: {objective}.\n"
            f"[Context] You are engaged with a customer to deliver the message: {objective}. "
            f"Once message is delivered, end the call by using 'endcallFunction'.\n"
            f"[Response Guidelines] Keep responses brief. Just deliver the message. "
            f"Maintain a calm, empathetic, and professional tone. "
            f"Answer only the question posed by the user. "
            f"Never say the word 'function' nor 'tools' nor the name of the Available functions. "
            f"Never say ending the call or 'triggering the end of the call'."
        )

        json_data = {
            "phoneNumberId": vapi_phone_number_id,
            "customer": {"name": contact_name, "number": formatted_number},
            "assistant": {
                "transcriber": {
                    "provider": "deepgram", "language": "en", "model": "nova-2",
                },
                "model": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "messages": [{"role": "system", "content": prompt}],
                },
                "voice": {
                    "provider": "11labs",
                    "voiceId": "wlmwDR77ptH6bKHZui0l",
                    "model": "eleven_turbo_v2",
                },
                "firstMessageMode": "assistant-speaks-first",
                "firstMessage": f"Hi I have a message for You from {contact_name}",
                "endCallMessage": "Thank You",
                "recordingEnabled": True,
                "endCallFunctionEnabled": True,
                "startSpeakingPlan": {"smartEndpointingEnabled": True},
            },
        }

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        print(
            f"[VAPI-General] Dispatching call | "
            f"agent: {agent_id} | number: {formatted_number} | "
            f"phone_id: {vapi_phone_number_id}"
        )

        response = requests.post(
            "https://api.vapi.ai/call/phone", headers=headers, json=json_data,
        )
        if response.status_code in [200, 201]:
            call_data = response.json()
            call_id = call_data.get("id", "")
            print(
                f"[VAPI-General] Call created | "
                f"call_id: {call_id} | number: {formatted_number} | status: in_progress"
            )
            return self._result_to_response(CallResult(
                status="in_progress",
                call_id=call_id,
                customer=contact_name,
                contact_number=data.get("contact_number"),
                data={
                    "call_id": call_id,
                    "call_type": "outbound",
                    "cost": 0,
                    "type": "outboundPhoneCall",
                },
            ))

        print(
            f"[VAPI-General] Call failed | "
            f"number: {formatted_number} | status: {response.status_code} | "
            f"response: {response.text[:200]}"
        )
        return self._result_to_response(CallResult(
            status="failed",
            error=f"Failed to start general VAPI call: {response.text}",
            data={"call_type": "outbound", "cost": 0, "type": "outboundPhoneCall"},
        ))

    # ── LiveKit ──────────────────────────────────────────────────────────

    def _execute_livekit(
        self, data: dict, task_id: str, agent_id: str, config: dict
    ) -> dict:
        """Execute a LiveKit call via livekit.api (API client only)."""
        from livekit import api
        from super_services.libs.core.utils import get_env_name

        contact_number = data.get("contact_number", "")
        contact_name = data.get("contact_name") or data.get("name") or "User"
        data["task_id"] = task_id
        data["agent_id"] = agent_id

        metadata = {"data": data, "model_config": config, "call_type": "outbound"}

        try:
            # Generate unique room ID
            sanitized_number = "".join(c for c in contact_number if c.isalnum())[-10:]
            unique_id = str(uuid.uuid4())[:8]
            room_name = (
                f"call_{sanitized_number}_{unique_id}"
                if sanitized_number
                else f"call_{unique_id}"
            )
            env = get_env_name()
            agent_name = os.environ.get(
                "AGENT_NAME", f"unpod-{env}-general-agent-v3"
            )

            print(
                f"[LiveKit] Dispatching call for task {task_id} | "
                f"agent: {agent_id} | lk_agent: {agent_name} | "
                f"room: {room_name} | number: {contact_number}"
            )

            _run_async(
                self._livekit_create_dispatch(
                    agent_name, room_name, json.dumps(metadata)
                )
            )

            print(
                f"[LiveKit] Dispatch created for task {task_id} | "
                f"room: {room_name} | status: in_progress"
            )

            return self._result_to_response(CallResult(
                status="in_progress",
                call_id="",
                customer=contact_name,
                contact_number=contact_number,
                data={
                    "call_type": "outbound",
                    "cost": 0,
                    "type": "outbound",
                    "thread_id": task_id,
                    "room_name": room_name,
                },
            ))

        except Exception as e:
            print(
                f"[LiveKit] Dispatch failed for task {task_id} | "
                f"error: {e}"
            )
            return self._result_to_response(CallResult(
                status="failed",
                error=str(e),
                data={
                    "call_type": "outbound",
                    "cost": 0,
                    "type": "outbound",
                    "error": str(e),
                },
            ))

    @staticmethod
    async def _livekit_create_dispatch(agent_name, room_name, metadata):
        """Create LiveKit agent dispatch and close API client.

        LiveKitAPI() is created here (not in the caller) so that the
        aiohttp session binds to the *current* event loop — required
        when _run_async spawns a new loop in a background thread.
        """
        from livekit import api

        lkapi = api.LiveKitAPI()
        try:
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=agent_name, room=room_name, metadata=metadata,
                )
            )
        finally:
            await lkapi.aclose()

    # ── Pipecat Fallback ────────────────────────────────────────────────

    def _execute_pipecat(
        self, data: dict, task_id: str, agent_id: str, instructions: str = None
    ) -> dict:
        """Pipecat runs locally — fallback to full execute_call (heavy deps)."""
        from super.app.call_execution import execute_call
        from super_services.voice.models.config import ModelConfig, MessageCallBack

        contact_number = data.get("contact_number", "")
        print(
            f"[Pipecat] Dispatching call for task {task_id} | "
            f"agent: {agent_id} | number: {contact_number}"
        )

        result = _run_async(
            execute_call(
                data=data,
                task_id=task_id,
                agent_id=agent_id,
                instructions=instructions,
                model_config=ModelConfig(),
                callback=MessageCallBack(),
            )
        )

        print(
            f"[Pipecat] Call completed for task {task_id} | "
            f"status: {result.get('status', 'unknown')}"
        )
        return result
