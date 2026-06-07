"""Tests for the WhatsApp personal authentication and approval gate."""

import sys
import types
from unittest.mock import AsyncMock, MagicMock
import pytest
import asyncio
from pathlib import Path

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult, BasePlatformAdapter
from gateway.session import SessionSource

class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        return {
            "final_response": "response_from_agent",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


class MockWhatsAppAdapter(BasePlatformAdapter):
    def __init__(self, config=None, platform=None):
        if config is None:
            config = PlatformConfig(enabled=True)
        if platform is None:
            platform = Platform.WHATSAPP
        super().__init__(config, platform)
        self.sent_messages = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.sent_messages.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": "Mock Chat", "type": "dm"}


@pytest.mark.asyncio
async def test_auth_gate_flow(monkeypatch, tmp_path):
    # Setup mocks
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    import gateway.run as gateway_run
    
    # Mock hermes home path
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    # Instantiate runner manually to avoid full setup
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = "Global prompt"
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner.hooks = types.SimpleNamespace(
        loaded_hooks=False,
        emit=AsyncMock(),
    )
    runner.config = types.SimpleNamespace(
        streaming=types.SimpleNamespace(enabled=False, transport="off"),
        get_connected_platforms=lambda: [],
    )
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._queued_events = {}
    runner._pending_native_image_paths_by_session = {}
    runner._busy_ack_ts = {}
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._agent_cache = {}
    import threading
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_approvals = {}
    runner._failed_platforms = {}
    runner._update_prompt_pending = {}
    runner._last_resolved_model = {}
    runner._session_db = None
    runner._voice_mode = {}

    runner.session_store = types.SimpleNamespace(
        get_or_create_session=lambda source: types.SimpleNamespace(
            session_id="session-1",
            session_key="session-1",
            created_at=100,
            updated_at=200,
        ),
        load_transcript=lambda session_id: [],
        has_any_sessions=lambda: False,
        append_to_transcript=lambda *args, **kwargs: None,
        update_session=lambda *args, **kwargs: None,
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._get_proxy_url = lambda: None

    # Auth gate attributes
    runner._approved_jids = set()
    runner._pending_auth_messages = {}
    runner._last_unauthorized_jid = None
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True

    # Monkeypatch helper methods
    monkeypatch.setattr(runner, "_load_approved_jids", lambda: None)
    
    def mock_save():
        # Just write to a dummy file to verify it was called
        (tmp_path / "saved").touch()
    monkeypatch.setattr(runner, "_save_approved_jids", mock_save)

    # Setup adapter mock
    whatsapp_adapter = MockWhatsAppAdapter()
    runner.adapters = {Platform.WHATSAPP: whatsapp_adapter}

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_platform_config_key", lambda p: "whatsapp")
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-mock")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    # 1. Message from Owner -> Allowed immediately
    owner_source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="918639228484@s.whatsapp.net",
        chat_type="dm",
        user_id="918639228484@s.whatsapp.net",
    )
    owner_event = MessageEvent(
        text="hi agent",
        message_type="text",
        source=owner_source,
        message_id="msg-owner-1",
    )

    _CapturingAgent.last_init = None
    res = await runner._handle_message(owner_event)
    assert res == "response_from_agent"
    assert _CapturingAgent.last_init is not None

    # 2. Message from Unauthorized Sender -> Blocked and owner notified
    unauth_source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="14086689990@s.whatsapp.net",
        chat_type="dm",
        user_id="14086689990@s.whatsapp.net",
    )
    unauth_event = MessageEvent(
        text="eval trigger",
        message_type="text",
        source=unauth_source,
        message_id="msg-unauth-1",
    )

    _CapturingAgent.last_init = None
    whatsapp_adapter.sent_messages = []
    res = await runner._handle_message(unauth_event)
    
    # Should return "" (blocked) and NOT call agent
    assert res == ""
    assert _CapturingAgent.last_init is None
    # Verify owner notified
    assert len(whatsapp_adapter.sent_messages) >= 1
    assert any("Unauthorized WhatsApp message" in msg[1] for msg in whatsapp_adapter.sent_messages)
    # Verify cached pending message
    assert runner._pending_auth_messages["14086689990"] == unauth_event
    assert runner._last_unauthorized_jid == "14086689990"

    # 3. Owner sends "start" command -> Target approved and pending message processed
    start_event = MessageEvent(
        text="start",
        message_type="text",
        source=owner_source,
        message_id="msg-owner-approve",
    )

    whatsapp_adapter.sent_messages = []
    # Mocking _handle_message recursively since create_task runs it in the loop
    original_handle_message = runner._handle_message
    captured_runs = []
    async def mock_handle_message_spy(evt):
        captured_runs.append(evt)
        # Call original but don't do real agent run to avoid background complexity
        if "start" in getattr(evt, "text", ""):
            return await original_handle_message(evt)
        return "mocked_agent_run"
    
    monkeypatch.setattr(runner, "_handle_message", mock_handle_message_spy)

    res = await runner._handle_message(start_event)
    assert res == ""  # Command interceptor returns "" for owner to prevent echo
    
    # Target JID should be in approved set
    assert "14086689990" in runner._approved_jids
    assert (tmp_path / "saved").exists()

    # Success notification should be sent to owner
    assert any("Approved WhatsApp user '14086689990'" in msg[1] for msg in whatsapp_adapter.sent_messages)

    # Give event loop a tick to process scheduled task
    await asyncio.sleep(0.1)
    
    # Pending auth message should be processed/re-dispatched
    assert any(evt.text == "eval trigger" for evt in captured_runs)

    # 4. Verification of Sandboxing (non-owner WhatsApp user)
    # Write a mock roleplay_persona.md to tmp_path
    persona_content = "# Test Roleplay Persona\nFollow this persona strictly."
    (tmp_path / "roleplay_persona.md").write_text(persona_content)

    # Restore original handle_message so we can test the real agent runner path
    monkeypatch.setattr(runner, "_handle_message", original_handle_message)

    # Approved user sends a message
    approved_user_event = MessageEvent(
        text="simulate roleplay",
        message_type="text",
        source=unauth_source,
        message_id="msg-auth-user-1",
    )

    _CapturingAgent.last_init = None
    whatsapp_adapter.sent_messages = []

    # Run handle_message for the approved user
    await runner._handle_message(approved_user_event)

    # Verify that the capturing agent was initialized
    assert _CapturingAgent.last_init is not None
    # Verify that toolsets are stripped for strict sandboxing
    assert _CapturingAgent.last_init.get("enabled_toolsets") == []
    # Verify that the system prompt was replaced with the roleplay persona content
    assert _CapturingAgent.last_init.get("ephemeral_system_prompt") == persona_content

    # 5. Revocation via "stop" command from owner
    real_analysis = runner._generate_and_send_analysis
    mock_analysis = AsyncMock()
    runner._generate_and_send_analysis = mock_analysis
    
    stop_event = MessageEvent(
        text="stop",
        message_type="text",
        source=owner_source,
        message_id="msg-owner-stop",
    )

    res = await runner._handle_message(stop_event)
    assert res == ""  # Command interceptor returns ""

    # JID should be removed from approved set
    assert "14086689990" not in runner._approved_jids

    # Analysis generation should be triggered
    mock_analysis.assert_called_once_with("14086689990", owner_source.chat_id, owner_source.platform)

    # Restore real method
    runner._generate_and_send_analysis = real_analysis

    # 6. Verify direct execution of _generate_and_send_analysis
    # Set up some dummy transcript history in session_store mock
    dummy_history = [
        {"role": "user", "content": "hello hermes"},
        {"role": "assistant", "content": "hello player"},
    ]
    runner.session_store.load_transcript = lambda sid: dummy_history

    whatsapp_adapter.sent_messages = []
    _CapturingAgent.last_init = None

    await runner._generate_and_send_analysis("14086689990", owner_source.chat_id, owner_source.platform)

    # Verify agent was invoked to evaluate transcript
    assert _CapturingAgent.last_init is not None
    # Verify no tools were passed
    assert _CapturingAgent.last_init.get("enabled_toolsets") == []
    # Verify report message sent back to owner
    assert len(whatsapp_adapter.sent_messages) >= 1
    assert any("Evaluation Report for WhatsApp User" in msg[1] for msg in whatsapp_adapter.sent_messages)
