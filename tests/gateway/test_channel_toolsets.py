"""Tests for per-channel toolsets configuration, resolution, and injection."""

import sys
import os
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, resolve_channel_toolsets
from gateway.session import SessionSource


class TestResolveChannelToolsets:
    def test_no_toolsets_returns_none(self):
        assert resolve_channel_toolsets({}, "123") is None

    def test_match_by_channel_id_empty_list(self):
        config_extra = {"channel_toolsets": {"12345": []}}
        assert resolve_channel_toolsets(config_extra, "12345") == []

    def test_match_by_channel_id_none_is_empty_list(self):
        config_extra = {"channel_toolsets": {"12345": None}}
        assert resolve_channel_toolsets(config_extra, "12345") == []

    def test_match_by_channel_id_list(self):
        config_extra = {"channel_toolsets": {"12345": ["web", "vision"]}}
        assert resolve_channel_toolsets(config_extra, "12345") == ["web", "vision"]

    def test_match_by_channel_id_string(self):
        config_extra = {"channel_toolsets": {"12345": "web, vision, clarify"}}
        assert resolve_channel_toolsets(config_extra, "12345") == ["web", "vision", "clarify"]

    def test_match_by_parent_id(self):
        config_extra = {"channel_toolsets": {"parent-1": ["todo"]}}
        assert resolve_channel_toolsets(config_extra, "child-1", parent_id="parent-1") == ["todo"]

    def test_exact_channel_overrides_parent(self):
        config_extra = {
            "channel_toolsets": {
                "child-1": ["web"],
                "parent-1": ["todo"]
            }
        }
        assert resolve_channel_toolsets(config_extra, "child-1", parent_id="parent-1") == ["web"]


from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


class MockWhatsAppAdapter(BasePlatformAdapter):
    def __init__(self, config=None, platform=None):
        if config is None:
            config = PlatformConfig(enabled=True, extra={"channel_toolsets": {"14086689990": ["web"]}})
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
async def test_run_agent_overrides_enabled_toolsets(monkeypatch, tmp_path):
    # Setup mocks for run_agent
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Mock gateway runner attributes
    import gateway.run as gateway_run
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = "Global prompt"
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner.hooks = types.SimpleNamespace(loaded_hooks=False)
    runner.config = types.SimpleNamespace(streaming=types.SimpleNamespace(enabled=False, transport="off"))
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    import threading
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.session_store = types.SimpleNamespace(
        get_or_create_session=lambda source: types.SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._get_proxy_url = lambda: None

    # Mock gateway config loader
    user_config_dict = {
        "platforms": {
            "whatsapp": {
                "extra": {
                    "channel_toolsets": {"14086689990": ["web"]}
                }
            }
        }
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: user_config_dict)
    
    whatsapp_adapter = MockWhatsAppAdapter()
    runner.adapters = {Platform.WHATSAPP: whatsapp_adapter}
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

    _CapturingAgent.last_init = None
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="14086689990",
        chat_type="dm",
        user_id="14086689990",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="Context prompt",
        history=[],
        source=source,
        session_id="session-1",
        session_key="agent:main:whatsapp:dm:14086689990",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    # Verify the tools are restricted to just "web"
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["web"]
