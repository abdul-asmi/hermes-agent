"""Tests for the make_phone_call tool."""

import os
import json
from unittest.mock import patch, MagicMock
import pytest

from tools.calling_tool import _clean_phone_number, make_phone_call

def test_clean_phone_number():
    assert _clean_phone_number("1234567890") == "+1234567890"
    assert _clean_phone_number("+1 (234) 567-8901") == "+12345678901"
    assert _clean_phone_number("  +123-456  ") == "+123456"

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
def test_unauthorized_platform(mock_load_env, mock_get_session_env):
    # If platform is whatsapp and chat_id/user_id is not the owner JID
    def side_effect(key, default=""):
        if key == "HERMES_SESSION_PLATFORM":
            return "whatsapp"
        if key == "HERMES_SESSION_CHAT_ID":
            return "987654321@lid"
        if key == "HERMES_SESSION_USER_ID":
            return "987654321@lid"
        return default
    mock_get_session_env.side_effect = side_effect

    res = make_phone_call({"phone_number": "+1234567890"})
    data = json.loads(res)
    assert "error" in data
    assert "Unauthorized" in data["error"]

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
def test_unauthorized_other_platform(mock_load_env, mock_get_session_env):
    # If platform is telegram, even if owner JID is passed
    def side_effect(key, default=""):
        if key == "HERMES_SESSION_PLATFORM":
            return "telegram"
        return default
    mock_get_session_env.side_effect = side_effect

    res = make_phone_call({"phone_number": "+1234567890"})
    data = json.loads(res)
    assert "error" in data
    assert "Unauthorized" in data["error"]

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
@patch.dict(os.environ, {}, clear=True)
def test_missing_config(mock_load_env, mock_get_session_env):
    # Authorized user (CLI) but missing env vars
    mock_get_session_env.return_value = "" # CLI
    
    res = make_phone_call({"phone_number": "+1234567890"})
    data = json.loads(res)
    assert "error" in data
    assert "Outbound call configuration is incomplete" in data["error"]

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
@patch("tools.calling_tool.requests.post")
@patch.dict(os.environ, {
    "ELEVENLABS_API_KEY": "test_key",
    "ELEVENLABS_AGENT_ID": "test_agent",
    "ELEVENLABS_PHONE_NUMBER_ID": "test_phone_id",
    "USER_PERSONAL_PHONE_NUMBER": "+19999999999"
})
def test_authorized_cli_success(mock_post, mock_load_env, mock_get_session_env):
    mock_get_session_env.return_value = "" # CLI
    
    # Mock successful ElevenLabs API response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"conversation_id": "convo_123"}
    mock_post.return_value = mock_resp

    # Make call to "me" (should resolve to USER_PERSONAL_PHONE_NUMBER)
    res = make_phone_call({"phone_number": "me", "reason": "Test outbound call"})
    data = json.loads(res)
    
    assert data.get("success") is True
    assert data.get("conversation_id") == "convo_123"
    
    # Verify post payload
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.elevenlabs.io/v1/convai/twilio/outbound-call"
    assert kwargs["headers"]["xi-api-key"] == "test_key"
    assert kwargs["json"]["to_number"] == "+19999999999"
    assert kwargs["json"]["agent_id"] == "test_agent"
    assert kwargs["json"]["agent_phone_number_id"] == "test_phone_id"
    assert kwargs["json"]["conversation_initiation_client_data"]["type"] == "conversation_initiation_client_data"
    assert kwargs["json"]["conversation_initiation_client_data"]["dynamic_variables"]["reason"] == "Test outbound call"
    assert kwargs["json"]["conversation_initiation_client_data"]["dynamic_variables"]["context"] == "Test outbound call"

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
@patch("tools.calling_tool.requests.post")
@patch.dict(os.environ, {
    "ELEVENLABS_API_KEY": "test_key",
    "ELEVENLABS_AGENT_ID": "test_agent",
    "ELEVENLABS_PHONE_NUMBER_ID": "test_phone_id"
})
def test_whatsapp_owner_success(mock_post, mock_load_env, mock_get_session_env):
    # Authorized user (WhatsApp Owner JID)
    def side_effect(key, default=""):
        if key == "HERMES_SESSION_PLATFORM":
            return "whatsapp"
        if key == "HERMES_SESSION_CHAT_ID":
            return "197602905739324@lid"
        if key == "HERMES_SESSION_USER_ID":
            return "197602905739324@lid"
        return default
    mock_get_session_env.side_effect = side_effect

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"conversation_id": "convo_whatsapp"}
    mock_post.return_value = mock_resp

    res = make_phone_call({"phone_number": "+14155551234"})
    data = json.loads(res)
    
    assert data.get("success") is True
    assert data.get("conversation_id") == "convo_whatsapp"
    mock_post.assert_called_once()
    assert mock_post.call_args[1]["json"]["to_number"] == "+14155551234"

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
@patch("tools.calling_tool.requests.post")
@patch.dict(os.environ, {
    "ELEVENLABS_API_KEY": "test_key",
    "ELEVENLABS_AGENT_ID": "test_agent",
    "ELEVENLABS_PHONE_NUMBER_ID": "test_phone_id"
})
def test_api_failure(mock_post, mock_load_env, mock_get_session_env):
    mock_get_session_env.return_value = "cli"
    
    # Mock failed API response
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request error detail"
    mock_resp.json.side_effect = ValueError("No JSON")
    mock_post.return_value = mock_resp

    res = make_phone_call({"phone_number": "+14155551234"})
    data = json.loads(res)
    
    assert "error" in data
    assert "ElevenLabs API error" in data["error"]
    assert "Bad Request error detail" in data["error"]

@patch("tools.calling_tool.get_session_env")
@patch("tools.calling_tool.load_hermes_dotenv")
@patch("tools.calling_tool.requests.post")
@patch.dict(os.environ, {
    "ELEVENLABS_API_KEY": "test_key",
    "ELEVENLABS_AGENT_ID": "default_agent",
    "ELEVENLABS_ASMI_AGENT_ID": "asmi_agent",
    "ELEVENLABS_PERSONAL_AGENT_ID": "personal_agent",
    "ELEVENLABS_PHONE_NUMBER_ID": "test_phone_id"
})
def test_make_phone_call_with_options(mock_post, mock_load_env, mock_get_session_env):
    mock_get_session_env.return_value = "cli"
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"conversation_id": "convo_opts"}
    mock_post.return_value = mock_resp

    # 1. Test agent_mode = "asmi" and overrides
    res = make_phone_call({
        "phone_number": "+14155551234",
        "agent_mode": "asmi",
        "custom_prompt": "Override prompt for Asmi",
        "first_message": "Override first message for Asmi"
    })
    data = json.loads(res)
    assert data.get("success") is True
    
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["agent_id"] == "asmi_agent"
    
    client_data = kwargs["json"]["conversation_initiation_client_data"]
    assert client_data["type"] == "conversation_initiation_client_data"
    assert client_data["conversation_config_override"]["agent"]["first_message"] == "Override first message for Asmi"
    assert client_data["conversation_config_override"]["agent"]["prompt"]["prompt"] == "Override prompt for Asmi"

    mock_post.reset_mock()

    # 2. Test agent_mode = "personal"
    res = make_phone_call({
        "phone_number": "+14155551234",
        "agent_mode": "personal"
    })
    data = json.loads(res)
    assert data.get("success") is True
    assert mock_post.call_args[1]["json"]["agent_id"] == "personal_agent"

    mock_post.reset_mock()

    # 3. Test explicit agent_id argument override
    res = make_phone_call({
        "phone_number": "+14155551234",
        "agent_id": "explicit_agent"
    })
    data = json.loads(res)
    assert data.get("success") is True
    assert mock_post.call_args[1]["json"]["agent_id"] == "explicit_agent"

