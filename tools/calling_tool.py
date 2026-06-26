"""Calling Tool -- initiate outbound phone calls via ElevenLabs Conversational AI."""

import asyncio
import concurrent.futures
import os
import re
import json
import logging
import requests
import subprocess
import threading
import time
from typing import Dict, Any, Optional

from tools.registry import registry, tool_error, tool_result
from gateway.session_context import get_session_env
from hermes_cli.env_loader import load_hermes_dotenv

logger = logging.getLogger(__name__)

# Load dotenv to ensure keys are available at module level
load_hermes_dotenv()

CALLING_SCHEMA = {
    "name": "make_phone_call",
    "description": (
        "Initiates an outbound phone call to the specified phone number using the ElevenLabs "
        "Conversational AI agent. "
        "If no phone_number is provided (or if the user requests to call 'me' or 'myself'), "
        "it will call the owner's configured personal phone number. "
        "Access to this tool is restricted: only the owner can trigger calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": (
                    "The recipient's phone number in E.164 format (e.g. +14155551234). "
                    "Omit or use 'me' / 'myself' to call the owner's configured personal number."
                ),
            },
            "reason": {
                "type": "string",
                "description": "The brief reason, context, or purpose for the call. Highly recommended to include.",
            },
            "agent_mode": {
                "type": "string",
                "enum": ["asmi", "personal"],
                "description": (
                    "The agent mode preset to run. 'asmi' targets the startup testing agent "
                    "(uses ELEVENLABS_ASMI_AGENT_ID), 'personal' targets your personal agent "
                    "(uses ELEVENLABS_PERSONAL_AGENT_ID). Falls back to ELEVENLABS_AGENT_ID if not found."
                ),
            },
            "agent_id": {
                "type": "string",
                "description": "Optional raw ElevenLabs agent ID to use directly, overriding agent_mode and defaults.",
            },
            "custom_prompt": {
                "type": "string",
                "description": (
                    "Optional system prompt instruction override to dynamically direct the agent's behavior. "
                    "Requires 'System prompt override' to be enabled in the ElevenLabs agent Security settings."
                ),
            },
            "first_message": {
                "type": "string",
                "description": (
                    "Optional custom first message for the agent to speak. "
                    "Requires 'First message override' to be enabled in the ElevenLabs agent Security settings."
                ),
            },
        },
        "required": [],
    },
}

def check_calling_requirements() -> bool:
    """Check if outbound calling requirements are met."""
    load_hermes_dotenv()
    return all(os.environ.get(k) for k in ["ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID", "ELEVENLABS_PHONE_NUMBER_ID"])

def _clean_phone_number(num: str) -> str:
    """Clean and format phone number to E.164 format."""
    cleaned = re.sub(r"[^\d+]", "", num)
    if not cleaned.startswith("+"):
        cleaned = "+" + cleaned
    return cleaned

def _format_transcript(transcript_list: list) -> str:
    """Format ElevenLabs transcript list into a readable string."""
    if not transcript_list:
        return "(no transcript captured)"
    lines = []
    for turn in transcript_list:
        role = str(turn.get("role") or turn.get("speaker") or "unknown").lower()
        text = str(turn.get("message") or turn.get("text") or "").strip()
        label = "User" if role in {"user", "caller"} else "ElevenLabs Agent"
        if text:
            lines.append(f"- *{label}*: {text}")
    return "\n".join(lines)

def _poll_and_deliver_call(conversation_id: str, platform: str, chat_id: str, api_key: str):
    """Background thread to poll ElevenLabs, fetch the call summary, download the recording, and send them to the user."""
    logger.info(f"Started background polling for ElevenLabs conversation {conversation_id}")
    
    url = f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}"
    headers = {"xi-api-key": api_key}
    
    # 1. Poll until the conversation is completed or timeout is reached
    timeout_seconds = 300
    poll_interval = 5
    elapsed = 0
    status = "unknown"
    detail = {}
    
    while elapsed < timeout_seconds:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                detail = resp.json()
                status = str(detail.get("status") or "").lower().strip()
                if status in {"done", "completed", "finished", "ended", "failed", "error", "aborted"}:
                    logger.info(f"Conversation {conversation_id} completed with status: {status}")
                    break
            else:
                logger.warning(f"Error polling conversation {conversation_id}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Exception polling conversation {conversation_id}: {e}")
            
        time.sleep(poll_interval)
        elapsed += poll_interval
        
    if status not in {"done", "completed", "finished", "ended"}:
        logger.warning(f"Conversation {conversation_id} did not complete successfully or timed out. Status: {status}")
        if platform == "whatsapp" and chat_id:
            # Notify user of failure
            _send_whatsapp_text(chat_id, f"📞 *Call Update*\n\nOutbound call failed or ended unexpectedly.\n*Status:* {status}")
        return
        
    # Wait another 5 seconds for ElevenLabs post-processing/summary/audio generation
    time.sleep(5)
    
    try:
        # Refetch final detail
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            detail = resp.json()
            
        analysis = detail.get("analysis") or {}
        title = analysis.get("call_summary_title") or "ElevenLabs Call"
        summary = analysis.get("transcript_summary") or "(No summary generated by ElevenLabs)"
        transcript = detail.get("transcript") or []
        
        metadata = detail.get("metadata") or {}
        duration = metadata.get("call_duration_secs") or detail.get("call_duration_secs") or detail.get("duration_secs") or "unknown"
        
        # Download call recording
        audio_url = f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}/audio"
        audio_resp = requests.get(audio_url, headers=headers, timeout=30)
        
        audio_path = None
        if audio_resp.status_code == 200:
            audio_dir = os.path.expanduser("~/.hermes/audio_cache")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"call_{conversation_id}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_resp.content)
            logger.info(f"Downloaded call recording to {audio_path}")
        else:
            logger.warning(f"Failed to download call recording: {audio_resp.status_code}")
            
        # Format the transcript text
        formatted_transcript = _format_transcript(transcript)
        
        # Build the summary report message
        report = (
            f"📞 *Call Summary: {title}*\n"
            f"──────────────\n"
            f"*Duration:* {duration}s\n"
            f"*Status:* {status}\n\n"
            f"*ElevenLabs Summary:*\n{summary}\n\n"
            f"*Transcript:*\n{formatted_transcript}"
        )
        
        # Deliver to the originating platform
        if platform == "bluebubbles" and chat_id:
            caf_path = _mp3_to_caf(audio_path) if audio_path else None
            _send_bluebubbles_audio(chat_id, caf_path, audio_path, report)
        elif platform == "whatsapp" and chat_id:
            # Send the text summary
            _send_whatsapp_text(chat_id, report)

            # Send the audio file if downloaded successfully
            if audio_path and os.path.exists(audio_path):
                _send_whatsapp_audio(chat_id, audio_path)
        else:
            logger.info(f"Call report generated for CLI:\n{report}")
            if audio_path:
                logger.info(f"Audio file saved at: {audio_path}")
                
    except Exception as e:
        logger.exception(f"Error during post-call delivery for {conversation_id}: {e}")

def _get_bridge_port() -> int:
    """Get the local WhatsApp bridge port from config."""
    bridge_port = 3000
    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.WHATSAPP)
        if pconfig and pconfig.extra:
            bridge_port = pconfig.extra.get("bridge_port", 3000)
    except Exception:
        pass
    return bridge_port

def _send_whatsapp_text(chat_id: str, message: str):
    """Post text message to local WhatsApp bridge."""
    port = _get_bridge_port()
    try:
        requests.post(
            f"http://localhost:{port}/send",
            json={"chatId": chat_id, "message": message},
            timeout=15
        )
    except Exception as e:
        logger.error(f"Failed to send background text to bridge: {e}")

def _send_whatsapp_audio(chat_id: str, file_path: str):
    """Post audio file to local WhatsApp bridge."""
    port = _get_bridge_port()
    try:
        requests.post(
            f"http://localhost:{port}/send-media",
            json={"chatId": chat_id, "filePath": file_path, "mediaType": "audio"},
            timeout=60
        )
    except Exception as e:
        logger.error(f"Failed to send background audio to bridge: {e}")


def _mp3_to_caf(mp3_path: str) -> Optional[str]:
    """Convert an MP3 file to CAF (Core Audio Format) for native iMessage audio bubbles.

    Tries afconvert (macOS built-in) first, then ffmpeg as a fallback.
    Returns the CAF file path on success, None if conversion is unavailable.
    """
    caf_path = mp3_path.rsplit(".", 1)[0] + ".caf"
    try:
        result = subprocess.run(
            ["afconvert", "-f", "caff", "-d", "LEI16", "-c", "1", mp3_path, caf_path],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.isfile(caf_path):
            logger.info(f"afconvert: {mp3_path} → {caf_path}")
            return caf_path
        logger.warning(
            "afconvert failed (rc=%d): %s",
            result.returncode,
            result.stderr.decode(errors="replace"),
        )
    except FileNotFoundError:
        logger.debug("afconvert not found, trying ffmpeg")
    except Exception as exc:
        logger.warning("afconvert error: %s", exc)

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1",
             "-acodec", "pcm_s16le", caf_path],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.isfile(caf_path):
            logger.info(f"ffmpeg: {mp3_path} → {caf_path}")
            return caf_path
        logger.warning(
            "ffmpeg failed (rc=%d): %s",
            result.returncode,
            result.stderr.decode(errors="replace"),
        )
    except FileNotFoundError:
        logger.debug("ffmpeg not found")
    except Exception as exc:
        logger.warning("ffmpeg error: %s", exc)

    return None


def _send_bluebubbles_audio(chat_id: str, caf_path: Optional[str], mp3_path: Optional[str], report: str) -> None:
    """Deliver a call recording to an iMessage chat via the live BlueBubbles adapter.

    Sends the text summary first, then attempts to upload the audio as a native
    iMessage voice-memo bubble (CAF + isAudioMessage=true). Falls back to sending
    the MP3 as a generic document attachment if native send fails or CAF is unavailable.
    """
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None

    if runner is None:
        logger.warning("[calling] No gateway runner available for BlueBubbles delivery")
        return

    try:
        from gateway.config import Platform
        adapter = runner.adapters.get(Platform.BLUEBUBBLES)
    except Exception:
        adapter = None

    if adapter is None:
        logger.warning("[calling] BlueBubbles adapter not available")
        return

    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        logger.warning("[calling] Gateway event loop not available for BlueBubbles delivery")
        return

    async def _deliver():
        await adapter.send(chat_id, report)

        # Try native audio bubble first (CAF)
        if caf_path and os.path.isfile(caf_path):
            result = await adapter.send_voice(chat_id, caf_path)
            if result.success:
                logger.info("[calling] Native iMessage audio bubble sent: %s", caf_path)
                return
            logger.warning("[calling] Native audio send failed: %s — falling back to attachment", result.error)

        # Fall back to generic attachment (MP3 or whatever we have)
        fallback = mp3_path or caf_path
        if fallback and os.path.isfile(fallback):
            await adapter.send_document(chat_id, fallback)

    future = asyncio.run_coroutine_threadsafe(_deliver(), loop)
    try:
        future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        logger.warning("[calling] BlueBubbles audio delivery timed out")
    except Exception as exc:
        logger.error("[calling] BlueBubbles audio delivery error: %s", exc)

def make_phone_call(args: dict, **kwargs) -> str:
    """Execute the outbound phone call."""
    # 1. Enforce access control / authorization gate
    platform = get_session_env("HERMES_SESSION_PLATFORM")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    user_id = get_session_env("HERMES_SESSION_USER_ID")

    is_authorized = False
    if not platform or platform.lower() in {"cli", "terminal"}:
        is_authorized = True
    elif platform.lower() == "whatsapp":
        owner_jid = "197602905739324@lid"
        if chat_id == owner_jid or user_id == owner_jid:
            is_authorized = True

    if not is_authorized:
        logger.warning(
            f"Unauthorized phone call attempt from platform={platform}, chat_id={chat_id}, user_id={user_id}"
        )
        return tool_error("Unauthorized. Calling features are restricted to the owner.")

    # 2. Check and load environment variables
    load_hermes_dotenv()
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    phone_number_id = os.environ.get("ELEVENLABS_PHONE_NUMBER_ID")
    personal_number = os.environ.get("USER_PERSONAL_PHONE_NUMBER")

    # Resolve agent ID based on mode or argument
    agent_mode = args.get("agent_mode", "").strip().lower()
    arg_agent_id = args.get("agent_id", "").strip()

    if arg_agent_id:
        agent_id = arg_agent_id
    elif agent_mode == "asmi":
        agent_id = os.environ.get("ELEVENLABS_ASMI_AGENT_ID") or os.environ.get("ELEVENLABS_AGENT_ID")
    elif agent_mode == "personal":
        agent_id = os.environ.get("ELEVENLABS_PERSONAL_AGENT_ID") or os.environ.get("ELEVENLABS_AGENT_ID")
    else:
        agent_id = os.environ.get("ELEVENLABS_AGENT_ID")

    if not all([api_key, agent_id, phone_number_id]):
        return tool_error(
            "Outbound call configuration is incomplete. "
            "Please check ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID (or mode-specific ID), and ELEVENLABS_PHONE_NUMBER_ID."
        )

    # 3. Determine target number
    phone_param = args.get("phone_number", "").strip()
    reason = args.get("reason", "").strip()

    target_number = ""
    if not phone_param or phone_param.lower() in {"me", "myself"}:
        if not personal_number:
            return tool_error("USER_PERSONAL_PHONE_NUMBER is not configured.")
        target_number = personal_number
    else:
        target_number = phone_param

    # Clean and normalize the number
    try:
        target_number = _clean_phone_number(target_number)
    except Exception as e:
        return tool_error(f"Failed to parse phone number: {e}")

    # Validate target number length/format
    if len(target_number) < 8 or not re.match(r"^\+\d+$", target_number):
        return tool_error(f"Invalid E.164 phone number: {target_number}")

    # 4. Trigger the outbound call
    url = "https://api.elevenlabs.io/v1/convai/twilio/outbound-call"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "agent_id": agent_id,
        "agent_phone_number_id": phone_number_id,
        "to_number": target_number
    }

    # Pass optional context if reason, custom_prompt, or first_message is provided
    client_data = {
        "type": "conversation_initiation_client_data"
    }
    if reason:
        client_data["dynamic_variables"] = {
            "reason": reason,
            "context": reason
        }

    custom_prompt = args.get("custom_prompt", "").strip()
    first_message = args.get("first_message", "").strip()

    config_override = {}
    if custom_prompt:
        config_override["agent"] = config_override.get("agent") or {}
        config_override["agent"]["prompt"] = {
            "prompt": custom_prompt
        }
    if first_message:
        config_override["agent"] = config_override.get("agent") or {}
        config_override["agent"]["first_message"] = first_message

    if config_override:
        client_data["conversation_config_override"] = config_override

    if "dynamic_variables" in client_data or "conversation_config_override" in client_data:
        payload["conversation_initiation_client_data"] = client_data

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            convo_id = data.get("conversation_id", "")
            logger.info(f"Phone call initiated successfully to {target_number}. Conversation ID: {convo_id}")
            
            # Start background thread to poll and deliver summary/recording
            threading.Thread(
                target=_poll_and_deliver_call,
                args=(convo_id, platform, chat_id or user_id, api_key),
                daemon=True
            ).start()
            
            return tool_result(
                success=True,
                message=f"Phone call successfully initiated to {target_number}.",
                conversation_id=convo_id
            )
        else:
            try:
                err_detail = resp.json().get("detail", {}).get("message") or resp.text
            except Exception:
                err_detail = resp.text
            logger.error(f"ElevenLabs outbound call failed: code={resp.status_code} detail={err_detail}")
            return tool_error(f"ElevenLabs API error (code {resp.status_code}): {err_detail}")
    except Exception as e:
        logger.exception(f"Exception triggered during outbound call: {e}")
        return tool_error(f"Failed to place outbound call: {e}")

# Register the tool
registry.register(
    name="make_phone_call",
    toolset="calling",
    schema=CALLING_SCHEMA,
    handler=make_phone_call,
    check_fn=check_calling_requirements,
    requires_env=["ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID", "ELEVENLABS_PHONE_NUMBER_ID"],
    emoji="📞",
)
