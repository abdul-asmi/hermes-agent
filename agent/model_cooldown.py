"""Persistent per-model cooldowns for quota-limited providers."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_STATE_VERSION = 1
_DEFAULT_COOLDOWN_SECONDS = 60.0
_PACIFIC = ZoneInfo("America/Los_Angeles")
_RETRY_RE = re.compile(
    r"(?:retry(?:ing)?(?:\s+suggested)?\s+in|retry\s+after)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|ms|s|sec(?:onds?)?|m|min(?:utes?)?)",
    re.IGNORECASE,
)


def _state_path() -> Path:
    return get_hermes_home() / "cache" / "model_cooldowns.json"


@contextmanager
def _state_lock():
    """Serialize state changes across gateway, CLI, and cron processes."""
    with _LOCK:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = path.with_suffix(".lock").open("a+", encoding="utf-8")
        try:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            lock_handle.close()


def _key(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}::{model.strip().lower()}"


def _load_state() -> dict:
    path = _state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("cooldowns"), dict):
            return payload
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"version": _STATE_VERSION, "cooldowns": {}}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _retry_delay_seconds(message: str) -> Optional[float]:
    match = _RETRY_RE.search(message or "")
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ns":
        return value / 1_000_000_000
    if unit in {"us", "µs"}:
        return value / 1_000_000
    if unit == "ms":
        return value / 1_000
    if unit.startswith("m") and unit != "ms":
        return value * 60
    return value


def _next_pacific_midnight(now: Optional[datetime] = None) -> float:
    current = now or datetime.now(tz=_PACIFIC)
    current = current.astimezone(_PACIFIC)
    tomorrow = current.date() + timedelta(days=1)
    reset = datetime.combine(tomorrow, datetime.min.time(), tzinfo=_PACIFIC)
    return (reset + timedelta(minutes=2)).timestamp()


def classify_cooldown(
    provider: str,
    message: str,
    *,
    retry_after: Optional[float] = None,
    now: Optional[float] = None,
) -> tuple[float, str]:
    """Return an absolute expiry timestamp and a human-readable reason."""
    timestamp = time.time() if now is None else now
    text = (message or "").lower()
    provider_name = (provider or "").strip().lower()

    if (
        provider_name == "gemini"
        and "generate_content_free_tier_requests" in text
    ):
        current = datetime.fromtimestamp(timestamp, tz=_PACIFIC)
        return _next_pacific_midnight(current), "Gemini free-tier daily request quota"

    delay = retry_after
    if delay is None:
        delay = _retry_delay_seconds(message)
    delay = max(float(delay or _DEFAULT_COOLDOWN_SECONDS), 1.0)
    return timestamp + delay + 1.0, "temporary provider rate limit"


def record_model_cooldown(
    provider: str,
    model: str,
    message: str,
    *,
    retry_after: Optional[float] = None,
) -> Optional[dict]:
    if not provider or not model:
        return None
    expires_at, reason = classify_cooldown(
        provider,
        message,
        retry_after=retry_after,
    )
    entry = {
        "provider": provider,
        "model": model,
        "expires_at": expires_at,
        "reason": reason,
        "updated_at": time.time(),
    }
    with _state_lock():
        state = _load_state()
        state["cooldowns"][_key(provider, model)] = entry
        _save_state(state)
    logger.info(
        "Model cooldown recorded: %s/%s until %s (%s)",
        provider,
        model,
        datetime.fromtimestamp(expires_at).astimezone().isoformat(),
        reason,
    )
    return entry


def get_active_model_cooldown(
    provider: str,
    model: str,
    *,
    now: Optional[float] = None,
) -> Optional[dict]:
    timestamp = time.time() if now is None else now
    with _state_lock():
        state = _load_state()
        entry = state["cooldowns"].get(_key(provider, model))
        if not isinstance(entry, dict):
            return None
        try:
            expires_at = float(entry.get("expires_at") or 0)
        except (TypeError, ValueError):
            expires_at = 0
        if expires_at > timestamp:
            return dict(entry)
        state["cooldowns"].pop(_key(provider, model), None)
        _save_state(state)
    return None


def format_cooldown(entry: dict) -> str:
    expires_at = float(entry.get("expires_at") or 0)
    expiry = datetime.fromtimestamp(expires_at).astimezone()
    return f"{entry.get('reason', 'rate limit')} until {expiry:%Y-%m-%d %H:%M %Z}"


def activate_available_fallback(agent: Any) -> bool:
    """Skip a cooled-down active model before the first API call of a turn."""
    entry = get_active_model_cooldown(
        getattr(agent, "provider", ""),
        getattr(agent, "model", ""),
    )
    if not entry:
        return False
    if getattr(agent, "_fallback_index", 0) >= len(
        getattr(agent, "_fallback_chain", []) or []
    ):
        return False
    description = format_cooldown(entry)
    logger.info(
        "Skipping cooled-down model %s/%s: %s",
        agent.provider,
        agent.model,
        description,
    )
    agent._buffer_status(
        f"⏭️ Skipping {agent.model}: {description}. Using fallback..."
    )
    return bool(agent._try_activate_fallback())
