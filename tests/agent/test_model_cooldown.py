from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from agent.model_cooldown import (
    activate_available_fallback,
    classify_cooldown,
    get_active_model_cooldown,
    record_model_cooldown,
)


def test_gemini_free_tier_requests_cool_down_until_pacific_midnight():
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 6, 6, 19, 0, tzinfo=pacific)
    expires_at, reason = classify_cooldown(
        "gemini",
        "Quota exceeded: generate_content_free_tier_requests, limit: 20",
        now=now.timestamp(),
    )

    expiry = datetime.fromtimestamp(expires_at, tz=pacific)
    assert expiry.date().isoformat() == "2026-06-07"
    assert (expiry.hour, expiry.minute) == (0, 2)
    assert "daily" in reason.lower()


def test_token_limit_uses_advertised_short_retry_delay():
    now = 1000.0
    expires_at, reason = classify_cooldown(
        "gemini",
        "generate_content_free_tier_input_token_count. Please retry in 6.5s.",
        now=now,
    )

    assert expires_at == 1007.5
    assert "temporary" in reason


def test_recorded_cooldown_survives_a_fresh_lookup():
    record_model_cooldown(
        "gemini",
        "gemini-test",
        "Please retry in 30s.",
    )

    entry = get_active_model_cooldown("gemini", "gemini-test")
    assert entry is not None
    assert entry["model"] == "gemini-test"


def test_active_primary_cooldown_activates_fallback():
    record_model_cooldown(
        "gemini",
        "gemini-primary",
        "Please retry in 30s.",
    )
    statuses = []
    agent = SimpleNamespace(
        provider="gemini",
        model="gemini-primary",
        _fallback_index=0,
        _fallback_chain=[{"provider": "gemini", "model": "gemini-backup"}],
        _buffer_status=statuses.append,
        _try_activate_fallback=lambda: True,
    )

    assert activate_available_fallback(agent) is True
    assert statuses
