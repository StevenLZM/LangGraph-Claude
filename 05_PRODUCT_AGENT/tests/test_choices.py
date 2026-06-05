from __future__ import annotations

from datetime import timedelta

from agent.choices import create_logistics_choice, create_refund_choice, resolve_choice, utc_now


def test_refund_choice_resolves_numeric_and_alias_input():
    choice_set = create_refund_choice(
        "ORD123456",
        {"order_id": "ORD123456", "refund_status": "confirmation_required"},
    )

    numeric = resolve_choice(choice_set, "选第1个")
    alias = resolve_choice(choice_set, "我确认退款")

    assert numeric["matched"] is True
    assert numeric["option"]["id"] == "confirm_refund"
    assert alias["matched"] is True
    assert alias["option"]["id"] == "confirm_refund"


def test_choice_resolution_reports_expired_choice_set():
    choice_set = create_logistics_choice([])
    choice_set["expires_at"] = (utc_now() - timedelta(seconds=1)).isoformat()

    resolved = resolve_choice(choice_set, "选第1个")

    assert resolved == {"matched": False, "expired": True, "option": None}
