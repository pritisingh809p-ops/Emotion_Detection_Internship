"""Unit tests for time window validator module."""
from datetime import datetime
from modules.time_validator import is_operational_time


def test_time_inside_window():
    # 8 PM (20:00) is inside 18:00 - 22:00
    dt = datetime(2026, 6, 1, 20, 30)
    allowed, msg = is_operational_time(dt, 18, 22)
    assert allowed is True
    assert "ACTIVE" in msg


def test_time_outside_window():
    # 2 PM (14:00) is outside 18:00 - 22:00
    dt = datetime(2026, 6, 1, 14, 0)
    allowed, msg = is_operational_time(dt, 18, 22)
    assert allowed is False
    assert "LOCKED" in msg
