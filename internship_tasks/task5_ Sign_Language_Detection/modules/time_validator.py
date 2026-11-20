"""
Time Window Validation Module.

Enforces that the sign language model / system is operational only during
a specific configured time window (e.g., from 6 PM to 10 PM / 18:00 to 22:00).
"""

from datetime import datetime
from typing import Tuple


def is_operational_time(
    current_time: datetime = None, start_hour: int = 18, end_hour: int = 22
) -> Tuple[bool, str]:
    """
    Check if the current system time falls within the allowed operational time window.

    Args:
        current_time: Datetime object to evaluate (defaults to datetime.now()).
        start_hour: Opening hour in 24-hour format (default 18 for 6 PM).
        end_hour: Closing hour in 24-hour format (default 22 for 10 PM).

    Returns:
        Tuple[bool, str]: (is_allowed, status_message)
    """
    if current_time is None:
        current_time = datetime.now()

    current_hour = current_time.hour
    current_minute = current_time.minute

    # Convert times to total minutes from midnight for precise boundary checks
    current_total_mins = current_hour * 60 + current_minute
    start_total_mins = start_hour * 60
    end_total_mins = end_hour * 60

    if start_total_mins <= current_total_mins < end_total_mins:
        return (
            True,
            f"System is ACTIVE. Current time ({current_time.strftime('%H:%M')}) is within the "
            f"operational window ({start_hour:02d}:00 – {end_hour:02d}:00)."
        )
    else:
        return (
            False,
            f"System is LOCKED. Current time ({current_time.strftime('%H:%M')}) is outside the "
            f"authorized operational window ({start_hour:02d}:00 – {end_hour:02d}:00). "
            f"Please access the system between 6:00 PM and 10:00 PM."
        )
