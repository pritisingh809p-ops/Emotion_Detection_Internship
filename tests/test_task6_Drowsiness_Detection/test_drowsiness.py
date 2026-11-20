"""Unit tests for drowsiness detection logic."""
import numpy as np
from modules.drowsiness_logic import analyze_vehicle_occupants


def test_analyze_vehicle_occupants_valid():
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    res = analyze_vehicle_occupants(dummy_img)
    assert res["success"] is True
    assert res["total_people"] >= 1
    assert "alert_message" in res
