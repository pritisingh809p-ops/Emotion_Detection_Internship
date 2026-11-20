"""Unit tests for conditional nationality/category routing logic."""

from unittest.mock import patch
import numpy as np
import pytest
from modules.nationality_logic import (
    CATEGORY_INDIAN,
    CATEGORY_US,
    CATEGORY_AFRICAN,
    CATEGORY_OTHER,
    get_required_attributes,
    validate_category_input,
    run_attribute_pipeline,
)


def test_required_attributes_mapping():
    assert get_required_attributes(CATEGORY_INDIAN) == ["age", "emotion", "dress_colour"]
    assert get_required_attributes(CATEGORY_US) == ["age", "emotion"]
    assert get_required_attributes(CATEGORY_AFRICAN) == ["emotion", "dress_colour"]
    assert get_required_attributes(CATEGORY_OTHER) == ["declared_nationality", "emotion"]


def test_invalid_category_raises():
    with pytest.raises(ValueError, match="Unsupported category"):
        get_required_attributes("Martian")


def test_other_category_requires_declared_nationality():
    with pytest.raises(ValueError, match="enter your declared nationality"):
        validate_category_input(CATEGORY_OTHER, "   ")

    assert validate_category_input(CATEGORY_OTHER, "  Kenyan ") == "Kenyan"


@patch("modules.nationality_logic.detect_dress_colour")
@patch("modules.nationality_logic.detect_emotion")
@patch("modules.nationality_logic.estimate_age")
def test_indian_category_executes_age_emotion_and_dress(mock_age, mock_emo, mock_dress):
    mock_age.return_value = {"success": True, "estimated_age": 25}
    mock_emo.return_value = {"success": True, "dominant_emotion": "Happy"}
    mock_dress.return_value = {"success": True, "colour_name": "Maroon"}

    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    res = run_attribute_pipeline(dummy_bgr, CATEGORY_INDIAN)

    assert set(res["outputs"].keys()) == {"age", "emotion", "dress_colour"}
    mock_age.assert_called_once()
    mock_emo.assert_called_once()
    mock_dress.assert_called_once()


@patch("modules.nationality_logic.detect_dress_colour")
@patch("modules.nationality_logic.detect_emotion")
@patch("modules.nationality_logic.estimate_age")
def test_us_category_skips_dress_colour(mock_age, mock_emo, mock_dress):
    mock_age.return_value = {"success": True, "estimated_age": 31}
    mock_emo.return_value = {"success": True, "dominant_emotion": "Neutral"}

    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    res = run_attribute_pipeline(dummy_bgr, CATEGORY_US)

    assert set(res["outputs"].keys()) == {"age", "emotion"}
    mock_age.assert_called_once()
    mock_emo.assert_called_once()
    mock_dress.assert_not_called()


@patch("modules.nationality_logic.detect_dress_colour")
@patch("modules.nationality_logic.detect_emotion")
@patch("modules.nationality_logic.estimate_age")
def test_african_category_skips_age(mock_age, mock_emo, mock_dress):
    mock_emo.return_value = {"success": True, "dominant_emotion": "Happy"}
    mock_dress.return_value = {"success": True, "colour_name": "Yellow"}

    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    res = run_attribute_pipeline(dummy_bgr, CATEGORY_AFRICAN)

    assert set(res["outputs"].keys()) == {"emotion", "dress_colour"}
    mock_age.assert_not_called()
    mock_emo.assert_called_once()
    mock_dress.assert_called_once()


@patch("modules.nationality_logic.detect_dress_colour")
@patch("modules.nationality_logic.detect_emotion")
@patch("modules.nationality_logic.estimate_age")
def test_other_category_skips_age_and_dress(mock_age, mock_emo, mock_dress):
    mock_emo.return_value = {"success": True, "dominant_emotion": "Surprise"}

    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    res = run_attribute_pipeline(dummy_bgr, CATEGORY_OTHER, declared_nationality="Japanese")

    assert set(res["outputs"].keys()) == {"declared_nationality", "emotion"}
    assert res["outputs"]["declared_nationality"]["value"] == "Japanese"
    mock_age.assert_not_called()
    mock_emo.assert_called_once()
    mock_dress.assert_not_called()
