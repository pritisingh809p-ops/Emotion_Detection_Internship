"""
Nationality / Category Conditional Routing Logic Module.

Ethical Design Principle:
- Nationality, ethnicity, race, or country is NEVER inferred from facial features.
- The user explicitly selects one of the four supported categories (and enters
  their declared nationality when 'Other' is selected).
- Only the models required for the chosen category are executed.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from internship_tasks.task4_Nationality_Detection.modules.emotion_detection import detect_emotion
from internship_tasks.task4_Nationality_Detection.modules.age_detection import estimate_age
from internship_tasks.task4_Nationality_Detection.modules.dress_colour import detect_dress_colour

CATEGORY_INDIAN = "Indian"
CATEGORY_US = "United States"
CATEGORY_AFRICAN = "African country/region"
CATEGORY_OTHER = "Other"

SUPPORTED_CATEGORIES: List[str] = [
    CATEGORY_INDIAN,
    CATEGORY_US,
    CATEGORY_AFRICAN,
    CATEGORY_OTHER,
]

# Maps each user-selected category to the exact required prediction modules
CATEGORY_REQUIREMENTS: Dict[str, List[str]] = {
    CATEGORY_INDIAN: ["age", "emotion", "dress_colour"],
    CATEGORY_US: ["age", "emotion"],
    CATEGORY_AFRICAN: ["emotion", "dress_colour"],
    CATEGORY_OTHER: ["declared_nationality", "emotion"],
}


def get_required_attributes(category: str) -> List[str]:
    """
    Return the ordered list of attribute keys required for a given category.

    Raises:
        ValueError: If `category` is not one of the supported categories.
    """
    if category not in CATEGORY_REQUIREMENTS:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Valid options are: {', '.join(SUPPORTED_CATEGORIES)}."
        )
    return list(CATEGORY_REQUIREMENTS[category])


def validate_category_input(
    category: str, declared_nationality: Optional[str] = None
) -> str:
    """
    Validate user-selected category and optional declared nationality text.

    Returns:
        str: Cleaned declared nationality string (empty string if not applicable).

    Raises:
        ValueError: If category is invalid or 'Other' is chosen without a declared nationality.
    """
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"Invalid category selected: '{category}'. Please choose a valid category."
        )

    cleaned_nationality = (declared_nationality or "").strip()
    if category == CATEGORY_OTHER:
        if not cleaned_nationality:
            raise ValueError(
                "Please enter your declared nationality in the text field when 'Other' is selected."
            )
        if len(cleaned_nationality) < 2:
            raise ValueError("Declared nationality must be at least 2 characters long.")
        if len(cleaned_nationality) > 80:
            raise ValueError("Declared nationality must not exceed 80 characters.")

    return cleaned_nationality


def run_attribute_pipeline(
    bgr_image: np.ndarray,
    category: str,
    declared_nationality: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute only the attribute detection models required for the user-selected category.

    Conditional Matrix:
    - Indian:                Age + Emotion + Dress Colour
    - United States:         Age + Emotion
    - African country/region: Emotion + Dress Colour
    - Other:                 Declared Nationality + Emotion

    Args:
        bgr_image: Preprocessed OpenCV BGR image array.
        category: User-selected category string.
        declared_nationality: User-entered nationality string (required when category == 'Other').

    Returns:
        Dict[str, Any]: Dictionary containing category metadata, executed tasks, and results.
    """
    cleaned_nationality = validate_category_input(category, declared_nationality)
    required_tasks = get_required_attributes(category)

    results: Dict[str, Any] = {
        "category": category,
        "executed_tasks": required_tasks,
        "outputs": {},
    }

    if "declared_nationality" in required_tasks:
        results["outputs"]["declared_nationality"] = {
            "success": True,
            "value": cleaned_nationality,
        }

    if "age" in required_tasks:
        results["outputs"]["age"] = estimate_age(bgr_image)

    if "emotion" in required_tasks:
        results["outputs"]["emotion"] = detect_emotion(bgr_image)

    if "dress_colour" in required_tasks:
        results["outputs"]["dress_colour"] = detect_dress_colour(bgr_image)

    return results
