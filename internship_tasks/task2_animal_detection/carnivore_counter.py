"""
Carnivore Counter Module
"""

CARNIVORES = {
    "cat",
    "dog",
    "bear"
}


def count_carnivores(detected_animals):
    """
    Count carnivorous animals.
    """
    return sum(1 for animal in detected_animals if animal in CARNIVORES)


def generate_summary(total_animals, carnivore_count):
    """
    Generate summary using total animals and carnivore count.
    """

    herbivore_count = total_animals - carnivore_count

    summary = f"""
🐾 Detection Summary

Total Animals Detected : {total_animals}

Carnivores : {carnivore_count}

Other Animals : {herbivore_count}
"""

    return summary


def get_alert_message(carnivore_count):
    """
    Generate alert message.
    """

    if carnivore_count == 0:
        return "✅ No carnivorous animals detected."

    elif carnivore_count == 1:
        return "⚠️ 1 carnivorous animal detected!"

    else:
        return f"🚨 {carnivore_count} carnivorous animals detected!"