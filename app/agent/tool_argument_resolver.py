import re


def resolve_calculator_arguments(
    user_message: str,
) -> dict[str, str]:
    """
    Extract the arithmetic expression from a natural-language
    calculator request.
    """

    text = user_message.lower().strip()

    # Remove common conversational prefixes.
    text = re.sub(
        r"^(what is|what's|calculate|can you calculate|please calculate)\s+",
        "",
        text,
    )

    # Remove question marks.
    text = text.rstrip("?").strip()

    return {
        "expression": text,
    }