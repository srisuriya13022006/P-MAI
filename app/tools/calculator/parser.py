import re


NUMBER = r"(-?\d+(?:\.\d+)?)"


def normalize_expression(text: str) -> str | None:
    """Convert common natural-language arithmetic into an expression."""

    text = text.lower().strip()

    patterns = [
        (
            rf"{NUMBER}\s*(?:multiplied by|times|x)\s*{NUMBER}",
            r"\1 * \2",
        ),
        (
            rf"{NUMBER}\s*(?:divided by|over)\s*{NUMBER}",
            r"\1 / \2",
        ),
        (
            rf"{NUMBER}\s*(?:plus|added to)\s*{NUMBER}",
            r"\1 + \2",
        ),
        (
            rf"{NUMBER}\s*(?:minus|subtracted from)\s*{NUMBER}",
            r"\1 - \2",
        ),
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, text)

        if match:
            return re.sub(
                pattern,
                replacement,
                text,
                count=1,
            )

    # Explicit mathematical expression.
    expression_pattern = r"[-+*/().%\d\s]+"

    match = re.search(
        expression_pattern,
        text,
    )

    if match:
        expression = match.group().strip()

        if any(char.isdigit() for char in expression):
            return expression

    return None