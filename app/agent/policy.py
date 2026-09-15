import re

from app.schemas.agent import AgentDecision


def apply_policy(
    user_message: str,
    decision: AgentDecision,
) -> AgentDecision:
    """Apply deterministic MAI routing and safety rules."""

    lowered = user_message.lower().strip()
    has_math_expression = bool(
        re.search(
            r"\d\s*[\+\-\*/%]\s*\d",
            lowered,
        )
    )

    arithmetic_keywords = (
        "calculate",
        "multiply",
        "multiplied",
        "divide",
        "divided",
        "plus",
        "minus",
        "subtract",
        "add",
        "times",
        "percent",
    )

    datetime_keywords = (
        "what time",
        "what's the time",
        "time in",
        "current time",
        "time is it",
        "today's date",
        "todays date",
        "what date",
        "date in",
        "today",
        "what day",
        "current date",
        "current datetime",
    )

    contains_arithmetic_keyword = any(
        keyword in lowered
        for keyword in arithmetic_keywords
    )

    contains_datetime_keyword = any(
        keyword in lowered
        for keyword in datetime_keywords
    )

    if contains_datetime_keyword:
        decision.intent = "datetime"
        decision.route = "tool"
        decision.tools = ["datetime"]
        decision.needs_clarification = False
        decision.reason = (
            "Current date and time requests are handled "
            "by the datetime tool."
        )
        return decision

    if contains_arithmetic_keyword or has_math_expression:
        decision.intent = "arithmetic"
        decision.route = "tool"
        decision.tools = ["calculator"]
        decision.needs_clarification = False
        decision.reason = (
            "Arithmetic requests are handled by "
            "the calculator tool."
        )
        return decision

    return decision