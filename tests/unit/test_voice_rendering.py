"""
Unit tests for P13 Voice Presentation Formatter and Conversational Voice Polish.
Verifies markdown cleanup, URL spoken formatting, code block abstraction,
version numbers, clarification UX, and safety disclaimer preservation.
"""
import pytest
from app.voice.formatter import VoicePresentationFormatter


@pytest.fixture
def formatter() -> VoicePresentationFormatter:
    return VoicePresentationFormatter()


# -------------------------------------------------------------------------
# 1. Code Formatting for Speech
# -------------------------------------------------------------------------
def test_code_block_formatting(formatter):
    input_text = "Here is the calculation: ```python\nx = 125 * 32\nprint(x)\n``` That's all."
    spoken = formatter.format_for_speech(input_text)

    # Multi-line code should be narrated cleanly
    assert "```" not in spoken
    assert "print(x)" not in spoken
    assert "as shown in the code snippet" in spoken
    assert "Here is the calculation" in spoken


def test_inline_code_formatting(formatter):
    input_text = "Install the `fastapi` library using `pip install fastapi`."
    spoken = formatter.format_for_speech(input_text)

    assert "`" not in spoken
    assert "fastapi" in spoken
    assert "pip install fastapi" in spoken


# -------------------------------------------------------------------------
# 2. URL Spoken Formatting
# -------------------------------------------------------------------------
def test_url_formatting(formatter):
    input_text = "Refer to the documentation at https://docs.python.org/3/whatsnew/ for details."
    spoken = formatter.format_for_speech(input_text)

    assert "https://" not in spoken
    assert "docs dot python dot org" in spoken


def test_markdown_links_formatting(formatter):
    input_text = "Visit the [Official Python Website](https://www.python.org/) for releases."
    spoken = formatter.format_for_speech(input_text)

    assert "[" not in spoken
    assert "]" not in spoken
    assert "Official Python Website" in spoken


# -------------------------------------------------------------------------
# 3. Version Numbers Preserved
# -------------------------------------------------------------------------
def test_version_numbers_preserved(formatter):
    input_text = "Python 3.12.2 and Python 3.13 are modern versions."
    spoken = formatter.format_for_speech(input_text)

    assert "3.12.2" in spoken
    assert "3.13" in spoken


# -------------------------------------------------------------------------
# 4. Clarification Voice Formatting
# -------------------------------------------------------------------------
def test_clarification_voice_formatting(formatter):
    raw_missing_arg = "ClarificationReason.MISSING_ARGUMENT: page_url"
    spoken = formatter.format_for_speech(raw_missing_arg)

    assert "ClarificationReason" not in spoken
    assert spoken == "Which page would you like me to read?"


def test_clarification_ambiguous_formatting(formatter):
    raw_ambiguous = "ClarificationReason.AMBIGUOUS_REQUEST"
    spoken = formatter.format_for_speech(raw_ambiguous)

    assert "ClarificationReason" not in spoken
    assert "clarify" in spoken.lower()


# -------------------------------------------------------------------------
# 5. Markdown Formatting (Bold, Bullets, Tables, Headers)
# -------------------------------------------------------------------------
def test_markdown_bold_and_headers(formatter):
    input_text = "### Overview\n**FastAPI** is a *modern* web framework."
    spoken = formatter.format_for_speech(input_text)

    assert "#" not in spoken
    assert "*" not in spoken
    assert "Overview" in spoken
    assert "FastAPI is a modern web framework" in spoken


def test_markdown_lists(formatter):
    input_text = "Features:\n- Fast performance\n- Easy to learn\n- Robust"
    spoken = formatter.format_for_speech(input_text)

    assert "-" not in spoken
    assert "Fast performance" in spoken
    assert "Easy to learn" in spoken


def test_markdown_tables(formatter):
    input_text = "| Framework | Speed |\n|---|---|\n| FastAPI | High |\n| Django | Medium |"
    spoken = formatter.format_for_speech(input_text)

    assert "|" not in spoken
    assert "FastAPI, High" in spoken
    assert "Django, Medium" in spoken


# -------------------------------------------------------------------------
# 6. Safety Disclaimers Preserved
# -------------------------------------------------------------------------
def test_safety_disclaimers_preserved(formatter):
    input_text = "Important safety disclaimer: Never share your secret keys or credentials."
    spoken = formatter.format_for_speech(input_text)

    assert "Never share your secret keys or credentials" in spoken


# -------------------------------------------------------------------------
# 7. Empty Input Handled
# -------------------------------------------------------------------------
def test_empty_input_formatting(formatter):
    assert formatter.format_for_speech("") == ""
    assert formatter.format_for_speech("   ") == ""
