"""
P13 — Conversational Voice Presentation Formatter.
Converts grounded markdown text into natural, spoken-ready speech.
Ensures TTS speaks clean, conversational audio without modifying underlying text data.
"""
import re
from urllib.parse import urlparse


class VoicePresentationFormatter:
    """
    Deterministic speech-rendering formatter.
    Pipeline: Grounded Text -> Voice Presentation Formatter -> TTS.
    Never modifies factual meaning or safety disclaimers.
    """

    def format_for_speech(self, text: str) -> str:
        """
        Render text into natural spoken narration.
        Strips markdown syntax, simplifies technical URLs, summarizes code blocks,
        and ensures conversational fluency.
        """
        if not text or not text.strip():
            return ""

        spoken = text.strip()

        # 1. Format raw clarification enum markers if present
        spoken = self._format_clarification_ux(spoken)

        # 2. Multi-line code blocks: replace with conversational description
        spoken = re.sub(
            r"```[a-zA-Z0-9_-]*\n[\s\S]*?```",
            "as shown in the code snippet.",
            spoken,
        )

        # 3. Inline code backticks: keep identifier, remove backticks
        spoken = re.sub(r"`([^`]+)`", r"\1", spoken)

        # 4. Markdown links: [Link Title](http://...) -> Link Title
        spoken = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", spoken)

        # 5. Raw URLs: format to conversational speech (e.g. domain dot org)
        spoken = self._format_urls_for_speech(spoken)

        # 6. Markdown tables: replace with comma-separated text
        spoken = self._format_tables_for_speech(spoken)

        # 7. Markdown headings (### Title) -> Title
        spoken = re.sub(r"^[#]+\s*(.+)$", r"\1.", spoken, flags=re.MULTILINE)

        # 8. Markdown bold/italics: **bold**, *italic*, __bold__
        spoken = re.sub(r"\*\*([^\*]+)\*\*", r"\1", spoken)
        spoken = re.sub(r"\*([^\*]+)\*", r"\1", spoken)
        spoken = re.sub(r"__([^_]+)__", r"\1", spoken)
        spoken = re.sub(r"_([^_]+)_", r"\1", spoken)

        # 9. Markdown list items (* item, - item, 1. item) -> item.
        spoken = re.sub(r"^\s*[\*\-\+]\s+", "", spoken, flags=re.MULTILINE)
        spoken = re.sub(r"^\s*\d+\.\s+", "", spoken, flags=re.MULTILINE)

        # 10. Horizontal rules (---, ***) -> empty
        spoken = re.sub(r"^\s*[-*_]{3,}\s*$", "", spoken, flags=re.MULTILINE)

        # 11. Emojis and decorative pictographs: strip so TTS does not speak Unicode names (e.g. 'sun with face')
        spoken = self._strip_emojis_for_speech(spoken)

        # 12. Normalize excessive whitespace and punctuation
        spoken = re.sub(r"\s+([,.!?])", r"\1", spoken)
        spoken = re.sub(r"\n+", ". ", spoken)
        spoken = re.sub(r"\s+", " ", spoken)
        spoken = re.sub(r"\.{2,}", ".", spoken)
        spoken = re.sub(r"\s+\.", ".", spoken)

        return spoken.strip()

    def _strip_emojis_for_speech(self, text: str) -> str:
        """
        Remove Unicode emojis and pictographs so speech synthesizers (TTS)
        do not vocalize character descriptions (e.g., 'sun with face', 'waving hand').
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs (includes 🌞 \U0001F31E)
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F680-\U0001F6FF"  # Transport & Map
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002600-\U000026FF"  # Misc symbols (e.g. ☀ \U00002600)
            "\U00002700-\U000027BF"  # Dingbats
            "\U0000FE00-\U0000FE0F"  # Variation Selectors
            "\U0000200D"             # Zero-width joiner
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def _format_clarification_ux(self, text: str) -> str:
        """Render any technical clarification reason into a natural question."""
        # e.g., "ClarificationReason.MISSING_ARGUMENT: page_url"
        match = re.search(r"ClarificationReason\.([A-Z_]+)(?::\s*([a-zA-Z0-9_]+))?", text)
        if match:
            reason = match.group(1)
            arg = match.group(2)
            if reason == "MISSING_ARGUMENT":
                if arg and "page" in arg.lower():
                    return "Which page would you like me to read?"
                elif arg and "url" in arg.lower():
                    return "Which URL would you like me to fetch?"
                return f"Could you please specify the {arg.replace('_', ' ') if arg else 'missing detail'}?"
            elif reason == "AMBIGUOUS_REQUEST":
                return "Could you please clarify your request?"
        return text

    def _format_urls_for_speech(self, text: str) -> str:
        """
        Convert URLs to human-friendly speech (e.g. 'docs dot python dot org').
        """
        url_regex = r"https?://(?:www\.)?([a-zA-Z0-9.\-_]+)(?:/[^\s]*)?"

        def replace_url(match):
            full_match = match.group(0)
            parsed = urlparse(full_match)
            domain = parsed.netloc or match.group(1)
            # Remove port if present
            domain = domain.split(":")[0]
            # Replace dots with ' dot '
            friendly = domain.replace(".", " dot ")
            return friendly

        return re.sub(url_regex, replace_url, text)

    def _format_tables_for_speech(self, text: str) -> str:
        """Convert markdown table rows to comma-separated speech."""
        lines = text.split("\n")
        out_lines = []
        for line in lines:
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip() and not set(c.strip()).issubset({"-", ":" })]
                if cells:
                    out_lines.append(", ".join(cells) + ".")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)
