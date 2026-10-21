"""
P15.5 — Speech-Text Preparation and Deterministic Tone Mapping Layer.
Transforms grounded assistant responses into clean, conversational spoken text
while strictly preserving displayed text and factual integrity.
"""
from dataclasses import dataclass
import re
from typing import Optional
from urllib.parse import urlparse


# Comprehensive Unicode emoji regex to avoid TTS reading descriptions aloud
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs (e.g. 🌞, 🚀, 💡)
    "\U0001F600-\U0001F64F"  # Emoticons (e.g. 😀, 😂, 👍)
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (e.g. 🤖, 🧠)
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Misc symbols (e.g. ☀, ☕, ⚡)
    "\U00002700-\U000027BF"  # Dingbats (e.g. ✔, ❌, ✨)
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0000200D"             # Zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Citation patterns: e.g. [1], [2], [1, 2], [source: 3], [ref: foo]
CITATION_PATTERN = re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*|source:[^\]]+|ref:[^\]]+)\]", re.IGNORECASE)

# ElevenLabs audio delivery tags pattern
AUDIO_TAG_PATTERN = re.compile(
    r"\[(friendly|excited|upbeat|calm|empathetic|curious|whispers|laughs|sighs|pause|restrained|cheerfully)\]\s*",
    re.IGNORECASE,
)

# Models known to support audio delivery tags
TAG_SUPPORTED_MODELS = {
    "eleven_v4_turbo",
    "eleven_v4",
    "eleven_v3",
    "eleven_turbo_v2_5",
}


from dataclasses import dataclass, field
import re
from typing import Optional
from urllib.parse import urlparse


@dataclass
class SpeechPreparationResult:
    """Output of the speech preparation pipeline."""
    clean_spoken_text: str
    expressive_text: str
    detected_tone: str
    audio_tag: Optional[str] = None
    segments: list[str] = field(default_factory=list)


class ToneMapper:
    """
    Lightweight, deterministic tone mapper.
    Maps assistant response patterns to controlled emotional delivery
    without requiring additional LLM calls.
    """

    GREETING_PATTERNS = [
        re.compile(r"^(?:hello|hi|hey|good morning|good afternoon|good evening|welcome)\b", re.IGNORECASE),
        re.compile(r"\b(?:how can i help|glad to help|happy to assist|nice to meet you)\b", re.IGNORECASE),
    ]

    SUCCESS_PATTERNS = [
        re.compile(r"\b(?:congratulations|congrats|great job|awesome|well done|hooray|fantastic)\b", re.IGNORECASE),
        re.compile(r"\b(?:successfully (?:created|completed|deployed|passed|updated|executed))\b", re.IGNORECASE),
        re.compile(r"\b(?:all tests passed|build succeeded)\b", re.IGNORECASE),
    ]

    SUPPORTIVE_PATTERNS = [
        re.compile(r"\b(?:i(?:'m| am) (?:sorry|apologize)|don't worry|no worries|take your time)\b", re.IGNORECASE),
        re.compile(r"\b(?:i understand this is (?:frustrating|confusing|difficult)|here for you)\b", re.IGNORECASE),
        re.compile(r"\b(?:let's fix this together|it's (?:totally )?okay)\b", re.IGNORECASE),
    ]

    CURIOUS_PATTERNS = [
        re.compile(r"\b(?:did you know|interestingly|curious|i wonder)\b", re.IGNORECASE),
        re.compile(r"\b(?:which (?:one|option) would you (?:prefer|like)|what do you think)\b", re.IGNORECASE),
        re.compile(r"\b(?:would you like me to|shall we explore)\b", re.IGNORECASE),
    ]

    def detect_tone(self, text: str) -> tuple[str, Optional[str]]:
        """
        Classifies tone and returns (tone_name, audio_tag).
        Defaults to neutral with no audio tag if uncertain.
        """
        if not text or not text.strip():
            return "neutral", None

        sample = text.strip()

        # 1. Greetings & Positive Conversational Opening
        for p in self.GREETING_PATTERNS:
            if p.search(sample):
                return "friendly", "[friendly]"

        # 2. Genuine Success / Celebration
        for p in self.SUCCESS_PATTERNS:
            if p.search(sample):
                return "celebration", "[excited]"

        # 3. Supportive / Empathetic
        for p in self.SUPPORTIVE_PATTERNS:
            if p.search(sample):
                return "supportive", "[calm]"

        # 4. Appropriate Questions or Discoveries
        for p in self.CURIOUS_PATTERNS:
            if p.search(sample):
                return "curious", "[curious]"

        # Direct questions that invite discovery
        if sample.endswith("?") and any(sample.lower().startswith(q) for q in ("would you", "could you", "what if", "have you")):
            return "curious", "[curious]"

        # 5. Neutral Factual Delivery (Default)
        return "neutral", None


def segment_text_for_speech(text: str, max_segment_chars: int = 1500) -> list[str]:
    """
    Split speech text into logical, complete segments at paragraph or sentence boundaries.
    Guarantees:
    - Never discards any words or text.
    - Preserves paragraph/sentence order.
    - Never splits in the middle of words.
    - Keeps each segment <= max_segment_chars (unless an individual unbroken word is longer).
    """
    if not text or not text.strip():
        return []
    clean = text.strip()
    if len(clean) <= max_segment_chars:
        return [clean]

    # Split first into paragraphs
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n|\n", clean) if p.strip()]
    if not raw_paras:
        raw_paras = [clean]

    segments = []
    current_chunk: list[str] = []
    current_len = 0

    for para in raw_paras:
        para_len = len(para)
        sep_len = 1 if current_chunk else 0

        if current_len + sep_len + para_len <= max_segment_chars:
            current_chunk.append(para)
            current_len += sep_len + para_len
        else:
            if current_chunk:
                segments.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            if para_len <= max_segment_chars:
                current_chunk.append(para)
                current_len = para_len
            else:
                # Paragraph itself exceeds max_segment_chars: split by sentence
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                for sent in sentences:
                    sent_len = len(sent)
                    s_sep = 1 if current_chunk else 0
                    if current_len + s_sep + sent_len <= max_segment_chars:
                        current_chunk.append(sent)
                        current_len += s_sep + sent_len
                    else:
                        if current_chunk:
                            segments.append(" ".join(current_chunk))
                            current_chunk = []
                            current_len = 0

                        if sent_len <= max_segment_chars:
                            current_chunk.append(sent)
                            current_len = sent_len
                        else:
                            # Sentence itself exceeds max_segment_chars: split by words
                            words = sent.split()
                            for w in words:
                                w_len = len(w)
                                w_sep = 1 if current_chunk else 0
                                if current_len + w_sep + w_len <= max_segment_chars:
                                    current_chunk.append(w)
                                    current_len += w_sep + w_len
                                else:
                                    if current_chunk:
                                        segments.append(" ".join(current_chunk))
                                        current_chunk = []
                                        current_len = 0
                                    current_chunk.append(w)
                                    current_len = w_len

    if current_chunk:
        sep = "\n\n" if any("\n" in s for s in current_chunk) else " "
        segments.append(sep.join(current_chunk))

    return segments


class SpeechTextPreprocessor:
    """
    Dedicated speech-text preparation layer.
    Preprocesses assistant responses for TTS consumption:
    - Normalizes emojis (strips so names are not read aloud).
    - Cleans Markdown, raw URLs, and citation brackets.
    - Preserves sentence boundaries and factual wording.
    - Adds controlled audio delivery tags only for supported ElevenLabs models.
    """

    def __init__(self, tone_mapper: Optional[ToneMapper] = None):
        self.tone_mapper = tone_mapper or ToneMapper()

    def clean_emojis(self, text: str) -> str:
        """Strip Unicode emojis so speech synthesizers do not vocalize descriptions."""
        if not text:
            return ""
        return EMOJI_PATTERN.sub("", text)

    def clean_markdown_and_citations(self, text: str) -> str:
        """
        Strip Markdown formatting and citations while preserving
        meaningful punctuation and sentence flow.
        """
        if not text:
            return ""

        spoken = text.strip()

        # 1. Strip citation brackets e.g. [1], [source: doc]
        spoken = CITATION_PATTERN.sub("", spoken)

        # 2. Multi-line code blocks -> conversational description
        spoken = re.sub(
            r"```[a-zA-Z0-9_-]*\n[\s\S]*?```",
            "as shown in the code snippet.",
            spoken,
        )

        # 3. Inline code backticks: keep identifier, remove backticks
        spoken = re.sub(r"`([^`]+)`", r"\1", spoken)

        # 4. Markdown links: [Link Title](http://...) -> Link Title
        spoken = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", spoken)

        # 5. Raw URLs: simplify to conversational domain
        spoken = self._format_urls_for_speech(spoken)

        # 6. Markdown tables: convert to clean comma-separated lines
        spoken = self._format_tables_for_speech(spoken)

        # 7. Markdown headings (### Title) -> Title.
        spoken = re.sub(r"^[#]+\s*(.+)$", r"\1.", spoken, flags=re.MULTILINE)

        # 8. Markdown bold/italics: **bold**, *italic*, __bold__, _italic_
        spoken = re.sub(r"\*\*([^\*]+)\*\*", r"\1", spoken)
        spoken = re.sub(r"\*([^\*]+)\*", r"\1", spoken)
        spoken = re.sub(r"__([^_]+)__", r"\1", spoken)
        spoken = re.sub(r"_([^_]+)_", r"\1", spoken)

        # 9. Markdown list bullets (* item, - item, 1. item)
        spoken = re.sub(r"^\s*[\*\-\+]\s+", "", spoken, flags=re.MULTILINE)
        spoken = re.sub(r"^\s*\d+\.\s+", "", spoken, flags=re.MULTILINE)

        # 10. Horizontal rules (---, ***) -> empty
        spoken = re.sub(r"^\s*[-*_]{3,}\s*$", "", spoken, flags=re.MULTILINE)

        # 11. Normalize excessive whitespace and punctuation
        spoken = re.sub(r"\s+([,.!?])", r"\1", spoken)
        spoken = re.sub(r"\n+", ". ", spoken)
        spoken = re.sub(r"\s+", " ", spoken)
        spoken = re.sub(r"\.{2,}", ".", spoken)
        spoken = re.sub(r"\s+\.", ".", spoken)

        return spoken.strip()

    def _format_urls_for_speech(self, text: str) -> str:
        """Convert raw URLs to natural speech (e.g. domain dot com)."""
        url_regex = r"https?://(?:www\.)?([a-zA-Z0-9.\-_]+)(?:/[^\s]*)?"

        def replace_url(match):
            full_match = match.group(0)
            parsed = urlparse(full_match)
            domain = parsed.netloc or match.group(1)
            domain = domain.split(":")[0]
            return domain.replace(".", " dot ")

        return re.sub(url_regex, replace_url, text)

    def _format_tables_for_speech(self, text: str) -> str:
        """Convert markdown table rows to comma-separated speech."""
        lines = text.split("\n")
        out_lines = []
        for line in lines:
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip() and not set(c.strip()).issubset({"-", ":"})]
                if cells:
                    out_lines.append(", ".join(cells) + ".")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    @staticmethod
    def strip_audio_tags(text: str) -> str:
        """
        Strip any bracketed audio delivery tags so non-ElevenLabs providers
        or fallback local engines never vocalize them aloud.
        """
        if not text:
            return ""
        return AUDIO_TAG_PATTERN.sub("", text).strip()

    def preprocess(
        self,
        text: str,
        target_provider: str = "elevenlabs",
        model_id: Optional[str] = None,
        max_segment_chars: int = 1500,
    ) -> SpeechPreparationResult:
        """
        Receives grounded assistant response and produces clean and expressive speech text.
        """
        if not text or not text.strip():
            return SpeechPreparationResult(
                clean_spoken_text="",
                expressive_text="",
                detected_tone="neutral",
                audio_tag=None,
                segments=[],
            )

        # 1. Clean emojis first
        no_emojis = self.clean_emojis(text)

        # 2. Clean markdown, citations, URLs
        clean_spoken = self.clean_markdown_and_citations(no_emojis)

        # 3. Detect tone
        tone, tag = self.tone_mapper.detect_tone(clean_spoken)

        # 4. Determine expressive text based on provider and model capabilities
        prov_normalized = (target_provider or "").lower().strip()
        model_normalized = (model_id or "eleven_v4_turbo").lower().strip()

        is_elevenlabs = "eleven" in prov_normalized
        supports_tags = is_elevenlabs and (model_normalized in TAG_SUPPORTED_MODELS or "v4" in model_normalized or "v3" in model_normalized)

        if supports_tags and tag:
            expressive = f"{tag} {clean_spoken}"
        else:
            expressive = clean_spoken

        # 5. Segment into logical parts if long
        segments = segment_text_for_speech(clean_spoken, max_segment_chars=max_segment_chars)

        return SpeechPreparationResult(
            clean_spoken_text=clean_spoken,
            expressive_text=expressive,
            detected_tone=tone,
            audio_tag=tag if supports_tags else None,
            segments=segments,
        )

    # Alias for backwards compatibility
    prepare = preprocess

