"""
P15.5 — Unit Tests for Expressive ElevenLabs TTS Integration.
Validates emoji normalization, markdown/citation cleaning, deterministic tone mapping,
ElevenLabs synthesis/streaming, authentication failure handling, and resilient local fallback.
"""
import io
import math
import struct
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
from app.voice.audio import CanonicalAudioFormat, is_wav_container
from app.voice.config import VoiceConfig
from app.voice.exceptions import TTSError, TTSTimeoutError, TTSUnavailableError
from app.voice.schemas import TTSResult
from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.preprocessor import (
    SpeechPreparationResult,
    SpeechTextPreprocessor,
    ToneMapper,
)
from app.voice.tts.resilient_adapter import ResilientTTSAdapter


# -------------------------------------------------------------------------
# 1. Emoji and Markdown Normalization Tests
# -------------------------------------------------------------------------

def test_emoji_normalization():
    """Verify emojis are stripped so synthesizers do not speak character descriptions."""
    prep = SpeechTextPreprocessor()
    raw_text = "Good morning! 🌞 The weather is sunny ☀️. Great job! 🚀👍"
    cleaned = prep.clean_emojis(raw_text)

    # Emoji characters stripped
    assert "🌞" not in cleaned
    assert "☀️" not in cleaned
    assert "🚀" not in cleaned
    assert "👍" not in cleaned
    assert "sun" not in cleaned.lower() or "sunny" in cleaned.lower()
    # Meaningful text preserved
    assert "Good morning!" in cleaned
    assert "The weather is sunny" in cleaned
    assert "Great job!" in cleaned


def test_markdown_normalization():
    """Verify Markdown markers (bold, italics, headings, backticks, bullets) are cleaned."""
    prep = SpeechTextPreprocessor()
    markdown_text = (
        "### Overview\n"
        "Here is the **important** update about `main.py`:\n"
        "- First *item*\n"
        "- Second __item__\n"
        "```python\nprint('hello')\n```\n"
        "---\nDone."
    )
    result = prep.clean_markdown_and_citations(markdown_text)

    assert "###" not in result
    assert "**" not in result
    assert "__" not in result
    assert "*" not in result
    assert "`" not in result
    assert "---" not in result
    assert "Overview." in result
    assert "important update about main.py" in result
    assert "First item" in result
    assert "Second item" in result
    assert "as shown in the code snippet." in result
    assert "Done." in result


# -------------------------------------------------------------------------
# 2. URL and Citation Cleaning Tests
# -------------------------------------------------------------------------

def test_url_and_citation_cleaning():
    """Verify bracketed citations and raw URLs are converted to natural spoken phrasing."""
    prep = SpeechTextPreprocessor()
    text = (
        "According to the documentation [1], Python is popular [source: official_docs]. "
        "Check [Python Docs](https://docs.python.org/3/) or visit https://www.github.com/org/repo."
    )
    result = prep.clean_markdown_and_citations(text)

    # Bracketed citation markers stripped
    assert "[1]" not in result
    assert "[source: official_docs]" not in result
    assert "Python Docs" in result
    # Raw URL replaced with spoken domain
    assert "github dot com" in result
    assert "https://" not in result


# -------------------------------------------------------------------------
# 3. Punctuation Preservation & Factual Integrity
# -------------------------------------------------------------------------

def test_punctuation_and_factual_preservation():
    """Verify punctuation and all factual words are strictly preserved."""
    prep = SpeechTextPreprocessor()
    factual = "At 10:30 AM, system latency was 45.2ms, CPU load at 78%, and 3 nodes were active. Is everything OK?"
    result = prep.preprocess(factual)

    assert result.clean_spoken_text == "At 10:30 AM, system latency was 45.2ms, CPU load at 78%, and 3 nodes were active. Is everything OK?"
    # Question mark preserved
    assert result.clean_spoken_text.endswith("?")
    # Numbers and punctuation preserved
    assert "10:30" in result.clean_spoken_text
    assert "45.2ms" in result.clean_spoken_text
    assert "78%" in result.clean_spoken_text


# -------------------------------------------------------------------------
# 4. Controlled Tone Mapping Tests
# -------------------------------------------------------------------------

def test_tone_mapping_categories():
    """Verify deterministic tone mapper correctly classifies emotional categories."""
    mapper = ToneMapper()

    # Greetings / friendly
    tone, tag = mapper.detect_tone("Hello! How can I help you today?")
    assert tone == "friendly"
    assert tag == "[friendly]"

    # Celebration / success
    tone, tag = mapper.detect_tone("Congratulations! All 757 tests passed successfully.")
    assert tone == "celebration"
    assert tag == "[excited]"

    # Supportive / empathetic
    tone, tag = mapper.detect_tone("I'm sorry to hear that. Don't worry, let's fix this together.")
    assert tone == "supportive"
    assert tag == "[calm]"

    # Questions / curious
    tone, tag = mapper.detect_tone("Which option would you prefer for the database migration?")
    assert tone == "curious"
    assert tag == "[curious]"

    # Neutral factual explanation (default)
    tone, tag = mapper.detect_tone(
        "PostgreSQL utilizes Multi-Version Concurrency Control to handle parallel transactions."
    )
    assert tone == "neutral"
    assert tag is None


def test_uncertain_tone_defaults_to_neutral():
    """Verify uncertain or ambiguous tone defaults to neutral without arbitrary emotion."""
    mapper = ToneMapper()
    tone, tag = mapper.detect_tone("The scheduled task has completed with status code 0.")
    assert tone == "neutral"
    assert tag is None


# -------------------------------------------------------------------------
# 5. Audio-Tag Placement and Stripping
# -------------------------------------------------------------------------

def test_audio_tag_placement_and_stripping():
    """Verify audio tags are placed only for supported models and stripped for fallback."""
    prep = SpeechTextPreprocessor()

    # Supported model (eleven_v4_turbo) receives tag
    res_v4 = prep.preprocess("Hello there!", target_provider="elevenlabs", model_id="eleven_v4_turbo")
    assert res_v4.audio_tag == "[friendly]"
    assert res_v4.expressive_text.startswith("[friendly]")

    # Non-supported provider (local) does NOT receive tag
    res_local = prep.preprocess("Hello there!", target_provider="local")
    assert res_local.audio_tag is None
    assert not res_local.expressive_text.startswith("[friendly]")

    # Tag stripping helper cleanly removes bracketed audio tags
    tagged_input = "[friendly] Hello there! [calm] It is okay."
    stripped = SpeechTextPreprocessor.strip_audio_tags(tagged_input)
    assert stripped == "Hello there! It is okay."
    assert "[" not in stripped and "]" not in stripped


# -------------------------------------------------------------------------
# 6. ElevenLabs Successful Synthesis and Streaming
# -------------------------------------------------------------------------

def _generate_synthetic_pcm_chunk(num_samples: int = 1600) -> bytes:
    """Generate 16kHz 16-bit mono PCM bytes."""
    buf = bytearray()
    for i in range(num_samples):
        val = int(4000.0 * math.sin(2.0 * math.pi * 440.0 * i / 16000.0))
        buf.extend(struct.pack("<h", val))
    return bytes(buf)


@pytest.mark.asyncio
async def test_elevenlabs_synthesize_success():
    """Verify synthesize produces canonical 16kHz mono 16-bit WAV audio."""
    adapter = ElevenLabsTTSAdapter(
        api_key="test-api-key",
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_v4_turbo",
    )

    mock_client = MagicMock()
    pcm_data = _generate_synthetic_pcm_chunk(3200)  # 0.2s of 16kHz audio

    async def mock_convert_generator(*args, **kwargs):
        yield pcm_data

    mock_client.text_to_dialogue.convert = MagicMock(return_value=mock_convert_generator())

    with patch.object(adapter, "_get_client", return_value=mock_client):
        res = await adapter.synthesize("Hello world!")

    assert isinstance(res, TTSResult)
    assert res.provider == "elevenlabs"
    assert res.content_type == "audio/wav"
    assert is_wav_container(res.audio_bytes)

    # Check WAV container header properties
    with wave.open(io.BytesIO(res.audio_bytes), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


@pytest.mark.asyncio
async def test_elevenlabs_synthesize_stream_success():
    """Verify synthesize_stream yields canonical PCM chunks with 2-byte alignment."""
    adapter = ElevenLabsTTSAdapter(
        api_key="test-api-key",
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_v4_turbo",
    )

    mock_client = MagicMock()
    chunk1 = _generate_synthetic_pcm_chunk(800)
    chunk2 = _generate_synthetic_pcm_chunk(800)

    async def mock_stream_generator(*args, **kwargs):
        yield chunk1
        yield chunk2

    mock_client.text_to_dialogue.stream = MagicMock(return_value=mock_stream_generator())

    with patch.object(adapter, "_get_client", return_value=mock_client):
        chunks = []
        async for chunk_res in adapter.synthesize_stream("Testing streaming synthesis"):
            chunks.append(chunk_res)

    assert len(chunks) == 2
    for c in chunks:
        assert c.provider == "elevenlabs"
        assert len(c.audio_bytes) % 2 == 0  # 16-bit aligned


# -------------------------------------------------------------------------
# 7. Missing API Key and Invalid Credentials
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_elevenlabs_missing_api_key_raises_hard_config():
    """Verify missing API key raises TTSUnavailableError classified as HARD_CONFIG."""
    adapter = ElevenLabsTTSAdapter(api_key=None)

    with pytest.raises(TTSUnavailableError) as exc_info:
        await adapter.synthesize("Hello")

    assert "API key is not configured" in str(exc_info.value)
    cat = classify_cloud_error(exc_info.value)
    assert cat == CloudErrorCategory.HARD_CONFIG


@pytest.mark.asyncio
async def test_elevenlabs_invalid_credentials_raises_hard_config():
    """Verify 401/403 ApiError raises TTSUnavailableError classified as HARD_CONFIG."""
    from elevenlabs.core.api_error import ApiError

    adapter = ElevenLabsTTSAdapter(api_key="invalid-key")
    mock_client = MagicMock()

    api_err = ApiError(status_code=401, body={"detail": "Unauthorized"})
    mock_client.text_to_dialogue.convert = MagicMock(side_effect=api_err)

    with patch.object(adapter, "_get_client", return_value=mock_client):
        with pytest.raises(TTSUnavailableError) as exc_info:
            await adapter.synthesize("Hello")

    assert "authentication failed" in str(exc_info.value).lower()
    cat = classify_cloud_error(exc_info.value)
    assert cat == CloudErrorCategory.HARD_CONFIG


# -------------------------------------------------------------------------
# 8. Network Failure and Resilient Local Fallback
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resilient_tts_adapter_network_failure_fallback():
    """Verify recoverable network failure triggers local fallback without flapping."""
    import httpx

    primary = ElevenLabsTTSAdapter(api_key="valid-key")
    mock_primary_client = MagicMock()
    mock_primary_client.text_to_dialogue.convert = MagicMock(
        side_effect=httpx.ConnectError("Network unreachable")
    )

    fallback = LocalTTSAdapter()
    fallback_wav = wave.open(io.BytesIO(), "wb")
    fallback_wav.setnchannels(1)
    fallback_wav.setsampwidth(2)
    fallback_wav.setframerate(16000)
    fallback_wav.writeframes(_generate_synthetic_pcm_chunk(1600))
    fallback_wav_bytes = fallback_wav._file.getvalue()

    fallback.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_bytes=fallback_wav_bytes,
            content_type="audio/wav",
            duration_seconds=0.1,
            provider="local_pyttsx3",
        )
    )

    resilient = ResilientTTSAdapter(
        primary_provider=primary,
        fallback_provider=fallback,
        fallback_enabled=True,
    )

    with patch.object(primary, "_get_client", return_value=mock_primary_client):
        res = await resilient.synthesize("Hello! We are testing fallback.")

    assert res.provider == "local_pyttsx3"
    assert resilient.provider_name == "local_pyttsx3"
    assert resilient.tts_mode == "LOCAL_FALLBACK"

    # Verify fallback method received cleaned text without bracketed audio tags
    fallback.synthesize.assert_called_once()
    called_text = fallback.synthesize.call_args[0][0]
    assert not called_text.startswith("[")


@pytest.mark.asyncio
async def test_resilient_tts_adapter_auth_error_preserves_speech_via_fallback():
    """Verify authentication errors log clearly, avoid remote retries, and preserve speech via local TTS."""
    from elevenlabs.core.api_error import ApiError

    primary = ElevenLabsTTSAdapter(api_key="bad-key")
    mock_primary_client = MagicMock()
    mock_primary_client.text_to_dialogue.convert = MagicMock(
        side_effect=ApiError(status_code=401, body={"detail": "Unauthorized"})
    )

    fallback = LocalTTSAdapter()
    fallback_wav = wave.open(io.BytesIO(), "wb")
    fallback_wav.setnchannels(1)
    fallback_wav.setsampwidth(2)
    fallback_wav.setframerate(16000)
    fallback_wav.writeframes(_generate_synthetic_pcm_chunk(1600))
    fallback_wav_bytes = fallback_wav._file.getvalue()

    fallback.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_bytes=fallback_wav_bytes,
            content_type="audio/wav",
            duration_seconds=0.1,
            provider="local_pyttsx3",
        )
    )

    resilient = ResilientTTSAdapter(
        primary_provider=primary,
        fallback_provider=fallback,
        fallback_enabled=True,
    )

    with patch.object(primary, "_get_client", return_value=mock_primary_client):
        res = await resilient.synthesize("Hello!")

    # Speech output is preserved via local fallback
    assert res.provider == "local_pyttsx3"
    assert resilient.provider_name == "local_pyttsx3"
    # Mode marked OFFLINE to prevent repeated remote retries on subsequent turns
    assert resilient.tts_mode == "OFFLINE"
    fallback.synthesize.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_tts_adapter_auth_error_raises_when_fallback_disabled():
    """Verify auth errors raise directly when fallback is disabled."""
    from elevenlabs.core.api_error import ApiError

    primary = ElevenLabsTTSAdapter(api_key="bad-key")
    mock_primary_client = MagicMock()
    mock_primary_client.text_to_dialogue.convert = MagicMock(
        side_effect=ApiError(status_code=401, body={"detail": "Unauthorized"})
    )

    resilient = ResilientTTSAdapter(
        primary_provider=primary,
        fallback_provider=None,
        fallback_enabled=False,
    )

    with patch.object(primary, "_get_client", return_value=mock_primary_client):
        with pytest.raises(TTSUnavailableError):
            await resilient.synthesize("Hello!")

    assert resilient.tts_mode == "OFFLINE"


# -------------------------------------------------------------------------
# 9. Factory Resolution Tests
# -------------------------------------------------------------------------

def test_create_tts_provider_factory_elevenlabs():
    """Verify create_tts_provider instantiates ResilientTTSAdapter wrapping ElevenLabs."""
    cfg = VoiceConfig(
        tts_provider="elevenlabs",
        elevenlabs_api_key="test-key",
        tts_fallback_enabled=True,
    )
    provider = create_tts_provider(cfg)

    assert isinstance(provider, ResilientTTSAdapter)
    assert isinstance(provider.primary_provider, ElevenLabsTTSAdapter)
    assert isinstance(provider.fallback_provider, LocalTTSAdapter)
    assert provider.primary_provider.provider_name == "elevenlabs"
