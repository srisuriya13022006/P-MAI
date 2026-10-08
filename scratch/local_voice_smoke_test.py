"""
P14 Local Voice Defaults Smoke Test.
Proves that when NO provider environment variables are set:
1. Settings defaults to VOICE_STT_PROVIDER='local' and VOICE_TTS_PROVIDER='local'.
2. VoiceConfig.from_settings() resolves to 'local'.
3. STT factory resolves to LocalWhisperSTTAdapter.
4. TTS factory resolves to LocalTTSAdapter.
5. RealtimeVoicePipeline resolves to LocalWhisperSTTAdapter and LocalTTSAdapter.
6. VoicePipeline (/voice/chat) resolves to LocalWhisperSTTAdapter and LocalTTSAdapter.
7. Cloud providers remain strictly opt-in.
"""
import os
import sys

# Ensure workspace in path
sys.path.insert(0, r"D:\suriya\projects\P-MAI")

# Ensure provider env vars are unset for this smoke test
os.environ.pop("VOICE_STT_PROVIDER", None)
os.environ.pop("VOICE_TTS_PROVIDER", None)

from app.core.config import Settings
from app.voice.config import VoiceConfig
from app.voice.pipeline import VoicePipeline
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.local_adapter import LocalTTSAdapter


def run_smoke_test():
    print("=" * 65)
    print("P14 CONFIGURATION AUDIT — LOCAL VOICE DEFAULTS SMOKE TEST")
    print("=" * 65)

    # 1. Settings Defaults
    settings = Settings()
    print(f"1. Settings.voice_stt_provider: '{settings.voice_stt_provider}'")
    print(f"   Settings.voice_tts_provider: '{settings.voice_tts_provider}'")
    assert settings.voice_stt_provider == "local", f"Expected 'local', got {settings.voice_stt_provider}"
    assert settings.voice_tts_provider == "local", f"Expected 'local', got {settings.voice_tts_provider}"
    print("   [PASS] Settings defaults verified as 'local'.")

    # 2. VoiceConfig.from_settings()
    cfg = VoiceConfig.from_settings(settings)
    print(f"\n2. VoiceConfig.stt_provider: '{cfg.stt_provider}'")
    print(f"   VoiceConfig.tts_provider: '{cfg.tts_provider}'")
    assert cfg.stt_provider == "local"
    assert cfg.tts_provider == "local"
    print("   [PASS] VoiceConfig.from_settings() resolves to 'local'.")

    # 3. STT Factory Default Dispatch
    stt_provider = create_stt_provider(cfg)
    print(f"\n3. create_stt_provider(cfg) -> {type(stt_provider).__name__}")
    assert isinstance(stt_provider, LocalWhisperSTTAdapter)
    print(f"   [PASS] Resolved to {type(stt_provider).__name__} (Local Whisper).")

    # 4. TTS Factory Default Dispatch
    tts_provider = create_tts_provider(cfg)
    print(f"\n4. create_tts_provider(cfg) -> {type(tts_provider).__name__}")
    assert isinstance(tts_provider, LocalTTSAdapter)
    print(f"   [PASS] Resolved to {type(tts_provider).__name__} (Local pyttsx3).")

    # 5. RealtimeVoicePipeline Resolution
    rt_pipeline = RealtimeVoicePipeline(config=cfg)
    print(f"\n5. RealtimeVoicePipeline.stt_provider -> {type(rt_pipeline.stt_provider).__name__}")
    print(f"   RealtimeVoicePipeline.tts_provider -> {type(rt_pipeline.tts_provider).__name__}")
    assert isinstance(rt_pipeline.stt_provider, LocalWhisperSTTAdapter)
    assert isinstance(rt_pipeline.tts_provider, LocalTTSAdapter)
    print("   [PASS] RealtimeVoicePipeline default resolution verified.")

    # 6. /voice/chat (VoicePipeline) Resolution
    rest_pipeline = VoicePipeline(config=cfg)
    print(f"\n6. VoicePipeline.stt_provider -> {type(rest_pipeline.stt_provider).__name__}")
    print(f"   VoicePipeline.tts_provider -> {type(rest_pipeline.tts_provider).__name__}")
    assert isinstance(rest_pipeline.stt_provider, LocalWhisperSTTAdapter)
    assert isinstance(rest_pipeline.tts_provider, LocalTTSAdapter)
    print("   [PASS] VoicePipeline (/voice/chat) default resolution verified.")

    # 7. Cloud Opt-In Verification
    groq_cfg = VoiceConfig(stt_provider="groq", stt_api_key="gsk-test-key")
    groq_stt = create_stt_provider(groq_cfg)
    openai_cfg = VoiceConfig(tts_provider="openai", tts_api_key="sk-test-key")
    openai_tts = create_tts_provider(openai_cfg)
    print(f"\n7. Opt-In Cloud Providers:")
    print(f"   Explicit VOICE_STT_PROVIDER='groq' -> {type(groq_stt).__name__}")
    print(f"   Explicit VOICE_TTS_PROVIDER='openai' -> {type(openai_tts).__name__}")
    assert groq_stt.provider_name == "whisper"
    assert openai_tts.provider_name == "openai_tts"
    print("   [PASS] Cloud providers remain strictly opt-in.")

    print("\n" + "=" * 65)
    print("ALL AUDIT CHECKS PASSED: LOCAL VOICE IS THE DEFAULT")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
