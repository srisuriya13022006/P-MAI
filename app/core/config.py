from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MAI"

    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 4096
    local_model: str = "qwen3:4b"
    local_llm_reserved_response_tokens: int = 512
    cloud_model: str = "openai/gpt-oss-20b"
    cloud_fallback_models: list[str] | str = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

    @field_validator("cloud_fallback_models")
    @classmethod
    def parse_cloud_fallback_models(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    # Gemini LLM Settings (P15.7 Quota-Aware Multi-Model Pool)
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_model_pool: list[str] | str = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    gemini_fallback_models: list[str] | str = ["gemini-3.1-flash-lite"]

    @field_validator("gemini_model_pool", "gemini_fallback_models")
    @classmethod
    def parse_gemini_models_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    use_cloud_fallback: bool = True
    groq_api_key: str | None = None
    tavily_api_key: str | None = None

    # LLM Provider Configuration (P15.4 / P15.5 Cloud-First Multi-Provider)
    llm_provider: str = "cloud"  # 'cloud' (primary) or 'ollama' / 'local'
    cloud_llm_provider: str = "auto"  # 'auto' (prefers gemini, fails over to groq), 'gemini', or 'groq'
    llm_fallback_enabled: bool = True

    database_url: str
    conversation_summary_interval: int = 10
    default_timezone: str = "Asia/Kolkata"

    web_fetch_timeout: int = 10
    web_fetch_max_bytes: int = 2 * 1024 * 1024  # 2 MB max response size
    web_fetch_max_chars: int = 8000  # Character budget for extracted text
    web_search_max_results: int = 5  # Configurable default result count
    web_search_candidate_pool_size: int = 10  # Bounded internal candidate pool for search ranking
    enable_llm_planner: bool = False  # Controlled LLM decomposition disabled by default for deterministic execution
    task_state_ttl_seconds: int = 300  # Multi-turn active task state TTL in seconds (P8)

    # Voice I/O Settings (P12 / P14 / P15.4)
    voice_enabled: bool = True
    voice_stt_provider: str = "assemblyai"  # AssemblyAI Realtime Streaming STT is primary (P15.4)
    voice_tts_provider: str = "local"
    assemblyai_api_key: str | None = None
    assemblyai_stt_model: str = "universal-3-6-pro"
    stt_fallback_enabled: bool = True
    voice_supported_formats: list[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/mp3",
        "audio/mpeg",
        "audio/ogg",
        "audio/webm",
        "audio/flac",
    ]
    voice_max_audio_size_bytes: int = 10 * 1024 * 1024  # 10 MB limit
    voice_max_duration_seconds: float = 60.0  # 60s max audio duration
    voice_timeout_seconds: float = 15.0
    voice_default_language: str = "en"
    voice_default_tts_voice: str = "natural-en-1"
    voice_vad_silence_threshold_db: float = -40.0
    voice_vad_max_silence_seconds: float = 2.0
    # Production Voice Provider Settings (P13 / P14)
    voice_stt_api_key: str | None = None
    voice_tts_api_key: str | None = None
    voice_stt_base_url: str = "https://api.groq.com/openai/v1"
    voice_tts_base_url: str = "https://api.openai.com/v1"
    voice_sample_rate: int = 16000
    voice_audio_format: str = "audio/wav"
    # Local-First Voice Settings (P14)
    voice_stt_local_model: str = "base"
    voice_tts_local_engine: str = "pyttsx3"

    # ElevenLabs Voice Settings (P15.5)
    tts_provider: str | None = None  # Direct alias for TTS_PROVIDER env var
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_v4_turbo"
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True
    tts_fallback_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()