from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MAI"

    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen3:4b"
    cloud_model: str = "openai/gpt-oss-20b"
    use_cloud_fallback: bool = True
    groq_api_key: str | None = None
    tavily_api_key: str | None = None

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

    # Voice I/O Settings (P12 / P14)
    voice_enabled: bool = True
    voice_stt_provider: str = "local"
    voice_tts_provider: str = "local"
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
    voice_stt_local_model: str = "tiny"
    voice_tts_local_engine: str = "pyttsx3"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()