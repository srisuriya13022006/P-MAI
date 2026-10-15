"""
P15.4 / P15.5 — LLM Provider Factory.
Instantiates and configures primary cloud provider (Gemini primary, Groq switch) and fallback local provider (Ollama).
"""
import os
from typing import Any

from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.cascading_provider import CascadingCloudProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider


def create_cloud_provider(settings_obj: Any = None) -> LLMProvider | None:
    """
    Construct configured cloud LLM provider or multi-provider failover chain.
    If Gemini is configured, it serves as the PRIMARY model, switching to Groq upon error.
    """
    cfg = settings_obj or settings
    preferred = getattr(cfg, "cloud_llm_provider", "auto").lower().strip()

    has_gemini_key = bool(
        getattr(cfg, "gemini_api_key", None)
        or getattr(cfg, "google_api_key", None)
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    has_groq_key = bool(
        getattr(cfg, "groq_api_key", None)
        or os.environ.get("GROQ_API_KEY")
    )

    cloud_providers: list[LLMProvider] = []

    def _build_gemini():
        return GeminiProvider(
            model=getattr(cfg, "gemini_model", None),
            fallback_models=getattr(cfg, "gemini_fallback_models", None),
            api_key=getattr(cfg, "gemini_api_key", None) or getattr(cfg, "google_api_key", None),
        )

    def _build_groq():
        return GroqProvider(
            model=getattr(cfg, "cloud_model", "openai/gpt-oss-20b"),
            fallback_models=getattr(cfg, "cloud_fallback_models", None),
            api_key=getattr(cfg, "groq_api_key", None),
        )

    if preferred == "groq":
        if has_groq_key:
            cloud_providers.append(_build_groq())
        if has_gemini_key:
            cloud_providers.append(_build_gemini())
    else:
        # Default ('auto' or 'gemini'): Gemini is PRIMARY, Groq is SWITCH/FAILOVER
        if has_gemini_key:
            cloud_providers.append(_build_gemini())
        if has_groq_key:
            cloud_providers.append(_build_groq())

    if len(cloud_providers) > 1:
        return CascadingCloudProvider(providers=cloud_providers)
    elif len(cloud_providers) == 1:
        return cloud_providers[0]
    return None


def create_llm_provider(
    provider_name: str | None = None,
    settings_obj: Any = None,
    allow_fallback: bool | None = None,
) -> LLMProvider:
    """
    Factory creating configured LLM provider according to runtime settings.
    Default: Cloud-First (Gemini primary -> Groq switch -> Ollama local fallback).
    """
    cfg = settings_obj or settings
    mode = (provider_name or getattr(cfg, "llm_provider", "cloud")).lower().strip()
    fallback_enabled = (
        allow_fallback if allow_fallback is not None else getattr(cfg, "llm_fallback_enabled", True)
    )

    if mode in ("local", "ollama"):
        return OllamaProvider(
            model=getattr(cfg, "local_model", None),
            base_url=getattr(cfg, "ollama_base_url", None),
            num_ctx=getattr(cfg, "ollama_num_ctx", None),
        )

    if mode == "gemini":
        return GeminiProvider(
            model=getattr(cfg, "gemini_model", None),
            fallback_models=getattr(cfg, "gemini_fallback_models", None),
            api_key=getattr(cfg, "gemini_api_key", None) or getattr(cfg, "google_api_key", None),
        )

    if mode == "groq":
        return GroqProvider(
            model=getattr(cfg, "cloud_model", "openai/gpt-oss-20b"),
            fallback_models=getattr(cfg, "cloud_fallback_models", None),
            api_key=getattr(cfg, "groq_api_key", None),
        )

    if mode == "cloud":
        primary = create_cloud_provider(cfg)

        if primary is None:
            if not fallback_enabled:
                raise ValueError("Cloud LLM requested but GROQ_API_KEY is not configured (and GEMINI_API_KEY is not configured).")

        fallback = OllamaProvider(
            model=getattr(cfg, "local_model", None),
            base_url=getattr(cfg, "ollama_base_url", None),
            num_ctx=getattr(cfg, "ollama_num_ctx", None),
        ) if fallback_enabled else None

        if primary is not None and not fallback_enabled:
            return primary

        return ResilientLLMProvider(
            primary_provider=primary,
            fallback_provider=fallback,
            fallback_enabled=fallback_enabled,
        )

    raise ValueError(f"Unsupported LLM provider '{mode}'.")


_canonical_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return shared canonical LLM provider instance for FastAPI dependencies and orchestrator."""
    global _canonical_llm_provider
    if _canonical_llm_provider is None:
        _canonical_llm_provider = create_llm_provider()
    return _canonical_llm_provider
