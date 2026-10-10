"""
P15.4 — LLM Provider Factory.
Instantiates and configures primary cloud provider (Groq) and fallback local provider (Ollama).
"""
import os
from typing import Any

from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider


def create_llm_provider(
    provider_name: str | None = None,
    settings_obj: Any = None,
    allow_fallback: bool | None = None,
) -> LLMProvider:
    """
    Factory creating configured LLM provider according to runtime settings.
    Default: Cloud-First (Groq primary, Ollama local fallback).
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
        )

    if mode == "cloud":
        primary = None
        has_groq_key = bool(getattr(cfg, "groq_api_key", None) or os.environ.get("GROQ_API_KEY"))

        if not has_groq_key:
            if not fallback_enabled:
                raise ValueError("Cloud LLM requested but GROQ_API_KEY is not configured.")
        else:
            primary = GroqProvider(
                model=getattr(cfg, "cloud_model", "openai/gpt-oss-20b"),
                fallback_models=getattr(cfg, "cloud_fallback_models", None),
                api_key=getattr(cfg, "groq_api_key", None),
            )

        fallback = OllamaProvider(
            model=getattr(cfg, "local_model", None),
            base_url=getattr(cfg, "ollama_base_url", None),
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
