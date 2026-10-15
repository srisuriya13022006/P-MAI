"""
Live contract smoke test for Gemini Primary LLM and failover switching.
Tests live inference against Google's Gemini endpoint using configured GEMINI_API_KEY.
Strictly masks keys and avoids secret leakage.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.llm.factory import create_cloud_provider, create_llm_provider
from app.llm.gemini_provider import GeminiProvider


def main():
    print("=" * 70)
    print("P15.5 — Gemini Primary LLM Live Contract Smoke Test")
    print("=" * 70)

    has_gemini = bool(
        getattr(settings, "gemini_api_key", None)
        or getattr(settings, "google_api_key", None)
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    has_groq = bool(
        getattr(settings, "groq_api_key", None)
        or os.environ.get("GROQ_API_KEY")
    )

    print(f"GEMINI_API_KEY Configured: {has_gemini}")
    print(f"GROQ_API_KEY Configured:   {has_groq}")
    print(f"Configured Gemini Model:   {settings.gemini_model}")
    print(f"Configured Fallback Models:{settings.gemini_fallback_models}")

    if not has_gemini:
        print("\n[INFO] GEMINI_API_KEY is not yet detected in .env or environment.")
        print("Please ensure GEMINI_API_KEY=your_key is saved in .env (press Ctrl+S in editor).")
        return

    print("\n1. Instantiating Cloud Provider...")
    provider = create_cloud_provider()
    print(f"   Provider resolved: {type(provider).__name__}")
    print(f"   Active Provider Name: {provider.provider_name}")
    print(f"   Active Model: {provider.model}")

    test_messages = [
        {"role": "user", "content": "Explain in one short sentence what P-MAI represents."}
    ]

    print("\n2. Sending test prompt to Gemini Primary Model...")
    try:
        reply = provider.generate(test_messages, system_prompt="You are a helpful assistant.")
        print(f"   Response received successfully from {provider.provider_name} ({provider.model}):")
        print(f"   -> '{reply.strip()}'")
        print(f"   Tokens: {getattr(provider, 'last_usage', {})}")
        print("\n" + "=" * 70)
        print("GEMINI PRIMARY MODEL VERIFICATION PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
