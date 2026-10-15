from app.llm.gemini_provider import GeminiProvider

p = GeminiProvider()
try:
    models = p.client.models.list()
    print("Available Gemini models via OpenAI compatibility endpoint:")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"List models failed: {type(e).__name__}: {e}")
