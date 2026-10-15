import time
from app.llm.gemini_provider import GeminiProvider

for model_name in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]:
    p = GeminiProvider(model=model_name, fallback_models=[])
    t0 = time.perf_counter()
    try:
        res = p.generate([{"role": "user", "content": "Say hello in one word"}], max_tokens=10)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model {model_name:25s}: OK ({ms:.1f}ms) -> {repr(res.strip())}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model {model_name:25s}: FAIL ({ms:.1f}ms) -> {type(e).__name__}: {e}")
