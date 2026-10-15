import time
from app.llm.gemini_provider import GeminiProvider

for m in ["gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"]:
    p = GeminiProvider(model=m, fallback_models=[])
    t0 = time.perf_counter()
    try:
        res = p.generate([{"role": "user", "content": "Reply with 'Hello'."}], max_tokens=10)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model: {m:30s} | STATUS: SUCCESS ({ms:.1f}ms) | Response: {repr(res.strip())}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model: {m:30s} | STATUS: FAILED ({ms:.1f}ms) | {type(e).__name__}: {e}")
