import time
from app.llm.gemini_provider import GeminiProvider

candidate_models = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]

print("=" * 70)
print("TESTING CANDIDATE GEMINI MODELS VIA OPENAI COMPATIBILITY ENDPOINT")
print("=" * 70)

for m in candidate_models:
    p = GeminiProvider(model=m, fallback_models=[])
    t0 = time.perf_counter()
    try:
        res = p.generate([{"role": "user", "content": "Reply with 'OK'."}], max_tokens=10)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model: {m:30s} | STATUS: SUCCESS ({ms:.1f}ms) | Response: {repr(res.strip())}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        err_msg = str(e)
        if len(err_msg) > 120:
            err_msg = err_msg[:120] + "..."
        print(f"Model: {m:30s} | STATUS: FAILED ({ms:.1f}ms) | {type(e).__name__}: {err_msg}")
