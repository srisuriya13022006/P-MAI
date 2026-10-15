import time
from app.llm.groq_provider import GroqProvider

p = GroqProvider()
print("Groq model:", p.primary_model)
print("Groq fallback models:", p.fallback_models)
t0 = time.perf_counter()
try:
    res = p.generate([{"role": "user", "content": "Say hello in one word"}], max_tokens=10)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Groq OK in {ms:.1f}ms: {repr(res.strip())}")
    print("Tokens:", getattr(p, "last_usage", {}))
except Exception as e:
    ms = (time.perf_counter() - t0) * 1000
    print(f"Groq FAILED in {ms:.1f}ms: {type(e).__name__}: {e}")
