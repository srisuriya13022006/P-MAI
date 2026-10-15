import time
from app.llm.ollama_provider import OllamaProvider

p = OllamaProvider()
print("Ollama model:", p.model)
print("Ollama num_ctx:", p.num_ctx)
t0 = time.perf_counter()
try:
    res = p.generate([{"role": "user", "content": "Say hello in one word"}], max_tokens=10)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Ollama OK in {ms:.1f}ms: {repr(res.strip())}")
except Exception as e:
    ms = (time.perf_counter() - t0) * 1000
    print(f"Ollama FAILED in {ms:.1f}ms: {type(e).__name__}: {e}")
