import time
from app.llm.gemini_provider import GeminiProvider

def test_gemini():
    p = GeminiProvider()
    print("Testing model:", p.primary_model)
    print("Fallback models:", p.fallback_models)
    
    for i in range(3):
        t0 = time.perf_counter()
        try:
            res = p.generate([{"role": "user", "content": "Hello! Reply with exactly two words."}])
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"Call {i+1}: Success in {elapsed:.1f}ms | Model: {p.model} | Res: {repr(res)}")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"Call {i+1}: FAILED in {elapsed:.1f}ms | Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_gemini()
