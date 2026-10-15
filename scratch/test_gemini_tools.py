import time
from app.llm.gemini_provider import GeminiProvider

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    }
]

for m in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
    p = GeminiProvider(model=m, fallback_models=[])
    t0 = time.perf_counter()
    try:
        response = p.client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "What is 42 multiplied by 17?"}],
            tools=tools_schema,
            tool_choice="auto",
        )
        ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        has_tool_call = bool(choice.message.tool_calls)
        calls_info = [(tc.function.name, tc.function.arguments) for tc in (choice.message.tool_calls or [])]
        print(f"Model: {m:25s} | Tool Call Success: {has_tool_call} ({ms:.1f}ms) | Calls: {calls_info}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"Model: {m:25s} | Tool Call FAILED ({ms:.1f}ms): {type(e).__name__}: {e}")
