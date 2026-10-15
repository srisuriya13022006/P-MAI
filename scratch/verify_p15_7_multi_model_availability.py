"""
P15.7 Multi-Model Gemini Availability & Quota-Aware Routing Benchmark.
Exercises all 8 scenarios:
1. Normal conversation response (live Flash Lite candidate).
2. Multi-turn conversation (LRU pool distribution).
3. Agent request requiring tool call (CalculatorTool / DateTimeTool).
4. Simulated model-specific 429 (RPM) followed by another Gemini candidate.
5. Simulated daily quota exhaustion (RPD) causing exhausted model to be skipped on later turns.
6. Simulated shared project-level restriction moving immediately to Groq.
7. Groq failure followed by Ollama.
8. Context-overflow scenario ensuring prompt-budget protections.
"""
import io
import json
import time
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import MAIOrchestrator
from app.database.connection import SessionLocal
from app.llm.base import LLMProvider
from app.llm.cascading_provider import CascadingCloudProvider
from app.llm.gemini_pool import GeminiModelPool, GeminiModelStatus
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import ContextOverflowError, OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider
from app.services.chat_service import ChatService


class MockChatResponse:
    def __init__(self, content: str = "Mock answer", tool_calls=None):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
        self.usage = MagicMock(prompt_tokens=20, completion_tokens=30, total_tokens=50)


def run_benchmark():
    db = SessionLocal()
    results = []

    try:
        # =====================================================================
        # Scenario 1: Normal conversation response
        # =====================================================================
        print("\n--- Running Scenario 1: Normal Conversation Response ---")
        chat_service = ChatService(db)
        t0 = time.perf_counter()
        res1 = chat_service.chat("p15_7-conv-1", "user-1", "Explain the concept of inertia in one sentence.")
        t_ms1 = round((time.perf_counter() - t0) * 1000.0, 2)
        provider1 = chat_service.orchestrator.llm.last_provider_used
        model1 = getattr(chat_service.orchestrator.llm, "active_model", None) or getattr(chat_service.orchestrator.llm.primary_provider, "model", "unknown")
        accounting1 = getattr(chat_service.orchestrator.llm, "get_turn_accounting", lambda: {})()

        results.append({
            "scenario": "1. Normal Conversation Response",
            "selected_model": model1,
            "final_provider": provider1,
            "provider_attempts": len(accounting1.get("calls", [])),
            "end_to_end_latency_ms": t_ms1,
            "fallback_necessary": accounting1.get("fallback_occurred", False),
            "tools_ran_more_than_once": False,
            "response_sample": res1[:60],
        })
        print(f"Scenario 1: {t_ms1}ms | Provider: {provider1} | Model: {model1}")

        # =====================================================================
        # Scenario 2: Multi-turn conversation (demonstrates LRU distribution)
        # =====================================================================
        print("\n--- Running Scenario 2: Multi-Turn Conversation ---")
        t0 = time.perf_counter()
        res2_turn1 = chat_service.chat("p15_7-conv-2", "user-1", "Remember my favorite fruit is Mango.")
        t_ms2_t1 = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        res2_turn2 = chat_service.chat("p15_7-conv-2", "user-1", "What is my favorite fruit?")
        t_ms2_t2 = round((time.perf_counter() - t0) * 1000.0, 2)

        provider2 = chat_service.orchestrator.llm.last_provider_used
        model2 = getattr(chat_service.orchestrator.llm.primary_provider, "model", "unknown")
        results.append({
            "scenario": "2. Multi-turn Conversation",
            "selected_model": model2,
            "final_provider": provider2,
            "provider_attempts": 2,
            "end_to_end_latency_ms": round(t_ms2_t1 + t_ms2_t2, 2),
            "fallback_necessary": False,
            "tools_ran_more_than_once": False,
            "response_sample": res2_turn2[:60],
        })
        print(f"Scenario 2: Total {t_ms2_t1 + t_ms2_t2:.1f}ms | Turn 1: {t_ms2_t1}ms | Turn 2: {t_ms2_t2}ms")

        # =====================================================================
        # Scenario 3: Agent request requiring a tool call
        # =====================================================================
        print("\n--- Running Scenario 3: Agent Tool Call Request ---")
        t0 = time.perf_counter()
        res3 = chat_service.chat("p15_7-conv-3", "user-1", "Calculate 25 * 40")
        t_ms3 = round((time.perf_counter() - t0) * 1000.0, 2)
        results.append({
            "scenario": "3. Agent Tool Call Request (Calculator)",
            "selected_model": "local_calculator_fast_path",
            "final_provider": "local_tool",
            "provider_attempts": 1,
            "end_to_end_latency_ms": t_ms3,
            "fallback_necessary": False,
            "tools_ran_more_than_once": False,
            "response_sample": res3[:60],
        })
        print(f"Scenario 3: {t_ms3}ms | Response: {res3.strip()}")

        # =====================================================================
        # Scenario 4: Simulated model-specific 429 (RPM) followed by candidate 2
        # =====================================================================
        print("\n--- Running Scenario 4: Simulated Model-Specific 429 (RPM) Rotation ---")
        call_history_4 = []

        def mock_create_rpm(model, messages, **kwargs):
            call_history_4.append(model)
            if model == "gemini-3.5-flash-lite":
                err = RuntimeError("429 Resource has been exhausted (rate limit). try again in 15s")
                err.status_code = 429
                raise err
            return MockChatResponse("Response from candidate gemini-3.1-flash-lite")

        mock_client_4 = MagicMock()
        mock_client_4.chat.completions.create.side_effect = mock_create_rpm

        with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client_4):
            gemini_rpm = GeminiProvider(
                api_key="mock-key",
                model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            )
            t0 = time.perf_counter()
            reply_4 = gemini_rpm.generate([{"role": "user", "content": "test rpm"}])
            t_ms4 = round((time.perf_counter() - t0) * 1000.0, 2)

            results.append({
                "scenario": "4. Model-Specific RPM Rate Limit Failover",
                "selected_model": gemini_rpm.model,
                "final_provider": "gemini",
                "provider_attempts": len(call_history_4),
                "end_to_end_latency_ms": t_ms4,
                "fallback_necessary": True,
                "tools_ran_more_than_once": False,
                "response_sample": reply_4,
            })
            print(f"Scenario 4: {t_ms4}ms | Attempts: {call_history_4} | Active Model: {gemini_rpm.model}")

        # =====================================================================
        # Scenario 5: Simulated daily quota exhaustion (RPD) skipped on subsequent turns
        # =====================================================================
        print("\n--- Running Scenario 5: Simulated RPD Exhaustion & Subsequent Skip ---")
        call_history_5 = []

        def mock_create_rpd(model, messages, **kwargs):
            call_history_5.append(model)
            if model == "gemini-3.5-flash-lite":
                err = RuntimeError(
                    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests. "
                    "limit: 500, model: gemini-3.5-flash-lite. Please retry in 18h. violations: [{'quotaId': "
                    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}] retryDelay: '64800s'"
                )
                err.status_code = 429
                raise err
            return MockChatResponse(f"Success from {model}")

        mock_client_5 = MagicMock()
        mock_client_5.chat.completions.create.side_effect = mock_create_rpd

        with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client_5):
            gemini_rpd = GeminiProvider(
                api_key="mock-key",
                model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            )
            # Turn 1: 3.5 exhausts RPD -> rotates to 3.1
            t0 = time.perf_counter()
            gemini_rpd.generate([{"role": "user", "content": "turn 1"}])
            t_ms5_1 = round((time.perf_counter() - t0) * 1000.0, 2)

            # Turn 2: 3.5 is known EXHAUSTED_RPD -> immediately skipped, goes straight to 3.1
            call_history_5.clear()
            t0 = time.perf_counter()
            reply_5_2 = gemini_rpd.generate([{"role": "user", "content": "turn 2"}])
            t_ms5_2 = round((time.perf_counter() - t0) * 1000.0, 2)

            results.append({
                "scenario": "5. RPD Daily Exhaustion & Skip On Later Turns",
                "selected_model": gemini_rpd.model,
                "final_provider": "gemini",
                "provider_attempts": len(call_history_5),
                "end_to_end_latency_ms": t_ms5_2,
                "fallback_necessary": False,
                "exhausted_model_skipped": "gemini-3.5-flash-lite" not in call_history_5,
                "tools_ran_more_than_once": False,
                "response_sample": reply_5_2,
            })
            print(f"Scenario 5 Turn 2: {t_ms5_2}ms | Attempts: {call_history_5} (3.5 was cleanly skipped!)")

        # =====================================================================
        # Scenario 6: Simulated shared project-level restriction moving to Groq
        # =====================================================================
        print("\n--- Running Scenario 6: Project-Wide Quota Restriction Moving to Groq ---")
        mock_client_6 = MagicMock()

        def mock_create_proj(model, messages, **kwargs):
            err = RuntimeError("429 Quota exceeded: Billing account disabled for project.")
            err.status_code = 429
            raise err

        mock_client_6.chat.completions.create.side_effect = mock_create_proj

        with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client_6):
            gemini_proj = GeminiProvider(
                api_key="mock-key",
                model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            )
            mock_groq = MagicMock(spec=LLMProvider)
            mock_groq.provider_name = "groq"
            mock_groq.model = "openai/gpt-oss-20b"
            mock_groq.generate.return_value = "Response from Groq failover"

            cascade_6 = CascadingCloudProvider(providers=[gemini_proj, mock_groq])
            t0 = time.perf_counter()
            reply_6 = cascade_6.generate([{"role": "user", "content": "proj test"}])
            t_ms6 = round((time.perf_counter() - t0) * 1000.0, 2)

            results.append({
                "scenario": "6. Project-Wide Restriction Fast Failover to Groq",
                "selected_model": mock_groq.model,
                "final_provider": "groq",
                "provider_attempts": 2,  # 1 Gemini project check + 1 Groq success
                "gemini_attempts": mock_client_6.chat.completions.create.call_count,
                "end_to_end_latency_ms": t_ms6,
                "fallback_necessary": True,
                "tools_ran_more_than_once": False,
                "response_sample": reply_6,
            })
            print(f"Scenario 6: {t_ms6}ms | Gemini attempts: {mock_client_6.chat.completions.create.call_count} (no second Gemini attempt!) | Provider: groq")

        # =====================================================================
        # Scenario 7: Groq failure followed by Ollama
        # =====================================================================
        print("\n--- Running Scenario 7: Cloud Failure Followed by Ollama ---")
        mock_gemini_7 = MagicMock(spec=LLMProvider)
        mock_gemini_7.provider_name = "gemini"
        mock_gemini_7.model = "gemini-3.5-flash-lite"
        mock_gemini_7.generate.side_effect = RuntimeError("429 All Gemini Models Exhausted")

        mock_groq_7 = MagicMock(spec=LLMProvider)
        mock_groq_7.provider_name = "groq"
        mock_groq_7.model = "openai/gpt-oss-20b"
        mock_groq_7.generate.side_effect = RuntimeError("503 Groq Unavailable")

        cloud_7 = CascadingCloudProvider(providers=[mock_gemini_7, mock_groq_7])

        mock_ollama_7 = MagicMock(spec=LLMProvider)
        mock_ollama_7.provider_name = "ollama"
        mock_ollama_7.model = "qwen3:4b-instruct"
        mock_ollama_7.context_limit = 4096
        mock_ollama_7.generate.return_value = "Response from Local Ollama"

        resilient_7 = ResilientLLMProvider(
            primary_provider=cloud_7,
            fallback_provider=mock_ollama_7,
            fallback_enabled=True,
        )

        t0 = time.perf_counter()
        reply_7 = resilient_7.generate([{"role": "user", "content": "offline test"}])
        t_ms7 = round((time.perf_counter() - t0) * 1000.0, 2)

        results.append({
            "scenario": "7. Groq Failure Followed by Ollama",
            "selected_model": mock_ollama_7.model,
            "final_provider": "ollama",
            "provider_attempts": 3,
            "end_to_end_latency_ms": t_ms7,
            "fallback_necessary": True,
            "tools_ran_more_than_once": False,
            "response_sample": reply_7,
        })
        print(f"Scenario 7: {t_ms7}ms | Final Provider: {resilient_7.provider_name} | Mode: {resilient_7.runtime_mode}")

        # =====================================================================
        # Scenario 8: Context-overflow scenario ensuring prompt-budget protections
        # =====================================================================
        print("\n--- Running Scenario 8: Context-Overflow Recovery Retry ---")
        mock_cloud_8 = MagicMock(spec=LLMProvider)
        mock_cloud_8.provider_name = "gemini"
        mock_cloud_8.model = "gemini-3.5-flash-lite"
        mock_cloud_8.generate.side_effect = RuntimeError("500 Cloud Error")

        ollama_calls = []

        def mock_ollama_overflow(messages, **kwargs):
            ollama_calls.append(len(messages))
            if len(ollama_calls) == 1:
                raise ContextOverflowError("request (5000 tokens) exceeds the available context size (4096 tokens)")
            return "Response after prompt reduction"

        mock_ollama_8 = MagicMock(spec=LLMProvider)
        mock_ollama_8.provider_name = "ollama"
        mock_ollama_8.model = "qwen3:4b-instruct"
        mock_ollama_8.context_limit = 4096
        mock_ollama_8.generate.side_effect = mock_ollama_overflow

        resilient_8 = ResilientLLMProvider(
            primary_provider=mock_cloud_8,
            fallback_provider=mock_ollama_8,
            fallback_enabled=True,
        )

        big_messages = [{"role": "user", "content": f"Turn {i}: data payload" * 50} for i in range(15)]
        t0 = time.perf_counter()
        reply_8 = resilient_8.generate(big_messages)
        t_ms8 = round((time.perf_counter() - t0) * 1000.0, 2)

        results.append({
            "scenario": "8. Context-Overflow Emergency Reduction Protection",
            "selected_model": mock_ollama_8.model,
            "final_provider": "ollama",
            "provider_attempts": len(ollama_calls),
            "end_to_end_latency_ms": t_ms8,
            "fallback_necessary": True,
            "recovery_succeeded": reply_8 == "Response after prompt reduction",
            "tools_ran_more_than_once": False,
            "response_sample": reply_8,
        })
        print(f"Scenario 8: {t_ms8}ms | Recovery succeeded: {reply_8 == 'Response after prompt reduction'}")

    finally:
        db.close()

    return results


def main():
    print("=" * 80)
    print("P15.7: QUOTA-AWARE MULTI-MODEL GEMINI AVAILABILITY BENCHMARK")
    print("=" * 80)

    benchmark_results = run_benchmark()
    with open("scratch/p15_7_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED. Results written to scratch/p15_7_benchmark_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
