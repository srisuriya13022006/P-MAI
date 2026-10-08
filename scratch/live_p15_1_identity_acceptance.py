"""
P15.1 — Live Acceptance Script: Conversational Self-Identity Routing & Fast Path.
Tests scenarios A through G directly against ChatService and Orchestrator.
"""
import sys
import os
import time

sys.path.insert(0, r"D:\suriya\projects\P-MAI")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

from app.database.connection import Base
from app.agent.policy import apply_policy, is_self_identity_query
from app.agent.orchestrator import MAIOrchestrator
from app.schemas.agent import AgentDecision
from app.services.chat_service import ChatService


def run_acceptance():
    print("=" * 70)
    print("P15.1 LIVE ACCEPTANCE: CONVERSATIONAL SELF-IDENTITY ROUTING")
    print("=" * 70)

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Your name is Alice according to memory."
    chat_service = ChatService(db=db, llm=mock_llm)

    # Save a mock user memory for testing user queries
    chat_service.memory_service.remember(
        user_id="user-p15-1",
        content="User's name is Alice",
        memory_type="fact",
    )

    scenarios = [
        ("A", "What is your name?", True),
        ("B", "Who are you?", True),
        ("C", "What can you do?", True),
        ("D", "What is my name?", False),
        ("E", "What do you remember about me?", False),
        ("F", "Hi, what's your name?", True),
        ("G", "Bye, what is your name?", True),
    ]

    results = {}
    for letter, prompt, is_self_id in scenarios:
        print(f"\n[Scenario {letter}] Query: '{prompt}'")
        decision = AgentDecision(
            intent="general",
            route="local",
            needs_clarification=False,
            tools=[],
            tool_arguments={},
            reason="Initial",
        )
        routed = apply_policy(prompt, decision)
        print(f"  Route: {routed.route} | Intent: {routed.intent} | Tools: {routed.tools}")

        t0 = time.perf_counter()
        response = chat_service.chat("conv-p15-1", "user-p15-1", prompt)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        print(f"  Response: {response}")
        print(f"  Elapsed: {elapsed_ms} ms")

        if is_self_id:
            assert routed.route == "local", f"Expected local route for {prompt}"
            assert routed.intent == "identify_self"
            assert routed.tools == []
            assert "I'm MAI" in response
            assert elapsed_ms < 50.0  # Fast path must be sub-50ms
        else:
            assert routed.route == "memory", f"Expected memory route for {prompt}"
            assert routed.intent != "identify_self"

        results[f"Scenario_{letter}"] = {
            "status": "PASS",
            "prompt": prompt,
            "route": routed.route,
            "intent": routed.intent,
            "response": response,
            "elapsed_ms": elapsed_ms,
        }
        print("  -> PASS")

    print("\n" + "=" * 70)
    print(f"ALL {len(results)} SCENARIOS PASSED SUCCESSFULLY")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_acceptance()
