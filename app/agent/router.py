import json
import logging
import re
from typing import Any

from app.llm.factory import get_llm_provider
from app.prompts.router import ROUTER_SYSTEM_PROMPT
from app.schemas.agent import AgentDecision

logger = logging.getLogger(__name__)


class RequestAnalyzer:
    """Analyzes user requests and produces a structured MAI decision."""

    def __init__(self, llm: Any = None):
        self.llm = llm if llm is not None else get_llm_provider()

    def analyze(self, user_message: str) -> AgentDecision:
        """Analyze a user request and return a validated decision."""
        from app.agent.policy import (
            is_self_identity_query,
            is_pure_datetime_query,
            is_greeting_query,
            is_acknowledgment_query,
            is_farewell_query,
            is_pure_calculator_query,
        )
        from app.agent.tool_argument_resolver import (
            resolve_calculator_arguments,
            resolve_datetime_arguments,
        )

        # 0. Deterministic fast paths: zero LLM invocation
        if is_self_identity_query(user_message):
            return AgentDecision(
                intent="identify_self",
                route="local",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Conversational self-identity query answered locally without tools or memory.",
            )

        if is_pure_datetime_query(user_message):
            return AgentDecision(
                intent="datetime",
                route="tool",
                needs_clarification=False,
                tools=["datetime"],
                tool_arguments=resolve_datetime_arguments(user_message),
                reason="Current date and time requests are handled by the datetime tool.",
            )

        if is_pure_calculator_query(user_message):
            return AgentDecision(
                intent="arithmetic",
                route="tool",
                needs_clarification=False,
                tools=["calculator"],
                tool_arguments=resolve_calculator_arguments(user_message),
                reason="Arithmetic requests are handled by the calculator tool.",
            )

        if is_farewell_query(user_message):
            return AgentDecision(
                intent="farewell",
                route="local",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Conversational farewell routed locally.",
            )

        if is_greeting_query(user_message):
            return AgentDecision(
                intent="greet",
                route="local",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Conversational greeting routed locally.",
            )

        if is_acknowledgment_query(user_message):
            return AgentDecision(
                intent="acknowledge",
                route="local",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Conversational acknowledgment routed locally.",
            )

        # Check deterministic policy rules before invoking any router LLM
        from app.agent.policy import apply_policy
        policy_decision = apply_policy(
            user_message,
            AgentDecision(
                intent="conversational",
                route="local",
                needs_clarification=False,
                tools=[],
                reason="preliminary",
            ),
        )
        if policy_decision.route != "local" or policy_decision.intent not in ("conversational", "unknown") or policy_decision.tools:
            return policy_decision

        # 1. Direct Ollama fast path if raw client is present and not wrapped in resilient/cloud
        if hasattr(self.llm, "client") and hasattr(self.llm.client, "chat") and not hasattr(self.llm, "primary_provider"):
            try:
                response = self.llm.client.chat(
                    model=getattr(self.llm, "model", "qwen3:4b-instruct"),
                    messages=[
                        {
                            "role": "system",
                            "content": ROUTER_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_message,
                        },
                    ],
                    format=AgentDecision.model_json_schema(),
                )
                content = response["message"]["content"]
                return AgentDecision.model_validate_json(content)
            except Exception as e:
                logger.warning("Ollama direct router chat failed: %s; falling back to provider.generate", e)

        # 2. General LLM Provider abstraction (Primary Cloud LLM / Resilient Provider)
        json_prompt = (
            f"{ROUTER_SYSTEM_PROMPT}\n\n"
            f"You MUST output valid JSON only matching this schema:\n"
            f"{json.dumps(AgentDecision.model_json_schema())}\n"
            f"Output strictly raw JSON without markdown code fences."
        )
        messages = [
            {"role": "system", "content": json_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            if hasattr(self.llm, "generate"):
                try:
                    content = self.llm.generate(messages=messages, system_prompt=json_prompt, stage="router", max_tokens=250)
                except TypeError:
                    content = self.llm.generate(messages=messages, system_prompt=json_prompt)
            else:
                raise RuntimeError(f"Unsupported LLM provider in RequestAnalyzer: {self.llm}")

            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.strip("`")
                if clean.startswith("json"):
                    clean = clean[4:].strip()

            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                clean = match.group(0)

            return AgentDecision.model_validate_json(clean)
        except Exception as exc:
            logger.info("Router LLM did not return schema JSON (%s); using deterministic routing fallback", exc)
            return self._deterministic_fallback(user_message)

    def _deterministic_fallback(self, user_message: str) -> AgentDecision:
        """Safe deterministic routing fallback when LLM output is not structured JSON."""
        msg_lower = user_message.lower().strip()
        if any(term in msg_lower for term in ("time", "current time", "what time", "date", "clock", "today's date")):
            return AgentDecision(
                intent="datetime_lookup",
                route="tool",
                needs_clarification=False,
                tools=["datetime"],
                tool_arguments={},
                reason="Deterministic datetime fallback",
            )
        if any(op in msg_lower for op in ("times", "plus", "minus", "divided by", "*", "+", "/", "%", "calculate")):
            return AgentDecision(
                intent="numeric_calculation",
                route="tool",
                needs_clarification=False,
                tools=["calculator"],
                tool_arguments={"expression": user_message},
                reason="Deterministic calculator fallback",
            )
        if any(p in msg_lower for p in ("remember", "my name is", "i prefer", "i like", "i use", "i live in")):
            return AgentDecision(
                intent="personal_fact",
                route="local",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Deterministic personal statement fallback",
            )
        if any(term in msg_lower for term in ("search", "web", "find online", "lookup online", "browse", "google")):
            return AgentDecision(
                intent="web_search",
                route="tool",
                needs_clarification=False,
                tools=["web_search"],
                tool_arguments={"query": user_message},
                reason="Deterministic web search fallback",
            )
        if any(p in msg_lower for p in ("what is my", "who am i", "what do you remember", "what do you know")):
            return AgentDecision(
                intent="personal_memory_retrieval",
                route="memory",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="Deterministic memory query fallback",
            )
        return AgentDecision(
            intent="general_conversation",
            route="cloud",
            needs_clarification=False,
            tools=[],
            tool_arguments={},
            reason="Deterministic general conversation fallback",
        )