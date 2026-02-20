"""Observability layer that traces every agent interaction.

Captures: tool calls (name, input, duration), total tokens (prompt +
completion), wall-clock time, and LLM model used.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    input: str
    output: str
    duration_ms: float


@dataclass
class QueryTrace:
    """Full trace of a single user question → agent answer cycle."""

    question: str
    answer: str = ""
    model: str = ""
    total_duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    agent_steps: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on gpt-4o-mini pricing."""
        INPUT_COST = 0.15 / 1_000_000
        OUTPUT_COST = 0.60 / 1_000_000
        return (self.prompt_tokens * INPUT_COST) + (self.completion_tokens * OUTPUT_COST)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "model": self.model,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "agent_steps": self.agent_steps,
            "tool_calls": [
                {
                    "name": tc.name,
                    "input": tc.input[:200],
                    "output": tc.output[:200],
                    "duration_ms": round(tc.duration_ms, 1),
                }
                for tc in self.tool_calls
            ],
        }


class QueryTracker:
    """Extracts trace data from LangGraph agent response messages."""

    def __init__(self, model_name: str = "") -> None:
        self._model_name = model_name
        self._history: list[QueryTrace] = []

    def trace_from_messages(
        self,
        question: str,
        answer: str,
        messages: list,
        wall_time_ms: float,
    ) -> QueryTrace:
        trace = QueryTrace(
            question=question,
            answer=answer,
            model=self._model_name,
            total_duration_ms=wall_time_ms,
        )

        tool_calls: list[ToolCall] = []
        prompt_tokens = 0
        completion_tokens = 0
        steps = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                usage = msg.usage_metadata or {}
                prompt_tokens += usage.get("input_tokens", 0)
                completion_tokens += usage.get("output_tokens", 0)

                if msg.tool_calls:
                    steps += 1
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall(
                            name=tc["name"],
                            input=str(tc.get("args", "")),
                            output="",
                            duration_ms=0.0,
                        ))

            elif isinstance(msg, ToolMessage):
                for tc in reversed(tool_calls):
                    if tc.name == msg.name and not tc.output:
                        tc.output = str(msg.content)[:500]
                        break

        trace.tool_calls = tool_calls
        trace.prompt_tokens = prompt_tokens
        trace.completion_tokens = completion_tokens
        trace.total_tokens = prompt_tokens + completion_tokens
        trace.agent_steps = steps

        self._history.append(trace)

        logger.info(
            "Trace: %d tokens (%d+%d) | %d tools | %.0fms | $%.6f",
            trace.total_tokens,
            trace.prompt_tokens,
            trace.completion_tokens,
            len(trace.tool_calls),
            trace.total_duration_ms,
            trace.estimated_cost_usd,
        )

        return trace

    @property
    def history(self) -> list[QueryTrace]:
        return self._history

    @property
    def total_tokens_used(self) -> int:
        return sum(t.total_tokens for t in self._history)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.estimated_cost_usd for t in self._history)

    @property
    def total_queries(self) -> int:
        return len(self._history)
