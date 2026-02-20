"""Integration tests for the SalesAgent.

These tests call the real LLM and consume tokens.
Run only when explicitly requested:
    pytest -m integration

Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture
def agent(sample_df: pd.DataFrame):
    if not has_api_key():
        pytest.skip("OPENAI_API_KEY not set — skipping integration test")

    from agent.csv_agent import SalesAgent

    return SalesAgent(sample_df)


class TestAgentIntegration:

    def test_agent_returns_answer(self, agent):
        result = agent.ask("Quantos registros existem no dataset?")
        assert result["answer"]
        assert len(result["answer"]) > 5
        assert result["conversation_id"]

    def test_agent_returns_trace(self, agent):
        result = agent.ask("Qual o produto mais vendido?")
        trace = result["trace"]
        assert trace.total_tokens > 0
        assert trace.total_duration_ms > 0

    def test_agent_uses_tools(self, agent):
        result = agent.ask("Qual o top 3 produtos por receita?")
        trace = result["trace"]
        assert len(trace.tool_calls) > 0
        tool_names = [tc.name for tc in trace.tool_calls]
        assert any("product" in name or "python" in name for name in tool_names)

    def test_agent_conversation_memory(self, agent):
        result1 = agent.ask("Qual o produto mais vendido?")
        cid = result1["conversation_id"]

        result2 = agent.ask("E o segundo?", conversation_id=cid)
        assert result2["answer"]
        assert result2["conversation_id"] == cid

    def test_tracker_accumulates(self, agent):
        agent.ask("Quantos produtos existem?")
        agent.ask("Quantos locais existem?")
        assert agent.tracker.total_queries == 2
        assert agent.tracker.total_tokens_used > 0
        assert agent.tracker.total_cost_usd > 0
