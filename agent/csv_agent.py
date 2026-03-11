"""Runtime-configurable LangGraph agent facade."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent.prompts import SYSTEM_PROMPT
from agent.prompt_builder import build_dynamic_system_prompt
from agent.tool_builder import build_dynamic_tool_bundle, build_sales_tool_bundle
from config.settings import Settings, get_settings
from observability.tracker import QueryTracker
from services.analytics import AnalyticsService
from services.dataset_profiler import DatasetProfile, profile_dataframe

logger = logging.getLogger(__name__)


class SalesAgent:
    """Stateful conversational agent for fixed or dynamic datasets."""

    def __init__(
        self,
        df: pd.DataFrame,
        settings: Settings | None = None,
        *,
        tools: list | None = None,
        prompt: str | None = None,
        dataset_id: str = "",
        tool_origin_map: dict[str, str] | None = None,
        profile: DatasetProfile | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._df = df
        self._analytics = AnalyticsService(df)
        self._profile = profile
        self._dataset_id = dataset_id or "sales_default"

        if tools is None:
            sales_bundle = build_sales_tool_bundle(df, self._analytics)
            self._tools = sales_bundle.tools
            self._tool_origin_map = sales_bundle.tool_origin_map
        else:
            self._tools = tools
            self._tool_origin_map = tool_origin_map or {t.name: "custom" for t in tools}

        self._prompt = prompt or SYSTEM_PROMPT
        self._llm = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_model,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        self._conversations: dict[str, list] = {}
        self._tracker = QueryTracker(model_name=self._settings.openai_model)
        self._agent = self._build_agent()

    def _build_agent(self):
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._prompt,
        )

    def _get_or_create_conversation(self, conversation_id: str | None) -> tuple[str, list]:
        cid = conversation_id or str(uuid.uuid4())
        if cid not in self._conversations:
            self._conversations[cid] = []
        return cid, self._conversations[cid]

    def ask(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        cid, history = self._get_or_create_conversation(conversation_id)

        history.append(HumanMessage(content=question))

        start = time.perf_counter()
        result = self._agent.invoke(
            {"messages": history},
        )
        wall_time_ms = (time.perf_counter() - start) * 1000

        response_messages = result["messages"]
        ai_answer = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                ai_answer = msg.content
                break

        trace = self._tracker.trace_from_messages(
            question=question,
            answer=ai_answer,
            messages=response_messages,
            wall_time_ms=wall_time_ms,
            dataset_id=self._dataset_id,
            tool_origin_map=self._tool_origin_map,
        )

        history.append(AIMessage(content=ai_answer))

        return {
            "answer": ai_answer,
            "conversation_id": cid,
            "trace": trace,
        }

    @property
    def tracker(self) -> QueryTracker:
        return self._tracker

    @property
    def analytics(self) -> AnalyticsService:
        return self._analytics

    @property
    def profile(self) -> DatasetProfile | None:
        return self._profile


def create_dynamic_agent(
    df: pd.DataFrame,
    dataset_name: str,
    settings: Settings | None = None,
) -> SalesAgent:
    profile = profile_dataframe(df, dataset_name=dataset_name)
    bundle = build_dynamic_tool_bundle(df, profile)
    prompt = build_dynamic_system_prompt(profile)
    return SalesAgent(
        df,
        settings=settings,
        tools=bundle.tools,
        prompt=prompt,
        dataset_id=profile.dataset_id,
        tool_origin_map=bundle.tool_origin_map,
        profile=profile,
    )
