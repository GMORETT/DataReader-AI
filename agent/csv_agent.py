"""High-level Sales Agent facade.

Creates and manages the LangChain ReAct agent backed by custom tools and an
optional pandas REPL.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import SYSTEM_PROMPT
from agent.tools import build_tools
from config.settings import Settings, get_settings
from services.analytics import AnalyticsService

logger = logging.getLogger(__name__)


class SalesAgent:
    """Stateful conversational agent for sales‑data Q&A."""

    def __init__(
        self,
        df: pd.DataFrame,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._df = df
        self._analytics = AnalyticsService(df)
        self._tools = build_tools(df, self._analytics)
        self._llm = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_model,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        self._conversations: dict[str, list] = {}
        self._agent = self._build_agent()

    def _build_agent(self):
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
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

        result = self._agent.invoke(
            {"messages": history},
        )

        response_messages = result["messages"]
        ai_answer = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                ai_answer = msg.content
                break

        history.append(AIMessage(content=ai_answer))

        return {
            "answer": ai_answer,
            "conversation_id": cid,
        }

    @property
    def analytics(self) -> AnalyticsService:
        return self._analytics
