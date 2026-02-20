"""API route definitions."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from agent.csv_agent import SalesAgent
from models.schemas import ChatRequest, ChatResponse
from services.data_loader import load_sales_data

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@lru_cache(maxsize=1)
def _get_agent() -> SalesAgent:
    df = load_sales_data()
    return SalesAgent(df)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the AI agent and receive an answer."""
    try:
        agent = _get_agent()
        result = agent.ask(
            question=request.question,
            conversation_id=request.conversation_id,
        )
        trace = result["trace"]
        return ChatResponse(
            answer=result["answer"],
            conversation_id=result["conversation_id"],
            metadata=trace.to_dict(),
        )
    except Exception as e:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/overview")
async def overview():
    """Quick dataset overview (no LLM call)."""
    agent = _get_agent()
    return agent.analytics.dataset_overview()


@router.get("/analytics/top-products")
async def top_products(n: int = 10, by: str = "quantity"):
    agent = _get_agent()
    if by == "revenue":
        return agent.analytics.top_products_by_revenue(n)
    return agent.analytics.top_products_by_quantity(n)


@router.get("/analytics/top-locations")
async def top_locations(n: int = 10, by: str = "quantity"):
    agent = _get_agent()
    if by == "revenue":
        return agent.analytics.top_locations_by_revenue(n)
    return agent.analytics.top_locations_by_quantity(n)


@router.get("/analytics/promotions")
async def promotions():
    agent = _get_agent()
    return agent.analytics.promotion_impact()


@router.get("/analytics/planned-vs-actual")
async def planned_vs_actual():
    agent = _get_agent()
    return agent.analytics.planned_vs_actual_summary()


@router.get("/analytics/service-level")
async def service_level():
    agent = _get_agent()
    return agent.analytics.service_level_stats()
