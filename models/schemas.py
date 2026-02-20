from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PromotionType(str, Enum):
    NONE = "None"
    PROMO_1 = "1"
    PROMO_2 = "2"
    PROMO_3 = "3"
    PROMO_4 = "4"
    PROMO_5 = "5"


class SalesRow(BaseModel):
    """Pydantic model that mirrors a single CSV row."""

    product_id: str
    local: str
    date: date
    planned_quantity: int
    actual_quantity: int
    planned_price: float
    promotion_type: str
    actual_price: float
    service_level: float

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> date:
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%d/%m/%Y").date()
        raise ValueError(f"Cannot parse date: {v}")


class AnalyticsResult(BaseModel):
    """Wrapper returned by analytics tools."""

    metric: str
    value: Any
    details: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
