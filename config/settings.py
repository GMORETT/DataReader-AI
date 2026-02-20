from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application‑wide configuration, loaded from env vars / .env file."""

    # --- LLM ------------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model name")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)

    # --- Data -----------------------------------------------------------------
    csv_path: Path = Field(
        default=PROJECT_ROOT / "sales.csv",
        description="Path to the sales CSV file",
    )
    csv_separator: str = Field(";", description="CSV column separator")
    csv_encoding: str = Field("utf-8", description="CSV file encoding")
    date_format: str = Field("%d/%m/%Y", description="Date column format")

    # --- API ------------------------------------------------------------------
    api_host: str = Field("0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(8000, ge=1, le=65535)

    # --- Streamlit ------------------------------------------------------------
    streamlit_port: int = Field(8501, ge=1, le=65535)

    # --- General --------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cache_ttl_seconds: int = Field(300, ge=0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("csv_path", mode="before")
    @classmethod
    def _resolve_csv_path(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
