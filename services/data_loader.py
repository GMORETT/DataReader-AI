from __future__ import annotations

import logging
from io import BytesIO
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _detect_separator(sample_text: str) -> str:
    candidates = [";", ",", "\t", "|"]
    scores = {sep: sample_text.count(sep) for sep in candidates}
    return max(scores, key=scores.get) if sample_text else ","


def _clean_dataframe(df: pd.DataFrame, date_format: str) -> pd.DataFrame:
    """Normalise column types after loading the raw CSV."""
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"], format=date_format, dayfirst=True)

    int_cols = ["planned_quantity", "actual_quantity"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = ["planned_price", "actual_price", "service_level"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["promotion_type"] = df["promotion_type"].astype(str).str.strip()

    df["quantity_diff"] = df["actual_quantity"] - df["planned_quantity"]
    df["price_diff"] = df["actual_price"] - df["planned_price"]
    df["actual_revenue"] = df["actual_quantity"] * df["actual_price"]
    df["planned_revenue"] = df["planned_quantity"] * df["planned_price"]
    df["revenue_diff"] = df["actual_revenue"] - df["planned_revenue"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["quarter"] = df["date"].dt.quarter

    logger.info("DataFrame loaded: %d rows × %d columns", len(df), len(df.columns))
    return df


def _infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="mixed")
            # Convert when we have enough parseable values to avoid accidental coercion.
            if converted.notna().mean() >= 0.8:
                df[col] = converted
    return df


def load_csv_generic(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load any CSV with best-effort type inference (no sales-specific logic)."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    sample = path.read_text(encoding=encoding, errors="ignore")[:4096]
    sep = _detect_separator(sample)
    df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Generic numeric/date inference keeps the dynamic mode dataset-agnostic.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() >= 0.8:
                df[col] = numeric

    df = _infer_datetime_columns(df)
    return df


def load_csv_generic_from_bytes(file_bytes: bytes, filename: str = "uploaded.csv") -> pd.DataFrame:
    """Load uploaded CSV bytes for Streamlit dynamic mode."""
    if not file_bytes:
        raise ValueError(f"Uploaded file {filename} is empty")

    head = file_bytes[:4096].decode("utf-8", errors="ignore")
    sep = _detect_separator(head)
    df = pd.read_csv(BytesIO(file_bytes), sep=sep, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() >= 0.8:
                df[col] = numeric

    df = _infer_datetime_columns(df)
    return df


@lru_cache(maxsize=1)
def load_sales_data(path: Path | None = None) -> pd.DataFrame:
    """Load, validate and enrich the sales CSV into a pandas DataFrame."""
    settings = get_settings()
    csv_path = Path(path) if path else settings.csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading CSV from %s …", csv_path)
    df = pd.read_csv(
        csv_path,
        sep=settings.csv_separator,
        encoding=settings.csv_encoding,
        low_memory=False,
    )
    return _clean_dataframe(df, settings.date_format)
