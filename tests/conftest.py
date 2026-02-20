"""Shared fixtures for the test suite.

Provides a small, deterministic DataFrame and pre-built service instances
so that tests never depend on the real sales.csv or an LLM API key.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from services.analytics import AnalyticsService


SAMPLE_CSV = """\
product_id;local;date;planned_quantity;actual_quantity;planned_price;promotion_type;actual_price;service_level
Product_A;Whse_A;01/01/2012;100;120;50;None;50;0.95
Product_A;Whse_A;15/01/2012;200;180;50;1;48;0.90
Product_A;Whse_B;01/02/2012;150;150;50;None;50;0.85
Product_B;Whse_A;01/01/2012;300;350;80;None;80;0.92
Product_B;Whse_B;01/03/2012;400;250;80;2;75;0.88
Product_B;Whse_C;15/03/2012;500;600;80;None;80;0.97
Product_C;Whse_A;01/02/2012;50;45;200;None;200;0.80
Product_C;Whse_C;01/04/2012;60;70;200;3;190;0.91
"""


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Write the sample CSV to a temp file and return its path."""
    csv_file = tmp_path / "test_sales.csv"
    csv_file.write_text(SAMPLE_CSV, encoding="utf-8")
    return csv_file


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a cleaned DataFrame matching the data loader output."""
    from io import StringIO

    df = pd.read_csv(StringIO(SAMPLE_CSV), sep=";", keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", dayfirst=True)
    df["planned_quantity"] = df["planned_quantity"].astype(int)
    df["actual_quantity"] = df["actual_quantity"].astype(int)
    df["planned_price"] = df["planned_price"].astype(float)
    df["actual_price"] = df["actual_price"].astype(float)
    df["service_level"] = df["service_level"].astype(float)
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

    return df


@pytest.fixture
def analytics(sample_df: pd.DataFrame) -> AnalyticsService:
    return AnalyticsService(sample_df)
