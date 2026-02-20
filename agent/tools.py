"""LangChain tools that wrap AnalyticsService methods.

Each tool is a thin bridge: it receives a text input from the LLM,
parses optional arguments, calls the analytics service, and returns
a JSON string that the LLM can interpret and summarise for the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from services.analytics import AnalyticsService

logger = logging.getLogger(__name__)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def build_tools(df: pd.DataFrame, analytics: AnalyticsService) -> list:
    """Build the full list of LangChain tools the agent can use."""

    @tool
    def dataset_overview(query: str = "") -> str:
        """Get a high-level overview of the dataset: row count, unique products,
        unique locations, date range, promotion types and column names."""
        return _json(analytics.dataset_overview())

    @tool
    def top_products_by_quantity(n: str = "10") -> str:
        """Return the top N products ranked by total actual quantity sold.
        Input: number of results (default 10)."""
        return _json(analytics.top_products_by_quantity(int(n)))

    @tool
    def top_locations_by_quantity(n: str = "10") -> str:
        """Return the top N locations ranked by total actual quantity sold.
        Input: number of results (default 10)."""
        return _json(analytics.top_locations_by_quantity(int(n)))

    @tool
    def top_products_by_revenue(n: str = "10") -> str:
        """Return the top N products ranked by total actual revenue.
        Input: number of results (default 10)."""
        return _json(analytics.top_products_by_revenue(int(n)))

    @tool
    def top_locations_by_revenue(n: str = "10") -> str:
        """Return the top N locations ranked by total actual revenue.
        Input: number of results (default 10)."""
        return _json(analytics.top_locations_by_revenue(int(n)))

    @tool
    def total_sales_in_period(period: str = "") -> str:
        """Get total sales (quantity and revenue) for a date range.
        Input: 'start_date,end_date' in DD/MM/YYYY format.
        Either date can be empty. Examples: '01/01/2012,31/12/2012' or ',31/06/2012'."""
        parts = [p.strip() for p in period.split(",")]
        start = parts[0] if len(parts) > 0 and parts[0] else None
        end = parts[1] if len(parts) > 1 and parts[1] else None
        return _json(analytics.total_sales_in_period(start, end))

    @tool
    def monthly_sales_summary(query: str = "") -> str:
        """Get a monthly breakdown of total quantity, revenue, average price
        and number of transactions."""
        return _json(analytics.monthly_sales_summary())

    @tool
    def planned_vs_actual_summary(query: str = "") -> str:
        """Compare planned vs actual quantities and revenue across the entire
        dataset.  Shows totals and averages."""
        return _json(analytics.planned_vs_actual_summary())

    @tool
    def planned_vs_actual_by_product(n: str = "10") -> str:
        """Show the top N products with the largest difference between planned
        and actual quantities.  Input: number of results (default 10)."""
        return _json(analytics.planned_vs_actual_by_product(int(n)))

    @tool
    def promotion_impact(query: str = "") -> str:
        """Analyse the impact of each promotion type on price, quantity and
        revenue.  Compares promo vs no-promo transactions."""
        return _json(analytics.promotion_impact())

    @tool
    def service_level_stats(query: str = "") -> str:
        """Return service-level statistics: mean, median, std, min, max."""
        return _json(analytics.service_level_stats())

    @tool
    def service_level_by_location(query: str = "") -> str:
        """Show service-level statistics broken down by warehouse/location."""
        return _json(analytics.service_level_by_location())

    python_repl = PythonAstREPLTool(
        locals={"df": df},
        name="python_repl",
        description=(
            "Execute Python/pandas code on the DataFrame `df` for custom "
            "analysis that the other tools cannot handle.  Always print() "
            "results so they appear in the output."
        ),
    )

    return [
        dataset_overview,
        top_products_by_quantity,
        top_locations_by_quantity,
        top_products_by_revenue,
        top_locations_by_revenue,
        total_sales_in_period,
        monthly_sales_summary,
        planned_vs_actual_summary,
        planned_vs_actual_by_product,
        promotion_impact,
        service_level_stats,
        service_level_by_location,
        python_repl,
    ]
