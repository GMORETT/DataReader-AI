from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from services.dataset_profiler import DatasetProfile


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value.strip().lower())


def build_dynamic_tools(df: pd.DataFrame, profile: DatasetProfile, max_tools: int = 12) -> list:
    tools = []

    cat_cols = profile.categorical_columns[:4]
    num_cols = profile.numeric_columns[:4]
    dt_cols = profile.datetime_columns[:2]

    # Top category by numeric sum.
    for c_col in cat_cols:
        for n_col in num_cols:
            tool_name = f"top_{_safe_name(c_col)}_by_{_safe_name(n_col)}"

            @tool(tool_name)
            def _top_tool(n: str = "10", category_col: str = c_col, value_col: str = n_col) -> str:
                """Rank categories by numeric sum. Input: top N (default 10)."""
                try:
                    n_int = int(n)
                except (ValueError, TypeError):
                    n_int = 10

                data = (
                    df.groupby(category_col)[value_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n_int)
                    .to_dict()
                )
                return _json(
                    {
                        "metric": "dynamic_top_by_sum",
                        "category_col": category_col,
                        "value_col": value_col,
                        "n": n_int,
                        "data": data,
                    }
                )

            tools.append(_top_tool)
            if len(tools) >= max_tools:
                return tools

    # Mean by category for primary numeric columns.
    for c_col in cat_cols:
        for n_col in num_cols[:2]:
            tool_name = f"mean_{_safe_name(n_col)}_by_{_safe_name(c_col)}"

            @tool(tool_name)
            def _mean_tool(query: str = "", category_col: str = c_col, value_col: str = n_col) -> str:
                """Average numeric value grouped by category."""
                data = (
                    df.groupby(category_col)[value_col]
                    .mean()
                    .sort_values(ascending=False)
                    .to_dict()
                )
                return _json(
                    {
                        "metric": "dynamic_mean_by_category",
                        "category_col": category_col,
                        "value_col": value_col,
                        "data": data,
                    }
                )

            tools.append(_mean_tool)
            if len(tools) >= max_tools:
                return tools

    # Trend tools by datetime.
    if dt_cols and num_cols:
        date_col = dt_cols[0]
        for n_col in num_cols[:2]:
            tool_name = f"trend_{_safe_name(n_col)}_over_{_safe_name(date_col)}"

            @tool(tool_name)
            def _trend_tool(
                freq: str = "M",
                date_column: str = date_col,
                value_col: str = n_col,
            ) -> str:
                """Trend of numeric values by datetime frequency (D, W, M, Q, Y)."""
                series = df[date_column]
                if not pd.api.types.is_datetime64_any_dtype(series):
                    return _json({"error": f"{date_column} is not datetime"})

                freq_val = (freq or "M").upper()
                tmp = df.copy()
                tmp["_period"] = tmp[date_column].dt.to_period(freq_val).astype(str)
                trend = tmp.groupby("_period")[value_col].sum().to_dict()
                return _json(
                    {
                        "metric": "dynamic_trend",
                        "date_column": date_column,
                        "value_col": value_col,
                        "frequency": freq_val,
                        "data": trend,
                    }
                )

            tools.append(_trend_tool)
            if len(tools) >= max_tools:
                return tools

    return tools
