from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _parse_json(input_str: str) -> dict[str, Any]:
    if not input_str or not input_str.strip():
        return {}
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        return {}


def build_generic_tools(df: pd.DataFrame) -> list:
    @tool
    def dataset_overview_generic(query: str = "") -> str:
        """Return dataset overview (rows, columns, dtypes, null percentages)."""
        null_pct = (df.isna().mean() * 100).round(2).to_dict()
        dtypes = {c: str(t) for c, t in df.dtypes.to_dict().items()}
        return _json(
            {
                "metric": "dataset_overview_generic",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": dtypes,
                "null_percentage": null_pct,
            }
        )

    @tool
    def aggregate_generic(input_json: str = "") -> str:
        """Aggregate by group.
        Input JSON: {"group_by":"col","value_col":"col","op":"sum|mean|min|max|count"}"""
        payload = _parse_json(input_json)
        group_by = payload.get("group_by")
        value_col = payload.get("value_col")
        op = str(payload.get("op", "sum")).lower()

        if not group_by or not value_col:
            return _json({"error": "group_by and value_col are required"})
        if group_by not in df.columns or value_col not in df.columns:
            return _json({"error": "invalid column name"})

        grouped = df.groupby(group_by)[value_col]
        if op == "sum":
            result = grouped.sum()
        elif op == "mean":
            result = grouped.mean()
        elif op == "min":
            result = grouped.min()
        elif op == "max":
            result = grouped.max()
        elif op == "count":
            result = grouped.count()
        else:
            return _json({"error": f"unsupported op '{op}'"})

        return _json(
            {
                "metric": "aggregate_generic",
                "group_by": group_by,
                "value_col": value_col,
                "op": op,
                "data": result.sort_values(ascending=False).to_dict(),
            }
        )

    @tool
    def top_n_generic(input_json: str = "") -> str:
        """Top N by aggregation.
        Input JSON: {"group_by":"col","value_col":"col","op":"sum|mean|count","n":10}"""
        payload = _parse_json(input_json)
        group_by = payload.get("group_by")
        value_col = payload.get("value_col")
        op = str(payload.get("op", "sum")).lower()
        n = int(payload.get("n", 10))

        if not group_by or not value_col:
            return _json({"error": "group_by and value_col are required"})
        if group_by not in df.columns or value_col not in df.columns:
            return _json({"error": "invalid column name"})

        grouped = df.groupby(group_by)[value_col]
        if op == "sum":
            result = grouped.sum()
        elif op == "mean":
            result = grouped.mean()
        elif op == "count":
            result = grouped.count()
        else:
            return _json({"error": f"unsupported op '{op}'"})

        ranking = result.sort_values(ascending=False).head(n)
        return _json(
            {
                "metric": "top_n_generic",
                "group_by": group_by,
                "value_col": value_col,
                "op": op,
                "n": n,
                "data": ranking.to_dict(),
            }
        )

    @tool
    def describe_column_generic(column_name: str = "") -> str:
        """Describe one column (stats for numeric; cardinality for categorical)."""
        col = (column_name or "").strip()
        if not col:
            return _json({"error": "column_name is required"})
        if col not in df.columns:
            return _json({"error": "invalid column name"})

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            data = series.describe().to_dict()
        else:
            data = {
                "unique_count": int(series.nunique(dropna=True)),
                "top_values": series.value_counts(dropna=False).head(10).to_dict(),
            }

        return _json({"metric": "describe_column_generic", "column": col, "data": data})

    @tool
    def filter_query_generic(input_json: str = "") -> str:
        """Filter rows using pandas query and optionally select columns.
        Input JSON: {"query":"col > 10","columns":["a","b"],"limit":20}"""
        payload = _parse_json(input_json)
        query = payload.get("query", "")
        columns = payload.get("columns", [])
        limit = int(payload.get("limit", 20))

        subset = df.query(query) if query else df.copy()
        if columns:
            valid_cols = [c for c in columns if c in subset.columns]
            subset = subset[valid_cols] if valid_cols else subset

        preview = subset.head(limit).to_dict(orient="records")
        return _json(
            {
                "metric": "filter_query_generic",
                "query": query,
                "rows_matched": int(len(subset)),
                "preview": preview,
            }
        )

    return [
        dataset_overview_generic,
        aggregate_generic,
        top_n_generic,
        describe_column_generic,
        filter_query_generic,
    ]
