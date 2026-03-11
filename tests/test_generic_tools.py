from __future__ import annotations

import json

import pandas as pd

from agent.generic_tools import build_generic_tools


def _tool_map(df: pd.DataFrame) -> dict:
    tools = build_generic_tools(df)
    return {t.name: t for t in tools}


def test_generic_tools_exist(sample_df: pd.DataFrame):
    tool_map = _tool_map(sample_df)
    expected = {
        "dataset_overview_generic",
        "aggregate_generic",
        "top_n_generic",
        "describe_column_generic",
        "filter_query_generic",
    }
    assert expected.issubset(set(tool_map.keys()))


def test_aggregate_generic(sample_df: pd.DataFrame):
    tool = _tool_map(sample_df)["aggregate_generic"]
    result = tool.invoke('{"group_by":"product_id","value_col":"actual_quantity","op":"sum"}')
    parsed = json.loads(result)
    assert parsed["metric"] == "aggregate_generic"
    assert "Product_B" in parsed["data"]


def test_describe_column_generic(sample_df: pd.DataFrame):
    tool = _tool_map(sample_df)["describe_column_generic"]
    result = tool.invoke("actual_quantity")
    parsed = json.loads(result)
    assert parsed["metric"] == "describe_column_generic"
    assert parsed["column"] == "actual_quantity"
