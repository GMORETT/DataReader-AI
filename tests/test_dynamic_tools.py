from __future__ import annotations

import json

import pandas as pd

from agent.dynamic_tools import build_dynamic_tools
from services.dataset_profiler import profile_dataframe


def test_dynamic_tools_created(sample_df: pd.DataFrame):
    profile = profile_dataframe(sample_df, dataset_name="sample.csv")
    tools = build_dynamic_tools(sample_df, profile)
    assert len(tools) > 0


def test_dynamic_tool_returns_json(sample_df: pd.DataFrame):
    profile = profile_dataframe(sample_df, dataset_name="sample.csv")
    tools = build_dynamic_tools(sample_df, profile)
    first = tools[0]
    out = first.invoke("5")
    parsed = json.loads(out)
    assert "metric" in parsed
    assert "data" in parsed


def test_dynamic_tools_respect_cap(sample_df: pd.DataFrame):
    profile = profile_dataframe(sample_df, dataset_name="sample.csv")
    tools = build_dynamic_tools(sample_df, profile, max_tools=3)
    assert len(tools) <= 3
