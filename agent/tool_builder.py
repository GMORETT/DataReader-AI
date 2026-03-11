from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from agent.dynamic_tools import build_dynamic_tools
from agent.generic_tools import build_generic_tools
from agent.tools import build_tools as build_sales_tools
from services.analytics import AnalyticsService
from services.dataset_profiler import DatasetProfile


@dataclass
class ToolBundle:
    tools: list
    tool_origin_map: dict[str, str]


def build_sales_tool_bundle(df: pd.DataFrame, analytics: AnalyticsService) -> ToolBundle:
    tools = build_sales_tools(df, analytics)
    return ToolBundle(
        tools=tools,
        tool_origin_map={t.name: "sales_specific" for t in tools},
    )


def build_dynamic_tool_bundle(df: pd.DataFrame, profile: DatasetProfile) -> ToolBundle:
    generic_tools = build_generic_tools(df)
    dynamic_tools = build_dynamic_tools(df, profile)
    python_repl = PythonAstREPLTool(
        locals={"df": df},
        name="python_repl",
        description="Execute ad-hoc Python/pandas against df. Use print() for output.",
    )

    tools = [*dynamic_tools, *generic_tools, python_repl]
    tool_origin_map: dict[str, str] = {}
    for t in dynamic_tools:
        tool_origin_map[t.name] = "dynamic"
    for t in generic_tools:
        tool_origin_map[t.name] = "generic"
    tool_origin_map[python_repl.name] = "python_repl"
    return ToolBundle(tools=tools, tool_origin_map=tool_origin_map)
