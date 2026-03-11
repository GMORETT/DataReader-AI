from __future__ import annotations

from services.dataset_profiler import DatasetProfile


def build_dynamic_system_prompt(profile: DatasetProfile) -> str:
    column_lines = []
    for c in profile.columns[:40]:
        sample = ", ".join(c.sample_values[:3]) if c.sample_values else "n/a"
        column_lines.append(
            f"- {c.name} ({c.semantic_type}, dtype={c.dtype}, unique={c.unique_count}, sample={sample})"
        )

    columns_block = "\n".join(column_lines)
    return f"""\
You are an expert data analyst working on an arbitrary uploaded dataset.
Dataset name: {profile.dataset_name}
Dataset id: {profile.dataset_id}
Rows: {profile.row_count}
Columns: {profile.column_count}

Column profile:
{columns_block}

Rules:
1. Always use tools before answering numeric questions.
2. Prefer dynamic/specific tools first, then generic tools, then python_repl.
3. If user asks for comparisons/rankings, include the exact values and method.
4. If no variation exists in data, explicitly say there is no meaningful difference.
5. Answer in the same language as the user.
6. If question is ambiguous, ask a concise clarification question.
"""
