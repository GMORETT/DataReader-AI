from __future__ import annotations

import os

from services.data_loader import load_csv_generic_from_bytes


def test_load_csv_generic_from_bytes(sample_csv_path):
    csv_bytes = sample_csv_path.read_bytes()
    df = load_csv_generic_from_bytes(csv_bytes, filename="sample.csv")
    assert len(df) == 8
    assert "product_id" in df.columns
    assert "actual_quantity" in df.columns


def test_create_dynamic_agent_without_invocation(sample_df, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test-key"))
    from agent.csv_agent import create_dynamic_agent

    agent = create_dynamic_agent(sample_df, dataset_name="sample.csv")
    assert agent.profile is not None
    assert agent.profile.dataset_name == "sample.csv"
    assert len(agent.profile.numeric_columns) > 0
