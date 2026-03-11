from __future__ import annotations

import pandas as pd

from services.dataset_profiler import build_dataset_id, profile_dataframe


def test_profile_dataframe_basic(sample_df: pd.DataFrame):
    profile = profile_dataframe(sample_df, dataset_name="sample.csv")
    assert profile.dataset_name == "sample.csv"
    assert profile.row_count == len(sample_df)
    assert profile.column_count == len(sample_df.columns)
    assert len(profile.columns) == len(sample_df.columns)


def test_profile_detects_semantic_types(sample_df: pd.DataFrame):
    profile = profile_dataframe(sample_df)
    assert "actual_quantity" in profile.numeric_columns
    assert "local" in profile.categorical_columns
    assert "date" in profile.datetime_columns


def test_dataset_id_is_stable(sample_df: pd.DataFrame):
    id1 = build_dataset_id(sample_df, "sample.csv")
    id2 = build_dataset_id(sample_df, "sample.csv")
    assert id1 == id2
    assert len(id1) == 12
