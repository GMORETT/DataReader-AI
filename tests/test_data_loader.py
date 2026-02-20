"""Tests for services.data_loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class TestCleanDataframe:
    """Test the _clean_dataframe helper (called internally by load_sales_data)."""

    def test_columns_are_lowercase(self, sample_df: pd.DataFrame):
        for col in sample_df.columns:
            assert col == col.lower()

    def test_date_column_is_datetime(self, sample_df: pd.DataFrame):
        assert pd.api.types.is_datetime64_any_dtype(sample_df["date"])

    def test_integer_columns(self, sample_df: pd.DataFrame):
        assert sample_df["planned_quantity"].dtype in ("int64", "int32")
        assert sample_df["actual_quantity"].dtype in ("int64", "int32")

    def test_float_columns(self, sample_df: pd.DataFrame):
        for col in ("planned_price", "actual_price", "service_level"):
            assert pd.api.types.is_float_dtype(sample_df[col])

    def test_derived_columns_exist(self, sample_df: pd.DataFrame):
        expected = [
            "quantity_diff", "price_diff", "actual_revenue",
            "planned_revenue", "revenue_diff", "year", "month",
            "month_name", "quarter",
        ]
        for col in expected:
            assert col in sample_df.columns, f"Missing derived column: {col}"

    def test_quantity_diff_calculation(self, sample_df: pd.DataFrame):
        row = sample_df.iloc[0]
        assert row["quantity_diff"] == row["actual_quantity"] - row["planned_quantity"]

    def test_actual_revenue_calculation(self, sample_df: pd.DataFrame):
        row = sample_df.iloc[0]
        assert row["actual_revenue"] == row["actual_quantity"] * row["actual_price"]

    def test_planned_revenue_calculation(self, sample_df: pd.DataFrame):
        row = sample_df.iloc[0]
        assert row["planned_revenue"] == row["planned_quantity"] * row["planned_price"]

    def test_revenue_diff_calculation(self, sample_df: pd.DataFrame):
        row = sample_df.iloc[0]
        assert row["revenue_diff"] == row["actual_revenue"] - row["planned_revenue"]

    def test_year_extracted(self, sample_df: pd.DataFrame):
        assert (sample_df["year"] == 2012).all()

    def test_row_count(self, sample_df: pd.DataFrame):
        assert len(sample_df) == 8


class TestLoadSalesData:
    """Test load_sales_data with real file I/O (temp files)."""

    def test_load_from_csv_file(self, sample_csv_path: Path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from services.data_loader import _clean_dataframe

        df = pd.read_csv(sample_csv_path, sep=";")
        result = _clean_dataframe(df, "%d/%m/%Y")

        assert len(result) == 8
        assert "actual_revenue" in result.columns

    def test_file_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from services.data_loader import load_sales_data

        load_sales_data.cache_clear()
        with pytest.raises(FileNotFoundError):
            load_sales_data(path=Path("/nonexistent/path/sales.csv"))

    def test_empty_csv_raises(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("", encoding="utf-8")

        from services.data_loader import _clean_dataframe

        with pytest.raises(Exception):
            df = pd.read_csv(empty_file, sep=";")
            _clean_dataframe(df, "%d/%m/%Y")
