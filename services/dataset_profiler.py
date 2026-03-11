from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    null_ratio: float
    unique_count: int
    sample_values: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    dataset_id: str
    dataset_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "semantic_type": c.semantic_type,
                    "null_ratio": c.null_ratio,
                    "unique_count": c.unique_count,
                    "sample_values": c.sample_values,
                }
                for c in self.columns
            ],
        }


def build_dataset_id(df: pd.DataFrame, dataset_name: str = "") -> str:
    basis = f"{dataset_name}|{len(df)}|{','.join(df.columns.astype(str))}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    return digest[:12]


def _semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def profile_dataframe(df: pd.DataFrame, dataset_name: str = "uploaded_dataset") -> DatasetProfile:
    columns: list[ColumnProfile] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        sem_type = _semantic_type(series)
        if sem_type == "numeric":
            numeric_columns.append(col)
        elif sem_type == "datetime":
            datetime_columns.append(col)
        else:
            categorical_columns.append(col)

        samples = series.dropna().astype(str).head(5).tolist()
        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                semantic_type=sem_type,
                null_ratio=float(series.isna().mean()) if len(series) else 0.0,
                unique_count=int(series.nunique(dropna=True)),
                sample_values=samples,
            )
        )

    return DatasetProfile(
        dataset_id=build_dataset_id(df, dataset_name),
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )
