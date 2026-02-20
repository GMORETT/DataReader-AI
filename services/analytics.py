from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Pre-built analytics on the sales DataFrame.

    Each public method returns a plain dict so it can be serialised by the
    LangChain tool layer and sent back to the LLM as structured context.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------
    # Top / Bottom helpers
    # ------------------------------------------------------------------

    def top_products_by_quantity(self, n: int = 10) -> dict[str, Any]:
        ranking = (
            self._df.groupby("product_id")["actual_quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return {
            "metric": "top_products_by_actual_quantity",
            "data": ranking.to_dict(),
        }

    def top_locations_by_quantity(self, n: int = 10) -> dict[str, Any]:
        ranking = (
            self._df.groupby("local")["actual_quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return {
            "metric": "top_locations_by_actual_quantity",
            "data": ranking.to_dict(),
        }

    def top_products_by_revenue(self, n: int = 10) -> dict[str, Any]:
        ranking = (
            self._df.groupby("product_id")["actual_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return {
            "metric": "top_products_by_revenue",
            "data": ranking.to_dict(),
        }

    def top_locations_by_revenue(self, n: int = 10) -> dict[str, Any]:
        ranking = (
            self._df.groupby("local")["actual_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return {
            "metric": "top_locations_by_revenue",
            "data": ranking.to_dict(),
        }

    # ------------------------------------------------------------------
    # Period analysis
    # ------------------------------------------------------------------

    def total_sales_in_period(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        df = self._df.copy()
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date, dayfirst=True)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date, dayfirst=True)]

        return {
            "metric": "total_sales_in_period",
            "start_date": start_date,
            "end_date": end_date,
            "total_actual_quantity": int(df["actual_quantity"].sum()),
            "total_actual_revenue": float(df["actual_revenue"].sum()),
            "total_planned_quantity": int(df["planned_quantity"].sum()),
            "total_planned_revenue": float(df["planned_revenue"].sum()),
            "num_transactions": len(df),
        }

    def monthly_sales_summary(self) -> dict[str, Any]:
        summary = (
            self._df.groupby(["year", "month"])
            .agg(
                total_quantity=("actual_quantity", "sum"),
                total_revenue=("actual_revenue", "sum"),
                avg_price=("actual_price", "mean"),
                num_transactions=("product_id", "count"),
            )
            .reset_index()
            .sort_values(["year", "month"])
        )
        return {
            "metric": "monthly_sales_summary",
            "data": summary.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Planned vs Actual
    # ------------------------------------------------------------------

    def planned_vs_actual_summary(self) -> dict[str, Any]:
        df = self._df
        return {
            "metric": "planned_vs_actual_summary",
            "total_planned_qty": int(df["planned_quantity"].sum()),
            "total_actual_qty": int(df["actual_quantity"].sum()),
            "total_quantity_diff": int(df["quantity_diff"].sum()),
            "avg_quantity_diff": float(df["quantity_diff"].mean()),
            "total_planned_revenue": float(df["planned_revenue"].sum()),
            "total_actual_revenue": float(df["actual_revenue"].sum()),
            "total_revenue_diff": float(df["revenue_diff"].sum()),
        }

    def planned_vs_actual_by_product(self, n: int = 10) -> dict[str, Any]:
        agg = (
            self._df.groupby("product_id")
            .agg(
                planned_qty=("planned_quantity", "sum"),
                actual_qty=("actual_quantity", "sum"),
                qty_diff=("quantity_diff", "sum"),
            )
            .sort_values("qty_diff", key=abs, ascending=False)
            .head(n)
        )
        return {
            "metric": "planned_vs_actual_by_product",
            "data": agg.to_dict(orient="index"),
        }

    # ------------------------------------------------------------------
    # Promotion analysis
    # ------------------------------------------------------------------

    def promotion_impact(self) -> dict[str, Any]:
        promo = (
            self._df.groupby("promotion_type")
            .agg(
                total_quantity=("actual_quantity", "sum"),
                avg_quantity=("actual_quantity", "mean"),
                avg_actual_price=("actual_price", "mean"),
                avg_planned_price=("planned_price", "mean"),
                price_diff=("price_diff", "mean"),
                num_transactions=("product_id", "count"),
                total_revenue=("actual_revenue", "sum"),
            )
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )
        return {
            "metric": "promotion_impact",
            "data": promo.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Service level
    # ------------------------------------------------------------------

    def service_level_stats(self) -> dict[str, Any]:
        df = self._df
        return {
            "metric": "service_level_statistics",
            "mean": float(df["service_level"].mean()),
            "median": float(df["service_level"].median()),
            "std": float(df["service_level"].std()),
            "min": float(df["service_level"].min()),
            "max": float(df["service_level"].max()),
        }

    def service_level_by_location(self) -> dict[str, Any]:
        agg = (
            self._df.groupby("local")["service_level"]
            .agg(["mean", "min", "max", "std"])
            .sort_values("mean", ascending=False)
        )
        return {
            "metric": "service_level_by_location",
            "data": agg.to_dict(orient="index"),
        }

    # ------------------------------------------------------------------
    # General dataset info
    # ------------------------------------------------------------------

    def dataset_overview(self) -> dict[str, Any]:
        df = self._df
        return {
            "metric": "dataset_overview",
            "total_rows": len(df),
            "unique_products": int(df["product_id"].nunique()),
            "unique_locations": int(df["local"].nunique()),
            "date_range": {
                "min": str(df["date"].min().date()),
                "max": str(df["date"].max().date()),
            },
            "promotion_types": df["promotion_type"].unique().tolist(),
            "columns": df.columns.tolist(),
        }
