"""Tests for services.analytics.AnalyticsService.

All tests use the sample_df / analytics fixtures from conftest.py.
The sample has 8 rows:
  Product_A: qty 120+180+150=450, Whse_A(2x), Whse_B(1x)
  Product_B: qty 350+250+600=1200, Whse_A(1x), Whse_B(1x), Whse_C(1x)
  Product_C: qty 45+70=115, Whse_A(1x), Whse_C(1x)
"""

from __future__ import annotations

from services.analytics import AnalyticsService


class TestTopProducts:

    def test_top_products_by_quantity_order(self, analytics: AnalyticsService):
        result = analytics.top_products_by_quantity(n=3)
        data = result["data"]
        values = list(data.values())
        assert values == sorted(values, reverse=True)

    def test_top_product_is_product_b(self, analytics: AnalyticsService):
        result = analytics.top_products_by_quantity(n=1)
        top_product = list(result["data"].keys())[0]
        assert top_product == "Product_B"

    def test_top_product_quantity_value(self, analytics: AnalyticsService):
        result = analytics.top_products_by_quantity(n=1)
        assert list(result["data"].values())[0] == 1200

    def test_respects_n_parameter(self, analytics: AnalyticsService):
        result = analytics.top_products_by_quantity(n=2)
        assert len(result["data"]) == 2

    def test_top_products_by_revenue(self, analytics: AnalyticsService):
        result = analytics.top_products_by_revenue(n=3)
        assert "metric" in result
        assert "data" in result
        assert len(result["data"]) == 3


class TestTopLocations:

    def test_top_locations_by_quantity(self, analytics: AnalyticsService):
        result = analytics.top_locations_by_quantity(n=3)
        assert len(result["data"]) == 3
        assert "Whse_A" in result["data"]

    def test_top_locations_by_revenue(self, analytics: AnalyticsService):
        result = analytics.top_locations_by_revenue(n=2)
        assert len(result["data"]) == 2


class TestPeriodAnalysis:

    def test_total_sales_no_filter(self, analytics: AnalyticsService):
        result = analytics.total_sales_in_period()
        assert result["num_transactions"] == 8
        assert result["total_actual_quantity"] == 450 + 1200 + 115

    def test_total_sales_with_date_filter(self, analytics: AnalyticsService):
        result = analytics.total_sales_in_period(
            start_date="01/02/2012",
            end_date="28/02/2012",
        )
        assert result["num_transactions"] < 8
        assert result["total_actual_quantity"] > 0

    def test_total_sales_empty_period(self, analytics: AnalyticsService):
        result = analytics.total_sales_in_period(
            start_date="01/01/2025",
            end_date="31/12/2025",
        )
        assert result["num_transactions"] == 0
        assert result["total_actual_quantity"] == 0

    def test_monthly_sales_summary(self, analytics: AnalyticsService):
        result = analytics.monthly_sales_summary()
        assert result["metric"] == "monthly_sales_summary"
        assert len(result["data"]) > 0
        first_month = result["data"][0]
        assert "total_quantity" in first_month
        assert "total_revenue" in first_month


class TestPlannedVsActual:

    def test_summary_keys(self, analytics: AnalyticsService):
        result = analytics.planned_vs_actual_summary()
        expected_keys = [
            "total_planned_qty", "total_actual_qty", "total_quantity_diff",
            "avg_quantity_diff", "total_planned_revenue", "total_actual_revenue",
            "total_revenue_diff",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_quantity_diff_is_consistent(self, analytics: AnalyticsService):
        result = analytics.planned_vs_actual_summary()
        assert result["total_quantity_diff"] == (
            result["total_actual_qty"] - result["total_planned_qty"]
        )

    def test_by_product_returns_correct_count(self, analytics: AnalyticsService):
        result = analytics.planned_vs_actual_by_product(n=2)
        assert len(result["data"]) == 2


class TestPromotionImpact:

    def test_returns_promotion_types(self, analytics: AnalyticsService):
        result = analytics.promotion_impact()
        promo_types = [r["promotion_type"] for r in result["data"]]
        assert "None" in promo_types

    def test_has_expected_fields(self, analytics: AnalyticsService):
        result = analytics.promotion_impact()
        record = result["data"][0]
        assert "total_quantity" in record
        assert "total_revenue" in record
        assert "num_transactions" in record


class TestServiceLevel:

    def test_stats_values(self, analytics: AnalyticsService):
        result = analytics.service_level_stats()
        assert 0 <= result["mean"] <= 1
        assert 0 <= result["min"] <= result["max"] <= 1
        assert result["std"] >= 0

    def test_by_location(self, analytics: AnalyticsService):
        result = analytics.service_level_by_location()
        assert "Whse_A" in result["data"]
        for loc_data in result["data"].values():
            assert "mean" in loc_data


class TestDatasetOverview:

    def test_overview_values(self, analytics: AnalyticsService):
        result = analytics.dataset_overview()
        assert result["total_rows"] == 8
        assert result["unique_products"] == 3
        assert result["unique_locations"] == 3

    def test_date_range(self, analytics: AnalyticsService):
        result = analytics.dataset_overview()
        assert "min" in result["date_range"]
        assert "max" in result["date_range"]

    def test_columns_list(self, analytics: AnalyticsService):
        result = analytics.dataset_overview()
        assert "product_id" in result["columns"]
        assert "actual_revenue" in result["columns"]
