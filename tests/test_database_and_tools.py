from __future__ import annotations

from pathlib import Path

import pytest

from fintool_rl.database import build_fixture_snapshot, metadata, snapshot_manifest
from fintool_rl.tools import FinancialTools


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    return build_fixture_snapshot(tmp_path / "snapshot.sqlite")


def test_fixture_is_explicitly_synthetic_and_manifested(fixture_db: Path):
    meta = metadata(fixture_db)
    assert meta["data_class"] == "synthetic_fixture"
    manifest = snapshot_manifest(fixture_db)
    assert manifest["row_counts"] == {
        "companies": 5,
        "financial_facts": 75,
        "daily_prices": 25,
        "market_indices": 5,
    }
    assert len(manifest["sha256"]) == 64


def test_same_call_produces_same_observation_hash(fixture_db: Path):
    first = FinancialTools(fixture_db).call(
        "get_financial_fact", symbol="ALFA", metric="revenue", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    second = FinancialTools(fixture_db).call(
        "get_financial_fact", symbol="ALFA", metric="revenue", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    assert first == second
    assert first["provenance"]["observation_id"].startswith("obs_")


def test_future_data_is_rejected(fixture_db: Path):
    result = FinancialTools(fixture_db).call(
        "get_daily_price", symbol="ALFA", trading_date="2025-01-08", as_of_time="2025-01-03"
    )
    assert result["ok"] is False
    assert "future_data_requested" in result["detail"]


def test_calculator_requires_observations_from_current_session(fixture_db: Path):
    tools = FinancialTools(fixture_db)
    missing = tools.call(
        "calculate_growth",
        current_observation_id="obs_missing_a",
        previous_observation_id="obs_missing_b",
    )
    assert missing["ok"] is False
    assert "unknown_observation" in missing["detail"]


def test_growth_preserves_parent_provenance(fixture_db: Path):
    tools = FinancialTools(fixture_db)
    current = tools.call(
        "get_financial_fact", symbol="ALFA", metric="revenue", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    previous = tools.call(
        "get_financial_fact", symbol="ALFA", metric="revenue", fiscal_year=2023,
        as_of_time="2025-03-31",
    )
    result = tools.call(
        "calculate_growth",
        current_observation_id=current["provenance"]["observation_id"],
        previous_observation_id=previous["provenance"]["observation_id"],
    )
    assert result["ok"] is True
    assert result["unit"] == "percent"
    assert result["provenance"]["parents"] == [
        current["provenance"]["observation_id"],
        previous["provenance"]["observation_id"],
    ]
    assert result["scalar"] == pytest.approx((1116.0 - 930.0) / 930.0 * 100.0)


def test_ratio_output_unit_is_explicit(fixture_db: Path):
    tools = FinancialTools(fixture_db)
    liabilities = tools.call(
        "get_financial_fact", symbol="ALFA", metric="total_liabilities", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    assets = tools.call(
        "get_financial_fact", symbol="ALFA", metric="total_assets", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    percent = tools.call(
        "calculate_ratio",
        numerator_observation_id=liabilities["provenance"]["observation_id"],
        denominator_observation_id=assets["provenance"]["observation_id"],
        scale=100.0,
        output_unit="percent",
    )
    ratio = tools.call(
        "calculate_ratio",
        numerator_observation_id=liabilities["provenance"]["observation_id"],
        denominator_observation_id=assets["provenance"]["observation_id"],
        scale=1.0,
        output_unit="ratio",
    )
    assert percent["ok"] is True
    assert percent["unit"] == "percent"
    assert ratio["unit"] == "ratio"
    invalid = tools.call(
        "calculate_ratio",
        numerator_observation_id=liabilities["provenance"]["observation_id"],
        denominator_observation_id=assets["provenance"]["observation_id"],
        output_unit="bps",
    )
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_arguments"


def test_ratio_scale_must_match_output_unit(fixture_db: Path):
    tools = FinancialTools(fixture_db)
    liabilities = tools.call(
        "get_financial_fact", symbol="ALFA", metric="total_liabilities", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    assets = tools.call(
        "get_financial_fact", symbol="ALFA", metric="total_assets", fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    kwargs = {
        "numerator_observation_id": liabilities["provenance"]["observation_id"],
        "denominator_observation_id": assets["provenance"]["observation_id"],
    }
    assert tools.call("calculate_ratio", **kwargs, scale=100.0, output_unit="ratio")["ok"] is False
    assert tools.call("calculate_ratio", **kwargs, scale=1.0, output_unit="percent")["ok"] is False
