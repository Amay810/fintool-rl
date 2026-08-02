from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from fintool_rl.database import connect, metadata
from fintool_rl.sec import (
    _download_json,
    assert_financial_fact_integrity,
    build_sec_snapshot,
    extract_annual_facts,
    resolve_universe,
)
from fintool_rl.tasks import generate_snapshot_tasks


def _companyfacts_payload():
    return {
        "cik": 1234,
        "entityName": "Example Corp",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-10",
                                "start": "2023-01-01", "end": "2023-12-31", "val": 1_000_000_000, "accn": "old",
                            },
                            {
                                "fy": 2023, "fp": "FY", "form": "10-K/A", "filed": "2024-03-01",
                                "start": "2023-01-01", "end": "2023-12-31", "val": 1_050_000_000, "accn": "amended",
                            },
                            {
                                "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-10",
                                "start": "2024-01-01", "end": "2024-12-31", "val": 1_200_000_000, "accn": "future",
                            },
                            {
                                "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-02-10",
                                "start": "2022-01-01", "end": "2022-12-31", "frame": "CY2022", "val": 900_000_000,
                                "accn": "comparative",
                            },
                            {
                                "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-05-01",
                                "start": "2023-10-01", "end": "2023-12-31", "frame": "CY2023Q4",
                                "val": 300_000_000, "accn": "quarter-as-fy",
                            },
                            {
                                "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-05-01",
                                "start": "2024-01-01", "end": "2024-03-31", "val": 300_000_000, "accn": "quarter",
                            },
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [{
                            "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-10",
                            "end": "2023-12-31", "val": 2_500_000_000, "accn": "assets",
                        }]
                    }
                },
            }
        },
    }


def test_extract_annual_facts_respects_cutoff_and_latest_amendment():
    facts = extract_annual_facts(_companyfacts_payload(), "2024-12-31")
    revenue = next(
        fact for fact in facts if fact["metric"] == "revenue" and fact["fiscal_year"] == 2023
    )
    assert revenue["fiscal_year"] == 2023
    assert revenue["period_end"] == "2023-12-31"
    assert revenue["value"] == pytest.approx(1050.0)
    assert revenue["accession"] == "amended"
    assert all(fact["fiscal_year"] != 2024 for fact in facts)
    assert any(fact["metric"] == "revenue" and fact["fiscal_year"] == 2022 for fact in facts)
    assert not any(fact["accession"] == "quarter-as-fy" for fact in facts)


def test_extract_annual_facts_dedupes_period_end_instead_of_cy_frame():
    payload = {
        "cik": 99,
        "entityName": "Retail Co",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-03-15",
                                "start": "2022-02-01", "end": "2023-01-31", "val": 605_881_000_000,
                                "accn": "primary",
                            },
                            {
                                "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-03-14",
                                "start": "2022-02-01", "end": "2023-01-31", "frame": "CY2022",
                                "val": 605_881_000_000, "accn": "comparative",
                            },
                            {
                                "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-03-15",
                                "start": "2021-02-01", "end": "2022-01-31", "frame": "CY2021",
                                "val": 567_762_000_000, "accn": "prior",
                            },
                            {
                                "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-03-14",
                                "start": "2022-11-01", "end": "2023-01-31", "frame": "CY2022Q4",
                                "val": 1_000_000, "accn": "quarterly-noise",
                            },
                        ]
                    }
                }
            }
        },
    }
    facts = extract_annual_facts(payload, "2025-03-31")
    revenue = sorted(
        (fact for fact in facts if fact["metric"] == "revenue"),
        key=lambda item: item["fiscal_year"],
    )
    assert [(fact["fiscal_year"], fact["period_end"], fact["value"]) for fact in revenue] == [
        (2022, "2022-01-31", 567762.0),
        (2023, "2023-01-31", 605881.0),
    ]


def test_early_january_close_maps_to_prior_fiscal_year():
    payload = {
        "cik": 3,
        "entityName": "Week Calendar Co",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2018, "fp": "FY", "form": "10-K", "filed": "2019-02-20",
                                "start": "2016-01-04", "end": "2017-01-01", "frame": "CY2016",
                                "val": 71_890_000_000, "accn": "prior",
                            },
                            {
                                "fy": 2018, "fp": "FY", "form": "10-K", "filed": "2019-02-20",
                                "start": "2017-01-02", "end": "2017-12-31",
                                "val": 76_450_000_000, "accn": "current",
                            },
                        ]
                    }
                }
            }
        },
    }
    facts = extract_annual_facts(payload, "2019-12-31")
    revenue = {
        fact["fiscal_year"]: (fact["period_end"], fact["value"])
        for fact in facts if fact["metric"] == "revenue"
    }
    assert revenue == {
        2016: ("2017-01-01", 71890.0),
        2017: ("2017-12-31", 76450.0),
    }


def test_extract_annual_facts_errors_on_conflicting_period_ends_for_same_fy():
    payload = {
        "cik": 1,
        "entityName": "Conflict Co",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01",
                                "end": "2023-06-30", "val": 1_000_000, "accn": "a",
                            },
                            {
                                "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-02",
                                "end": "2023-12-31", "val": 2_000_000, "accn": "b",
                            },
                        ]
                    }
                }
            }
        },
    }
    with pytest.raises(ValueError, match="conflicting period_end"):
        extract_annual_facts(payload, "2024-12-31")


def test_preferred_tag_is_not_overridden_by_later_fallback_tag():
    payload = {
        "cik": 2,
        "entityName": "Tag Priority Co",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [{
                            "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-01-01",
                            "start": "2023-01-01", "end": "2023-12-31", "val": 100_000_000,
                            "accn": "preferred",
                        }]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [{
                            "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-06-01",
                            "start": "2023-01-01", "end": "2023-12-31", "val": 999_000_000,
                            "accn": "fallback-later",
                        }]
                    }
                },
            }
        },
    }
    facts = extract_annual_facts(payload, "2024-12-31")
    revenue = next(fact for fact in facts if fact["metric"] == "revenue")
    assert revenue["value"] == pytest.approx(100.0)
    assert revenue["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert revenue["accession"] == "preferred"


def test_build_sec_snapshot_from_local_json(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "example.json").write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([{
        "symbol": "EXM", "cik": 1234, "file": "example.json", "sector": "Technology",
    }]), encoding="utf-8")
    database = build_sec_snapshot(mapping, raw, tmp_path / "sec.sqlite", as_of_time="2024-12-31")
    assert metadata(database)["data_class"] == "sec_companyfacts"
    conn = connect(database, read_only=True)
    try:
        rows = conn.execute(
            "SELECT metric, fiscal_year, value, source_ref FROM financial_facts ORDER BY metric"
        ).fetchall()
    finally:
        conn.close()
    assert {(row["metric"], row["fiscal_year"]) for row in rows} == {
        ("revenue", 2022), ("revenue", 2023), ("total_assets", 2023)
    }
    assert all("CIK0000001234" in row["source_ref"] for row in rows)
    integrity_conn = connect(database, read_only=True)
    try:
        assert_financial_fact_integrity(integrity_conn)
    finally:
        integrity_conn.close()
    tasks = generate_snapshot_tasks(database, {"EXM": "test"}, recent_years=3)
    assert tasks
    assert all(task.split == "test" for task in tasks)
    assert all(task.metadata["fact_keys"] for task in tasks)
    assert all(task.metadata["generator"] == "sec_snapshot_v1" for task in tasks)
    assert all(
        step["arguments"].get("output_unit") == "percent"
        for task in tasks if task.template_family == "liabilities_to_assets"
        for step in task.oracle_steps if step["tool"] == "calculate_ratio"
    )


def test_resolve_universe_uses_official_ticker_fields(tmp_path: Path):
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps([
        {"symbol": "EXM", "sector": "Technology", "split": "train"}
    ]), encoding="utf-8")
    tickers = tmp_path / "tickers.json"
    tickers.write_text(json.dumps({
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[1234, "Example Corp", "EXM", "Nasdaq"]],
    }), encoding="utf-8")
    mapping = resolve_universe(universe, tickers)
    assert mapping == [{
        "symbol": "EXM", "sector": "Technology", "split": "train",
        "cik": 1234, "name": "Example Corp", "exchange": "Nasdaq", "file": "EXM.json",
    }]


def test_download_json_retries_and_replaces_atomically(tmp_path: Path, monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    responses = iter([Response(OSError("connection reset")), Response(b'{"ok": true}')])
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    destination = tmp_path / "payload.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    _download_json(urllib.request.Request("https://example.test/data"), destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": True}
    assert not destination.with_suffix(".json.part").exists()
