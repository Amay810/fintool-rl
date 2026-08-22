"""Offline SEC Company Facts importer and explicit download helper.

Downloads are separate from snapshot construction.  Compute jobs consume only
the frozen JSON inputs and the resulting immutable SQLite snapshot.
"""

from __future__ import annotations

import json
import http.client
import random
import re
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .database import SCHEMA_VERSION, connect, initialize_schema
from .metrics import CANONICAL_METRIC_TAGS

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKER_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

METRIC_TAGS = CANONICAL_METRIC_TAGS

_QUARTER_FRAME = re.compile(r"^CY\d{4}Q[1-4]$")
_CALENDAR_FRAME = re.compile(r"^CY(\d{4})$")
_MIN_ANNUAL_DAYS = 300


def _is_annual_fact_row(row: dict[str, Any]) -> bool:
    """Keep annual duration facts and instant balance-sheet facts; drop quarters."""
    frame = row.get("frame") or ""
    if _QUARTER_FRAME.fullmatch(frame):
        return False
    start = row.get("start")
    end = row.get("end")
    if not end:
        return False
    if not start:
        # Instant facts (assets/liabilities) only carry an end date.
        return True
    try:
        span = (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days
    except ValueError:
        return False
    return span >= _MIN_ANNUAL_DAYS


def _fiscal_year_from_period_end(row: dict[str, Any]) -> int:
    """Map period_end to M1 fiscal-year labels.

    Default: ``int(period_end[:4])``.
    52/53-week calendars that close in the first week of January are assigned to
    the prior calendar year (preferring an exact ``CYxxxx`` frame when present).
    Retail January month-ends (e.g. WMT/NVDA on Jan 28-31) keep the period-end
    year, so comparative ``CYxxxx`` frames cannot shift the label.
    """
    period_end = str(row["end"])
    year = int(period_end[:4])
    month = int(period_end[5:7])
    day = int(period_end[8:10])
    if month == 1 and day <= 7:
        frame_match = _CALENDAR_FRAME.fullmatch(row.get("frame") or "")
        if frame_match:
            return int(frame_match.group(1))
        return year - 1
    return year


def _annual_candidates(payload: dict[str, Any], tag: str, cutoff: str) -> list[dict[str, Any]]:
    concept = payload.get("facts", {}).get("us-gaap", {}).get(tag, {})
    units = concept.get("units", {}).get("USD", [])
    return [
        row for row in units
        if row.get("form") in {"10-K", "10-K/A"}
        and row.get("fp") == "FY"
        and isinstance(row.get("fy"), int)
        and isinstance(row.get("val"), (int, float))
        and row.get("filed", "9999-99-99") <= cutoff
        and _is_annual_fact_row(row)
    ]


def _valid_json_file(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def _download_json(
    request: urllib.request.Request,
    destination: Path,
    *,
    max_attempts: int = 4,
    retry_base_seconds: float = 1.0,
) -> Path:
    """Download JSON with validation, bounded retries, and atomic replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            json.loads(data)
            temporary.write_bytes(data)
            temporary.replace(destination)
            return destination
        except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < max_attempts:
                delay = retry_base_seconds * (2 ** attempt) + random.uniform(0.0, 0.25)
                time.sleep(delay)
    raise RuntimeError(f"download failed after {max_attempts} attempts: {destination}") from last_error


def load_company_mapping(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("company mapping must be a JSON list")
    required = {"symbol", "cik", "file"}
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"company mapping row {index} requires {sorted(required)}")
    return payload


def download_companyfacts(
    mapping: Iterable[dict[str, Any]],
    output_dir: Path | str,
    *,
    user_agent: str,
    pause_seconds: float = 0.2,
    skip_existing: bool = True,
) -> None:
    """Download explicitly requested Company Facts files with SEC identification."""
    if not user_agent.strip() or "@" not in user_agent:
        raise ValueError("SEC user_agent must identify an application and contact email")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for row in mapping:
        destination = root / row["file"]
        if skip_existing and _valid_json_file(destination):
            continue
        cik = int(row["cik"])
        request = urllib.request.Request(
            SEC_COMPANYFACTS_URL.format(cik=cik),
            headers={"User-Agent": user_agent, "Host": "data.sec.gov"},
        )
        _download_json(request, destination)
        time.sleep(max(0.0, pause_seconds))


def download_ticker_exchange(output: Path | str, *, user_agent: str) -> Path:
    if not user_agent.strip() or "@" not in user_agent:
        raise ValueError("SEC user_agent must identify an application and contact email")
    request = urllib.request.Request(SEC_TICKER_EXCHANGE_URL, headers={"User-Agent": user_agent})
    destination = Path(output)
    return _download_json(request, destination)


def resolve_universe(
    universe_path: Path | str,
    ticker_exchange_path: Path | str,
) -> list[dict[str, Any]]:
    universe = json.loads(Path(universe_path).read_text(encoding="utf-8"))
    ticker_payload = json.loads(Path(ticker_exchange_path).read_text(encoding="utf-8"))
    fields = ticker_payload.get("fields", [])
    rows = ticker_payload.get("data", [])
    lookup = {
        str(dict(zip(fields, row)).get("ticker", "")).upper(): dict(zip(fields, row))
        for row in rows
    }
    resolved: list[dict[str, Any]] = []
    for company in universe:
        symbol = str(company["symbol"]).upper()
        match = lookup.get(symbol)
        if not match:
            raise ValueError(f"symbol absent from SEC ticker mapping: {symbol}")
        resolved.append({
            **company,
            "symbol": symbol,
            "cik": int(match["cik"]),
            "name": match.get("name", symbol),
            "exchange": match.get("exchange", ""),
            "file": company.get("file", f"{symbol}.json"),
        })
    return resolved


def _latest_row(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    current_key = (current["filed"], current.get("accn", ""), current.get("end", ""))
    candidate_key = (candidate["filed"], candidate.get("accn", ""), candidate.get("end", ""))
    return candidate if candidate_key > current_key else current


def extract_annual_facts(payload: dict[str, Any], cutoff: str) -> list[dict[str, Any]]:
    """Select one latest-filed annual USD fact per canonical metric and period end.

    Fiscal-year labels default to ``int(period_end[:4])``.  Early-January closes
    from 52/53-week calendars are mapped to the prior year.  Comparative CY*
    frames do not relabel ordinary January month-ends (WMT/NVDA).  Preferred
    taxonomy tags win; fallback tags only fill uncovered period ends.
    """
    selected: list[dict[str, Any]] = []
    entity = payload.get("entityName") or payload.get("cik") or "unknown"
    for metric, tags in METRIC_TAGS.items():
        by_period_end: dict[str, tuple[str, dict[str, Any]]] = {}
        for tag in tags:
            tag_periods: dict[str, dict[str, Any]] = {}
            for row in _annual_candidates(payload, tag, cutoff):
                period_end = str(row["end"])
                tag_periods[period_end] = _latest_row(tag_periods.get(period_end), row)
            # Earlier tags in METRIC_TAGS are preferred; fallback tags fill gaps.
            for period_end, row in tag_periods.items():
                by_period_end.setdefault(period_end, (tag, row))

        by_fiscal_year: dict[int, tuple[str, dict[str, Any]]] = {}
        for period_end, (chosen_tag, row) in sorted(by_period_end.items()):
            fiscal_year = _fiscal_year_from_period_end(row)
            existing = by_fiscal_year.get(fiscal_year)
            if existing is not None and existing[1]["end"] != period_end:
                raise ValueError(
                    f"{entity} {metric} FY{fiscal_year} has conflicting period_end "
                    f"values {existing[1]['end']} and {period_end}"
                )
            by_fiscal_year[fiscal_year] = (chosen_tag, row)

        for fiscal_year, (chosen_tag, row) in sorted(by_fiscal_year.items()):
            period_end = str(row["end"])
            selected.append({
                "metric": metric,
                "tag": chosen_tag,
                "fiscal_year": fiscal_year,
                "period_end": period_end,
                "filed_at": row["filed"],
                "value": float(row["val"]) / 1_000_000.0,
                "unit": "USD_million",
                "accession": row.get("accn", "unknown"),
            })
    return selected


def assert_financial_fact_integrity(conn: Any) -> None:
    """Fail closed if period_end or fiscal_year uniqueness is violated."""
    duplicate_periods = conn.execute(
        """
        SELECT symbol, metric, period_end, COUNT(*) AS n
        FROM financial_facts
        GROUP BY symbol, metric, period_end
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    if duplicate_periods:
        sample = dict(duplicate_periods[0])
        raise ValueError(f"duplicate (symbol, metric, period_end): {sample}")

    duplicate_years = conn.execute(
        """
        SELECT symbol, metric, fiscal_year, COUNT(*) AS n
        FROM financial_facts
        GROUP BY symbol, metric, fiscal_year
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    if duplicate_years:
        sample = dict(duplicate_years[0])
        raise ValueError(f"duplicate (symbol, metric, fiscal_year): {sample}")

    mismatched = conn.execute(
        """
        SELECT symbol, metric, fiscal_year, period_end
        FROM financial_facts
        WHERE NOT (
            fiscal_year = CAST(substr(period_end, 1, 4) AS INTEGER)
            OR (
                CAST(substr(period_end, 6, 2) AS INTEGER) = 1
                AND CAST(substr(period_end, 9, 2) AS INTEGER) <= 7
                AND fiscal_year = CAST(substr(period_end, 1, 4) AS INTEGER) - 1
            )
        )
        LIMIT 1
        """
    ).fetchone()
    if mismatched is not None:
        raise ValueError(
            "fiscal_year/period_end labeling invariant broken: "
            f"{dict(mismatched)}"
        )


def build_sec_snapshot(
    mapping_path: Path | str,
    input_dir: Path | str,
    output_db: Path | str,
    *,
    as_of_time: str,
    overwrite: bool = False,
) -> Path:
    output = Path(output_db)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"snapshot already exists: {output}")
        output.unlink()
    mapping = load_company_mapping(mapping_path)
    conn = connect(output)
    try:
        initialize_schema(conn)
        snapshot_id = f"sec-companyfacts-{as_of_time}"
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("snapshot_id", snapshot_id),
                ("as_of_time", as_of_time),
                ("data_class", "sec_companyfacts"),
                ("upstream", "https://data.sec.gov/api/xbrl/companyfacts/"),
            ],
        )
        for company in mapping:
            raw_path = Path(input_dir) / company["file"]
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            symbol = str(company["symbol"]).upper()
            cik = int(company["cik"])
            conn.execute(
                "INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    payload.get("entityName") or company.get("name") or symbol,
                    company.get("sector", "Unknown"),
                    "USD",
                    company.get("listed_at", "1900-01-01"),
                    f"sec:companyfacts:CIK{cik:010d}",
                ),
            )
            for fact in extract_annual_facts(payload, as_of_time):
                conn.execute(
                    "INSERT INTO financial_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        symbol,
                        fact["metric"],
                        fact["fiscal_year"],
                        fact["period_end"],
                        fact["filed_at"],
                        fact["value"],
                        fact["unit"],
                        (
                            f"sec:xbrl:CIK{cik:010d}:{fact['accession']}:"
                            f"us-gaap:{fact['tag']}:{fact['filed_at']}"
                        ),
                    ),
                )
        assert_financial_fact_integrity(conn)
        conn.commit()
    finally:
        conn.close()
    return output
