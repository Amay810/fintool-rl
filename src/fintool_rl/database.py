"""Build and inspect immutable financial data snapshots.

The bundled fixture is deliberately synthetic.  It exists to exercise the
environment without claiming that example numbers are real market data.  Real
SEC/OpenFinData importers can populate the same schema later.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
FIXTURE_SNAPSHOT_ID = "synthetic-us-equities-v1"
FIXTURE_AS_OF = "2025-03-31"


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(path).resolve()
    if read_only:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE companies (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            currency TEXT NOT NULL,
            listed_at TEXT NOT NULL,
            source_ref TEXT NOT NULL
        );
        CREATE TABLE financial_facts (
            symbol TEXT NOT NULL,
            metric TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_end TEXT NOT NULL,
            filed_at TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            PRIMARY KEY (symbol, metric, fiscal_year)
        );
        CREATE TABLE daily_prices (
            symbol TEXT NOT NULL,
            trading_date TEXT NOT NULL,
            close REAL NOT NULL,
            currency TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            PRIMARY KEY (symbol, trading_date)
        );
        CREATE TABLE market_indices (
            index_symbol TEXT NOT NULL,
            trading_date TEXT NOT NULL,
            close REAL NOT NULL,
            source_ref TEXT NOT NULL,
            PRIMARY KEY (index_symbol, trading_date)
        );
        CREATE INDEX facts_lookup
            ON financial_facts(symbol, metric, fiscal_year, filed_at);
        CREATE INDEX prices_lookup
            ON daily_prices(symbol, trading_date);
        """
    )


def build_fixture_snapshot(path: Path | str, *, overwrite: bool = False) -> Path:
    """Create a tiny deterministic snapshot for development and CI."""
    db_path = Path(path)
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(f"snapshot already exists: {db_path}")
        db_path.unlink()
    conn = connect(db_path)
    try:
        initialize_schema(conn)
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("snapshot_id", FIXTURE_SNAPSHOT_ID),
                ("as_of_time", FIXTURE_AS_OF),
                ("data_class", "synthetic_fixture"),
            ],
        )
        companies = [
            ("ALFA", "Alpha Systems", "Technology", "USD", "2014-05-01", "fixture:company:ALFA"),
            ("BETA", "Beta Retail", "Consumer", "USD", "2011-09-12", "fixture:company:BETA"),
            ("GAMA", "Gamma Energy", "Energy", "USD", "2008-02-20", "fixture:company:GAMA"),
            ("DELT", "Delta Health", "Healthcare", "USD", "2017-11-03", "fixture:company:DELT"),
            ("EPSI", "Epsilon Finance", "Financials", "USD", "2005-07-18", "fixture:company:EPSI"),
        ]
        conn.executemany("INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?)", companies)

        base: dict[str, dict[int, tuple[float, float, float, float, float]]] = {
            "ALFA": {
                2022: (820.0, 310.0, 112.0, 1460.0, 520.0),
                2023: (930.0, 365.0, 139.0, 1580.0, 548.0),
                2024: (1116.0, 452.0, 181.0, 1740.0, 575.0),
            },
            "BETA": {
                2022: (1250.0, 338.0, 91.0, 1020.0, 640.0),
                2023: (1325.0, 351.0, 96.0, 1090.0, 682.0),
                2024: (1298.5, 324.6, 72.0, 1125.0, 715.0),
            },
            "GAMA": {
                2022: (680.0, 244.0, 77.0, 2120.0, 890.0),
                2023: (745.0, 261.0, 82.0, 2250.0, 925.0),
                2024: (812.0, 300.0, 101.0, 2380.0, 970.0),
            },
            "DELT": {
                2022: (510.0, 285.0, 62.0, 870.0, 260.0),
                2023: (602.0, 349.0, 80.0, 940.0, 275.0),
                2024: (710.0, 418.0, 99.0, 1030.0, 300.0),
            },
            "EPSI": {
                2022: (440.0, 190.0, 88.0, 3250.0, 2710.0),
                2023: (482.0, 211.0, 96.0, 3480.0, 2860.0),
                2024: (535.0, 229.0, 104.0, 3720.0, 3015.0),
            },
        }
        metric_names = ("revenue", "gross_profit", "net_income", "total_assets", "total_liabilities")
        facts: list[tuple[Any, ...]] = []
        for symbol, years in base.items():
            for year, values in years.items():
                filed_at = f"{year + 1}-02-{20 + (len(symbol) % 5):02d}"
                for metric, value in zip(metric_names, values):
                    facts.append(
                        (
                            symbol,
                            metric,
                            year,
                            f"{year}-12-31",
                            filed_at,
                            value,
                            "USD_million",
                            f"fixture:xbrl:{symbol}:{year}:{metric}",
                        )
                    )
        conn.executemany("INSERT INTO financial_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", facts)

        dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        starts = {"ALFA": 100.0, "BETA": 62.0, "GAMA": 48.0, "DELT": 75.0, "EPSI": 41.0}
        prices: list[tuple[Any, ...]] = []
        for symbol, start in starts.items():
            for offset, trading_date in enumerate(dates):
                close = round(start * (1 + (offset - 1) * 0.012 + (len(symbol) - 4) * 0.003), 2)
                prices.append((symbol, trading_date, close, "USD", f"fixture:price:{symbol}:{trading_date}"))
        conn.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?)", prices)
        index_rows = [
            ("MKT", trading_date, round(5000.0 * (1 + offset * 0.004), 2), f"fixture:index:MKT:{trading_date}")
            for offset, trading_date in enumerate(dates)
        ]
        conn.executemany("INSERT INTO market_indices VALUES (?, ?, ?, ?)", index_rows)
        conn.commit()
    finally:
        conn.close()
    return db_path


def metadata(path: Path | str) -> dict[str, str]:
    conn = connect(path, read_only=True)
    try:
        return {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM metadata")}
    finally:
        conn.close()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_manifest(path: Path | str) -> dict[str, Any]:
    db_path = Path(path)
    conn = connect(db_path, read_only=True)
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("companies", "financial_facts", "daily_prices", "market_indices")
        }
    finally:
        conn.close()
    return {
        "path": db_path.name,
        "sha256": file_sha256(db_path),
        "metadata": metadata(db_path),
        "row_counts": counts,
    }


def write_manifest(path: Path | str, output: Path | str) -> None:
    Path(output).write_text(json.dumps(snapshot_manifest(path), indent=2, sort_keys=True), encoding="utf-8")
