"""Programmatic task generation with executable numeric oracles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import TaskSpec
from .database import connect
from .oracle import execute_oracle
from .tools import FinancialTools

SYMBOL_SPLITS = {
    "ALFA": "train",
    "BETA": "train",
    "GAMA": "dev",
    "DELT": "test",
    "EPSI": "challenge",
}


def _task_id(parts: list[Any]) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return "task_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_task(
    db_path: Path | str,
    *,
    symbol: str,
    question: str,
    as_of_time: str,
    difficulty: str,
    template_family: str,
    oracle_steps: list[dict[str, Any]],
    required_tool_families: list[str],
    fact_keys: list[str],
    answer_tolerance: float = 1e-6,
    split: str | None = None,
) -> TaskSpec:
    tools = FinancialTools(db_path)
    _, results = execute_oracle(tools, oracle_steps)
    final = results[oracle_steps[-1]["id"]]
    if "scalar" not in final:
        raise ValueError(f"final oracle result is not scalar: {template_family}")
    assigned_split = split or SYMBOL_SPLITS[symbol]
    return TaskSpec(
        task_id=_task_id([symbol, question, as_of_time, template_family]),
        question=question,
        split=assigned_split,  # type: ignore[arg-type]
        as_of_time=as_of_time,
        difficulty=difficulty,  # type: ignore[arg-type]
        template_family=template_family,
        answer={
            "value": final["scalar"],
            "unit": final["unit"],
            "tolerance": answer_tolerance,
        },
        oracle_steps=oracle_steps,
        required_tool_families=required_tool_families,
        metadata={
            "symbol": symbol,
            "fact_keys": fact_keys,
            "generator": "synthetic_fixture_v1",
        },
    )


def generate_fixture_tasks(db_path: Path | str) -> list[TaskSpec]:
    """Generate deterministic tasks over the bundled synthetic snapshot.

    Company-disjoint splits prevent the same underlying fact from crossing
    train/dev/test in the smoke dataset.  Real importers will add IID,
    compositional, and held-out-tool protocols separately.
    """
    conn = connect(db_path, read_only=True)
    try:
        symbols = [row[0] for row in conn.execute("SELECT symbol FROM companies ORDER BY symbol")]
    finally:
        conn.close()
    tasks: list[TaskSpec] = []
    as_of = "2025-03-31"
    for symbol in symbols:
        for metric in ("revenue", "net_income"):
            for year in (2022, 2023, 2024):
                steps = [{
                    "id": "fact",
                    "tool": "get_financial_fact",
                    "arguments": {"symbol": symbol, "metric": metric, "fiscal_year": year, "as_of_time": as_of},
                }]
                tasks.append(_build_task(
                    db_path,
                    symbol=symbol,
                    question=f"As of {as_of}, what was {symbol}'s {metric} in fiscal year {year}?",
                    as_of_time=as_of,
                    difficulty="single_tool",
                    template_family="financial_fact_lookup",
                    oracle_steps=steps,
                    required_tool_families=["financial_statement"],
                    fact_keys=[f"{symbol}:{metric}:{year}"],
                ))

        for metric in ("revenue", "net_income"):
            for previous_year, current_year in ((2022, 2023), (2023, 2024)):
                steps = [
                    {
                        "id": "current",
                        "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": metric, "fiscal_year": current_year, "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "previous",
                        "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": metric, "fiscal_year": previous_year, "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "answer",
                        "tool": "calculate_growth",
                        "arguments": {
                            "current_observation_id": "$current.observation_id",
                            "previous_observation_id": "$previous.observation_id",
                        },
                    },
                ]
                tasks.append(_build_task(
                    db_path,
                    symbol=symbol,
                    question=(
                        f"Using information available by {as_of}, calculate {symbol}'s {metric} growth "
                        f"from FY{previous_year} to FY{current_year}."
                    ),
                    as_of_time=as_of,
                    difficulty="multi_tool",
                    template_family="year_over_year_growth",
                    oracle_steps=steps,
                    required_tool_families=["financial_statement", "calculator"],
                    fact_keys=[f"{symbol}:{metric}:{previous_year}", f"{symbol}:{metric}:{current_year}"],
                    answer_tolerance=1e-4,
                ))

        for year in (2022, 2023, 2024):
            margin_steps = [
                {
                    "id": "profit",
                    "tool": "get_financial_fact",
                    "arguments": {
                        "symbol": symbol, "metric": "gross_profit", "fiscal_year": year, "as_of_time": as_of,
                    },
                },
                {
                    "id": "revenue",
                    "tool": "get_financial_fact",
                    "arguments": {
                        "symbol": symbol, "metric": "revenue", "fiscal_year": year, "as_of_time": as_of,
                    },
                },
                {
                    "id": "answer",
                    "tool": "calculate_margin",
                    "arguments": {
                        "profit_observation_id": "$profit.observation_id",
                        "revenue_observation_id": "$revenue.observation_id",
                    },
                },
            ]
            tasks.append(_build_task(
                db_path,
                symbol=symbol,
                question=f"Calculate {symbol}'s gross margin for FY{year} using data available by {as_of}.",
                as_of_time=as_of,
                difficulty="multi_tool",
                template_family="gross_margin",
                oracle_steps=margin_steps,
                required_tool_families=["financial_statement", "calculator"],
                fact_keys=[f"{symbol}:gross_profit:{year}", f"{symbol}:revenue:{year}"],
                answer_tolerance=1e-4,
            ))

            ratio_steps = [
                {
                    "id": "liabilities",
                    "tool": "get_financial_fact",
                    "arguments": {
                        "symbol": symbol, "metric": "total_liabilities", "fiscal_year": year, "as_of_time": as_of,
                    },
                },
                {
                    "id": "assets",
                    "tool": "get_financial_fact",
                    "arguments": {
                        "symbol": symbol, "metric": "total_assets", "fiscal_year": year, "as_of_time": as_of,
                    },
                },
                {
                    "id": "answer",
                    "tool": "calculate_ratio",
                    "arguments": {
                        "numerator_observation_id": "$liabilities.observation_id",
                        "denominator_observation_id": "$assets.observation_id",
                        "scale": 100.0,
                        "output_unit": "percent",
                    },
                },
            ]
            task = _build_task(
                db_path,
                symbol=symbol,
                question=f"What was {symbol}'s liabilities-to-assets percentage for FY{year} as of {as_of}?",
                as_of_time=as_of,
                difficulty="compositional",
                template_family="liabilities_to_assets",
                oracle_steps=ratio_steps,
                required_tool_families=["financial_statement", "calculator"],
                fact_keys=[f"{symbol}:total_liabilities:{year}", f"{symbol}:total_assets:{year}"],
                answer_tolerance=1e-4,
            )
            tasks.append(task)

        price_steps = [
            {
                "id": "end",
                "tool": "get_daily_price",
                "arguments": {"symbol": symbol, "trading_date": "2025-01-08", "as_of_time": as_of},
            },
            {
                "id": "start",
                "tool": "get_daily_price",
                "arguments": {"symbol": symbol, "trading_date": "2025-01-02", "as_of_time": as_of},
            },
            {
                "id": "answer",
                "tool": "calculate_growth",
                "arguments": {
                    "current_observation_id": "$end.observation_id",
                    "previous_observation_id": "$start.observation_id",
                },
            },
        ]
        tasks.append(_build_task(
            db_path,
            symbol=symbol,
            question=f"Calculate {symbol}'s close-price return from 2025-01-02 to 2025-01-08.",
            as_of_time=as_of,
            difficulty="multi_tool",
            template_family="price_return",
            oracle_steps=price_steps,
            required_tool_families=["market_data", "calculator"],
            fact_keys=[f"{symbol}:price:2025-01-02", f"{symbol}:price:2025-01-08"],
            answer_tolerance=1e-4,
        ))
    return sorted(tasks, key=lambda task: task.task_id)


def generate_snapshot_tasks(
    db_path: Path | str,
    company_splits: dict[str, str],
    *,
    recent_years: int = 3,
) -> list[TaskSpec]:
    """Generate real-snapshot tasks using only facts present at the cutoff."""
    conn = connect(db_path, read_only=True)
    try:
        as_of = conn.execute("SELECT value FROM metadata WHERE key='as_of_time'").fetchone()[0]
        available: dict[str, dict[str, list[int]]] = {}
        zero_values: set[tuple[str, str, int]] = set()
        for row in conn.execute(
            "SELECT symbol, metric, fiscal_year, value FROM financial_facts "
            "ORDER BY symbol, metric, fiscal_year"
        ):
            available.setdefault(row["symbol"], {}).setdefault(row["metric"], []).append(row["fiscal_year"])
            if row["value"] == 0:
                zero_values.add((row["symbol"], row["metric"], row["fiscal_year"]))
    finally:
        conn.close()

    tasks: list[TaskSpec] = []
    for symbol, split in sorted(company_splits.items()):
        metrics = available.get(symbol.upper(), {})
        if not metrics:
            raise ValueError(f"company has no imported facts: {symbol}")

        for metric in ("revenue", "net_income"):
            years = sorted(set(metrics.get(metric, [])))[-recent_years:]
            for year in years:
                tasks.append(_build_task(
                    db_path,
                    symbol=symbol,
                    split=split,
                    question=f"As of {as_of}, what was {symbol}'s {metric} for fiscal year {year}?",
                    as_of_time=as_of,
                    difficulty="single_tool",
                    template_family="financial_fact_lookup",
                    oracle_steps=[{
                        "id": "fact",
                        "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": metric, "fiscal_year": year, "as_of_time": as_of,
                        },
                    }],
                    required_tool_families=["financial_statement"],
                    fact_keys=[f"{symbol}:{metric}:{year}"],
                ))
            for previous_year, current_year in zip(years, years[1:]):
                if current_year != previous_year + 1:
                    continue
                if (symbol, metric, previous_year) in zero_values:
                    continue
                tasks.append(_build_task(
                    db_path,
                    symbol=symbol,
                    split=split,
                    question=(
                        f"Using filings available by {as_of}, calculate {symbol}'s {metric} growth "
                        f"from FY{previous_year} to FY{current_year}."
                    ),
                    as_of_time=as_of,
                    difficulty="multi_tool",
                    template_family="year_over_year_growth",
                    oracle_steps=[
                        {
                            "id": "current", "tool": "get_financial_fact",
                            "arguments": {
                                "symbol": symbol, "metric": metric, "fiscal_year": current_year,
                                "as_of_time": as_of,
                            },
                        },
                        {
                            "id": "previous", "tool": "get_financial_fact",
                            "arguments": {
                                "symbol": symbol, "metric": metric, "fiscal_year": previous_year,
                                "as_of_time": as_of,
                            },
                        },
                        {
                            "id": "answer", "tool": "calculate_growth",
                            "arguments": {
                                "current_observation_id": "$current.observation_id",
                                "previous_observation_id": "$previous.observation_id",
                            },
                        },
                    ],
                    required_tool_families=["financial_statement", "calculator"],
                    fact_keys=[f"{symbol}:{metric}:{previous_year}", f"{symbol}:{metric}:{current_year}"],
                    answer_tolerance=1e-4,
                ))

        margin_years = sorted(
            set(metrics.get("gross_profit", [])) & set(metrics.get("revenue", []))
        )[-recent_years:]
        for year in margin_years:
            if (symbol, "revenue", year) in zero_values:
                continue
            tasks.append(_build_task(
                db_path,
                symbol=symbol,
                split=split,
                question=f"Calculate {symbol}'s gross margin for FY{year} using filings available by {as_of}.",
                as_of_time=as_of,
                difficulty="multi_tool",
                template_family="gross_margin",
                oracle_steps=[
                    {
                        "id": "profit", "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": "gross_profit", "fiscal_year": year,
                            "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "revenue", "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": "revenue", "fiscal_year": year,
                            "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "answer", "tool": "calculate_margin",
                        "arguments": {
                            "profit_observation_id": "$profit.observation_id",
                            "revenue_observation_id": "$revenue.observation_id",
                        },
                    },
                ],
                required_tool_families=["financial_statement", "calculator"],
                fact_keys=[f"{symbol}:gross_profit:{year}", f"{symbol}:revenue:{year}"],
                answer_tolerance=1e-4,
            ))

        ratio_years = sorted(
            set(metrics.get("total_liabilities", [])) & set(metrics.get("total_assets", []))
        )[-recent_years:]
        for year in ratio_years:
            if (symbol, "total_assets", year) in zero_values:
                continue
            tasks.append(_build_task(
                db_path,
                symbol=symbol,
                split=split,
                question=f"What was {symbol}'s liabilities-to-assets percentage for FY{year} as of {as_of}?",
                as_of_time=as_of,
                difficulty="compositional",
                template_family="liabilities_to_assets",
                oracle_steps=[
                    {
                        "id": "liabilities", "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": "total_liabilities", "fiscal_year": year,
                            "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "assets", "tool": "get_financial_fact",
                        "arguments": {
                            "symbol": symbol, "metric": "total_assets", "fiscal_year": year,
                            "as_of_time": as_of,
                        },
                    },
                    {
                        "id": "answer", "tool": "calculate_ratio",
                        "arguments": {
                            "numerator_observation_id": "$liabilities.observation_id",
                            "denominator_observation_id": "$assets.observation_id",
                            "scale": 100.0,
                            "output_unit": "percent",
                        },
                    },
                ],
                required_tool_families=["financial_statement", "calculator"],
                fact_keys=[f"{symbol}:total_liabilities:{year}", f"{symbol}:total_assets:{year}"],
                answer_tolerance=1e-4,
            ))
    for task in tasks:
        task.metadata["generator"] = "sec_snapshot_v1"
    tasks.sort(key=lambda task: task.task_id)
    assert_no_fact_leakage(tasks)
    return tasks


def select_split_targets(
    tasks: Iterable[TaskSpec],
    targets: dict[str, int],
) -> list[TaskSpec]:
    """Deterministically downsample each split while cycling template families."""
    all_tasks = list(tasks)
    selected: list[TaskSpec] = []
    for split, requested in targets.items():
        if requested < 0:
            raise ValueError(f"negative target for {split}: {requested}")
        candidates = [task for task in all_tasks if task.split == split]
        if len(candidates) < requested:
            raise ValueError(
                f"split {split} has {len(candidates)} candidates, fewer than target {requested}"
            )
        by_family: dict[str, list[TaskSpec]] = {}
        for task in candidates:
            by_family.setdefault(task.template_family, []).append(task)
        for family_tasks in by_family.values():
            family_tasks.sort(key=lambda task: task.task_id, reverse=True)
        remaining = requested
        families = sorted(by_family)
        while remaining:
            made_progress = False
            for family in families:
                if remaining == 0:
                    break
                if by_family[family]:
                    selected.append(by_family[family].pop())
                    remaining -= 1
                    made_progress = True
            if not made_progress:
                raise AssertionError("candidate accounting error")
    selected.sort(key=lambda task: task.task_id)
    assert_no_fact_leakage(selected)
    return selected


def assert_no_fact_leakage(tasks: Iterable[TaskSpec]) -> None:
    owners: dict[str, str] = {}
    for task in tasks:
        for fact_key in task.metadata.get("fact_keys", []):
            previous = owners.setdefault(fact_key, task.split)
            if previous != task.split:
                raise ValueError(
                    f"fact leakage: {fact_key} occurs in both {previous} and {task.split}"
                )


def write_tasks(tasks: Iterable[TaskSpec], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) for task in tasks]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_tasks(path: Path | str) -> list[TaskSpec]:
    return [
        TaskSpec.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
