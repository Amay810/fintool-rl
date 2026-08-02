"""Deterministic, read-only financial tools over an immutable snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import ToolCall
from .database import connect, metadata
from .schema import SCHEMA_BY_NAME, ToolArgumentError, validate_arguments


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class FinancialTools:
    """One isolated tool session with an in-memory observation ledger."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.snapshot = metadata(self.db_path)
        self.calls: list[ToolCall] = []
        self.observations: dict[str, dict[str, Any]] = {}
        self._registry: dict[str, Callable[..., dict[str, Any]]] = {
            name: getattr(self, name) for name in SCHEMA_BY_NAME
        }

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        started = time.perf_counter()
        call_id = hashlib.sha256(_canonical([name, arguments, len(self.calls)]).encode()).hexdigest()[:16]
        argument_valid = True
        error: str | None = None
        try:
            validate_arguments(name, arguments)
            result = self._registry[name](**arguments)
        except ToolArgumentError as exc:
            argument_valid = False
            error = str(exc)
            result = {"ok": False, "error": "invalid_arguments", "detail": str(exc)}
        except Exception as exc:  # environment errors must become auditable observations
            detail = str(exc)
            if isinstance(exc, ValueError) and detail:
                error = detail.partition(":")[0]
            else:
                error = type(exc).__name__
            result = {"ok": False, "error": error, "detail": detail}
        self.calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                call_id=call_id,
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000,
                argument_valid=argument_valid,
                error=error,
            )
        )
        return result

    def _cutoff_guard(self, requested_date: str, as_of_time: str) -> None:
        snapshot_cutoff = self.snapshot["as_of_time"]
        if as_of_time > snapshot_cutoff:
            raise ValueError(f"as_of_time_after_snapshot:{as_of_time}>{snapshot_cutoff}")
        if requested_date > as_of_time:
            raise ValueError(f"future_data_requested:{requested_date}>{as_of_time}")

    def _observe(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        data: Any,
        *,
        as_of_time: str,
        source_refs: list[str],
        scalar: float | None = None,
        unit: str | None = None,
        parents: list[str] | None = None,
    ) -> dict[str, Any]:
        core = {
            "tool": tool_name,
            "arguments": arguments,
            "data": data,
            "snapshot_id": self.snapshot["snapshot_id"],
            "as_of_time": as_of_time,
            "source_refs": sorted(source_refs),
            "parents": parents or [],
        }
        observation_id = "obs_" + hashlib.sha256(_canonical(core).encode()).hexdigest()[:20]
        result: dict[str, Any] = {
            "ok": True,
            "data": data,
            "provenance": {
                "observation_id": observation_id,
                "snapshot_id": self.snapshot["snapshot_id"],
                "as_of_time": as_of_time,
                "source_refs": sorted(source_refs),
                "parents": parents or [],
                "query_hash": hashlib.sha256(_canonical([tool_name, arguments]).encode()).hexdigest(),
            },
        }
        if scalar is not None:
            result["scalar"] = float(scalar)
            result["unit"] = unit
        self.observations[observation_id] = result
        return result

    def _scalar(self, observation_id: str) -> tuple[float, str | None, dict[str, Any]]:
        observation = self.observations.get(observation_id)
        if observation is None:
            raise ValueError(f"unknown_observation:{observation_id}")
        if "scalar" not in observation:
            raise ValueError(f"non_scalar_observation:{observation_id}")
        return float(observation["scalar"]), observation.get("unit"), observation

    def get_company_profile(self, symbol: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard("1900-01-01", as_of_time)
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute("SELECT * FROM companies WHERE symbol=?", (symbol.upper(),)).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"ok": False, "error": "company_not_found"}
        data = dict(row)
        source = data.pop("source_ref")
        return self._observe(
            "get_company_profile", {"symbol": symbol, "as_of_time": as_of_time}, data,
            as_of_time=as_of_time, source_refs=[source],
        )

    def list_available_periods(self, symbol: str, metric: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard("1900-01-01", as_of_time)
        conn = connect(self.db_path, read_only=True)
        try:
            rows = conn.execute(
                "SELECT fiscal_year, source_ref FROM financial_facts "
                "WHERE symbol=? AND metric=? AND filed_at<=? ORDER BY fiscal_year",
                (symbol.upper(), metric, as_of_time),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return {"ok": False, "error": "periods_not_found"}
        return self._observe(
            "list_available_periods",
            {"symbol": symbol, "metric": metric, "as_of_time": as_of_time},
            {"symbol": symbol.upper(), "metric": metric, "fiscal_years": [row["fiscal_year"] for row in rows]},
            as_of_time=as_of_time,
            source_refs=[row["source_ref"] for row in rows],
        )

    def get_financial_fact(self, symbol: str, metric: str, fiscal_year: int, as_of_time: str) -> dict[str, Any]:
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute(
                "SELECT * FROM financial_facts WHERE symbol=? AND metric=? AND fiscal_year=? AND filed_at<=?",
                (symbol.upper(), metric, fiscal_year, as_of_time),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"ok": False, "error": "fact_not_available_at_cutoff"}
        data = dict(row)
        self._cutoff_guard(data["filed_at"], as_of_time)
        source = data.pop("source_ref")
        return self._observe(
            "get_financial_fact",
            {"symbol": symbol, "metric": metric, "fiscal_year": fiscal_year, "as_of_time": as_of_time},
            data,
            as_of_time=as_of_time,
            source_refs=[source],
            scalar=data["value"],
            unit=data["unit"],
        )

    def get_daily_price(self, symbol: str, trading_date: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard(trading_date, as_of_time)
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute(
                "SELECT * FROM daily_prices WHERE symbol=? AND trading_date=?",
                (symbol.upper(), trading_date),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"ok": False, "error": "price_not_found"}
        data = dict(row)
        source = data.pop("source_ref")
        return self._observe(
            "get_daily_price",
            {"symbol": symbol, "trading_date": trading_date, "as_of_time": as_of_time},
            data,
            as_of_time=as_of_time,
            source_refs=[source],
            scalar=data["close"],
            unit=data["currency"],
        )

    def get_price_series(self, symbol: str, start_date: str, end_date: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard(end_date, as_of_time)
        if start_date > end_date:
            raise ValueError("start_date_after_end_date")
        conn = connect(self.db_path, read_only=True)
        try:
            rows = conn.execute(
                "SELECT * FROM daily_prices WHERE symbol=? AND trading_date BETWEEN ? AND ? ORDER BY trading_date",
                (symbol.upper(), start_date, end_date),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return {"ok": False, "error": "price_series_not_found"}
        data_rows = [dict(row) for row in rows]
        refs = [row.pop("source_ref") for row in data_rows]
        return self._observe(
            "get_price_series",
            {"symbol": symbol, "start_date": start_date, "end_date": end_date, "as_of_time": as_of_time},
            {"symbol": symbol.upper(), "prices": data_rows},
            as_of_time=as_of_time,
            source_refs=refs,
        )

    def get_market_index_level(self, index_symbol: str, trading_date: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard(trading_date, as_of_time)
        conn = connect(self.db_path, read_only=True)
        try:
            row = conn.execute(
                "SELECT * FROM market_indices WHERE index_symbol=? AND trading_date=?",
                (index_symbol.upper(), trading_date),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"ok": False, "error": "index_level_not_found"}
        data = dict(row)
        source = data.pop("source_ref")
        return self._observe(
            "get_market_index_level",
            {"index_symbol": index_symbol, "trading_date": trading_date, "as_of_time": as_of_time},
            data,
            as_of_time=as_of_time,
            source_refs=[source],
            scalar=data["close"],
            unit="index_points",
        )

    def get_trading_days(self, start_date: str, end_date: str, as_of_time: str) -> dict[str, Any]:
        self._cutoff_guard(end_date, as_of_time)
        conn = connect(self.db_path, read_only=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT trading_date FROM daily_prices WHERE trading_date BETWEEN ? AND ? "
                "ORDER BY trading_date",
                (start_date, end_date),
            ).fetchall()
        finally:
            conn.close()
        dates = [row["trading_date"] for row in rows]
        if not dates:
            return {"ok": False, "error": "trading_days_not_found"}
        return self._observe(
            "get_trading_days",
            {"start_date": start_date, "end_date": end_date, "as_of_time": as_of_time},
            {"trading_days": dates, "count": len(dates)},
            as_of_time=as_of_time,
            source_refs=[f"snapshot:{self.snapshot['snapshot_id']}:calendar"],
        )

    def calculate_growth(self, current_observation_id: str, previous_observation_id: str) -> dict[str, Any]:
        current, current_unit, current_obs = self._scalar(current_observation_id)
        previous, previous_unit, previous_obs = self._scalar(previous_observation_id)
        if current_unit != previous_unit:
            raise ValueError("incompatible_units")
        if math.isclose(previous, 0.0):
            raise ValueError("division_by_zero")
        value = (current - previous) / abs(previous) * 100.0
        as_of = max(current_obs["provenance"]["as_of_time"], previous_obs["provenance"]["as_of_time"])
        refs = current_obs["provenance"]["source_refs"] + previous_obs["provenance"]["source_refs"]
        return self._observe(
            "calculate_growth",
            {"current_observation_id": current_observation_id, "previous_observation_id": previous_observation_id},
            {"value": value, "formula": "(current - previous) / abs(previous) * 100"},
            as_of_time=as_of,
            source_refs=refs,
            scalar=value,
            unit="percent",
            parents=[current_observation_id, previous_observation_id],
        )

    def calculate_margin(self, profit_observation_id: str, revenue_observation_id: str) -> dict[str, Any]:
        profit, profit_unit, profit_obs = self._scalar(profit_observation_id)
        revenue, revenue_unit, revenue_obs = self._scalar(revenue_observation_id)
        if profit_unit != revenue_unit:
            raise ValueError("incompatible_units")
        if math.isclose(revenue, 0.0):
            raise ValueError("division_by_zero")
        value = profit / revenue * 100.0
        as_of = max(profit_obs["provenance"]["as_of_time"], revenue_obs["provenance"]["as_of_time"])
        refs = profit_obs["provenance"]["source_refs"] + revenue_obs["provenance"]["source_refs"]
        return self._observe(
            "calculate_margin",
            {"profit_observation_id": profit_observation_id, "revenue_observation_id": revenue_observation_id},
            {"value": value, "formula": "profit / revenue * 100"},
            as_of_time=as_of,
            source_refs=refs,
            scalar=value,
            unit="percent",
            parents=[profit_observation_id, revenue_observation_id],
        )

    def calculate_difference(self, left_observation_id: str, right_observation_id: str) -> dict[str, Any]:
        left, left_unit, left_obs = self._scalar(left_observation_id)
        right, right_unit, right_obs = self._scalar(right_observation_id)
        if left_unit != right_unit:
            raise ValueError("incompatible_units")
        value = left - right
        as_of = max(left_obs["provenance"]["as_of_time"], right_obs["provenance"]["as_of_time"])
        refs = left_obs["provenance"]["source_refs"] + right_obs["provenance"]["source_refs"]
        return self._observe(
            "calculate_difference",
            {"left_observation_id": left_observation_id, "right_observation_id": right_observation_id},
            {"value": value, "formula": "left - right"},
            as_of_time=as_of,
            source_refs=refs,
            scalar=value,
            unit=left_unit,
            parents=[left_observation_id, right_observation_id],
        )

    def calculate_ratio(
        self,
        numerator_observation_id: str,
        denominator_observation_id: str,
        scale: float = 1.0,
        output_unit: str = "ratio",
    ) -> dict[str, Any]:
        if output_unit not in {"ratio", "percent"}:
            raise ValueError("invalid_output_unit")
        numerator, _, numerator_obs = self._scalar(numerator_observation_id)
        denominator, _, denominator_obs = self._scalar(denominator_observation_id)
        if math.isclose(denominator, 0.0):
            raise ValueError("division_by_zero")
        value = numerator / denominator * scale
        as_of = max(numerator_obs["provenance"]["as_of_time"], denominator_obs["provenance"]["as_of_time"])
        refs = numerator_obs["provenance"]["source_refs"] + denominator_obs["provenance"]["source_refs"]
        return self._observe(
            "calculate_ratio",
            {
                "numerator_observation_id": numerator_observation_id,
                "denominator_observation_id": denominator_observation_id,
                "scale": scale,
                "output_unit": output_unit,
            },
            {"value": value, "formula": "numerator / denominator * scale", "output_unit": output_unit},
            as_of_time=as_of,
            source_refs=refs,
            scalar=value,
            unit=output_unit,
            parents=[numerator_observation_id, denominator_observation_id],
        )

    def compare_values(self, left_observation_id: str, right_observation_id: str) -> dict[str, Any]:
        left, left_unit, left_obs = self._scalar(left_observation_id)
        right, right_unit, right_obs = self._scalar(right_observation_id)
        if left_unit != right_unit:
            raise ValueError("incompatible_units")
        winner = "left" if left > right else "right" if right > left else "equal"
        value = max(left, right)
        as_of = max(left_obs["provenance"]["as_of_time"], right_obs["provenance"]["as_of_time"])
        refs = left_obs["provenance"]["source_refs"] + right_obs["provenance"]["source_refs"]
        return self._observe(
            "compare_values",
            {"left_observation_id": left_observation_id, "right_observation_id": right_observation_id},
            {"winner": winner, "left": left, "right": right},
            as_of_time=as_of,
            source_refs=refs,
            scalar=value,
            unit=left_unit,
            parents=[left_observation_id, right_observation_id],
        )
