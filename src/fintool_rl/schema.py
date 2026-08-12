"""JSON-schema-like tool contracts with strict local validation.

The project intentionally keeps validation dependency-free.  The contract is
also rendered into prompts and checked against Python signatures in tests.
"""

from __future__ import annotations

import re
from typing import Any

FINANCIAL_METRICS = (
    "revenue",
    "gross_profit",
    "net_income",
    "total_assets",
    "total_liabilities",
)


class ToolArgumentError(ValueError):
    """Arguments do not satisfy a declared tool contract."""


def _string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_company_profile",
        "family": "company",
        "description": "Return one listed company's profile from the frozen snapshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": _string("Canonical uppercase ticker."),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["symbol", "as_of_time"],
        },
    },
    {
        "name": "list_available_periods",
        "family": "financial_statement",
        "description": "List fiscal years available for a company and metric at the cutoff.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": _string("Canonical uppercase ticker."),
                "metric": _string(
                    "Canonical financial metric.",
                    enum=list(FINANCIAL_METRICS),
                ),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["symbol", "metric", "as_of_time"],
        },
    },
    {
        "name": "get_financial_fact",
        "family": "financial_statement",
        "description": "Read one filed annual financial fact available by the cutoff.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": _string("Canonical uppercase ticker."),
                "metric": _string(
                    "Canonical financial metric.",
                    enum=list(FINANCIAL_METRICS),
                ),
                "fiscal_year": {"type": "integer", "minimum": 1900, "maximum": 2200},
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["symbol", "metric", "fiscal_year", "as_of_time"],
        },
    },
    {
        "name": "get_daily_price",
        "family": "market_data",
        "description": "Read a security close price on one trading date.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": _string("Canonical uppercase ticker."),
                "trading_date": _string("Trading date in YYYY-MM-DD format."),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["symbol", "trading_date", "as_of_time"],
        },
    },
    {
        "name": "get_price_series",
        "family": "market_data",
        "description": "Read close prices over an inclusive date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": _string("Canonical uppercase ticker."),
                "start_date": _string("Inclusive start date."),
                "end_date": _string("Inclusive end date."),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["symbol", "start_date", "end_date", "as_of_time"],
        },
    },
    {
        "name": "get_market_index_level",
        "family": "market_data",
        "description": "Read a benchmark index close level on one date.",
        "parameters": {
            "type": "object",
            "properties": {
                "index_symbol": _string("Canonical index symbol."),
                "trading_date": _string("Trading date in YYYY-MM-DD format."),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["index_symbol", "trading_date", "as_of_time"],
        },
    },
    {
        "name": "get_trading_days",
        "family": "calendar",
        "description": "List trading dates in an inclusive range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": _string("Inclusive start date."),
                "end_date": _string("Inclusive end date."),
                "as_of_time": _string("Information cutoff in YYYY-MM-DD format."),
            },
            "required": ["start_date", "end_date", "as_of_time"],
        },
    },
    {
        "name": "calculate_growth",
        "family": "calculator",
        "description": "Calculate percentage growth from two scalar observations.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_observation_id": _string("Observation containing the current value."),
                "previous_observation_id": _string("Observation containing the previous value."),
            },
            "required": ["current_observation_id", "previous_observation_id"],
        },
    },
    {
        "name": "calculate_margin",
        "family": "calculator",
        "description": "Calculate a percentage margin from profit and revenue observations.",
        "parameters": {
            "type": "object",
            "properties": {
                "profit_observation_id": _string("Observation containing profit."),
                "revenue_observation_id": _string("Observation containing revenue."),
            },
            "required": ["profit_observation_id", "revenue_observation_id"],
        },
    },
    {
        "name": "calculate_difference",
        "family": "calculator",
        "description": "Subtract the right scalar observation from the left one.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_observation_id": _string("Observation containing the minuend."),
                "right_observation_id": _string("Observation containing the subtrahend."),
            },
            "required": ["left_observation_id", "right_observation_id"],
        },
    },
    {
        "name": "calculate_ratio",
        "family": "calculator",
        "description": "Divide one scalar observation by another, multiply by scale, and label the unit.",
        "parameters": {
            "type": "object",
            "properties": {
                "numerator_observation_id": _string("Observation containing the numerator."),
                "denominator_observation_id": _string("Observation containing the denominator."),
                "scale": {"type": "number", "default": 1.0},
                "output_unit": _string(
                    "Unit label for the result: ratio or percent.",
                    pattern="^(ratio|percent)$",
                    default="ratio",
                ),
            },
            "required": ["numerator_observation_id", "denominator_observation_id"],
        },
    },
    {
        "name": "compare_values",
        "family": "comparison",
        "description": "Compare two scalar observations and report the larger one.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_observation_id": _string("First scalar observation."),
                "right_observation_id": _string("Second scalar observation."),
            },
            "required": ["left_observation_id", "right_observation_id"],
        },
    },
]

SCHEMA_BY_NAME = {schema["name"]: schema for schema in TOOL_SCHEMAS}
TOOL_FAMILY_BY_NAME = {schema["name"]: schema["family"] for schema in TOOL_SCHEMAS}

_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _type_ok(value: Any, declared: str | list[str]) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if name == "null" and value is None:
            return True
        expected = _PYTHON_TYPES.get(name)
        if expected and not isinstance(value, bool) and isinstance(value, expected):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
    return False


def validate_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    schema = SCHEMA_BY_NAME.get(tool_name)
    if schema is None:
        raise ToolArgumentError(f"unknown tool: {tool_name}")
    if not isinstance(arguments, dict):
        raise ToolArgumentError("arguments must be an object")
    params = schema["parameters"]
    properties = params["properties"]
    missing = [name for name in params.get("required", []) if name not in arguments]
    if missing:
        raise ToolArgumentError(f"{tool_name}: missing required argument(s): {', '.join(missing)}")
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ToolArgumentError(f"{tool_name}: unknown argument(s): {', '.join(unknown)}")
    for name, value in arguments.items():
        spec = properties[name]
        if not _type_ok(value, spec["type"]):
            raise ToolArgumentError(
                f"{tool_name}.{name}: expected {spec['type']}, got {type(value).__name__}"
            )
        if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
            raise ToolArgumentError(f"{tool_name}.{name}: does not match {spec['pattern']}")
        if "enum" in spec and value not in spec["enum"]:
            legal = ", ".join(str(item) for item in spec["enum"])
            raise ToolArgumentError(f"{tool_name}.{name}: must be one of: {legal}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ToolArgumentError(f"{tool_name}.{name}: below minimum")
        if "maximum" in spec and value > spec["maximum"]:
            raise ToolArgumentError(f"{tool_name}.{name}: above maximum")


def prompt_block() -> str:
    lines: list[str] = []
    for schema in TOOL_SCHEMAS:
        params = schema["parameters"]
        required = set(params.get("required", []))
        rendered_fields: list[str] = []
        for name, spec in params["properties"].items():
            enum_hint = f" enum={spec['enum']}" if "enum" in spec else ""
            optional_hint = "" if name in required else " (optional)"
            rendered_fields.append(f"{name}: {spec['type']}{enum_hint}{optional_hint}")
        fields = ", ".join(rendered_fields)
        lines.append(f"- {schema['name']}({fields}) — {schema['description']}")
    return "\n".join(lines)
