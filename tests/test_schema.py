from __future__ import annotations

import inspect

import pytest

from fintool_rl.schema import SCHEMA_BY_NAME, TOOL_SCHEMAS, ToolArgumentError, prompt_block, validate_arguments
from fintool_rl.tools import FinancialTools


def test_every_schema_has_exactly_one_implementation():
    assert len(SCHEMA_BY_NAME) == len(TOOL_SCHEMAS)
    for schema in TOOL_SCHEMAS:
        method = getattr(FinancialTools, schema["name"], None)
        assert callable(method)
        signature = inspect.signature(method)
        python_params = {name for name in signature.parameters if name != "self"}
        schema_params = set(schema["parameters"]["properties"])
        assert python_params == schema_params


def test_required_arguments_match_signature_defaults():
    for schema in TOOL_SCHEMAS:
        signature = inspect.signature(getattr(FinancialTools, schema["name"]))
        required = {
            name for name, param in signature.parameters.items()
            if name != "self" and param.default is inspect.Parameter.empty
        }
        assert required == set(schema["parameters"].get("required", []))


def test_strict_argument_validation():
    validate_arguments(
        "get_financial_fact",
        {"symbol": "ALFA", "metric": "revenue", "fiscal_year": 2024, "as_of_time": "2025-03-31"},
    )
    with pytest.raises(ToolArgumentError):
        validate_arguments("get_financial_fact", {"symbol": "ALFA"})
    with pytest.raises(ToolArgumentError):
        validate_arguments(
            "get_financial_fact",
            {
                "symbol": "ALFA", "metric": "revenue", "fiscal_year": True,
                "as_of_time": "2025-03-31",
            },
        )
    with pytest.raises(ToolArgumentError):
        validate_arguments("unknown_tool", {})


def test_prompt_contract_is_deterministic():
    assert prompt_block() == prompt_block()
    for name in SCHEMA_BY_NAME:
        assert name in prompt_block()

