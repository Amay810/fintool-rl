"""Executable task oracles and variable resolution."""

from __future__ import annotations

from typing import Any

from .tools import FinancialTools


def resolve_references(value: Any, step_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        expression = value[1:]
        step_id, separator, path = expression.partition(".")
        if not separator or step_id not in step_results:
            raise ValueError(f"unresolved oracle reference: {value}")
        current: Any = step_results[step_id]
        for part in path.split("."):
            if part == "observation_id":
                current = current["provenance"]["observation_id"]
            else:
                current = current[part]
        return current
    if isinstance(value, dict):
        return {key: resolve_references(item, step_results) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_references(item, step_results) for item in value]
    return value


def execute_oracle(
    tools: FinancialTools, steps: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rendered: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for step in steps:
        arguments = resolve_references(step["arguments"], results)
        result = tools.call(step["tool"], **arguments)
        if not result.get("ok"):
            raise RuntimeError(f"oracle step {step['id']} failed: {result}")
        results[step["id"]] = result
        rendered.append({"id": step["id"], "tool": step["tool"], "arguments": arguments, "result": result})
    return rendered, results

