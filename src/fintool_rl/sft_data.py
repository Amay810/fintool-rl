"""Export oracle programs as native function-calling SFT conversations."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .contracts import TaskSpec
from .oracle import execute_oracle
from .policies import NATIVE_TOOL_SYSTEM_PROMPT, _openai_tools
from .readiness import graph_stratum
from .schema import TOOL_SCHEMAS
from .tools import FinancialTools


def oracle_conversation(task: TaskSpec, db_path: Path | str) -> dict:
    rendered, _ = execute_oracle(FinancialTools(db_path), task.oracle_steps)
    messages: list[dict] = [
        {"role": "system", "content": NATIVE_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(task.public_view(), ensure_ascii=False, sort_keys=True)},
    ]
    for index, step in enumerate(rendered):
        call_id = f"oracle_{index}_{step['id']}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": step["tool"],
                    "arguments": json.dumps(step["arguments"], ensure_ascii=False, sort_keys=True),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(step["result"], ensure_ascii=False, sort_keys=True),
        })
    final = rendered[-1]["result"]
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "oracle_submit_answer",
            "type": "function",
            "function": {
                "name": "submit_answer",
                "arguments": json.dumps({
                    "value": final["scalar"],
                    "unit": final["unit"],
                    "observation_ids": [final["provenance"]["observation_id"]],
                }, ensure_ascii=False, sort_keys=True),
            },
        }],
    })
    return {
        "task_id": task.task_id,
        "split": task.split,
        "template_family": task.template_family,
        "graph_stratum": graph_stratum(task),
        "tools": _openai_tools(TOOL_SCHEMAS),
        "messages": messages,
    }


def write_sft_data(tasks: Iterable[TaskSpec], db_path: Path | str, output: Path | str) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [oracle_conversation(task, db_path) for task in tasks]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return len(records)
