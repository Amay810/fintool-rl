from __future__ import annotations

from pathlib import Path

from fintool_rl.database import build_fixture_snapshot
from fintool_rl.sft_data import oracle_conversation
from fintool_rl.tasks import generate_fixture_tasks


def test_oracle_conversation_uses_native_tool_messages(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    tasks = generate_fixture_tasks(db)
    task = next(task for task in tasks if len(task.oracle_steps) == 3)
    record = oracle_conversation(task, db)
    messages = record["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert sum(message["role"] == "tool" for message in messages) == 3
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "submit_answer"
    rendered = str(messages)
    assert "oracle_steps" not in rendered
    assert "tolerance" not in rendered
