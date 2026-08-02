"""Stable data contracts shared by tools, tasks, policies, and graders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    result: dict[str, Any]
    latency_ms: float = 0.0
    argument_valid: bool = True
    error: str | None = None


@dataclass(frozen=True)
class AgentAction:
    kind: Literal["tool", "answer"]
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: dict[str, Any] | None = None

    @classmethod
    def tool(cls, name: str, **arguments: Any) -> "AgentAction":
        return cls(kind="tool", tool_name=name, arguments=arguments)

    @classmethod
    def final(cls, value: float, unit: str, observation_ids: list[str]) -> "AgentAction":
        return cls(
            kind="answer",
            answer={"value": value, "unit": unit, "observation_ids": observation_ids},
        )


@dataclass
class TaskSpec:
    task_id: str
    question: str
    split: Literal["train", "dev", "test", "challenge"]
    as_of_time: str
    difficulty: Literal["single_tool", "multi_tool", "compositional", "held_out_tool"]
    template_family: str
    answer: dict[str, Any]
    oracle_steps: list[dict[str, Any]]
    required_tool_families: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_view(self) -> dict[str, Any]:
        """Return only fields an evaluated policy is allowed to observe."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "as_of_time": self.as_of_time,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        return cls(**payload)


@dataclass
class Trajectory:
    trajectory_id: str
    task_id: str
    policy_name: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: dict[str, Any] | None = None
    terminal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

