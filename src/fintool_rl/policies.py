"""Policies for OpenAI-compatible local or hosted model baselines."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .contracts import AgentAction, TaskSpec
from .harness import AgentObservation
from .schema import prompt_block

SYSTEM_PROMPT = """You are a financial tool-use agent operating on a frozen data snapshot.
Use tools for every factual number. Never use information after the task's as_of_time.
Respond with exactly one JSON object and no prose. Emit exactly ONE action in each
response: either one tool action or one final answer.

Interaction protocol:
- If more tools are needed, call only the NEXT tool, then stop.
- Wait for the environment to return that tool's observation before deciding the next action.
- The interaction is: assistant action -> environment observation -> assistant action -> ... -> final answer.
- Never emit multiple JSON objects, JSONL, a JSON array of actions, or a plan containing future tool calls.

Tool action:
{"kind":"tool","tool_name":"<name>","arguments":{...}}

The tool-action key is exactly "arguments". Do not use "tool_arguments", "parameters",
"args", or another alias.

Return exactly one JSON object and nothing else: no prose before or after it, no Markdown
fences, and no additional JSON object.

Final answer:
{"kind":"answer","answer":{"value":<number>,"unit":"<unit>","observation_ids":["obs_..."]}}

Available tools:
{tools}
"""


class ActionParseError(ValueError):
    """Model output could not be parsed into a valid agent action."""

    def __init__(self, message: str, *, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


class ModelCallError(RuntimeError):
    """The underlying model HTTP/API call failed."""

    def __init__(self, message: str, *, cause: BaseException | None = None):
        super().__init__(message)
        self.cause = cause


def parse_action(text: str) -> AgentAction:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ActionParseError(
            f"model output is not one JSON object: {exc}", raw_text=text
        ) from exc
    if not isinstance(payload, dict) or payload.get("kind") not in {"tool", "answer"}:
        raise ActionParseError(
            "model output must declare kind=tool or kind=answer", raw_text=text
        )
    if payload["kind"] == "tool":
        if not isinstance(payload.get("tool_name"), str) or not isinstance(payload.get("arguments"), dict):
            raise ActionParseError(
                "tool action requires string tool_name and object arguments", raw_text=text
            )
        return AgentAction.tool(payload["tool_name"], **payload["arguments"])
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        raise ActionParseError("answer action requires an answer object", raw_text=text)
    return AgentAction(kind="answer", answer=answer)


class JsonActionPolicy:
    name = "JsonActionPolicy"
    privileged = False

    def __init__(self, completion_fn: Callable[[list[dict[str, str]]], str], *, name: str | None = None):
        self.completion_fn = completion_fn
        self.name = name or self.name
        self.task: dict[str, Any] = {}

    def reset(self, task: TaskSpec | dict[str, Any]) -> None:
        if isinstance(task, TaskSpec):
            raise ValueError("non-privileged policy received hidden TaskSpec")
        self.task = task

    def act(self, observation: AgentObservation) -> AgentAction:
        user_payload = {
            "task": self.task,
            "tool_results": observation.tool_results,
            "remaining_steps": observation.remaining_steps,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.replace("{tools}", prompt_block())},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
        ]
        return parse_action(self.completion_fn(messages))


class OpenAICompatiblePolicy(JsonActionPolicy):
    """Minimal dependency-free client for vLLM and compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        super().__init__(self._complete, name=f"OpenAICompatiblePolicy:{model}")

    @classmethod
    def from_env(cls) -> "OpenAICompatiblePolicy":
        return cls(
            base_url=os.environ.get("FINTOOL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("FINTOOL_LLM_MODEL", "Qwen3-1.7B"),
            api_key=os.environ.get("FINTOOL_LLM_API_KEY", ""),
        )

    def _complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ModelCallError(f"HTTP {exc.code}: {detail}", cause=exc) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            raise ModelCallError(str(exc), cause=exc) from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelCallError(f"unexpected completion payload: {exc}", cause=exc) from exc
