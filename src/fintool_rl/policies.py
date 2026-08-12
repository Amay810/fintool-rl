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
from .schema import TOOL_SCHEMAS, prompt_block

SYSTEM_PROMPT = """You are a financial tool-use agent operating on a frozen data snapshot.
Use tools for every factual number. Never use information after the task's as_of_time.
Respond with exactly one JSON object and no prose.

Tool action:
{"kind":"tool","tool_name":"<name>","arguments":{...}}

Final answer:
{"kind":"answer","answer":{"value":<number>,"unit":"<unit>","observation_ids":["obs_..."]}}

Available tools:
{tools}
"""

NATIVE_TOOL_SYSTEM_PROMPT = """You are a financial tool-use agent operating on a frozen data snapshot.
Use the provided functions for every factual number. Never use information after the task's as_of_time.
Calculator functions accept observation_id references from prior tool results, never raw numeric values.
When the calculation is complete, call submit_answer. Do not write a prose answer.
"""

SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final grounded numeric answer and end the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["value", "unit", "observation_ids"],
        },
    },
}


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
            raise TypeError("non-privileged policy received hidden TaskSpec")
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


def _openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in schemas
    ]
    tools.append(SUBMIT_ANSWER_TOOL)
    return tools


def _parse_native_answer(content: Any) -> AgentAction:
    if not isinstance(content, str):
        raise ActionParseError("final response has no text content", raw_text=str(content))
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"final response is not one JSON object: {exc}", raw_text=content) from exc
    if isinstance(payload, dict) and payload.get("kind") == "answer":
        payload = payload.get("answer")
    if not isinstance(payload, dict):
        raise ActionParseError("final response must be an answer object", raw_text=content)
    return AgentAction(kind="answer", answer=payload)


class OpenAICompatiblePolicy:
    """Dependency-free native function-calling client for vLLM-compatible endpoints."""

    privileged = False

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        seed: int = 0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed
        self.name = f"OpenAICompatiblePolicy:{model}:native-tools"
        self.messages: list[dict[str, Any]] = []
        self._queued_tool_calls: list[dict[str, Any]] = []
        self._awaiting_call_id: str | None = None
        self._seen_tool_results = 0
        self.generated_tokens = 0
        self.policy_version = os.environ.get("FINTOOL_POLICY_VERSION", model)

    @classmethod
    def from_env(cls) -> OpenAICompatiblePolicy:
        return cls(
            base_url=os.environ.get("FINTOOL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("FINTOOL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            api_key=os.environ.get("FINTOOL_LLM_API_KEY", ""),
            temperature=float(os.environ.get("FINTOOL_LLM_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("FINTOOL_LLM_TOP_P", "0.8")),
            top_k=int(os.environ.get("FINTOOL_LLM_TOP_K", "20")),
            min_p=float(os.environ.get("FINTOOL_LLM_MIN_P", "0")),
            seed=int(os.environ.get("FINTOOL_LLM_SEED", "0")),
        )

    def reset(self, task: TaskSpec | dict[str, Any]) -> None:
        if isinstance(task, TaskSpec):
            raise TypeError("non-privileged policy received hidden TaskSpec")
        self.messages = [
            {"role": "system", "content": NATIVE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(task, ensure_ascii=False, sort_keys=True)},
        ]
        self._queued_tool_calls = []
        self._awaiting_call_id = None
        self._seen_tool_results = 0
        self.generated_tokens = 0

    @staticmethod
    def _action_from_tool_call(tool_call: dict[str, Any]) -> AgentAction:
        try:
            name = tool_call["function"]["name"]
            raw_arguments = tool_call["function"]["arguments"]
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ActionParseError(f"malformed native tool call: {exc}", raw_text=str(tool_call)) from exc
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ActionParseError("native tool call requires a name and object arguments", raw_text=str(tool_call))
        if name == "submit_answer":
            return AgentAction(kind="answer", answer=arguments)
        return AgentAction.tool(name, **arguments)

    def _dispatch_tool_call(self, tool_call: dict[str, Any]) -> AgentAction:
        action = self._action_from_tool_call(tool_call)
        if action.kind == "tool":
            call_id = str(tool_call.get("id", ""))
            if not call_id:
                raise ActionParseError("native tool call is missing id", raw_text=str(tool_call))
            self._awaiting_call_id = call_id
        else:
            self._awaiting_call_id = None
            self._queued_tool_calls = []
        return action

    def act(self, observation: AgentObservation) -> AgentAction:
        if self._awaiting_call_id is not None:
            if len(observation.tool_results) <= self._seen_tool_results:
                raise ModelCallError("tool result missing for the previous native function call")
            result = observation.tool_results[self._seen_tool_results]
            self.messages.append({
                "role": "tool",
                "tool_call_id": self._awaiting_call_id,
                "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
            })
            self._seen_tool_results += 1
            self._awaiting_call_id = None

        if self._queued_tool_calls:
            tool_call = self._queued_tool_calls.pop(0)
            return self._dispatch_tool_call(tool_call)

        message = self._chat()
        assistant_message = {
            key: message.get(key)
            for key in ("role", "content", "tool_calls")
            if message.get(key) is not None
        }
        assistant_message.setdefault("role", "assistant")
        self.messages.append(assistant_message)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise ActionParseError("tool_calls must be a list", raw_text=str(tool_calls))
            self._queued_tool_calls = tool_calls
            tool_call = self._queued_tool_calls.pop(0)
            return self._dispatch_tool_call(tool_call)
        return _parse_native_answer(message.get("content"))

    def _chat(self) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "messages": self.messages,
            "tools": _openai_tools(TOOL_SCHEMAS),
            "tool_choice": "auto",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
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
            message = payload["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            usage = payload.get("usage") or {}
            completion_tokens = usage.get("completion_tokens", 0)
            if isinstance(completion_tokens, int) and completion_tokens >= 0:
                self.generated_tokens += completion_tokens
            return message
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelCallError(f"unexpected completion payload: {exc}", cause=exc) from exc
