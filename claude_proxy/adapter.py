from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

from claude_proxy.config import ModelConfig


def anthropic_messages_to_openai_chat(
    payload: dict[str, Any],
    model: ModelConfig,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_text = _content_blocks_to_text(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for message in payload.get("messages", []):
        converted = _convert_anthropic_message(message)
        if isinstance(converted, list):
            messages.extend(converted)
        else:
            messages.append(converted)

    result: dict[str, Any] = {
        "model": model.upstream_model,
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }
    if result["stream"]:
        result["stream_options"] = {"include_usage": True}
    for key in ("max_tokens", "temperature", "top_p", "stop"):
        if key in payload:
            result[key] = payload[key]
    if "stop_sequences" in payload:
        result["stop"] = payload["stop_sequences"]

    tools = payload.get("tools")
    if tools and model.capabilities.get("tools", True):
        result["tools"] = [_anthropic_tool_to_openai_tool(tool) for tool in tools]
        result["tool_choice"] = _anthropic_tool_choice_to_openai(payload.get("tool_choice"))

    extra_body: dict[str, Any] = {}
    if model.capabilities.get("reasoning"):
        reasoning = _anthropic_reasoning_config(payload)
        if reasoning:
            if model.capabilities.get("reasoning_exclude"):
                reasoning["exclude"] = True
            extra_body["reasoning"] = reasoning
    result.update(extra_body)
    return result


def openai_chat_to_anthropic_message(
    response: dict[str, Any],
    requested_model: str,
    *,
    expose_reasoning: bool = True,
) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = _openai_message_content_to_anthropic_blocks(
        message,
        expose_reasoning=expose_reasoning,
    )
    usage = response.get("usage") or {}
    return {
        "id": response.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _openai_finish_reason_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_stream_chunks_to_anthropic_events(
    chunks: Iterable[dict[str, Any]],
    requested_model: str,
    *,
    expose_reasoning: bool = True,
) -> Iterator[dict[str, Any]]:
    builder = AnthropicStreamBuilder(
        requested_model=requested_model,
        expose_reasoning=expose_reasoning,
    )
    for chunk in chunks:
        yield from builder.consume(chunk)
    yield from builder.close()


class AnthropicStreamBuilder:
    def __init__(self, requested_model: str, *, expose_reasoning: bool = True) -> None:
        self.requested_model = requested_model
        self.expose_reasoning = expose_reasoning
        self.message_started = False
        self.text_started = False
        self.text_index: int | None = None
        self.text_kind: str | None = None
        self.next_index = 0
        self.stop_reason = "end_turn"
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.reasoning_output_chars = 0
        self.reasoning_prefix_sent = False

    def consume(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.message_started:
            self.message_started = True
            events.append(
                _event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": chunk.get("id") or f"msg_{uuid.uuid4().hex}",
                            "type": "message",
                            "role": "assistant",
                            "model": self.requested_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self.usage,
                        },
                    },
                )
            )

        usage = chunk.get("usage") or {}
        if usage:
            self.usage = {
                "input_tokens": usage.get("prompt_tokens", self.usage["input_tokens"]),
                "output_tokens": usage.get("completion_tokens", self.usage["output_tokens"]),
            }

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if text := delta.get("content"):
                events.extend(self._append_text_delta(text, kind="content"))

            reasoning = _delta_reasoning_to_text(delta)
            if reasoning:
                self.reasoning_output_chars += len(reasoning)
                if self.expose_reasoning:
                    if not self.reasoning_prefix_sent:
                        reasoning = f"Reasoning:\n{reasoning}"
                        self.reasoning_prefix_sent = True
                    events.extend(self._append_text_delta(reasoning, kind="reasoning"))

            for tool_call in delta.get("tool_calls") or []:
                index = int(tool_call.get("index", 0))
                current = self.tool_calls.setdefault(
                    index,
                    {"id": None, "name": None, "arguments": "", "started": False, "index": None},
                )
                if tool_call.get("id"):
                    current["id"] = tool_call["id"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    current["name"] = function["name"]
                partial_json = function.get("arguments")
                if not current["started"] and (current["id"] or current["name"] or partial_json):
                    events.extend(self._stop_text_block())
                    current["started"] = True
                    current["index"] = self.next_index
                    self.next_index += 1
                    events.append(
                        _event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": current["index"],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": current["id"] or f"toolu_{uuid.uuid4().hex}",
                                    "name": current["name"] or "unknown_tool",
                                    "input": {},
                                },
                            },
                        )
                    )
                if partial_json:
                    current["arguments"] += partial_json
                    events.append(
                        _event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": current["index"],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": partial_json,
                                },
                            },
                        )
                    )

            if choice.get("finish_reason"):
                self.stop_reason = _openai_finish_reason_to_anthropic(choice["finish_reason"])

        return events

    def close(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.message_started:
            events.extend(self.consume({"id": f"msg_{uuid.uuid4().hex}", "choices": []}))

        events.extend(self._stop_text_block())

        for _, tool_call in sorted(self.tool_calls.items()):
            if tool_call["started"]:
                events.append(
                    _event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": tool_call["index"]},
                    )
                )

        events.append(
            _event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": self.usage["output_tokens"]},
                },
            )
        )
        events.append(_event("message_stop", {"type": "message_stop"}))
        return events

    def _stop_text_block(self) -> list[dict[str, Any]]:
        if not self.text_started:
            return []
        self.text_started = False
        self.text_kind = None
        return [
            _event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self.text_index},
            )
        ]

    def _append_text_delta(self, text: str, *, kind: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.text_started and self.text_kind != kind:
            events.extend(self._stop_text_block())
        if not self.text_started:
            self.text_started = True
            self.text_kind = kind
            self.text_index = self.next_index
            self.next_index += 1
            events.append(
                _event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self.text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        events.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.text_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        return events


def estimate_anthropic_tokens(payload: dict[str, Any]) -> int:
    text_parts: list[str] = []
    if system := payload.get("system"):
        text_parts.append(system if isinstance(system, str) else _content_blocks_to_text(system))
    for message in payload.get("messages", []):
        content = message.get("content", "")
        text_parts.append(content if isinstance(content, str) else _content_blocks_to_text(content))
    text = "\n".join(text_parts)
    return max(1, (len(text) + 3) // 4)


def _convert_anthropic_message(message: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    role = message.get("role", "user")
    content = message.get("content", "")
    if isinstance(content, str):
        return {"role": role, "content": content}

    if role == "user":
        tool_messages = []
        text_blocks = []
        for block in content:
            if block.get("type") == "tool_result":
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _block_content_to_text(block.get("content", "")),
                    }
                )
            else:
                text_blocks.append(block)
        messages: list[dict[str, Any]] = []
        text = _content_blocks_to_text(text_blocks)
        if text:
            messages.append({"role": "user", "content": text})
        messages.extend(tool_messages)
        return messages

    result: dict[str, Any] = {"role": role, "content": _assistant_text(content)}
    tool_calls = [
        _anthropic_tool_use_to_openai_tool_call(block)
        for block in content
        if block.get("type") == "tool_use"
    ]
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _anthropic_tool_to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return "auto"

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def _anthropic_reasoning_config(payload: dict[str, Any]) -> dict[str, Any]:
    thinking = payload.get("thinking")
    output_effort = _output_config_effort(payload.get("output_config"))

    if thinking:
        reasoning = _anthropic_thinking_to_openai_reasoning(thinking)
    elif output_effort:
        reasoning = {"enabled": output_effort != "none"}
    else:
        return {}

    if output_effort and "max_tokens" not in reasoning:
        reasoning["effort"] = output_effort
    return reasoning


def _anthropic_thinking_to_openai_reasoning(thinking: Any) -> dict[str, Any]:
    if not isinstance(thinking, dict):
        return {"enabled": True}

    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return {"enabled": False}

    reasoning: dict[str, Any] = {"enabled": True}
    if effort := thinking.get("effort"):
        reasoning["effort"] = effort
    if budget_tokens := thinking.get("budget_tokens"):
        reasoning["max_tokens"] = budget_tokens
    return reasoning


def _output_config_effort(output_config: Any) -> str | None:
    if not isinstance(output_config, dict):
        return None
    effort = output_config.get("effort")
    return str(effort) if effort else None


def _anthropic_tool_use_to_openai_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": block["name"],
            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
        },
    }


def _openai_message_content_to_anthropic_blocks(
    message: dict[str, Any],
    *,
    expose_reasoning: bool,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        text = _block_content_to_text(content)
        if text:
            blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": function.get("name") or "unknown_tool",
                "input": _json_loads_object(function.get("arguments", "{}")),
            }
        )
    if expose_reasoning and not blocks and (reasoning := _message_reasoning_to_text(message)):
        blocks.append({"type": "text", "text": f"Reasoning:\n{reasoning}"})
    return blocks or [{"type": "text", "text": ""}]


def _openai_finish_reason_to_anthropic(reason: str | None) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }.get(reason or "stop", "end_turn")


def _content_blocks_to_text(blocks: list[Any]) -> str:
    parts = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            parts.append(_block_content_to_text(block.get("content", "")))
    return "\n".join(part for part in parts if part)


def _assistant_text(blocks: list[dict[str, Any]]) -> str | None:
    text = _content_blocks_to_text([block for block in blocks if block.get("type") == "text"])
    return text or None


def _block_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_blocks_to_text(content)
    return json.dumps(content, ensure_ascii=False)


def _json_loads_object(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {"_raw": value}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _message_reasoning_to_text(message: dict[str, Any]) -> str:
    if reasoning := message.get("reasoning"):
        return str(reasoning)
    return _reasoning_details_to_text(message.get("reasoning_details"))


def _delta_reasoning_to_text(delta: dict[str, Any]) -> str:
    if reasoning := delta.get("reasoning"):
        return str(reasoning)
    return _reasoning_details_to_text(delta.get("reasoning_details"))


def _reasoning_details_to_text(reasoning_details: Any) -> str:
    if not isinstance(reasoning_details, list):
        return ""
    parts = []
    for item in reasoning_details:
        if isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "".join(parts)


def _event(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event, "data": data}
