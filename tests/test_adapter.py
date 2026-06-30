import json

from claude_proxy.adapter import (
    anthropic_messages_to_openai_chat,
    openai_chat_to_anthropic_message,
    openai_stream_chunks_to_anthropic_events,
)
from claude_proxy.config import ModelConfig


def test_anthropic_messages_request_maps_tools_and_tool_results_to_openai_chat() -> None:
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://kubeflow.example/v1",
        upstream_model="glm-5.2-serving",
        api_key_env="KUBEFLOW_API_KEY",
        capabilities={"streaming": True, "tools": True},
    )

    payload = {
        "model": "glm-5.2",
        "max_tokens": 1024,
        "temperature": 0.2,
        "stop_sequences": ["</done>"],
        "system": "You are a coding assistant.",
        "stream": True,
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Inspect app.py"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "app.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "print('hello')",
                    }
                ],
            },
        ],
    }

    result = anthropic_messages_to_openai_chat(payload, model)

    assert result["model"] == "glm-5.2-serving"
    assert result["stream"] is True
    assert result["max_tokens"] == 1024
    assert result["temperature"] == 0.2
    assert result["stop"] == ["</done>"]
    assert result["messages"][0] == {
        "role": "system",
        "content": "You are a coding assistant.",
    }
    assert result["messages"][1] == {"role": "user", "content": "Inspect app.py"}
    assistant = result["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "I will inspect it."
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"path": "app.py"}
    assert result["messages"][3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "print('hello')",
    }
    assert result["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": payload["tools"][0]["input_schema"],
            },
        }
    ]


def test_anthropic_tool_choice_maps_to_openai_tool_choice() -> None:
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://kubeflow.example/v1",
        upstream_model="glm-5.2-serving",
        api_key_env="KUBEFLOW_API_KEY",
        capabilities={"tools": True},
    )
    base_payload = {
        "model": "glm-5.2",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }

    any_choice = anthropic_messages_to_openai_chat(
        {**base_payload, "tool_choice": {"type": "any"}},
        model,
    )
    named_choice = anthropic_messages_to_openai_chat(
        {**base_payload, "tool_choice": {"type": "tool", "name": "read_file"}},
        model,
    )
    none_choice = anthropic_messages_to_openai_chat(
        {**base_payload, "tool_choice": {"type": "none"}},
        model,
    )

    assert any_choice["tool_choice"] == "required"
    assert named_choice["tool_choice"] == {
        "type": "function",
        "function": {"name": "read_file"},
    }
    assert none_choice["tool_choice"] == "none"


def test_anthropic_thinking_maps_to_openrouter_reasoning_shape() -> None:
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://openrouter.ai/api/v1",
        upstream_model="deepseek/deepseek-v4-flash",
        api_key_env="OPENROUTER_API_KEY",
        capabilities={"reasoning": True},
    )

    result = anthropic_messages_to_openai_chat(
        {
            "model": "glm-5.2",
            "max_tokens": 128,
            "thinking": {"type": "enabled", "budget_tokens": 512},
            "messages": [{"role": "user", "content": "think"}],
        },
        model,
    )

    assert result["reasoning"] == {"enabled": True, "max_tokens": 512}


def test_anthropic_output_config_effort_maps_to_openai_reasoning_effort() -> None:
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://openrouter.ai/api/v1",
        upstream_model="deepseek/deepseek-v4-flash",
        api_key_env="OPENROUTER_API_KEY",
        capabilities={"reasoning": True, "reasoning_exclude": True},
    )

    result = anthropic_messages_to_openai_chat(
        {
            "model": "glm-5.2",
            "max_tokens": 128,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
            "messages": [{"role": "user", "content": "think hard"}],
        },
        model,
    )

    assert result["reasoning"] == {
        "enabled": True,
        "effort": "xhigh",
        "exclude": True,
    }


def test_openai_response_maps_text_and_tool_calls_to_anthropic_message() -> None:
    response = {
        "id": "chatcmpl_123",
        "model": "glm-5.2-serving",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I need the file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"app.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = openai_chat_to_anthropic_message(response, requested_model="glm-5.2")

    assert result["type"] == "message"
    assert result["id"] == "chatcmpl_123"
    assert result["role"] == "assistant"
    assert result["model"] == "glm-5.2"
    assert result["stop_reason"] == "tool_use"
    assert result["content"] == [
        {"type": "text", "text": "I need the file."},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "app.py"},
        },
    ]
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_openai_reasoning_only_response_maps_to_visible_text_fallback() -> None:
    response = {
        "id": "chatcmpl_reasoning",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "I should answer with pong.",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 9},
    }

    result = openai_chat_to_anthropic_message(response, requested_model="glm-5.2")

    assert result["content"] == [
        {
            "type": "text",
            "text": "Reasoning:\nI should answer with pong.",
        }
    ]
    assert result["stop_reason"] == "max_tokens"


def test_openai_response_prefers_visible_content_over_reasoning_fallback() -> None:
    response = {
        "id": "chatcmpl_content",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "pong",
                    "reasoning": "Hidden-ish reasoning should not be shown when content exists.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    result = openai_chat_to_anthropic_message(response, requested_model="glm-5.2")

    assert result["content"] == [{"type": "text", "text": "pong"}]


def test_openai_reasoning_only_response_can_disable_visible_fallback() -> None:
    response = {
        "id": "chatcmpl_reasoning",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "This should stay hidden in normal mode.",
                },
                "finish_reason": "length",
            }
        ],
    }

    result = openai_chat_to_anthropic_message(
        response,
        requested_model="glm-5.2",
        expose_reasoning=False,
    )

    assert result["content"] == [{"type": "text", "text": ""}]


def test_openai_stream_chunks_are_reframed_as_anthropic_sse_events() -> None:
    chunks = [
        {"id": "chatcmpl_1", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path"'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ':"app.py"}'},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    events = list(openai_stream_chunks_to_anthropic_events(chunks, requested_model="glm-5.2"))

    assert [event["event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["data"]["content_block"] == {"type": "text", "text": ""}
    assert events[2]["data"]["delta"] == {"type": "text_delta", "text": "Hel"}
    assert events[3]["data"]["delta"] == {"type": "text_delta", "text": "lo"}
    assert events[5]["data"]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {},
    }
    assert events[6]["data"]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"path"',
    }
    assert events[7]["data"]["delta"] == {
        "type": "input_json_delta",
        "partial_json": ':"app.py"}',
    }
    assert events[9]["data"]["delta"]["stop_reason"] == "tool_use"


def test_openai_reasoning_stream_delta_maps_to_visible_text_fallback() -> None:
    chunks = [
        {"id": "chatcmpl_reasoning", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning": "I should "}}]},
        {"choices": [{"delta": {"reasoning": "answer."}, "finish_reason": "length"}]},
    ]

    events = list(openai_stream_chunks_to_anthropic_events(chunks, requested_model="glm-5.2"))

    assert [event["event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["data"]["delta"] == {
        "type": "text_delta",
        "text": "Reasoning:\nI should ",
    }
    assert events[3]["data"]["delta"] == {"type": "text_delta", "text": "answer."}
    assert events[5]["data"]["delta"]["stop_reason"] == "max_tokens"


def test_openai_reasoning_stream_then_content_uses_separate_text_blocks() -> None:
    chunks = [
        {"id": "chatcmpl_reasoning", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning": "I should answer. "}}]},
        {"choices": [{"delta": {"content": "안녕하세요!"}, "finish_reason": "stop"}]},
    ]

    events = list(openai_stream_chunks_to_anthropic_events(chunks, requested_model="glm-5.2"))

    assert [event["event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["data"]["delta"] == {
        "type": "text_delta",
        "text": "Reasoning:\nI should answer. ",
    }
    assert events[5]["data"]["delta"] == {"type": "text_delta", "text": "안녕하세요!"}


def test_openai_reasoning_stream_delta_can_disable_visible_fallback() -> None:
    chunks = [
        {"id": "chatcmpl_reasoning", "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning": "hidden"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ]

    events = list(
        openai_stream_chunks_to_anthropic_events(
            chunks,
            requested_model="glm-5.2",
            expose_reasoning=False,
        )
    )

    assert [event["event"] for event in events] == [
        "message_start",
        "message_delta",
        "message_stop",
    ]


def test_openai_tool_call_stream_uses_anthropic_input_json_delta_events() -> None:
    chunks = [
        {"id": "chatcmpl_tool", "choices": [{"delta": {"role": "assistant"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path"'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ':"app.py"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    events = list(openai_stream_chunks_to_anthropic_events(chunks, requested_model="glm-5.2"))

    assert [event["event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["data"] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {},
        },
    }
    assert events[2]["data"]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"path"',
    }
    assert events[3]["data"]["delta"] == {
        "type": "input_json_delta",
        "partial_json": ':"app.py"}',
    }
    assert events[5]["data"]["delta"]["stop_reason"] == "tool_use"
