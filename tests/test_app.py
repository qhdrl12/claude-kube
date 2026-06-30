import json
import logging

import httpx
import pytest

from claude_proxy.app import create_app
from claude_proxy.config import ModelConfig, ModelRegistry, Settings


class FakeUpstreamTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


@pytest.mark.asyncio
async def test_models_route_lists_registered_model_aliases() -> None:
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={
                        "streaming": True,
                        "tools": True,
                        "reasoning": True,
                        "max_context_tokens": 131072,
                    },
                ),
                ModelConfig(
                    alias="qwen-coder",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="qwen-coder-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    routing_tier="fast",
                ),
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "glm-5.2",
                "object": "model",
                "owned_by": "claude-proxy",
                "upstream_model": "glm-5.2-serving",
                "upstream_path": "/v1/chat/completions",
                "routing_tier": "default",
                "capabilities": {
                    "streaming": True,
                    "tools": True,
                    "reasoning": True,
                    "max_context_tokens": 131072,
                },
            },
            {
                "id": "qwen-coder",
                "object": "model",
                "owned_by": "claude-proxy",
                "upstream_model": "qwen-coder-serving",
                "upstream_path": "/v1/chat/completions",
                "routing_tier": "fast",
                "capabilities": {},
            },
        ],
    }


@pytest.mark.asyncio
async def test_readyz_includes_registered_aliases_and_upstream_paths() -> None:
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/serving/model/v1/chat/completions",
                    upstream_model="glm-5.2",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "models": ["glm-5.2"],
        "model_details": [
            {
                "id": "glm-5.2",
                "object": "model",
                "owned_by": "claude-proxy",
                "upstream_model": "glm-5.2",
                "upstream_path": "/serving/model/v1/chat/completions",
                "routing_tier": "default",
                "capabilities": {},
            }
        ],
    }


@pytest.mark.asyncio
async def test_messages_route_proxies_non_stream_request_without_auth_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    registry = ModelRegistry(
        models=[
            ModelConfig(
                alias="glm-5.2",
                upstream_base_url="https://kubeflow.example/v1",
                upstream_model="glm-5.2-serving",
                api_key_env="KUBEFLOW_API_KEY",
            )
        ]
    )
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=registry,
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
    assert transport.requests[0].headers["authorization"] == "Bearer upstream-secret"
    assert transport.requests[0].url == "https://kubeflow.example/v1/chat/completions"
    assert transport.requests[0].content


@pytest.mark.asyncio
async def test_messages_route_streams_anthropic_sse_from_openai_chunks(monkeypatch) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            text=(
                'data: {"id":"chatcmpl_1","choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert "event: message_stop" in response.text


@pytest.mark.asyncio
async def test_messages_route_logs_stream_usage_when_upstream_provides_usage(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            text=(
                'data: {"id":"chatcmpl_1","choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"prompt_tokens":5,'
                '"completion_tokens":7,"total_tokens":12,"completion_tokens_details":'
                '{"reasoning_tokens":3}}}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "stream": True,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "xhigh"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert '"stream": true' in logs
    assert '"upstream_reasoning_effort": "xhigh"' in logs
    assert '"input_tokens": 5' in logs
    assert '"output_tokens": 7' in logs
    assert '"total_tokens": 12' in logs
    assert '"reasoning_tokens": 3' in logs


@pytest.mark.asyncio
async def test_messages_route_requests_stream_usage_from_openai_compatible_upstream(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            text='data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    upstream_payload = json.loads(transport.requests[0].content)
    assert upstream_payload["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_messages_route_logs_reasoning_chars_when_upstream_usage_lacks_reasoning_tokens(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            text=(
                'data: {"id":"chatcmpl_1","choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning":"think "}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning":"more"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ok"}}],"usage":'
                '{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "stream": True,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert '"input_tokens": 5' in logs
    assert '"output_tokens": 7' in logs
    assert '"total_tokens": 12' in logs
    assert '"reasoning_tokens": null' in logs
    assert '"reasoning_tokens_estimated": 3' in logs
    assert '"reasoning_output_chars": 10' in logs
    assert '"usage_keys": ["completion_tokens", "prompt_tokens", "total_tokens"]' in logs
    assert "think more" not in logs


@pytest.mark.asyncio
async def test_messages_route_keeps_exact_reasoning_tokens_null_when_only_estimate_exists(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            text=(
                'data: {"id":"chatcmpl_1","choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning":"abcdefghijkl"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ok"}}],"usage":'
                '{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={
                        "reasoning": True,
                        "reasoning_format": "vllm",
                    },
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "stream": True,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert '"reasoning_output_chars": 12' in logs
    assert '"reasoning_tokens": null' in logs
    assert '"reasoning_tokens_estimated": 3' in logs
    assert "abcdefghijkl" not in logs


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
async def test_messages_route_preserves_upstream_error_status_and_body(
    monkeypatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status_code, json={"error": {"message": "upstream"}})
            )
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == status_code
    assert response.json() == {"error": {"message": "upstream"}}


@pytest.mark.asyncio
async def test_messages_route_logs_upstream_400_body_summary_without_secrets(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "bad_request",
                            "message": "reasoning.exclude is not supported",
                            "api_key": "upstream-secret",
                        }
                    },
                )
            )
        ),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "secret prompt text"}],
                },
            )

    assert response.status_code == 400
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "claude_proxy.upstream_error" in logs
    assert '"status_code": 400' in logs
    assert '"upstream_path": "/v1/chat/completions"' in logs
    assert '"upstream_error_message": "reasoning.exclude is not supported"' in logs
    assert '"upstream_error_type": "bad_request"' in logs
    assert "secret prompt text" not in logs
    assert "upstream-secret" not in logs


@pytest.mark.asyncio
async def test_messages_route_logs_unknown_model_bad_request(caplog) -> None:
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(models=[]),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "missing-model",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "secret prompt text"}],
                },
            )

    assert response.status_code == 400
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "claude_proxy.bad_request" in logs
    assert '"reason": "unknown_model"' in logs
    assert '"requested_model": "missing-model"' in logs
    assert "secret prompt text" not in logs


@pytest.mark.asyncio
async def test_messages_route_falls_back_unknown_model_when_configured(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    transport = FakeUpstreamTransport(
        httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    app = create_app(
        settings=Settings(
            gateway_auth_token=None,
            unknown_model_fallback_alias="glm-5.2",
        ),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=transport),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-5.3-codex(minimal)",
                    "max_tokens": 128,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    upstream_payload = json.loads(transport.requests[0].content)
    assert upstream_payload["model"] == "glm-5.2-serving"
    assert upstream_payload["reasoning"] == {
        "enabled": True,
        "effort": "high",
        "exclude": True,
    }
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "claude_proxy.model_fallback" in logs
    assert '"requested_model": "gpt-5.3-codex(minimal)"' in logs
    assert '"fallback_model_alias": "glm-5.2"' in logs


@pytest.mark.asyncio
async def test_messages_route_captures_reasoning_request_and_response_without_secrets(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    capture_path = tmp_path / "capture.jsonl"
    app = create_app(
        settings=Settings(gateway_auth_token=None, capture_path=str(capture_path)),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "expose_reasoning": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_reasoning",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "reasoning": "Reasoning visible in fallback.",
                                },
                                "finish_reason": "length",
                            }
                        ],
                    },
                )
            )
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer gateway-secret"},
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "thinking": {"type": "enabled", "budget_tokens": 256},
                "messages": [{"role": "user", "content": "think"}],
            },
        )

    assert response.status_code == 200
    assert "Reasoning visible in fallback." in response.json()["content"][0]["text"]
    lines = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == [
        "anthropic_request",
        "upstream_request",
        "upstream_response",
        "anthropic_response",
    ]
    assert lines[0]["payload"]["thinking"] == {"type": "enabled", "budget_tokens": 256}
    assert lines[1]["payload"]["reasoning"] == {"enabled": True, "max_tokens": 256}
    serialized = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    assert "upstream-secret" not in serialized
    assert "gateway-secret" not in serialized


@pytest.mark.asyncio
async def test_messages_route_suppresses_reasoning_when_model_capability_is_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_payload = json.loads(request.content)
        assert "reasoning" not in upstream_payload
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_reasoning",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "This should stay hidden.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": False},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "thinking": {"type": "adaptive"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": ""}]


@pytest.mark.asyncio
async def test_messages_route_forwards_effort_but_hides_reasoning_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_payload = json.loads(request.content)
        assert upstream_payload["reasoning"] == {
            "enabled": True,
            "effort": "high",
            "exclude": True,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_reasoning",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "This should not be shown.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": ""}]


@pytest.mark.asyncio
async def test_messages_route_forwards_effort_as_vllm_reasoning_fields(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")

    async def handler(request: httpx.Request) -> httpx.Response:
        upstream_payload = json.loads(request.content)
        assert "reasoning" not in upstream_payload
        assert upstream_payload["reasoning_effort"] == "xhigh"
        assert upstream_payload["include_reasoning"] is True
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_reasoning",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning": "Hidden reasoning.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_format": "vllm"},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "xhigh"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert '"upstream_reasoning_format": "vllm"' in logs
    assert '"upstream_reasoning_enabled": true' in logs
    assert '"upstream_reasoning_effort": "xhigh"' in logs
    assert '"upstream_include_reasoning": true' in logs


@pytest.mark.asyncio
async def test_messages_route_logs_reasoning_config_and_usage_without_payload_text(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                    capabilities={"reasoning": True, "reasoning_exclude": True},
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_1",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 17,
                            "total_tokens": 28,
                            "completion_tokens_details": {"reasoning_tokens": 9},
                        },
                    },
                )
            )
        ),
    )

    with caplog.at_level(logging.INFO, logger="claude_proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "glm-5.2",
                    "max_tokens": 128,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "secret prompt text"}],
                },
            )

    assert response.status_code == 200
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "claude_proxy.reasoning_config" in logs
    assert '"claude_effort": "high"' in logs
    assert '"upstream_reasoning_effort": "high"' in logs
    assert '"upstream_reasoning_exclude": true' in logs
    assert "claude_proxy.usage" in logs
    assert '"input_tokens": 11' in logs
    assert '"output_tokens": 17' in logs
    assert '"reasoning_tokens": 9' in logs
    assert "secret prompt text" not in logs
    assert "upstream-secret" not in logs


@pytest.mark.asyncio
async def test_messages_route_captures_headers_and_query_when_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    capture_path = tmp_path / "capture.jsonl"
    app = create_app(
        settings=Settings(
            gateway_auth_token=None,
            capture_path=str(capture_path),
            capture_headers=True,
        ),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_1",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages?beta=true",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Claude-Code-Effort": "high",
                "Anthropic-Beta": "test-beta",
            },
            json={
                "model": "glm-5.2",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    first = json.loads(capture_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["event"] == "anthropic_request"
    assert first["headers"]["authorization"] == "<redacted>"
    assert first["headers"]["x-claude-code-effort"] == "high"
    assert first["headers"]["anthropic-beta"] == "test-beta"
    assert first["query"] == {"beta": "true"}


@pytest.mark.asyncio
async def test_messages_route_rejects_invalid_gateway_token_when_auth_is_configured(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    app = create_app(
        settings=Settings(gateway_auth_token="gateway-secret"),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "glm-5.2", "max_tokens": 128, "messages": []},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_count_tokens_returns_stable_estimate(monkeypatch) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "upstream-secret")
    app = create_app(
        settings=Settings(gateway_auth_token="gateway-secret"),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://kubeflow.example/v1",
                    upstream_model="glm-5.2-serving",
                    api_key_env="KUBEFLOW_API_KEY",
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages/count_tokens",
            headers={"Authorization": "Bearer gateway-secret"},
            json={
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "hello world"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["input_tokens"] >= 1
