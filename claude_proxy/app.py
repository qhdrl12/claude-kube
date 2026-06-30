from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from claude_proxy.adapter import (
    AnthropicStreamBuilder,
    anthropic_messages_to_openai_chat,
    estimate_anthropic_tokens,
    openai_chat_to_anthropic_message,
)
from claude_proxy.config import ModelRegistry, Settings, load_registry, load_settings

logger = logging.getLogger("claude_proxy")


def create_app(
    *,
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    upstream_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    registry = registry or load_registry(settings)
    _configure_logger()
    owns_client = upstream_client is None
    upstream_client = upstream_client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_client:
                await upstream_client.aclose()

    app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.upstream_client = upstream_client

    @app.head("/")
    async def head_root() -> Response:
        return Response(status_code=200)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        return {
            "status": "ready",
            "models": [model.alias for model in registry.models],
            "model_details": [_model_detail(model) for model in registry.models],
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return _model_list_response(registry)

    @app.get("/models")
    async def list_models_shortcut() -> dict[str, Any]:
        return _model_list_response(registry)

    @app.post("/v1/messages")
    async def messages(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        started_at = time.perf_counter()
        _require_gateway_auth(settings, authorization)
        payload = await request.json()
        _capture(
            settings,
            "anthropic_request",
            payload,
            headers=dict(request.headers) if settings.capture_headers else None,
            query=dict(request.query_params) if settings.capture_headers else None,
        )
        try:
            resolution = registry.resolve(
                payload["model"],
                fallback_alias=settings.unknown_model_fallback_alias,
            )
        except KeyError as error:
            _log_bad_request(
                request_id,
                reason="unknown_model",
                requested_model=payload.get("model"),
                detail=str(error),
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        model = resolution.model
        if resolution.fallback_used:
            _log_model_fallback(request_id, resolution)
        upstream_payload = anthropic_messages_to_openai_chat(payload, model)
        _capture(settings, "upstream_request", upstream_payload)
        _log_reasoning_config(request_id, model, payload, upstream_payload)

        if upstream_payload.get("stream"):
            return StreamingResponse(
                _stream_upstream_events(
                    upstream_client,
                    model,
                    upstream_payload,
                    payload["model"],
                    settings,
                    request_id,
                    started_at,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        upstream_response = await upstream_client.post(
            model.upstream_chat_completions_url,
            headers=_upstream_headers(model),
            json=upstream_payload,
        )
        if upstream_response.status_code >= 400:
            _capture(
                settings,
                "upstream_error",
                {
                    "status_code": upstream_response.status_code,
                    "body": _response_body(upstream_response),
                },
            )
            _log_upstream_error(
                request_id,
                model,
                upstream_response.status_code,
                started_at,
                _response_body(upstream_response),
            )
            return _upstream_error_response(upstream_response)
        upstream_body = upstream_response.json()
        _capture(settings, "upstream_response", upstream_body)
        _log_usage(request_id, model, upstream_body.get("usage"), started_at, stream=False)
        anthropic_body = openai_chat_to_anthropic_message(
            upstream_body,
            payload["model"],
            expose_reasoning=bool(model.capabilities.get("expose_reasoning")),
        )
        _capture(settings, "anthropic_response", anthropic_body)
        return JSONResponse(anthropic_body)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, int]:
        _require_gateway_auth(settings, authorization)
        payload = await request.json()
        return {"input_tokens": estimate_anthropic_tokens(payload)}

    return app


async def _stream_upstream_events(
    upstream_client: httpx.AsyncClient,
    model,
    upstream_payload: dict[str, Any],
    requested_model: str,
    settings: Settings,
    request_id: str,
    started_at: float,
) -> AsyncIterator[str]:
    builder = AnthropicStreamBuilder(
        requested_model=requested_model,
        expose_reasoning=bool(model.capabilities.get("expose_reasoning")),
    )
    async with upstream_client.stream(
        "POST",
        model.upstream_chat_completions_url,
        headers=_upstream_headers(model),
        json=upstream_payload,
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            _capture(
                settings,
                "upstream_stream_error",
                {
                    "status_code": response.status_code,
                    "body": body.decode("utf-8", errors="replace"),
                },
            )
            _log_upstream_error(
                request_id,
                model,
                response.status_code,
                started_at,
                _response_body_from_bytes(body),
            )
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "upstream_error",
                        "message": body.decode("utf-8", errors="replace"),
                    },
                },
            )
            return
        stream_usage: dict[str, Any] | None = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                stream_usage = chunk["usage"]
            _capture(settings, "upstream_stream_chunk", chunk)
            for event in builder.consume(chunk):
                _capture(settings, "anthropic_stream_event", event)
                yield _sse(event["event"], event["data"])

    for event in builder.close():
        _capture(settings, "anthropic_stream_event", event)
        yield _sse(event["event"], event["data"])
    _log_usage(
        request_id,
        model,
        stream_usage
        or {
            "prompt_tokens": builder.usage["input_tokens"],
            "completion_tokens": builder.usage["output_tokens"],
        },
        started_at,
        stream=True,
    )


def _require_gateway_auth(settings: Settings, authorization: str | None) -> None:
    if not settings.gateway_auth_token:
        return
    expected = f"Bearer {settings.gateway_auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid gateway token")


def _upstream_headers(model) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {model.upstream_api_key}",
        "Content-Type": "application/json",
    }
    headers.update(model.extra_headers)
    return headers


def _upstream_error_response(response: httpx.Response) -> JSONResponse:
    body = _response_body(response)
    return JSONResponse(status_code=response.status_code, content=body)


def _response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": response.text}


def _response_body_from_bytes(body: bytes) -> Any:
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _model_list_response(registry: ModelRegistry) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [_model_detail(model) for model in registry.models],
    }


def _model_detail(model) -> dict[str, Any]:
    return {
        "id": model.alias,
        "object": "model",
        "owned_by": "claude-proxy",
        "upstream_model": model.upstream_model,
        "upstream_path": _url_path(model.upstream_chat_completions_url),
        "routing_tier": model.routing_tier,
        "capabilities": model.capabilities,
    }


def _configure_logger() -> None:
    logger.setLevel(logging.INFO)
    if any(getattr(handler, "_claude_proxy_handler", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    handler._claude_proxy_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = True


def _log_reasoning_config(
    request_id: str,
    model,
    payload: dict[str, Any],
    upstream_payload: dict[str, Any],
) -> None:
    output_config_raw = payload.get("output_config")
    thinking_raw = payload.get("thinking")
    reasoning_raw = upstream_payload.get("reasoning")
    output_config = output_config_raw if isinstance(output_config_raw, dict) else {}
    thinking = thinking_raw if isinstance(thinking_raw, dict) else {}
    reasoning = reasoning_raw if isinstance(reasoning_raw, dict) else {}
    _log_json(
        "claude_proxy.reasoning_config",
        {
            "request_id": request_id,
            "model_alias": model.alias,
            "upstream_model": model.upstream_model,
            "stream": bool(upstream_payload.get("stream")),
            "claude_thinking_type": thinking.get("type"),
            "claude_effort": output_config.get("effort"),
            "upstream_reasoning_enabled": reasoning.get("enabled", False),
            "upstream_reasoning_effort": reasoning.get("effort"),
            "upstream_reasoning_max_tokens": reasoning.get("max_tokens"),
            "upstream_reasoning_exclude": reasoning.get("exclude", False),
            "expose_reasoning": bool(model.capabilities.get("expose_reasoning")),
        },
    )


def _log_usage(
    request_id: str,
    model,
    usage: Any,
    started_at: float,
    *,
    stream: bool,
) -> None:
    summary = _usage_summary(usage)
    summary.update(
        {
            "request_id": request_id,
            "model_alias": model.alias,
            "upstream_model": model.upstream_model,
            "stream": stream,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
    )
    _log_json("claude_proxy.usage", summary)


def _log_upstream_error(
    request_id: str,
    model,
    status_code: int,
    started_at: float,
    body: Any,
) -> None:
    error_type, error_message = _error_summary(body)
    _log_json(
        "claude_proxy.upstream_error",
        {
            "request_id": request_id,
            "model_alias": model.alias,
            "upstream_model": model.upstream_model,
            "upstream_path": _url_path(model.upstream_chat_completions_url),
            "status_code": status_code,
            "upstream_error_type": error_type,
            "upstream_error_message": error_message,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )


def _log_bad_request(
    request_id: str,
    *,
    reason: str,
    requested_model: Any,
    detail: str,
) -> None:
    _log_json(
        "claude_proxy.bad_request",
        {
            "request_id": request_id,
            "reason": reason,
            "requested_model": requested_model,
            "detail": _truncate_text(detail),
        },
    )


def _url_path(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def _log_model_fallback(request_id: str, resolution) -> None:
    _log_json(
        "claude_proxy.model_fallback",
        {
            "request_id": request_id,
            "requested_model": resolution.requested_alias,
            "fallback_model_alias": resolution.model.alias,
            "upstream_model": resolution.model.upstream_model,
        },
    )


def _error_summary(body: Any) -> tuple[str | None, str | None]:
    sanitized = _sanitize_capture(body)
    if isinstance(sanitized, dict):
        error = sanitized.get("error")
        if isinstance(error, dict):
            return _string_or_none(error.get("type")), _truncate_text(error.get("message"))
        if isinstance(error, str):
            return None, _truncate_text(error)
        return _string_or_none(sanitized.get("type")), _truncate_text(sanitized.get("message"))
    if isinstance(sanitized, str):
        return None, _truncate_text(sanitized)
    return None, _truncate_text(json.dumps(sanitized, ensure_ascii=False))


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _truncate_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _usage_summary(usage: Any) -> dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    completion_details = _first_dict(
        usage.get("completion_tokens_details"),
        usage.get("output_tokens_details"),
        usage.get("completion_details"),
    )
    prompt_details = _first_dict(
        usage.get("prompt_tokens_details"),
        usage.get("input_tokens_details"),
        usage.get("prompt_details"),
    )
    reasoning_tokens = _first_int(
        usage.get("reasoning_tokens"),
        completion_details.get("reasoning_tokens"),
        usage.get("reasoning"),
    )
    return {
        "input_tokens": _first_int(usage.get("prompt_tokens"), usage.get("input_tokens")),
        "output_tokens": _first_int(usage.get("completion_tokens"), usage.get("output_tokens")),
        "total_tokens": _first_int(usage.get("total_tokens")),
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": _first_int(
            prompt_details.get("cached_tokens"),
            prompt_details.get("cached_input_tokens"),
        ),
    }


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _log_json(event: str, payload: dict[str, Any]) -> None:
    _configure_logger()
    logger.info("%s %s", event, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _capture(
    settings: Settings,
    event: str,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
) -> None:
    if not settings.capture_path:
        return
    path = Path(settings.capture_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "payload": _sanitize_capture(payload),
    }
    if headers is not None:
        record["headers"] = _sanitize_capture(_normalize_headers(headers))
    if query is not None:
        record["query"] = _sanitize_capture(query)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sanitize_capture(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if _is_secret_capture_key(key_text):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_capture(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_capture(item) for item in value]
    return value


def _is_secret_capture_key(key: str) -> bool:
    return (
        key == "authorization"
        or key == "token"
        or key.endswith("_token")
        or "api_key" in key
        or "apikey" in key
        or "secret" in key
    )


def _normalize_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}
