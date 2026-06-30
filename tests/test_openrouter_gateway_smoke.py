import os

import httpx
import pytest

from claude_proxy.app import create_app
from claude_proxy.config import ModelConfig, ModelRegistry, Settings, load_dotenv_file


@pytest.mark.asyncio
async def test_gateway_can_call_openrouter_as_openai_compatible_upstream() -> None:
    load_dotenv_file(".env")
    if os.getenv("RUN_OPENROUTER_SMOKE") != "1":
        pytest.skip("RUN_OPENROUTER_SMOKE=1 is not set")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")

    app = create_app(
        settings=Settings(gateway_auth_token=None),
        registry=ModelRegistry(
            models=[
                ModelConfig(
                    alias="glm-5.2",
                    upstream_base_url="https://openrouter.ai/api/v1",
                    upstream_model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5"),
                    api_key_env="OPENROUTER_API_KEY",
                    capabilities={"streaming": True, "tools": True},
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8000",
                        "X-OpenRouter-Title": "claude-proxy-gateway-smoke",
                    },
                )
            ]
        ),
        upstream_client=httpx.AsyncClient(timeout=60),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "glm-5.2",
                "max_tokens": 16,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            },
        )

    assert response.status_code == 200, response.text
    blocks = response.json()["content"]
    assert blocks and blocks[0]["type"] == "text"
