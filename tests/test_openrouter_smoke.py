import os

import httpx
import pytest

from claude_proxy.config import load_dotenv_file


@pytest.mark.asyncio
async def test_openrouter_openai_compatible_chat_completions_smoke() -> None:
    load_dotenv_file(".env")
    if os.getenv("RUN_OPENROUTER_SMOKE") != "1":
        pytest.skip("RUN_OPENROUTER_SMOKE=1 is not set")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-OpenRouter-Title": "claude-proxy-smoke",
            },
            json={
                "model": os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5"),
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"]
    assert body["choices"]
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message.get("content") is not None or message.get("reasoning")
