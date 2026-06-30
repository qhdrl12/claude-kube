import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from claude_proxy.config import load_dotenv_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def claude_gateway_url() -> str:
    load_dotenv_file(PROJECT_ROOT / ".env")
    if os.getenv("RUN_CLAUDE_CODE_E2E") != "1":
        pytest.skip("RUN_CLAUDE_CODE_E2E=1 is not set")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    if not shutil.which("claude"):
        pytest.skip("claude CLI is not installed")

    process, url = _start_gateway()
    try:
        _wait_for_healthz(url)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_claude_code_core_e2e_scenarios(claude_gateway_url: str, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    input_file = workspace / "input.txt"
    sample_file = workspace / "sample.py"
    input_file.write_text("alpha-123\n", encoding="utf-8")
    sample_file.write_text('print("ok")\n', encoding="utf-8")

    assert "pong" in _normalized_result(
        _claude_result(claude_gateway_url, workspace, "Reply with exactly: pong")
    )
    code = _claude_result(
        claude_gateway_url,
        workspace,
        "Return only a Python function named add that returns the sum of a and b. No markdown.",
        extra_args=["--tools", ""],
    )
    namespace: dict[str, object] = {}
    exec(_extract_python_code(code), namespace)
    assert namespace["add"](2, 3) == 5
    assert "alpha-123" in _normalized_result(
        _claude_result(
            claude_gateway_url,
            workspace,
            f"Read {input_file} using the Read tool, then reply with the file content.",
            extra_args=[f"--add-dir={workspace}", "--allowedTools=Read"],
        )
    )
    assert "edited" in _normalized_result(
        _claude_result(
            claude_gateway_url,
            workspace,
            (
                f"Use the Edit tool to replace alpha-123 with beta-456 in {input_file}. "
                "After editing, reply exactly: edited"
            ),
            extra_args=[f"--add-dir={workspace}", "--allowedTools=Read,Edit"],
        )
    )
    assert input_file.read_text(encoding="utf-8") == "beta-456\n"
    assert "ok" in _normalized_result(
        _claude_result(
            claude_gateway_url,
            workspace,
            f"Use Bash to run python3 {sample_file}. Then reply with the command output.",
            extra_args=[f"--add-dir={workspace}", "--allowedTools=Bash"],
        )
    )


def test_claude_code_effort_reasoning_capture_and_upstream_mapping(tmp_path: Path) -> None:
    load_dotenv_file(PROJECT_ROOT / ".env")
    if os.getenv("RUN_CLAUDE_CODE_REASONING_E2E") != "1":
        pytest.skip("RUN_CLAUDE_CODE_REASONING_E2E=1 is not set")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    if not shutil.which("claude"):
        pytest.skip("claude CLI is not installed")

    capture_path = tmp_path / "reasoning-capture.jsonl"
    registry_path = _write_reasoning_registry(tmp_path)
    process, url = _start_gateway(capture_path=capture_path, registry_path=registry_path)
    try:
        _wait_for_healthz(url)
        results = {
            effort: _claude_result(
                url,
                tmp_path,
                "Think briefly and answer with a one sentence explanation of why 2+2=4.",
                effort=effort,
                disable_adaptive_thinking=False,
            )
            for effort in ("low", "high", "xhigh")
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    assert all(result for result in results.values())
    assert all("Reasoning:" not in result for result in results.values())

    records = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    anthropic_requests = [
        record["payload"] for record in records if record["event"] == "anthropic_request"
    ]
    upstream_requests = [
        record["payload"] for record in records if record["event"] == "upstream_request"
    ]
    assert len(anthropic_requests) == 3
    assert len(upstream_requests) == 3
    assert {request.get("thinking", {}).get("type") for request in anthropic_requests} == {
        "adaptive"
    }
    assert [request.get("output_config", {}).get("effort") for request in anthropic_requests] == [
        "low",
        "high",
        "xhigh",
    ]
    assert [request.get("reasoning", {}).get("effort") for request in upstream_requests] == [
        "low",
        "high",
        "xhigh",
    ]
    assert all(request.get("reasoning", {}).get("enabled") is True for request in upstream_requests)
    assert all(request.get("reasoning", {}).get("exclude") is True for request in upstream_requests)


def _claude_result(
    gateway_url: str,
    cwd: Path,
    prompt: str,
    *,
    extra_args: list[str] | None = None,
    effort: str | None = None,
    disable_adaptive_thinking: bool = True,
) -> str:
    env = os.environ.copy()
    env.update(
        {
            "ANTHROPIC_BASE_URL": gateway_url,
            "ANTHROPIC_AUTH_TOKEN": "dummy-local-token",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.2",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.2",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    if disable_adaptive_thinking:
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
    else:
        env.pop("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", None)
    effort_args = ["--effort", effort] if effort else []
    command = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        "glm-5.2",
        *effort_args,
        "--permission-mode",
        "bypassPermissions",
        *(extra_args or []),
        "--",
        prompt,
    ]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)["result"].strip()


def _normalized_result(value: str) -> str:
    return value.strip().strip("`").strip()


def _extract_python_code(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    if "```" in stripped:
        fenced = stripped.split("```", 2)[1]
        lines = fenced.splitlines()
        if lines and lines[0].strip().lower() == "python":
            lines = lines[1:]
        return "\n".join(lines).strip()
    if "def add" in stripped:
        return stripped[stripped.index("def add") :].strip()
    return stripped


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_reasoning_registry(tmp_path: Path) -> Path:
    registry_path = tmp_path / "models.reasoning.yaml"
    registry_path.write_text(
        """
models:
  - alias: glm-5.2
    upstream_base_url: https://openrouter.ai/api/v1
    upstream_model: ${OPENROUTER_MODEL}
    api_key_env: OPENROUTER_API_KEY
    routing_tier: default
    capabilities:
      streaming: true
      tools: true
      reasoning: true
      reasoning_exclude: true
      expose_reasoning: false
      max_context_tokens: 131072
    extra_headers:
      HTTP-Referer: http://localhost:8000
      X-OpenRouter-Title: claude-proxy-reasoning-e2e
""",
        encoding="utf-8",
    )
    return registry_path


def _start_gateway(
    capture_path: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["CLAUDE_PROXY_REGISTRY_PATH"] = str(registry_path or PROJECT_ROOT / "config/models.yaml")
    if capture_path:
        env["CLAUDE_PROXY_CAPTURE_PATH"] = str(capture_path)
    process = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "uvicorn",
            "claude_proxy.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, url


def _wait_for_healthz(url: str) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("claude-proxy test server did not become healthy")
