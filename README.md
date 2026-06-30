# claude-proxy

FastAPI gateway that exposes a Claude Code-facing Anthropic Messages API and forwards
requests to OpenAI-compatible upstreams such as OpenRouter or Kubeflow/vLLM.

## Runtime

This project is pinned to Python 3.13.

Recommended `uv` setup:

```bash
uv venv --python 3.13
uv pip install -e '.[dev]'
```

If you need to reset the local environment:

```bash
rm -rf .venv
uv venv --python 3.13
uv pip install -e '.[dev]'
```

Equivalent standard `venv` setup:

```bash
/Users/bongkilee/.local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Kubeflow/vLLM Setup

`config/models.yaml` defaults the Claude Code model alias `glm-5.2` to a Kubeflow/vLLM
OpenAI-compatible endpoint. The default config is driven by three environment values:
`KUBEFLOW_ENDPOINT`, `KUBEFLOW_MODEL`, and `KUBEFLOW_API_KEY`.

## API Key Setup

Create a local `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit this line in `.env`:

```bash
KUBEFLOW_ENDPOINT=https://<kubeflow-route>/v1
KUBEFLOW_MODEL=glm-5.2
KUBEFLOW_API_KEY=<shared-internal-key>
```

For local-only use, remove this line if it exists in your `.env`:

```bash
CLAUDE_PROXY_GATEWAY_AUTH_TOKEN=...
```

The gateway reads `.env` automatically at startup. The same key can also be passed as a
shell environment variable if preferred:

```bash
export KUBEFLOW_ENDPOINT=https://<kubeflow-route>/v1
export KUBEFLOW_MODEL=glm-5.2
export KUBEFLOW_API_KEY=<shared-internal-key>
```

If any `${KUBEFLOW_...}` placeholder in `config/models.yaml` is not resolved, the gateway
fails at startup with the missing variable name.

```bash
export CLAUDE_PROXY_REGISTRY_PATH=config/models.yaml

uv run uvicorn claude_proxy.main:app
```

The direct `.venv` command is equivalent:

```bash
.venv/bin/uvicorn claude_proxy.main:app
```

Uvicorn defaults to `127.0.0.1:8000`. Pass `--host` or `--port` only when you need
different binding:

```bash
uv run uvicorn claude_proxy.main:app --host 127.0.0.1 --port 8001
```

In another shell:

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}]
  }'
```

List registered gateway model aliases:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## Claude Code Wiring

Point Claude Code at this gateway:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

`CLAUDE_PROXY_GATEWAY_AUTH_TOKEN` is optional. Leave it unset for local-only use.
If the gateway is exposed beyond localhost, set `CLAUDE_PROXY_GATEWAY_AUTH_TOKEN` on
the server and set Claude Code's `ANTHROPIC_AUTH_TOKEN` to the same value.

Claude Code may send internal model labels such as `gpt-5.3-codex(minimal)` for some
auxiliary requests. Unknown-model fallback is disabled by default. If you want those
internal labels to route to a known alias, explicitly set:

```bash
export CLAUDE_PROXY_UNKNOWN_MODEL_FALLBACK_ALIAS=glm-5.2
```

If your local shell already has `claude-switch bedrock` and `claude-switch copilot`,
add a kube profile with the same pattern:

```bash
claude-switch() {
  case "$1" in
    kube)
      export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
      export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
      export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
      export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2
      unset ANTHROPIC_AUTH_TOKEN
      ;;
    kube-list)
      curl -s "${ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}/v1/models"
      ;;
    bedrock)
      # existing bedrock exports
      ;;
    copilot)
      # existing copilot exports
      ;;
  esac
}
```

## Model Registry

```yaml
models:
  - alias: glm-5.2
    upstream_base_url: ${KUBEFLOW_ENDPOINT}
    upstream_model: ${KUBEFLOW_MODEL}
    api_key_env: KUBEFLOW_API_KEY
    routing_tier: default
    capabilities:
      streaming: true
      tools: true
      reasoning: true
      reasoning_exclude: true
      expose_reasoning: false
```

Additional models are added as more `models[]` entries. Claude Code keeps using the
`alias`; the upstream model name and endpoint stay behind the registry.

## Verification

Fast local tests:

```bash
uv run pytest
```

The direct `.venv` command is equivalent:

```bash
.venv/bin/python -m pytest
```

Optional real OpenRouter smoke, useful before internal Kubeflow access is available:

```bash
RUN_OPENROUTER_SMOKE=1 \
OPENROUTER_API_KEY=sk-or-v1-... \
OPENROUTER_MODEL=z-ai/glm-4.5 \
.venv/bin/python -m pytest tests/test_openrouter_smoke.py tests/test_openrouter_gateway_smoke.py
```

Optional Claude Code E2E against OpenRouter:

```bash
RUN_CLAUDE_CODE_E2E=1 .venv/bin/python -m pytest tests/test_claude_code_e2e.py
```

The Claude Code E2E test starts a temporary local gateway, points `claude -p` at it,
and verifies general chat, code generation, file read, file edit, and Bash execution.

Optional Claude Code effort/reasoning E2E:

```bash
RUN_CLAUDE_CODE_REASONING_E2E=1 .venv/bin/python -m pytest tests/test_claude_code_e2e.py::test_claude_code_effort_reasoning_capture_and_upstream_mapping
```

This starts a temporary gateway with `CLAUDE_PROXY_CAPTURE_PATH`, runs Claude Code with
`--effort low`, `--effort high`, and `--effort xhigh`, and verifies:

- Claude Code sends `thinking: {"type": "adaptive"}` for the captured requests.
- Claude Code sends `output_config.effort` as `low`, `high`, and `xhigh`.
- The gateway maps adaptive thinking and effort to upstream
  `reasoning: {"enabled": true, "effort": "...", "exclude": true}` when the model
  registry capability has `reasoning: true` and `reasoning_exclude: true`.
- Upstream reasoning fields are hidden from Claude Code output by default when
  `expose_reasoning: false`.
- If upstream streams reasoning before answer text, the gateway closes the visible
  reasoning fallback block before starting the answer text block. This only matters when
  `expose_reasoning: true`.

See [docs/reasoning-validation.md](docs/reasoning-validation.md) for the detailed
reasoning policy, capture stages, and observed effort behavior.

Use `capabilities.reasoning: true` when the upstream endpoint supports OpenAI-compatible
reasoning and you want Claude Code `--effort` to affect the upstream call. Keep
`expose_reasoning: false` for normal Claude Code use so reasoning is applied but not
shown as `Reasoning:\n...`.

## Runtime Logs

The gateway writes structured INFO logs through the `claude_proxy` logger. The logs do
not include prompt text, tool arguments, API keys, or authorization headers.

Reasoning application log:

```text
claude_proxy.reasoning_config {"claude_effort": "high", "claude_thinking_type": "adaptive", "expose_reasoning": false, "model_alias": "glm-5.2", "stream": true, "upstream_model": "...", "upstream_reasoning_effort": "high", "upstream_reasoning_enabled": true, "upstream_reasoning_exclude": true}
```

Usage log:

```text
claude_proxy.usage {"elapsed_ms": 1240.5, "input_tokens": 1800, "output_tokens": 420, "reasoning_tokens": 96, "stream": true, "total_tokens": 2220}
```

Use these two lines to compare `--effort low`, `--effort high`, and `--effort xhigh`.
Some upstreams do not report `reasoning_tokens`; in that case the field is logged as
`null`, but `upstream_reasoning_effort` still confirms what the gateway sent.

400 diagnostics:

```text
claude_proxy.model_fallback {"requested_model": "gpt-5.3-codex(minimal)", "fallback_model_alias": "glm-5.2", ...}
claude_proxy.bad_request {"reason": "unknown_model", "requested_model": "missing-model", ...}
claude_proxy.upstream_error {"status_code": 400, "upstream_error_type": "bad_request", "upstream_error_message": "...", ...}
```

`model_fallback` means the gateway accepted an unknown Claude Code model label and routed
it to the configured fallback alias. `bad_request` means the gateway rejected the request
before calling upstream, usually because no fallback was available. `upstream_error`
means the upstream endpoint returned the error and the gateway preserved that status for
Claude Code.
