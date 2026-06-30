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

`KUBEFLOW_ENDPOINT` may be either the OpenAI-compatible base URL or the final chat
completions URL:

```bash
# Base URL. The gateway calls <base>/chat/completions.
KUBEFLOW_ENDPOINT=https://<kubeflow-route>/v1

# Final URL. The gateway uses this as-is.
KUBEFLOW_ENDPOINT=https://<kubeflow-route>/v1/chat/completions
```

When each Kubeflow endpoint already points to one served model, `KUBEFLOW_MODEL` can stay
the same as the Claude Code alias, for example `glm-5.2`; the endpoint routing is what
selects the actual model.

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

Confirm the exact upstream path the gateway will call:

```bash
curl -s http://127.0.0.1:8000/readyz
```

`model_details[].upstream_path` should match the Kubeflow OpenAI-compatible route, for
example `/serving/ai-platform/common-model-glm-5-2-fp8-1-0-0/v1/chat/completions`.

## Claude Code Wiring

Point Claude Code at this gateway:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export ANTHROPIC_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

With `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`, Claude Code can populate its
`/model` picker from this gateway's `/v1/models` response. If discovery is not enabled,
Claude Code may only show its built-in model choices even though `/readyz` and `/v1/models`
list `glm-5.2`.

Run Claude Code after setting the exports:

```bash
claude --model glm-5.2
```

Or start Claude Code without `--model` and pick the gateway alias from `/model`.

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
      export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
      export ANTHROPIC_MODEL=glm-5.2
      export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
      export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
      export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2
      export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
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

## One-command Claude Code Launcher

If you switch between Bedrock, Copilot, and this local gateway often, prefer a wrapper
script instead of exporting `ANTHROPIC_BASE_URL` in your interactive shell. The wrapper
starts the gateway when needed, waits for `/readyz`, injects Claude Code environment
variables only into the child `claude` process, and leaves your current shell unchanged.

Create `~/bin/claude-kube`:

```bash
mkdir -p ~/bin
$EDITOR ~/bin/claude-kube
chmod +x ~/bin/claude-kube
```

Paste this script:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAUDE_KUBE_ROOT:-$HOME/Project/claude-proxy}"
HOST="${CLAUDE_KUBE_HOST:-127.0.0.1}"
PORT="${CLAUDE_KUBE_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
MODEL="${CLAUDE_KUBE_MODEL:-glm-5.2}"
LOG_FILE="${CLAUDE_KUBE_LOG_FILE:-$ROOT/tmp/claude-kube.log}"

mkdir -p "$ROOT/tmp"

if ! curl -fsS "$BASE_URL/readyz" >/dev/null 2>&1; then
  echo "Starting claude-kube gateway at $BASE_URL ..."
  (
    cd "$ROOT"
    uv run uvicorn claude_proxy.main:app --host "$HOST" --port "$PORT"
  ) >"$LOG_FILE" 2>&1 &

  for _ in {1..60}; do
    if curl -fsS "$BASE_URL/readyz" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
fi

curl -fsS "$BASE_URL/readyz" >/dev/null

export ANTHROPIC_BASE_URL="$BASE_URL"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
unset ANTHROPIC_AUTH_TOKEN

exec claude --model "$MODEL" "$@"
```

Use it like this:

```bash
claude-kube
claude-kube -p "Reply with pong"
CLAUDE_KUBE_MODEL=glm-5.2 claude-kube --effort high
CLAUDE_KUBE_PORT=8001 claude-kube
```

Useful tips:

- Add `~/bin` to `PATH` if your shell cannot find `claude-kube`:
  `export PATH="$HOME/bin:$PATH"`.
- The gateway log is written to `tmp/claude-kube.log` by default.
- The script reuses an already-running gateway if `/readyz` succeeds.
- The script does not stop the gateway when Claude Code exits. This keeps the next
  Claude Code startup fast. Stop it manually with `pkill -f 'uvicorn claude_proxy.main:app'`
  if you want a clean shutdown.
- If you edited `.env`, restart the existing gateway before testing the new values.
- If port `8000` is already occupied by another service, run
  `CLAUDE_KUBE_PORT=8001 claude-kube`.
- If `/model` does not show `glm-5.2`, confirm:
  `curl -s http://127.0.0.1:8000/v1/models`.
- For effort comparison, run the same prompt with `--effort low`, `--effort high`, and
  `--effort xhigh`, then compare `claude_proxy.reasoning_config` and
  `claude_proxy.usage` in the gateway log.
- Keep shared Kubeflow secrets in `.env`; do not put API keys in the wrapper script.

Use `claude-switch kube` when you want to change the current shell profile for multiple
commands. Use `claude-kube` when you want one isolated Claude Code session without
touching your Bedrock or Copilot shell state.

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
      reasoning_format: vllm
      expose_reasoning: false
```

Additional models are added as more `models[]` entries. Claude Code keeps using the
`alias`; the upstream model name and endpoint stay behind the registry.

For Kubeflow/vLLM, keep `reasoning_format: vllm`. The gateway maps Claude Code effort to
vLLM's top-level `reasoning_effort`, sends `include_reasoning: true` so reasoning can be
observed by the gateway, and still hides reasoning from Claude Code when
`expose_reasoning: false`.

The vLLM server must also be started with the reasoning support required by the served
model, such as the appropriate `--reasoning-parser` or model-specific chat template
thinking configuration. If vLLM is not reasoning-enabled, it may accept the request but
answer immediately without `delta.reasoning`.

For GLM reasoning models, configure the Kubeflow/vLLM serving command with the GLM parser
used by your vLLM version. For GLM-4.5 style reasoning output this is commonly:

```bash
vllm serve <glm-model> --reasoning-parser glm45
```

In Kubeflow, the same flag should be present in the container args for the model server.
Without this server-side parser, the gateway can still send `reasoning_effort`, but vLLM
may return only normal `content` and no `delta.reasoning`.

After enabling the parser, check the gateway logs:

```text
claude_proxy.reasoning_config ... "upstream_reasoning_format": "vllm" ... "upstream_reasoning_effort": "high" ... "upstream_include_reasoning": true
claude_proxy.usage ... "reasoning_tokens": 7144 ...
```

`reasoning_tokens` greater than zero means vLLM emitted reasoning through
`delta.reasoning` or legacy `delta.reasoning_content` and the gateway observed it. If
vLLM also includes an exact reasoning token count in the usage object, that upstream
usage value is preserved. Otherwise the gateway counts non-empty reasoning deltas as
reasoning tokens, matching vLLM setups that stream reasoning one token per delta.

The gateway reads vLLM-separated reasoning fields instead of parsing `<think>...</think>`
as the primary path. It prefers the current `reasoning` field and falls back to legacy
`reasoning_content` for older vLLM-compatible responses. If a parser bug leaks
`<think>...</think>` into visible `content`, the gateway strips that leaked block as a
defensive fallback.

If a served model requires extra vLLM request fields, add them per model:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

Use this only when the specific model/server expects those fields. The gateway also sends
`reasoning_effort` for Claude Code `/effort` values when `reasoning_format: vllm`.

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
claude_proxy.reasoning_config {"claude_effort": "high", "claude_thinking_type": "adaptive", "expose_reasoning": false, "model_alias": "glm-5.2", "stream": true, "upstream_include_reasoning": true, "upstream_model": "...", "upstream_reasoning_effort": "high", "upstream_reasoning_enabled": true, "upstream_reasoning_exclude": false, "upstream_reasoning_format": "vllm"}
```

Usage log:

```text
claude_proxy.usage {"elapsed_ms": 1240.5, "input_tokens": 1800, "output_tokens": 420, "reasoning_tokens": 96, "stream": true, "total_tokens": 2220}
```

Use these two lines to compare `--effort low`, `--effort high`, and `--effort xhigh`.
When the upstream usage object reports `reasoning_tokens`, that exact value is logged. If
streaming vLLM omits the usage field but emits reasoning text, the gateway counts each
non-empty `delta.reasoning` or legacy `delta.reasoning_content` as one
`reasoning_tokens` unit. If no reasoning deltas are observed and upstream usage does not
report reasoning tokens, `reasoning_tokens` remains `null`.

For debugging upstream stream shapes, enable sanitized per-chunk logs:

```bash
export CLAUDE_PROXY_LOG_UPSTREAM_STREAM_CHUNKS=true
uv run uvicorn claude_proxy.main:app
```

This writes `claude_proxy.upstream_stream_chunk` lines with chunk keys, delta keys,
reasoning/content character counts, tool call counts, and the upstream `usage` object.
It does not log prompt text, visible content, reasoning text, tool arguments, API keys, or
authorization headers.

If you need full raw upstream stream chunks for local-only debugging, write them to a
dedicated JSONL file:

```bash
export CLAUDE_PROXY_RAW_UPSTREAM_STREAM_PATH=tmp/upstream-stream-raw.jsonl
uv run uvicorn claude_proxy.main:app
```

Each line contains the raw upstream chunk exactly as the gateway received it, plus
metadata such as `request_id`, `model_alias`, and `chunk_index`. This is the best option
when you need to inspect whether vLLM sends reasoning token fields under a different key.

The broader capture file records all gateway stages, including requests and converted
responses:

```bash
export CLAUDE_PROXY_CAPTURE_PATH=tmp/claude-proxy-capture.jsonl
```

Raw stream and capture files may contain model output text, hidden reasoning, and tool
payloads, so do not share them without reviewing/redacting them first.

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

For upstream 404s, check `upstream_path` in the log and `/readyz`. If `.env` was edited
but the path did not change, restart the gateway and check whether an existing shell
export is overriding `.env`:

```bash
env | grep '^KUBEFLOW_'
```
