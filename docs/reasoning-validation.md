# Reasoning and Effort Validation

## Scope

This gateway does not synthesize Anthropic native `thinking` blocks from upstream
OpenAI-compatible reasoning fields. Anthropic thinking streams include signed thinking
metadata, so fabricating that shape would be more likely to break Claude Code than help.

Instead, the gateway uses this policy:

- Forward Claude Code `thinking` to upstream `reasoning` only when the model registry has
  `capabilities.reasoning: true`.
- Convert Anthropic-style `thinking: {"type": "enabled", "budget_tokens": N}` to
  OpenRouter-compatible `reasoning: {"enabled": true, "max_tokens": N}`.
- Convert `thinking: {"type": "adaptive"}` to `reasoning: {"enabled": true}`.
- Convert Claude Code `output_config: {"effort": "low|high|xhigh|..."}` to upstream
  `reasoning.effort` when no explicit reasoning token budget is present.
- Add `reasoning.exclude: true` when the model registry has
  `capabilities.reasoning_exclude: true`, so the upstream may spend reasoning tokens
  without returning reasoning text.
- Prefer visible assistant `content` over upstream `reasoning` when both are present.
- If `capabilities.reasoning: false`, do not send upstream `reasoning` and do not expose
  upstream reasoning fields in Claude Code output.
- If `capabilities.expose_reasoning: false`, do not expose upstream reasoning fields in
  Claude Code output.
- If `capabilities.expose_reasoning: true` and upstream returns reasoning-only output,
  render it as a visible text fallback: `Reasoning:\n...`
- If upstream streams reasoning before answer content, close the visible reasoning
  fallback text block before starting the answer text block. This prevents the first
  answer tokens from being appended to the `Reasoning:` area.

Claude Code can visually collapse native Anthropic thinking because it is a distinct
`thinking` content block in the Anthropic protocol. OpenAI-compatible upstream
`reasoning` fields are not the same thing, and this gateway does not fabricate signed
Anthropic thinking blocks from those fields.

## Verification Stages

1. Unit contract:
   - `tests/test_adapter.py` verifies non-stream reasoning-only fallback.
   - `tests/test_adapter.py` verifies streaming `delta.reasoning` fallback.
   - `tests/test_adapter.py` verifies reasoning fallback and answer text are separate
     streaming text blocks.
   - `tests/test_adapter.py` verifies reasoning fallback can be disabled.
   - `tests/test_adapter.py` verifies Claude Code `output_config.effort` maps to
     upstream `reasoning.effort`.
   - `tests/test_adapter.py` verifies Anthropic `thinking` to OpenRouter `reasoning`.

2. Route capture:
   - `tests/test_app.py` verifies `CLAUDE_PROXY_CAPTURE_PATH` writes JSONL events for
     `anthropic_request`, `upstream_request`, `upstream_response`, and
     `anthropic_response`.
   - Capture redacts secret-like keys while preserving `budget_tokens`.

3. OpenRouter smoke:
   - `RUN_OPENROUTER_SMOKE=1 .venv/bin/python -m pytest tests/test_openrouter_smoke.py tests/test_openrouter_gateway_smoke.py`
   - This verifies direct OpenRouter shape and gateway-through-OpenRouter shape.

4. Claude Code core E2E:
   - `RUN_CLAUDE_CODE_E2E=1 .venv/bin/python -m pytest tests/test_claude_code_e2e.py::test_claude_code_core_e2e_scenarios`
   - This verifies chat, code generation, Read, Edit, and Bash still work when reasoning
     fallback is enabled.

5. Claude Code effort/reasoning E2E:
   - `RUN_CLAUDE_CODE_REASONING_E2E=1 .venv/bin/python -m pytest tests/test_claude_code_e2e.py::test_claude_code_effort_reasoning_capture_and_upstream_mapping`
   - This starts a temporary gateway with capture enabled and runs Claude Code with
     `--effort low`, `--effort high`, and `--effort xhigh`.
   - It verifies captured Claude Code `output_config.effort` values and upstream
     `reasoning.effort` values are distinct and ordered as low/high/xhigh.

6. Header/query capture:
   - Enable `CLAUDE_PROXY_CAPTURE_HEADERS=true` together with
     `CLAUDE_PROXY_CAPTURE_PATH`.
   - The gateway captures masked inbound headers and query params on
     `anthropic_request` events so effort-related transport metadata can be compared.

7. Runtime logs:
   - The `claude_proxy.reasoning_config` INFO log records Claude Code effort, thinking
     type, upstream reasoning effort, `exclude`, and whether reasoning is exposed.
   - The `claude_proxy.usage` INFO log records elapsed milliseconds, input tokens, output
     tokens, total tokens, cached input tokens when available, and reasoning tokens when
     the upstream reports them.
   - Prompt text, tool arguments, API keys, and authorization headers are not included in
     these logs.

## Observed OpenRouter Result

Using the current local `.env` model (`OPENROUTER_MODEL`) during validation:

- Claude Code sent `thinking: {"type": "adaptive"}` for `--effort low`, `--effort high`,
  `--effort xhigh`, and `--effort max`.
- Claude Code also sent `output_config.effort` with the selected effort value.
- The gateway maps those values to upstream `reasoning.effort`.
- Server logs show the same mapping in `claude_proxy.reasoning_config`, and token usage
  in `claude_proxy.usage`.
- Header/query capture showed the same query params for all four requests:
  `{"beta": "true"}`.
- Header/query capture showed no explicit low/high/xhigh/max value in request headers.
  The only effort-related transport signal was the common beta flag
  `effort-2025-11-24` inside `anthropic-beta`.
- The selected OpenRouter model can return reasoning-only output or reasoning before
  answer text.
- With `capabilities.expose_reasoning: false`, the gateway suppresses upstream reasoning
  output and avoids `Reasoning:` noise in normal Claude Code use.

This means `--effort` is observable in the Claude Code request body as
`output_config.effort` and can be forwarded to an OpenAI-compatible upstream as
`reasoning.effort`. Re-run the effort E2E after changing Claude Code versions or
upstream models.

## Kubeflow/vLLM Follow-up

When switching to Kubeflow/vLLM:

- Set `capabilities.reasoning: true` only if the specific vLLM route supports a compatible
  `reasoning` request body.
- Keep `capabilities.expose_reasoning: false` unless debugging visible reasoning output.
- If the route supports hidden reasoning output, set `capabilities.reasoning_exclude: true`.
- Rerun the effort E2E and verify `low`, `high`, and `xhigh` are captured as distinct
  `upstream_request.reasoning.effort` values.
- Compare captured `upstream_request.reasoning`, upstream stream chunk keys, and visible
  Claude Code output.
