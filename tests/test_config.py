import pytest

from claude_proxy.config import ModelConfig, ModelRegistry, load_dotenv_file


def test_model_registry_resolves_alias_and_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("KUBEFLOW_API_KEY", "shared-secret")
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

    model = registry.get("glm-5.2")

    assert model.upstream_api_key == "shared-secret"
    assert model.upstream_chat_completions_url == "https://kubeflow.example/v1/chat/completions"


def test_model_config_accepts_final_chat_completions_endpoint() -> None:
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://kubeflow.example/v1/chat/completions",
        upstream_model="glm-5.2-serving",
        api_key_env="KUBEFLOW_API_KEY",
    )

    assert model.upstream_chat_completions_url == "https://kubeflow.example/v1/chat/completions"


def test_model_registry_resolves_api_key_from_dotenv_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-v1-test\n", encoding="utf-8")

    load_dotenv_file(env_file)
    model = ModelConfig(
        alias="glm-5.2",
        upstream_base_url="https://openrouter.ai/api/v1",
        upstream_model="z-ai/glm-4.5",
        api_key_env="OPENROUTER_API_KEY",
    )

    assert model.upstream_api_key == "sk-or-v1-test"


def test_model_registry_expands_environment_placeholders_from_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")
    registry_file = tmp_path / "models.yaml"
    registry_file.write_text(
        """
models:
  - alias: glm-5.2
    upstream_base_url: https://openrouter.ai/api/v1
    upstream_model: ${OPENROUTER_MODEL}
    api_key_env: OPENROUTER_API_KEY
""",
        encoding="utf-8",
    )

    registry = ModelRegistry.from_yaml(registry_file)

    assert registry.get("glm-5.2").upstream_model == "openrouter/test-model"


def test_model_registry_rejects_unresolved_environment_placeholders(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KUBEFLOW_ENDPOINT", raising=False)
    registry_file = tmp_path / "models.yaml"
    registry_file.write_text(
        """
models:
  - alias: glm-5.2
    upstream_base_url: ${KUBEFLOW_ENDPOINT}
    upstream_model: glm-5.2
    api_key_env: KUBEFLOW_API_KEY
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unresolved environment placeholders"):
        ModelRegistry.from_yaml(registry_file)


def test_model_registry_rejects_unknown_model() -> None:
    registry = ModelRegistry(models=[])

    with pytest.raises(KeyError, match="Unknown model alias"):
        registry.get("missing")


def test_model_registry_rejects_unknown_alias_by_default_even_with_one_model() -> None:
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

    with pytest.raises(KeyError, match="Unknown model alias"):
        registry.resolve("gpt-5.3-codex(minimal)")


def test_model_registry_falls_back_when_fallback_alias_is_configured() -> None:
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

    resolved = registry.resolve("gpt-5.3-codex(minimal)", fallback_alias="glm-5.2")

    assert resolved.model.alias == "glm-5.2"
    assert resolved.requested_alias == "gpt-5.3-codex(minimal)"
    assert resolved.fallback_used is True


def test_model_registry_uses_configured_fallback_alias_when_multiple_models() -> None:
    registry = ModelRegistry(
        models=[
            ModelConfig(
                alias="glm-5.2",
                upstream_base_url="https://kubeflow.example/v1",
                upstream_model="glm-5.2-serving",
                api_key_env="KUBEFLOW_API_KEY",
            ),
            ModelConfig(
                alias="other-model",
                upstream_base_url="https://kubeflow.example/v1",
                upstream_model="other-serving",
                api_key_env="KUBEFLOW_API_KEY",
            ),
        ]
    )

    resolved = registry.resolve("gpt-5.3-codex(minimal)", fallback_alias="glm-5.2")

    assert resolved.model.alias == "glm-5.2"
    assert resolved.fallback_used is True


def test_model_registry_rejects_unknown_alias_when_multiple_models_without_fallback() -> None:
    registry = ModelRegistry(
        models=[
            ModelConfig(
                alias="glm-5.2",
                upstream_base_url="https://kubeflow.example/v1",
                upstream_model="glm-5.2-serving",
                api_key_env="KUBEFLOW_API_KEY",
            ),
            ModelConfig(
                alias="other-model",
                upstream_base_url="https://kubeflow.example/v1",
                upstream_model="other-serving",
                api_key_env="KUBEFLOW_API_KEY",
            ),
        ]
    )

    with pytest.raises(KeyError, match="Unknown model alias"):
        registry.resolve("gpt-5.3-codex(minimal)")
