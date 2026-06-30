from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ModelConfig(BaseModel):
    alias: str
    upstream_base_url: str
    upstream_model: str
    api_key_env: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    routing_tier: str = "default"
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @property
    def upstream_api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise RuntimeError(f"Missing upstream API key env var: {self.api_key_env}")
        return value

    @property
    def upstream_chat_completions_url(self) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/chat/completions"


class ModelResolution(BaseModel):
    requested_alias: str
    model: ModelConfig
    fallback_used: bool = False


class ModelRegistry(BaseModel):
    models: list[ModelConfig] = Field(default_factory=list)

    def get(self, alias: str) -> ModelConfig:
        for model in self.models:
            if model.alias == alias:
                return model
        aliases = ", ".join(sorted(model.alias for model in self.models)) or "<none>"
        raise KeyError(f"Unknown model alias: {alias}. Available aliases: {aliases}")

    def resolve(
        self,
        alias: str,
        *,
        fallback_alias: str | None = None,
    ) -> ModelResolution:
        try:
            return ModelResolution(requested_alias=alias, model=self.get(alias))
        except KeyError:
            fallback = self._fallback_model(fallback_alias)
            if fallback:
                return ModelResolution(
                    requested_alias=alias,
                    model=fallback,
                    fallback_used=True,
                )
            raise

    def _fallback_model(self, fallback_alias: str | None) -> ModelConfig | None:
        if fallback_alias:
            return self.get(fallback_alias)
        return None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelRegistry:
        content = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
        unresolved = sorted(set(_UNRESOLVED_ENV_PATTERN.findall(content)))
        if unresolved:
            variables = ", ".join(unresolved)
            raise ValueError(f"Unresolved environment placeholders in {path}: {variables}")
        raw = yaml.safe_load(content) or {}
        return cls.model_validate(raw)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_PROXY_", env_file=".env", extra="ignore")

    gateway_auth_token: str | None = None
    registry_path: str = "config/models.yaml"
    request_timeout_seconds: float = 120.0
    service_name: str = "claude-proxy"
    capture_path: str | None = None
    capture_headers: bool = False
    unknown_model_fallback_alias: str | None = None


def load_settings() -> Settings:
    load_dotenv_file(".env")
    return Settings()


def load_registry(settings: Settings) -> ModelRegistry:
    return ModelRegistry.from_yaml(settings.registry_path)


def load_dotenv_file(path: str | Path) -> None:
    load_dotenv(dotenv_path=path, override=False)
