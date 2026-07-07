"""OpenAI-compatible embedding providers for retrieval artifacts."""

from __future__ import annotations

from typing import Any, Protocol
import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key_env: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing openai package. Install it with: pip install openai") from exc

        self.model = model
        self.dimension = dimension
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension,
        )
        return [list(item.embedding) for item in result.data]


class OpenRouterEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """OpenRouter embeddings provider using its OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = OPENROUTER_BASE_URL,
        app_title: str | None = None,
        http_referer_env: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if app_title:
            headers["X-Title"] = app_title

        referer = os.getenv(http_referer_env) if http_referer_env else None
        if referer:
            headers["HTTP-Referer"] = referer

        super().__init__(
            model=model,
            dimension=dimension,
            api_key_env=api_key_env,
            base_url=base_url,
            default_headers=headers or None,
        )
