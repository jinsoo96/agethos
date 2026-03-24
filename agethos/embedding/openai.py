"""OpenAI 임베딩 어댑터."""

from __future__ import annotations

from agethos.embedding.base import EmbeddingAdapter


class OpenAIEmbedder(EmbeddingAdapter):
    """OpenAI Embeddings API 어댑터."""

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("pip install agethos[openai]") from e
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = self.DIMENSIONS.get(model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension
