"""Ollama 임베딩 어댑터 — 로컬 Ollama 서버 기반 임베딩.

Usage::

    from agethos.embedding.ollama import OllamaEmbedder

    embedder = OllamaEmbedder()  # 기본: nomic-embed-text
    embedder = OllamaEmbedder(model="mxbai-embed-large", base_url="http://localhost:11434")

    brain = Brain(persona=spec, llm=llm, embedder=embedder)
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from agethos.embedding.base import EmbeddingAdapter


class OllamaEmbedder(EmbeddingAdapter):
    """Ollama API 기반 로컬 임베딩."""

    DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimension = self.DIMENSIONS.get(model, 768)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            data = json.dumps({"model": self._model, "input": text}).encode()
            req = Request(
                f"{self._base_url}/api/embed",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req) as resp:
                body = json.loads(resp.read())
            embeddings = body.get("embeddings", [])
            if embeddings:
                results.append(embeddings[0])
                if self._dimension != len(embeddings[0]):
                    self._dimension = len(embeddings[0])
            else:
                results.append([0.0] * self._dimension)
        return results

    @property
    def dimension(self) -> int:
        return self._dimension
