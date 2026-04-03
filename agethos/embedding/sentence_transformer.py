"""SentenceTransformer 임베딩 어댑터 — 로컬/오픈소스 모델 지원.

Usage::

    from agethos.embedding.sentence_transformer import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()  # 기본: all-MiniLM-L6-v2
    embedder = SentenceTransformerEmbedder(model="intfloat/multilingual-e5-small")

    brain = Brain(persona=spec, llm=llm, embedder=embedder)
"""

from __future__ import annotations

from agethos.embedding.base import EmbeddingAdapter


class SentenceTransformerEmbedder(EmbeddingAdapter):
    """sentence-transformers 기반 로컬 임베딩."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e

        self._model = SentenceTransformer(model, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
