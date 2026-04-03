from agethos.embedding.base import EmbeddingAdapter

__all__ = ["EmbeddingAdapter"]


def resolve_embedder(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingAdapter:
    """임베딩 어댑터를 프로바이더 이름으로 생성.

    Args:
        provider: "openai", "ollama", "sentence-transformer" 등.
        model: 모델명 오버라이드.
        api_key: API 키 (openai 전용).
        base_url: 커스텀 API URL (ollama 전용).

    Usage::

        embedder = resolve_embedder("openai")
        embedder = resolve_embedder("ollama", model="nomic-embed-text")
        embedder = resolve_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
    """
    provider = provider.lower().replace("_", "-")

    if provider == "openai":
        from agethos.embedding.openai import OpenAIEmbedder
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIEmbedder(**kwargs)

    elif provider == "ollama":
        from agethos.embedding.ollama import OllamaEmbedder
        kwargs = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbedder(**kwargs)

    elif provider in ("sentence-transformer", "sentence-transformers", "sbert"):
        from agethos.embedding.sentence_transformer import SentenceTransformerEmbedder
        kwargs = {}
        if model:
            kwargs["model"] = model
        return SentenceTransformerEmbedder(**kwargs)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider!r}. "
            "Supported: 'openai', 'ollama', 'sentence-transformer'"
        )
