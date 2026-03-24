"""Brain — the agent's 'brain' facade.

Unifies persona, memory, and cognition as a single entry point.
"""

from __future__ import annotations

from agethos.cognition.perceive import Perceiver
from agethos.cognition.plan import Planner
from agethos.cognition.reflect import Reflector
from agethos.cognition.retrieve import Retriever
from agethos.embedding.base import EmbeddingAdapter
from agethos.llm.base import LLMAdapter
from agethos.memory.store import StorageBackend
from agethos.memory.stream import MemoryStream
from agethos.models import DailyPlan, EmotionalState, MemoryNode, NodeType, PersonaSpec, RetrievalResult
from agethos.persona.renderer import PersonaRenderer
from agethos.storage.memory_store import InMemoryStore


def _resolve_llm(provider: str, model: str | None = None, api_key: str | None = None) -> LLMAdapter:
    """Resolve an LLM adapter by provider name string."""
    provider = provider.lower()
    if provider == "openai":
        from agethos.llm.openai import OpenAIAdapter
        return OpenAIAdapter(model=model or "gpt-4o-mini", api_key=api_key)
    elif provider in ("anthropic", "claude"):
        from agethos.llm.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=model or "claude-sonnet-4-20250514", api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'anthropic'.")


class Brain:
    """The agent's brain.

    Usage::

        # Quick start — from dict + provider string
        brain = Brain.build(
            persona={"name": "Minsoo", "ocean": {"O": 0.8, "C": 0.9, "E": 0.2, "A": 0.6, "N": 0.3}},
            llm="openai",
        )
        reply = await brain.chat("Hello!")

        # Or traditional style
        brain = Brain(persona=spec, llm=llm_adapter)
    """

    def __init__(
        self,
        persona: PersonaSpec,
        llm: LLMAdapter,
        embedder: EmbeddingAdapter | None = None,
        store: StorageBackend | None = None,
        reflection_threshold: float = 150.0,
        retrieval_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        decay_factor: float = 0.995,
        max_history: int = 20,
    ):
        self._persona = persona
        self._llm = llm
        self._renderer = PersonaRenderer(persona)
        self._current_plan: DailyPlan | None = None
        self._history: list[dict[str, str]] = []
        self._max_history = max_history

        # Memory
        self._memory = MemoryStream(
            store=store or InMemoryStore(),
            embedder=embedder,
            decay_factor=decay_factor,
        )

        # Cognition
        self._perceiver = Perceiver(llm)
        self._retriever = Retriever(self._memory, default_weights=retrieval_weights)
        self._reflector = Reflector(
            llm=llm,
            memory=self._memory,
            retriever=self._retriever,
            threshold=reflection_threshold,
        )
        self._planner = Planner(llm=llm, persona=persona)

        # Init emotion from OCEAN if available
        self._persona.init_emotion_from_ocean()

        # Seed memories
        self._seed_loaded = False

    # ── Factory methods ──

    @classmethod
    def build(
        cls,
        persona: dict | PersonaSpec,
        llm: str | LLMAdapter,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> Brain:
        """Convenience factory — build a Brain from dicts and strings.

        Args:
            persona: A dict (flat shorthand) or PersonaSpec instance.
            llm: An LLMAdapter instance, or a provider string ("openai", "anthropic").
            model: Model name override (e.g. "gpt-4o", "claude-opus-4-20250514").
            api_key: API key override (defaults to env var).
            **kwargs: Passed to Brain.__init__ (e.g. max_history, reflection_threshold).

        Examples::

            brain = Brain.build(
                persona={"name": "Luna", "ocean": {"O": 0.9, "E": 0.3}},
                llm="openai",
            )

            brain = Brain.build(
                persona="personas/minsoo.yaml",  # YAML file path
                llm="openai",
                model="gpt-4o",
            )
        """
        # Resolve persona
        if isinstance(persona, dict):
            persona = PersonaSpec.from_dict(persona)
        elif isinstance(persona, str):
            persona = PersonaSpec.from_yaml(persona)

        # Resolve LLM
        if isinstance(llm, str):
            llm = _resolve_llm(llm, model=model, api_key=api_key)

        return cls(persona=persona, llm=llm, **kwargs)

    async def _ensure_seed(self) -> None:
        """시드 메모리 로드 (최초 1회)."""
        if self._seed_loaded:
            return
        self._seed_loaded = True
        if self._persona.seed_memory:
            sentences = [
                s.strip()
                for s in self._persona.seed_memory.replace(";", ".").split(".")
                if s.strip()
            ]
            for sentence in sentences:
                node = MemoryNode(
                    node_type=NodeType.EVENT,
                    description=sentence,
                    importance=5.0,
                )
                await self._memory.append(node)

    # ── 핵심 API ──

    async def observe(self, observation: str) -> MemoryNode:
        """외부 관찰 기록. 자동 reflection 트리거 포함."""
        await self._ensure_seed()
        node = await self._perceiver.perceive(observation, NodeType.EVENT)
        await self._memory.append(node)

        # Auto-reflect if threshold exceeded
        if await self._reflector.should_reflect():
            await self._reflector.reflect()

        return node

    async def chat(
        self,
        user_message: str,
        context: str = "",
    ) -> str:
        """Conversation with full cognitive loop.

        Flow:
        1. perceive: user_message → MemoryNode
        2. retrieve: search related memories
        3. render: persona + memories + plan → system prompt
        4. generate: LLM call with multi-turn history
        5. store: save own response as MemoryNode
        6. reflect: auto-reflection if importance threshold exceeded
        """
        await self._ensure_seed()

        # 1. Perceive user message
        user_node = await self._perceiver.perceive(user_message, NodeType.CHAT)
        await self._memory.append(user_node)

        # 2. Retrieve relevant memories
        results = await self._retriever.retrieve(user_message, top_k=10)
        context_memories = [r.node for r in results]

        # 3. Render system prompt
        system_prompt = self._renderer.render_full(
            context_memories=context_memories,
            current_plan=self._current_plan,
        )

        # 4. Generate response (with conversation history)
        user_prompt = user_message
        if context:
            user_prompt = f"[Context: {context}]\n\n{user_message}"

        response = await self._llm.generate_with_history(
            system_prompt=system_prompt,
            history=list(self._history),
            user_prompt=user_prompt,
        )

        # 5. Update conversation history
        self._history.append({"role": "user", "content": user_prompt})
        self._history.append({"role": "assistant", "content": response})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        # 6. Store own response as memory
        response_node = MemoryNode(
            node_type=NodeType.CHAT,
            description=f"{self._persona.name} responded: {response[:200]}",
            importance=3.0,
        )
        await self._memory.append(response_node)

        # 7. Auto-reflect if needed
        if await self._reflector.should_reflect():
            await self._reflector.reflect()

        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Current conversation history."""
        return list(self._history)

    async def plan_day(
        self,
        date: str,
        context: str = "",
    ) -> DailyPlan:
        """일일 계획 수립."""
        await self._ensure_seed()
        recent = await self._memory.get_recent(10)
        plan = await self._planner.create_daily_plan(
            date=date,
            context=context,
            existing_memories=recent,
        )
        self._current_plan = plan

        # Store plan as memory
        plan_node = MemoryNode(
            node_type=NodeType.PLAN,
            description=f"{date} plan: {plan.summary}",
            importance=5.0,
        )
        await self._memory.append(plan_node)

        return plan

    async def reflect(self) -> list[MemoryNode]:
        """반성 수동 트리거."""
        await self._ensure_seed()
        return await self._reflector.reflect()

    async def recall(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """기억 검색."""
        await self._ensure_seed()
        return await self._retriever.retrieve(query, top_k=top_k)

    # ── 상태 접근 ──

    @property
    def persona(self) -> PersonaSpec:
        return self._persona

    @property
    def memory(self) -> MemoryStream:
        return self._memory

    @property
    def current_plan(self) -> DailyPlan | None:
        return self._current_plan

    @property
    def emotion(self) -> EmotionalState | None:
        return self._persona.emotion

    def update_situation(self, **l2_traits: str) -> None:
        """L2 (현재 상황) 레이어 업데이트."""
        self._persona.l2_situation.traits.update(l2_traits)
        self._renderer = PersonaRenderer(self._persona)

    def apply_event_emotion(
        self,
        event_pad: tuple[float, float, float],
        sensitivity: float | None = None,
    ) -> None:
        """이벤트에 의한 감정 변화 적용."""
        self._persona.apply_event(event_pad, sensitivity)

    def decay_emotion(self, rate: float = 0.1) -> None:
        """감정 감쇠."""
        self._persona.decay_emotion(rate)
