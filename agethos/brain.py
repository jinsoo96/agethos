"""Brain — 에이전트의 '뇌' 파사드.

인격, 기억, 인지를 하나로 묶는 메인 진입점.
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


class Brain:
    """에이전트의 '뇌'.

    Usage::

        brain = Brain(persona=spec, llm=llm_adapter)

        # 대화
        reply = await brain.chat("안녕하세요!")

        # 관찰 기록
        await brain.observe("팀 미팅에서 마감일이 변경되었다")

        # 계획 수립
        plan = await brain.plan_day("2026-03-25")

        # 반성
        insights = await brain.reflect()

        # 기억 검색
        results = await brain.recall("마감일 관련")
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
    ):
        self._persona = persona
        self._llm = llm
        self._renderer = PersonaRenderer(persona)
        self._current_plan: DailyPlan | None = None

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
        """대화. 기억 검색 + 인격 반영 + 응답 생성.

        흐름:
        1. perceive: user_message → MemoryNode 저장
        2. retrieve: 관련 기억 검색
        3. render: persona + 기억 + 계획 → system prompt
        4. generate: LLM 호출
        5. perceive: 자신의 응답 저장
        6. reflect check: 자동 반성 트리거
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

        # 4. Generate response
        user_prompt = user_message
        if context:
            user_prompt = f"[Context: {context}]\n\n{user_message}"

        response = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 5. Store own response
        response_node = MemoryNode(
            node_type=NodeType.CHAT,
            description=f"{self._persona.name} responded: {response[:200]}",
            importance=3.0,
        )
        await self._memory.append(response_node)

        # 6. Auto-reflect if needed
        if await self._reflector.should_reflect():
            await self._reflector.reflect()

        return response

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
