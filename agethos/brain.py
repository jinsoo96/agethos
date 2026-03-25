"""Brain — the agent's 'brain' facade.

Unifies persona, memory, and cognition as a single entry point.
"""

from __future__ import annotations

import json
import time

from agethos.cognition.perceive import Perceiver
from agethos.cognition.plan import Planner
from agethos.cognition.reflect import Reflector
from agethos.cognition.retrieve import Retriever
from agethos.embedding.base import EmbeddingAdapter
from agethos.llm.base import LLMAdapter
from agethos.memory.store import StorageBackend
from agethos.memory.stream import MemoryStream
from agethos.models import (
    BrainState,
    CommunityProfile,
    DailyPlan,
    EmotionalState,
    MentalModel,
    MemoryNode,
    NodeType,
    PersonaSpec,
    RetrievalResult,
    SelfRefineConfig,
    SocialPattern,
)
from agethos.persona.renderer import PersonaRenderer
from agethos.storage.memory_store import InMemoryStore


def _resolve_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMAdapter:
    """Resolve an LLM adapter by provider name string."""
    provider = provider.lower()
    if provider == "openai":
        from agethos.llm.openai import OpenAIAdapter
        return OpenAIAdapter(model=model or "gpt-4o-mini", api_key=api_key, base_url=base_url)
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
        self_refine: SelfRefineConfig | None = None,
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

        # Social learning state
        self._social_patterns: list[SocialPattern] = []
        self._community_profiles: list[CommunityProfile] = []

        # Theory of Mind
        self._mental_models: dict[str, MentalModel] = {}

        # Self-Refine
        self._self_refine_config = self_refine or SelfRefineConfig()

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
        base_url: str | None = None,
        **kwargs,
    ) -> Brain:
        """Convenience factory — build a Brain from dicts and strings.

        Args:
            persona: A dict (flat shorthand) or PersonaSpec instance.
            llm: An LLMAdapter instance, or a provider string ("openai", "anthropic").
            model: Model name override (e.g. "gpt-4o", "claude-opus-4-20250514").
            api_key: API key override (defaults to env var).
            base_url: Custom API endpoint for OpenAI-compatible providers.
            **kwargs: Passed to Brain.__init__ (e.g. max_history, reflection_threshold).

        Examples::

            # OpenAI
            brain = Brain.build(persona={...}, llm="openai")

            # Ollama (local)
            brain = Brain.build(
                persona={...}, llm="openai",
                model="qwen2.5:7b",
                base_url="http://localhost:11434/v1",
            )

            # Qwen via DashScope
            brain = Brain.build(
                persona={...}, llm="openai",
                model="qwen-plus",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-your-dashscope-key",
            )
        """
        # Resolve persona
        if isinstance(persona, dict):
            persona = PersonaSpec.from_dict(persona)
        elif isinstance(persona, str):
            persona = PersonaSpec.from_yaml(persona)

        # Resolve LLM
        if isinstance(llm, str):
            llm = _resolve_llm(llm, model=model, api_key=api_key, base_url=base_url)

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

        # 4.5. Self-Refine (optional)
        if self._self_refine_config.enabled:
            from agethos.cognition.refine import SelfRefiner
            refiner = SelfRefiner(llm=self._llm, config=self._self_refine_config)
            result = await refiner.refine(response, user_message, self._renderer.render_iss())
            response = result.refined

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

    # ── 저장/복원 ──

    async def save(self, path: str) -> None:
        """인격 상태 전체를 .brain.json 파일로 저장.

        PersonaSpec + 기억 + 학습 패턴 + 대화 기록을 직렬화.
        """
        all_memories = await self._memory.store.get_all()

        state = BrainState(
            version="0.5.0",
            last_active=time.time(),
            total_interactions=len([h for h in self._history if h.get("role") == "user"]),
            persona=self._persona,
            memories=all_memories,
            social_patterns=list(self._social_patterns),
            community_profiles=list(self._community_profiles),
            history=list(self._history),
            mental_models=list(self._mental_models.values()),
        )

        data = state.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    async def load(
        cls,
        path: str,
        llm: str | LLMAdapter,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> Brain:
        """저장된 .brain.json에서 Brain 복원.

        경험/기억/학습 패턴이 그대로 복원됨.

        Args:
            path: .brain.json 파일 경로.
            llm: LLM 프로바이더 또는 어댑터 인스턴스.
            **kwargs: Brain.__init__에 전달.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        state = BrainState.model_validate(data)

        # Resolve LLM
        if isinstance(llm, str):
            llm = _resolve_llm(llm, model=model, api_key=api_key, base_url=base_url)

        brain = cls(persona=state.persona, llm=llm, **kwargs)

        # Restore memories
        for mem in state.memories:
            await brain._memory.store.save(mem)

        # Restore social patterns & community profiles
        brain._social_patterns = list(state.social_patterns)
        brain._community_profiles = list(state.community_profiles)

        # Restore mental models
        for mm in state.mental_models:
            brain._mental_models[mm.target] = mm

        # Restore history
        brain._history = list(state.history)
        brain._seed_loaded = True

        return brain

    # ── Export 어댑터 ──

    def export(self, format: str, **kwargs) -> str | dict:
        """인격을 다양한 플랫폼 형식으로 내보내기.

        Args:
            format: 출력 형식. 지원:
                - "system_prompt": 범용 시스템 프롬프트 텍스트
                - "anthropic": Anthropic Messages API system 필드용
                - "openai_assistant": OpenAI Assistants API instructions
                - "crewai": CrewAI agent config dict
                - "bedrock_agent": AWS Bedrock instruction (4000자 제한)
                - "a2a_card": A2A Agent Card dict

        Returns:
            str (프롬프트 형식) 또는 dict (API 설정 형식).
        """
        from agethos.export.adapters import export_brain
        return export_brain(self, format, **kwargs)

    # ── 관찰 학습 ──

    async def observe_community(
        self,
        env,
        max_messages: int = 500,
        community_name: str = "",
    ) -> list[SocialPattern]:
        """외부 채팅을 관찰하여 사회적 패턴 추출.

        Args:
            env: Environment (ChatLogEnvironment 등).
            max_messages: 최대 관찰 메시지 수.
            community_name: 커뮤니티 이름.

        Returns:
            추출된 SocialPattern 목록.
        """
        from agethos.cognition.observer import Observer
        observer = Observer(
            brain=self,
            llm=self._llm,
            community_name=community_name,
        )
        patterns = await observer.observe(env, max_messages=max_messages)
        self._social_patterns.extend(patterns)

        # Update or create community profile
        if community_name:
            existing = next(
                (cp for cp in self._community_profiles if cp.name == community_name),
                None,
            )
            if existing:
                existing.norms.extend(patterns)
                existing.observed_count += max_messages
                existing.last_updated = time.time()
            else:
                self._community_profiles.append(
                    CommunityProfile(
                        name=community_name,
                        norms=patterns,
                        observed_count=max_messages,
                    )
                )

        return patterns

    # ── Theory of Mind ──

    async def infer_mental_model(self, target: str, conversation: str) -> MentalModel:
        """대화에서 상대의 멘탈 모델 추론 또는 갱신.

        이미 모델이 있으면 업데이트, 없으면 새로 생성.
        """
        from agethos.cognition.tom import TheoryOfMind
        tom = TheoryOfMind(self._llm)

        existing = self._mental_models.get(target)
        if existing:
            model = await tom.update(existing, conversation)
        else:
            model = await tom.infer(target, conversation)

        self._mental_models[target] = model
        return model

    def get_mental_model(self, target: str) -> MentalModel | None:
        """저장된 멘탈 모델 조회."""
        return self._mental_models.get(target)

    @property
    def mental_models(self) -> dict[str, MentalModel]:
        return dict(self._mental_models)

    # ── 상태 접근 (social) ──

    @property
    def social_patterns(self) -> list[SocialPattern]:
        return list(self._social_patterns)

    @property
    def community_profiles(self) -> list[CommunityProfile]:
        return list(self._community_profiles)

    # ── 기존 ──

    def autopilot(self, env, **kwargs):
        """이 Brain에 연결된 Autopilot 생성.

        Usage::

            env = QueueEnvironment()
            pilot = brain.autopilot(env, tick_interval=2.0)
            actions = await pilot.step()
        """
        from agethos.autopilot import Autopilot
        return Autopilot(brain=self, env=env, **kwargs)
