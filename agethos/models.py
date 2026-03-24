"""agethos 핵심 데이터 모델."""

from __future__ import annotations

import math
import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


# ────────────────────────── Enums ──────────────────────────


class NodeType(str, Enum):
    """기억 노드 유형."""

    EVENT = "event"
    THOUGHT = "thought"
    CHAT = "chat"
    PLAN = "plan"


# ────────────────────────── Memory ──────────────────────────


class MemoryNode(BaseModel):
    """기억 스트림의 단일 노드.

    Generative Agents의 ConceptNode + Synaptic Memory의 Node 융합.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_type: NodeType = NodeType.EVENT

    # SPO triple
    subject: str = ""
    predicate: str = ""
    obj: str = ""

    description: str
    keywords: list[str] = Field(default_factory=list)

    # Scoring axes
    importance: float = 5.0
    depth: int = 1

    # Temporal
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0

    # Embedding
    embedding: list[float] | None = None

    # Evidence pointers (for reflections)
    evidence_ids: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """검색 결과 + 점수 분해."""

    node: MemoryNode
    score: float
    recency_score: float = 0.0
    importance_score: float = 0.0
    relevance_score: float = 0.0


# ────────────────────────── OCEAN (Big Five) ──────────────────────────


class OceanTraits(BaseModel):
    """Big Five (OCEAN) 성격 특성 모델.

    각 특성은 0.0~1.0 범위의 연속값.
    Mehrabian (1996), BIG5-CHAT (2024) 기반.

    행동 영향:
    - High O: 창의적, 새로운 경험 개방 → 비유적 표현, 참신한 관점 제시
    - High C: 체계적, 규율 → 구조화된 응답, 계획적 행동
    - High E: 사교적, 에너지 → 적극적 대화, 긍정적 톤
    - High A: 협조적, 따뜻 → 공감적 응답, 갈등 회피
    - High N: 감정적 불안정 → 걱정 표현, 부정적 사건에 과민 반응
    """

    openness: float = Field(0.5, ge=0.0, le=1.0, description="개방성")
    conscientiousness: float = Field(0.5, ge=0.0, le=1.0, description="성실성")
    extraversion: float = Field(0.5, ge=0.0, le=1.0, description="외향성")
    agreeableness: float = Field(0.5, ge=0.0, le=1.0, description="우호성")
    neuroticism: float = Field(0.5, ge=0.0, le=1.0, description="신경성")

    def to_prompt(self) -> str:
        """OCEAN 특성을 LLM 프롬프트 텍스트로 변환."""
        labels = {
            "openness": ("개방성", self._describe_o),
            "conscientiousness": ("성실성", self._describe_c),
            "extraversion": ("외향성", self._describe_e),
            "agreeableness": ("우호성", self._describe_a),
            "neuroticism": ("신경성", self._describe_n),
        }
        lines = []
        for attr, (name, describer) in labels.items():
            val = getattr(self, attr)
            level = "HIGH" if val > 0.66 else ("LOW" if val < 0.33 else "MID")
            lines.append(f"- {name} ({level}, {val:.2f}): {describer()}")
        return "\n".join(lines)

    def _describe_o(self) -> str:
        if self.openness > 0.66:
            return "creative, curious, open to new experiences, uses metaphors"
        if self.openness < 0.33:
            return "practical, conventional, prefers routine, concrete thinking"
        return "balanced between creativity and practicality"

    def _describe_c(self) -> str:
        if self.conscientiousness > 0.66:
            return "organized, disciplined, structured responses, thorough"
        if self.conscientiousness < 0.33:
            return "spontaneous, flexible, casual, less structured"
        return "moderately organized, adaptable"

    def _describe_e(self) -> str:
        if self.extraversion > 0.66:
            return "outgoing, energetic, talkative, enthusiastic"
        if self.extraversion < 0.33:
            return "reserved, quiet, prefers listening, contemplative"
        return "ambivert, socially adaptable"

    def _describe_a(self) -> str:
        if self.agreeableness > 0.66:
            return "cooperative, warm, empathetic, conflict-avoidant"
        if self.agreeableness < 0.33:
            return "competitive, direct, skeptical, confrontational"
        return "balanced between cooperation and assertiveness"

    def _describe_n(self) -> str:
        if self.neuroticism > 0.66:
            return "emotionally reactive, prone to worry, sensitive"
        if self.neuroticism < 0.33:
            return "emotionally stable, calm, stress-resistant"
        return "moderately emotional, generally composed"


# ────────────────────────── PAD Emotional State ──────────────────────────


class EmotionalState(BaseModel):
    """PAD (Pleasure-Arousal-Dominance) 감정 상태 모델.

    Mehrabian (1996) 3차원 감정 공간.
    각 축은 -1.0 ~ +1.0 범위.

    Pleasure:  쾌-불쾌 (+1=최대 쾌감, -1=최대 불쾌)
    Arousal:   각성 수준 (+1=최대 각성, -1=최대 이완)
    Dominance: 지배감 (+1=완전 통제, -1=완전 무력)
    """

    pleasure: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.0, ge=-1.0, le=1.0)
    dominance: float = Field(0.0, ge=-1.0, le=1.0)

    # 기본 감정별 PAD 좌표 (Mehrabian, 1996)
    EMOTION_MAP: dict[str, tuple[float, float, float]] = {
        "joy": (0.76, 0.48, 0.35),
        "anger": (-0.51, 0.59, 0.25),
        "sadness": (-0.63, -0.27, -0.33),
        "fear": (-0.64, 0.60, -0.43),
        "disgust": (-0.60, 0.35, 0.11),
        "surprise": (0.40, 0.67, -0.13),
        "contempt": (-0.55, 0.20, 0.45),
        "shame": (-0.57, 0.20, -0.56),
        "pride": (0.40, 0.30, 0.55),
        "calm": (0.20, -0.50, 0.30),
        "excitement": (0.62, 0.75, 0.38),
        "boredom": (-0.30, -0.60, -0.15),
    }

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_ocean(cls, ocean: OceanTraits) -> EmotionalState:
        """OCEAN 성격 → 기본 감정 상태 (baseline).

        Mehrabian (1996) 변환 공식:
        P = 0.21E + 0.59A + 0.19(-N)
        A = 0.15O + 0.30N - 0.57A_trait
        D = 0.25O + 0.17C + 0.60E - 0.32A_trait
        """
        return cls(
            pleasure=max(-1, min(1,
                0.21 * ocean.extraversion
                + 0.59 * ocean.agreeableness
                - 0.19 * ocean.neuroticism
            )),
            arousal=max(-1, min(1,
                0.15 * ocean.openness
                + 0.30 * ocean.neuroticism
                - 0.57 * ocean.agreeableness
            )),
            dominance=max(-1, min(1,
                0.25 * ocean.openness
                + 0.17 * ocean.conscientiousness
                + 0.60 * ocean.extraversion
                - 0.32 * ocean.agreeableness
            )),
        )

    def apply_stimulus(
        self,
        stimulus_pad: tuple[float, float, float],
        sensitivity: float = 0.3,
        personality_bias: tuple[float, float, float] | None = None,
        bias_weight: float = 0.2,
    ) -> EmotionalState:
        """자극에 의한 감정 전이.

        E(t+1) = E(t) + α·(E_stimulus - E(t)) + β·E_personality_bias

        Args:
            stimulus_pad: 자극의 PAD 값 (P, A, D).
            sensitivity: 감정 감수성 계수 α (0~1).
            personality_bias: 성격 기본 PAD 편향.
            bias_weight: 성격 편향 가중치 β.
        """
        sp, sa, sd = stimulus_pad
        bp, ba, bd = personality_bias or (0, 0, 0)

        new_p = self.pleasure + sensitivity * (sp - self.pleasure) + bias_weight * bp
        new_a = self.arousal + sensitivity * (sa - self.arousal) + bias_weight * ba
        new_d = self.dominance + sensitivity * (sd - self.dominance) + bias_weight * bd

        return EmotionalState(
            pleasure=max(-1, min(1, new_p)),
            arousal=max(-1, min(1, new_a)),
            dominance=max(-1, min(1, new_d)),
        )

    def decay(self, baseline: EmotionalState | None = None, rate: float = 0.1) -> EmotionalState:
        """감정 감쇠 — 시간에 따라 기본 상태로 회귀.

        E(t) = E_baseline + (E_current - E_baseline) · (1 - rate)
        """
        base = baseline or EmotionalState()
        factor = 1 - rate
        return EmotionalState(
            pleasure=max(-1, min(1, base.pleasure + (self.pleasure - base.pleasure) * factor)),
            arousal=max(-1, min(1, base.arousal + (self.arousal - base.arousal) * factor)),
            dominance=max(-1, min(1, base.dominance + (self.dominance - base.dominance) * factor)),
        )

    def closest_emotion(self) -> str:
        """현재 PAD 좌표에 가장 가까운 감정 라벨."""
        best_emotion = "neutral"
        best_dist = float("inf")
        for emotion, (ep, ea, ed) in self.EMOTION_MAP.items():
            dist = math.sqrt(
                (self.pleasure - ep) ** 2
                + (self.arousal - ea) ** 2
                + (self.dominance - ed) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_emotion = emotion
        return best_emotion

    def to_prompt(self) -> str:
        """현재 감정 상태를 프롬프트 텍스트로 변환."""
        emotion = self.closest_emotion()
        return (
            f"Current emotional state: {emotion} "
            f"(P={self.pleasure:+.2f}, A={self.arousal:+.2f}, D={self.dominance:+.2f})"
        )


# ────────────────────────── Character Card ──────────────────────────


class CharacterCard(BaseModel):
    """Character Card V2 (Tavern Card) 호환 캐릭터 정의.

    SillyTavern / TavernAI 표준 스펙.
    W++, SBF, PList, 자연어 등 어떤 포맷이든 description 필드에 사용 가능.
    """

    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    creator: str = ""
    character_version: str = "1.0"

    def to_persona_spec(self) -> PersonaSpec:
        """CharacterCard → PersonaSpec 변환."""
        return PersonaSpec(
            name=self.name,
            identity=self.description,
            tone=self.personality,
            seed_memory=self.scenario,
            behavioral_rules=[
                line.strip()
                for line in self.system_prompt.split("\n")
                if line.strip()
            ] if self.system_prompt else [],
        )

    @classmethod
    def from_wpp(cls, wpp_text: str) -> CharacterCard:
        """W++ 포맷 텍스트를 CharacterCard로 파싱.

        W++ 예시:
        [character("Luna")
        {
          Personality("analytical" + "curious")
          Age("25")
        }]
        """
        import re

        name_match = re.search(r'character\("([^"]+)"\)', wpp_text)
        name = name_match.group(1) if name_match else "Unknown"

        categories: dict[str, list[str]] = {}
        for match in re.finditer(r'(\w+)\(([^)]+)\)', wpp_text):
            cat = match.group(1).lower()
            if cat == "character":
                continue
            values = [v.strip().strip('"') for v in match.group(2).split("+")]
            categories[cat] = values

        personality_parts = []
        for key in ("personality", "mind", "traits"):
            if key in categories:
                personality_parts.extend(categories[key])

        description_parts = []
        for key, vals in categories.items():
            if key not in ("personality", "mind", "traits"):
                description_parts.append(f"{key.capitalize()}: {', '.join(vals)}")

        return cls(
            name=name,
            description="\n".join(description_parts),
            personality=", ".join(personality_parts),
        )

    @classmethod
    def from_sbf(cls, sbf_text: str) -> CharacterCard:
        """SBF (Square Bracket Format) 파싱.

        [character: Luna;
        personality: analytical, curious;
        age: 25]
        """
        import re

        fields: dict[str, str] = {}
        content = sbf_text.strip().strip("[]")
        for part in content.split(";"):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                fields[key.strip().lower()] = val.strip()

        name = fields.pop("character", fields.pop("name", "Unknown"))
        personality = fields.pop("personality", "")
        description = "\n".join(f"{k.capitalize()}: {v}" for k, v in fields.items())

        return cls(
            name=name,
            description=description,
            personality=personality,
        )


# ────────────────────────── Persona ──────────────────────────


class PersonaLayer(BaseModel):
    """계층별 정체성 정보."""

    traits: dict[str, str] = Field(default_factory=dict)


class PersonaSpec(BaseModel):
    """에이전트 인격 전체 명세.

    구성 요소:
    - 3계층 정체성: L0(선천적) / L1(학습된) / L2(현재 상황)
    - 6면 페르소나: identity, tone, values, boundaries, conversation_style, transparency
    - OCEAN (Big Five): 수치 기반 성격 특성
    - PAD: 동적 감정 상태
    - 행동 규칙: 상황별 규칙 기반 행동 정의
    """

    name: str

    # 3-layer identity (Generative Agents ISS)
    l0_innate: PersonaLayer = Field(default_factory=PersonaLayer)
    l1_learned: PersonaLayer = Field(default_factory=PersonaLayer)
    l2_situation: PersonaLayer = Field(default_factory=PersonaLayer)

    # 6-facet persona (system prompt analysis)
    identity: str = ""
    tone: str = ""
    values: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    conversation_style: str = ""
    transparency: str = ""

    # OCEAN personality traits (Big Five)
    ocean: OceanTraits | None = None

    # PAD emotional state (dynamic)
    emotion: EmotionalState | None = None

    # Behavioral rules
    behavioral_rules: list[str] = Field(default_factory=list)

    # Seed memory paragraph
    seed_memory: str = ""

    def init_emotion_from_ocean(self) -> None:
        """OCEAN 특성에서 감정 baseline 초기화."""
        if self.ocean and not self.emotion:
            self.emotion = EmotionalState.from_ocean(self.ocean)

    def apply_event(
        self,
        event_pad: tuple[float, float, float],
        sensitivity: float | None = None,
    ) -> None:
        """이벤트에 의한 감정 상태 갱신.

        sensitivity는 Neuroticism에 의해 자동 결정됨:
        High N → 더 강한 감정 반응 (sensitivity ↑)
        """
        if not self.emotion:
            self.emotion = EmotionalState()

        if sensitivity is None:
            n = self.ocean.neuroticism if self.ocean else 0.5
            sensitivity = 0.15 + 0.35 * n  # Range: 0.15 ~ 0.50

        baseline_pad = None
        if self.ocean:
            baseline = EmotionalState.from_ocean(self.ocean)
            baseline_pad = (baseline.pleasure, baseline.arousal, baseline.dominance)

        self.emotion = self.emotion.apply_stimulus(
            stimulus_pad=event_pad,
            sensitivity=sensitivity,
            personality_bias=baseline_pad,
            bias_weight=0.1,
        )

    def decay_emotion(self, rate: float = 0.1) -> None:
        """감정 감쇠 — 성격 baseline으로 회귀."""
        if not self.emotion:
            return
        baseline = EmotionalState.from_ocean(self.ocean) if self.ocean else None
        self.emotion = self.emotion.decay(baseline=baseline, rate=rate)


# ────────────────────────── Plan ──────────────────────────


class PlanItem(BaseModel):
    """계획 항목."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    time_range: str = ""
    duration_minutes: int = 30
    status: str = "pending"
    sub_items: list[PlanItem] = Field(default_factory=list)
    parent_id: str | None = None


class DailyPlan(BaseModel):
    """일일 계획."""

    date: str
    summary: str = ""
    items: list[PlanItem] = Field(default_factory=list)
