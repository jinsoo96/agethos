""".brain ZIP 포맷 — 포터블 뇌 패키징.

.brain 파일 구조:
    manifest.json       — 메타데이터 (버전, 이름, 생성 시각, 통계)
    persona.json        — PersonaSpec 전체
    memories.jsonl      — 기억 노드 (줄 단위 JSON)
    patterns.json       — 학습된 사회적 패턴
    mental_models.json  — Theory of Mind 모델
    history.json        — 대화 기록
    fingerprint.svg     — Neural Fingerprint 시각화
"""

from __future__ import annotations

import io
import json
import math
import time
import zipfile
from pathlib import Path

from agethos.models import (
    BrainState,
    MemoryNode,
    OceanTraits,
    PersonaSpec,
)

BRAIN_FORMAT_VERSION = "1.0"


# ── Neural Fingerprint SVG ──────────────────────────────────────────


def _generate_fingerprint_svg(persona: PersonaSpec, memories: list[MemoryNode]) -> str:
    """OCEAN + 감정 + 기억 통계를 기반으로 Neural Fingerprint SVG 생성.

    5각형 레이더 차트(OCEAN) + 감정 위치 점 + 기억 밀도 링.
    """
    width, height = 400, 400
    cx, cy = 200, 200
    radius = 140

    # OCEAN values (default 0.5)
    ocean = persona.ocean or OceanTraits()
    traits = [ocean.openness, ocean.conscientiousness, ocean.extraversion,
              ocean.agreeableness, ocean.neuroticism]
    labels = ["O", "C", "E", "A", "N"]

    # 5각형 꼭짓점 좌표 계산 (위에서 시작, 시계방향)
    def polar(angle_deg: float, r: float) -> tuple[float, float]:
        rad = math.radians(angle_deg - 90)  # 12시 방향 시작
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    angles = [i * 72 for i in range(5)]

    # SVG 시작
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        '<style>',
        '  .grid { stroke: #e0e0e0; stroke-width: 0.5; fill: none; }',
        '  .shape { fill: rgba(99, 102, 241, 0.25); stroke: #6366f1; stroke-width: 2; }',
        '  .label { font: bold 12px sans-serif; fill: #374151; text-anchor: middle; }',
        '  .value { font: 10px sans-serif; fill: #6b7280; text-anchor: middle; }',
        '  .title { font: bold 14px sans-serif; fill: #111827; text-anchor: middle; }',
        '  .emotion-dot { fill: #ef4444; }',
        '  .memory-ring { fill: none; stroke-width: 3; stroke-linecap: round; }',
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white" rx="16"/>',
    ]

    # 그리드 원 (3단계)
    for scale in (0.33, 0.66, 1.0):
        r = radius * scale
        points = " ".join(f"{polar(a, r)[0]},{polar(a, r)[1]}" for a in angles)
        svg_parts.append(f'<polygon points="{points}" class="grid"/>')

    # 그리드 축선
    for a in angles:
        x2, y2 = polar(a, radius)
        svg_parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" class="grid"/>')

    # OCEAN 다각형
    trait_points = []
    for i, val in enumerate(traits):
        x, y = polar(angles[i], radius * val)
        trait_points.append(f"{x},{y}")
    svg_parts.append(f'<polygon points="{" ".join(trait_points)}" class="shape"/>')

    # 꼭짓점 점 + 라벨
    for i, (val, label) in enumerate(zip(traits, labels)):
        # 라벨 위치 (약간 바깥)
        lx, ly = polar(angles[i], radius + 20)
        svg_parts.append(f'<text x="{lx}" y="{ly}" class="label">{label}</text>')
        # 값 표시
        vx, vy = polar(angles[i], radius + 34)
        svg_parts.append(f'<text x="{vx}" y="{vy}" class="value">{val:.2f}</text>')
        # 점
        px, py = polar(angles[i], radius * val)
        svg_parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#6366f1"/>')

    # 감정 상태 점 (PAD → 2D: P=x, A=y)
    if persona.emotion:
        ex = cx + persona.emotion.pleasure * 30
        ey = cy - persona.emotion.arousal * 30
        svg_parts.append(f'<circle cx="{ex}" cy="{ey}" r="6" class="emotion-dot" opacity="0.7"/>')
        emo_label = persona.emotion.closest_emotion()
        svg_parts.append(f'<text x="{ex}" y="{ey - 10}" class="value" fill="#ef4444">{emo_label}</text>')

    # 기억 밀도 링 (기억 수에 비례하는 호)
    mem_count = len(memories)
    if mem_count > 0:
        # 기억 수 → 0~1 비율 (100개 기준)
        mem_ratio = min(1.0, mem_count / 100)
        arc_length = 2 * math.pi * (radius + 50) * mem_ratio
        dash = f"{arc_length:.1f} {2 * math.pi * (radius + 50):.1f}"
        color_intensity = int(99 + 156 * mem_ratio)
        svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius + 50}" '
            f'class="memory-ring" stroke="rgba(99, {color_intensity}, 241, 0.4)" '
            f'stroke-dasharray="{dash}" '
            f'transform="rotate(-90 {cx} {cy})"/>'
        )

    # 타이틀 + 통계
    svg_parts.append(f'<text x="{cx}" y="30" class="title">{persona.name}</text>')
    svg_parts.append(f'<text x="{cx}" y="{height - 15}" class="value">'
                     f'Memories: {mem_count} | '
                     f'Emotion: {persona.emotion.closest_emotion() if persona.emotion else "none"}'
                     f'</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


# ── Pack / Unpack ──────────────────────────────────────────


async def pack_brain(
    brain,
    path: str | Path,
    include_history: bool = True,
    include_fingerprint: bool = True,
) -> Path:
    """Brain → .brain ZIP 파일로 패키징.

    Args:
        brain: Brain 인스턴스.
        path: 저장 경로 (.brain 확장자 권장).
        include_history: 대화 기록 포함 여부.
        include_fingerprint: Neural Fingerprint SVG 포함 여부.

    Returns:
        저장된 파일의 Path.
    """
    path = Path(path)
    all_memories = await brain.memory.store.get_all()

    # manifest
    manifest = {
        "format_version": BRAIN_FORMAT_VERSION,
        "agethos_version": "0.8.0",
        "name": brain.persona.name,
        "created_at": time.time(),
        "stats": {
            "memory_count": len(all_memories),
            "pattern_count": len(brain.social_patterns),
            "mental_model_count": len(brain.mental_models),
            "interaction_count": len([h for h in brain.history if h.get("role") == "user"]),
        },
    }

    # persona
    persona_data = brain.persona.model_dump(mode="json")

    # memories → JSONL
    memories_lines = []
    for mem in all_memories:
        memories_lines.append(json.dumps(mem.model_dump(mode="json"), ensure_ascii=False))

    # patterns
    patterns_data = [p.model_dump(mode="json") for p in brain.social_patterns]

    # mental models
    mm_data = [m.model_dump(mode="json") for m in brain.mental_models.values()]

    # history
    history_data = list(brain.history) if include_history else []

    # Write ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("persona.json", json.dumps(persona_data, ensure_ascii=False, indent=2))
        zf.writestr("memories.jsonl", "\n".join(memories_lines))
        zf.writestr("patterns.json", json.dumps(patterns_data, ensure_ascii=False, indent=2))
        zf.writestr("mental_models.json", json.dumps(mm_data, ensure_ascii=False, indent=2))
        zf.writestr("history.json", json.dumps(history_data, ensure_ascii=False, indent=2))

        if include_fingerprint:
            svg = _generate_fingerprint_svg(brain.persona, all_memories)
            zf.writestr("fingerprint.svg", svg)

    path.write_bytes(buf.getvalue())
    return path


async def unpack_brain(
    path: str | Path,
    llm,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
):
    """`.brain` ZIP → Brain 복원.

    Args:
        path: .brain 파일 경로.
        llm: LLM 프로바이더 문자열 또는 어댑터 인스턴스.
        **kwargs: Brain.__init__에 전달.

    Returns:
        복원된 Brain 인스턴스.
    """
    from agethos.brain import Brain, _resolve_llm
    from agethos.models import MentalModel, SocialPattern

    path = Path(path)

    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        persona_data = json.loads(zf.read("persona.json"))
        memories_raw = zf.read("memories.jsonl").decode("utf-8")
        patterns_data = json.loads(zf.read("patterns.json"))
        mm_data = json.loads(zf.read("mental_models.json"))
        history_data = json.loads(zf.read("history.json"))

    # Rebuild persona
    persona = PersonaSpec.model_validate(persona_data)

    # Rebuild memories
    memories = []
    for line in memories_raw.strip().split("\n"):
        if line.strip():
            memories.append(MemoryNode.model_validate(json.loads(line)))

    # Resolve LLM
    if isinstance(llm, str):
        llm = _resolve_llm(llm, model=model, api_key=api_key, base_url=base_url)

    # Build brain
    brain = Brain(persona=persona, llm=llm, **kwargs)

    # Restore memories
    for mem in memories:
        await brain._memory.store.save(mem)

    # Restore patterns
    brain._social_patterns = [SocialPattern.model_validate(p) for p in patterns_data]

    # Restore mental models
    for mm in mm_data:
        model_obj = MentalModel.model_validate(mm)
        brain._mental_models[model_obj.target] = model_obj

    # Restore history
    brain._history = list(history_data)
    brain._seed_loaded = True

    return brain


def inspect_brain(path: str | Path) -> dict:
    """`.brain` 파일의 메타데이터를 열람 (Brain 인스턴스 없이).

    Returns:
        manifest + 기본 통계 dict.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        manifest["files"] = zf.namelist()
        manifest["total_size_bytes"] = sum(info.file_size for info in zf.infolist())
        manifest["compressed_size_bytes"] = sum(info.compress_size for info in zf.infolist())
    return manifest


def extract_fingerprint(path: str | Path) -> str | None:
    """`.brain` 파일에서 Neural Fingerprint SVG 추출.

    Returns:
        SVG 문자열 또는 None.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        if "fingerprint.svg" in zf.namelist():
            return zf.read("fingerprint.svg").decode("utf-8")
    return None
