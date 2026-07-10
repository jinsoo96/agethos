"""Deterministic description → trait signals (EN + KO substring lexicons).

The LLM compiler is the primary path; this lexicon powers the zero-infra fallback and
offline tests. Korean is agglutinative, so matching is substring-based ("까칠하지만"
matches "까칠"), not token-based.
"""
from __future__ import annotations

import re

# trait → (high-pole keywords, low-pole keywords)
OCEAN_LEXICON: dict[str, tuple[list[str], list[str]]] = {
    "openness": (
        ["creative", "curious", "imaginative", "novel", "unconventional", "artistic",
         "experimental", "창의", "호기심", "상상력", "모험", "실험적", "예술", "새로운 것"],
        ["practical", "conventional", "traditional", "routine", "down-to-earth",
         "보수적", "실용적", "관습", "루틴", "현실적", "전통적"],
    ),
    "conscientiousness": (
        ["organized", "disciplined", "meticulous", "thorough", "punctual", "systematic",
         "diligent", "꼼꼼", "체계적", "성실", "철저", "계획적", "규율", "정리"],
        ["spontaneous", "careless", "messy", "impulsive", "즉흥", "덜렁", "산만", "자유분방"],
    ),
    "extraversion": (
        ["outgoing", "energetic", "talkative", "sociable", "enthusiastic", "lively",
         "외향", "사교적", "활발", "수다", "적극적", "에너지 넘"],
        ["reserved", "quiet", "introverted", "shy", "withdrawn",
         "내향", "조용", "과묵", "수줍", "낯가림", "말수가 적"],
    ),
    "agreeableness": (
        ["warm", "kind", "empathetic", "cooperative", "gentle", "caring", "compassionate",
         "따뜻", "다정", "친절", "공감", "배려", "협조", "속정", "온화"],
        ["blunt", "direct", "competitive", "skeptical", "confrontational", "sarcastic",
         "grumpy", "cynical", "까칠", "직설", "퉁명", "시니컬", "냉소", "경쟁적", "비판적", "무뚝뚝"],
    ),
    "neuroticism": (
        ["anxious", "worried", "sensitive", "moody", "insecure", "nervous",
         "불안", "예민", "걱정", "소심", "감정기복", "신경질"],
        ["calm", "stable", "composed", "resilient", "unflappable", "relaxed",
         "침착", "차분", "안정", "담담", "태연", "평온"],
    ),
}

OCCUPATION_LEXICON = [
    "developer", "engineer", "designer", "researcher", "scientist", "teacher", "doctor",
    "writer", "artist", "manager", "analyst", "consultant", "barista", "chef", "detective",
    "개발자", "엔지니어", "디자이너", "연구원", "과학자", "교사", "선생님", "의사",
    "작가", "예술가", "매니저", "분석가", "컨설턴트", "바리스타", "요리사", "탐정", "기획자",
]

_AGE = re.compile(r"(\d{1,2})\s*(?:대|살|세)|(\d{1,2})[-\s]?(?:years?[-\s]?old|yo\b)", re.IGNORECASE)


def estimate_ocean(description: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Substring-lexicon OCEAN estimate → ({trait: 0..1}, {trait: matched keywords})."""
    low = description.lower()
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for trait, (hi_words, lo_words) in OCEAN_LEXICON.items():
        hi = [w for w in hi_words if w in low]
        lo = [w for w in lo_words if w in low]
        score = 0.5 + 0.18 * len(hi) - 0.18 * len(lo)
        scores[trait] = max(0.05, min(0.95, round(score, 2)))
        evidence[trait] = hi + lo
    return scores, evidence


def extract_innate(description: str) -> dict[str, str]:
    """Pull age / occupation hints out of the description (best-effort, deterministic)."""
    innate: dict[str, str] = {}
    m = _AGE.search(description)
    if m:
        innate["age"] = m.group(1) or m.group(2)
    low = description.lower()
    for occ in OCCUPATION_LEXICON:
        if occ in low:
            innate["occupation"] = occ
            break
    return innate
