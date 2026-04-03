"""60명 가상 고객 시뮬레이션 데모.

사용법:
    # 1) 환경변수 설정
    export OPENAI_API_KEY="sk-..."

    # 2) 실행
    python examples/demo_60_customers.py

    # anthropic 사용 시
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/demo_60_customers.py --llm anthropic

    # 로컬 ollama 사용 시 (무료)
    python examples/demo_60_customers.py --llm openai --model qwen2.5:7b --base-url http://localhost:11434/v1
"""

import argparse
import asyncio
import random

from dotenv import load_dotenv

load_dotenv()

from agethos import Brain, OceanTraits, PersonaLayer, PersonaSpec

# ── 60명 고객 프로필 생성 ──────────────────────────────────────────

NAMES = [
    "김민수", "박지영", "이준혁", "최수진", "정태윤", "한소희", "오재현", "윤미래",
    "장동건", "서예린", "강현우", "문채원", "임도현", "배수정", "송지호", "홍다은",
    "권혁진", "조은비", "유승민", "신하영", "남궁원", "황보람", "전지훈", "고은서",
    "류태양", "안서윤", "차민기", "노현정", "하준서", "양미경", "구자윤", "방시혁",
    "우지원", "탁재훈", "피지영", "엄태웅", "변정수", "길은혜", "도경수", "라미란",
    "마동석", "사공진", "아이린", "자효진", "차태현", "카라엘", "파니엘", "나윤선",
    "다니엘", "가은지", "이소라", "박보검", "김태리", "정해인", "손예진", "공유진",
    "이병헌", "전도연", "송강호", "김혜수",
]

OCCUPATIONS = [
    "대학생", "직장인(사무직)", "직장인(영업)", "자영업자", "프리랜서",
    "주부", "공무원", "교사", "간호사", "엔지니어",
    "디자이너", "요리사", "운동선수", "유튜버", "퇴직자",
]

SHOPPING_STYLES = [
    "꼼꼼히 비교하고 리뷰를 반드시 확인함",
    "충동구매 성향, 감성에 끌리면 바로 결제",
    "가성비를 최우선으로 따짐",
    "브랜드 충성도가 높아 새 제품도 같은 브랜드 선호",
    "주변 추천에 크게 영향받음",
    "최신 트렌드를 빠르게 따라감",
    "필요한 것만 최소한으로 구매",
    "프리미엄 품질이면 가격은 상관없음",
]


def make_customers(n: int = 60) -> list[dict]:
    """N명의 다양한 고객 프로필 생성."""
    customers = []
    for i in range(n):
        customers.append({
            "name": NAMES[i % len(NAMES)],
            "age": str(random.randint(18, 65)),
            "occupation": random.choice(OCCUPATIONS),
            "shopping_style": random.choice(SHOPPING_STYLES),
            "ocean": {
                "O": round(random.uniform(0.1, 0.9), 1),  # 개방성
                "C": round(random.uniform(0.1, 0.9), 1),  # 성실성
                "E": round(random.uniform(0.1, 0.9), 1),  # 외향성
                "A": round(random.uniform(0.1, 0.9), 1),  # 우호성
                "N": round(random.uniform(0.1, 0.9), 1),  # 신경성
            },
        })
    return customers


def build_brain(customer: dict, llm: str, model: str | None, base_url: str | None) -> Brain:
    """고객 프로필로 Brain 생성."""
    o = customer["ocean"]
    persona = PersonaSpec(
        name=customer["name"],
        ocean=OceanTraits(
            openness=o["O"],
            conscientiousness=o["C"],
            extraversion=o["E"],
            agreeableness=o["A"],
            neuroticism=o["N"],
        ),
        l0_innate=PersonaLayer(traits={
            "age": customer["age"],
            "occupation": customer["occupation"],
            "shopping_style": customer["shopping_style"],
        }),
        tone="자연스러운 한국어, 본인 성격대로 솔직하게",
        behavioral_rules=[
            "실제 고객처럼 자연스럽게 반응할 것",
            "2~3문장으로 짧게 답변할 것",
            "본인의 직업, 나이, 소비 습관에 맞게 반응할 것",
        ],
    )

    kwargs = {"model": model} if model else {}
    if base_url:
        kwargs["base_url"] = base_url

    return Brain.build(persona=persona, llm=llm, **kwargs)


async def simulate(
    question: str,
    customers: list[dict],
    llm: str,
    model: str | None,
    base_url: str | None,
    concurrency: int = 5,
):
    """고객들에게 질문하고 응답 수집."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def ask_one(c: dict):
        async with sem:
            brain = build_brain(c, llm, model, base_url)
            reply = await brain.chat(question)
            return {**c, "reply": reply}

    tasks = [ask_one(c) for c in customers]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        o = result["ocean"]
        print(f"\n[{result['name']} / {result['age']}세 / {result['occupation']}]")
        print(f"  성격: O={o['O']} C={o['C']} E={o['E']} A={o['A']} N={o['N']}")
        print(f"  소비: {result['shopping_style']}")
        print(f"  → {result['reply']}")
        results.append(result)

    return results


def summarize(results: list[dict]):
    """간단한 통계 요약."""
    positive = [r for r in results if any(kw in r["reply"] for kw in ["좋", "관심", "써볼", "해볼", "구매", "신청", "괜찮", "궁금"])]
    negative = [r for r in results if any(kw in r["reply"] for kw in ["필요없", "비싸", "별로", "안 ", "글쎄", "아니"])]

    print(f"\n{'='*60}")
    print(f"  총 {len(results)}명 응답")
    print(f"  긍정 반응 (키워드 기반 추정): ~{len(positive)}명")
    print(f"  부정 반응 (키워드 기반 추정): ~{len(negative)}명")
    print(f"  모호/중립: ~{len(results) - len(positive) - len(negative)}명")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="60명 가상 고객 시뮬레이션")
    parser.add_argument("--llm", default="openai", help="LLM 제공자 (openai/anthropic)")
    parser.add_argument("--model", default=None, help="모델명 (기본: gpt-4o-mini)")
    parser.add_argument("--base-url", default=None, help="커스텀 API URL (ollama 등)")
    parser.add_argument("--count", type=int, default=60, help="고객 수 (기본: 60)")
    parser.add_argument("--concurrency", type=int, default=5, help="동시 요청 수 (기본: 5)")
    parser.add_argument("--question", default=None, help="고객에게 물어볼 질문")
    args = parser.parse_args()

    question = args.question or "새로 나온 AI 영어 학습 앱이 월 9,900원인데요, 관심 있으세요? 첫 달 무료 체험도 가능합니다."

    print(f"{'='*60}")
    print(f"  가상 고객 시뮬레이션")
    print(f"  고객 수: {args.count}명 | LLM: {args.llm} | 동시성: {args.concurrency}")
    print(f"  질문: {question}")
    print(f"{'='*60}")

    customers = make_customers(args.count)
    results = await simulate(question, customers, args.llm, args.model, args.base_url, args.concurrency)
    summarize(results)


if __name__ == "__main__":
    asyncio.run(main())
