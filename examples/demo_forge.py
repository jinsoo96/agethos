"""Persona forge demo — a free-text description becomes a typed persona config.

Run offline (deterministic lexicon path):
    python examples/demo_forge.py

Run with a real LLM (any OpenAI-compatible endpoint):
    OPENAI_API_KEY=... python examples/demo_forge.py --llm openai --model gpt-4o-mini
"""
import argparse
import asyncio

from agethos import Brain, forge
from agethos.forge import plan_vectors
from agethos.steering import MockSteeringBackend

DESCRIPTION = (
    "까칠한데 은근 정 많은 시니어 백엔드 개발자 아저씨. "
    "A gruff but secretly caring senior backend developer."
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=None, help="provider ('openai', 'anthropic', ...)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--description", default=DESCRIPTION)
    args = parser.parse_args()

    adapter = None
    if args.llm:
        from agethos.brain import _resolve_llm
        adapter = _resolve_llm(args.llm, model=args.model, base_url=args.base_url)

    result = await forge(args.description, llm=adapter, name="Minsoo")

    print(f"converged={result.converged} rounds={result.rounds}")
    for t in result.trace:
        print(f"  round {t.round}: fidelity={t.overall:.3f} weak={t.weak}")

    spec = result.spec
    print(f"\nOCEAN: {spec.ocean}")
    print(f"tone: {spec.tone}")
    print(f"rules: {spec.behavioral_rules}")
    print(f"hard constraints: {spec.hard_constraints}")

    print("\nsteering plan (activation-layer mount for open-weight models):")
    plan = result.steering_plan()
    for p in plan:
        print(f"  {p.trait}: direction={p.direction:+d} strength={p.strength}")
    vectors = plan_vectors(plan, MockSteeringBackend())
    print(f"  → {len(vectors)} scaled persona vector(s) extracted")

    print("\nprompt-layer mount (first 400 chars):")
    print(result.render()[:400])

    if adapter:
        brain = Brain(persona=spec, llm=adapter)
        reply = await brain.chat("제 코드 리뷰 좀 해주실 수 있나요? 좀 급해서요...")
        print(f"\n[chat] {reply[:400]}")


if __name__ == "__main__":
    asyncio.run(main())
