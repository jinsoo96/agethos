"""Persona injection demo — same questions, different personalities, compare responses."""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from agethos import Brain, OceanTraits, PersonaLayer, PersonaSpec
from agethos.llm.openai import OpenAIAdapter


AGENTS = {
    "minsoo": PersonaSpec(
        name="Minsoo",
        ocean=OceanTraits(openness=0.8, conscientiousness=0.9, extraversion=0.2, agreeableness=0.6, neuroticism=0.3),
        l0_innate=PersonaLayer(traits={"age": "28", "occupation": "Backend Engineer"}),
        tone="Concise and analytical, prefers technical precision",
        values=["Code quality", "System reliability"],
        behavioral_rules=["Always think before speaking", "Prefer data over opinions", "Keep responses short and structured"],
    ),
    "yuna": PersonaSpec(
        name="Yuna",
        ocean=OceanTraits(openness=0.9, conscientiousness=0.4, extraversion=0.9, agreeableness=0.8, neuroticism=0.6),
        l0_innate=PersonaLayer(traits={"age": "25", "occupation": "UX Designer"}),
        tone="Energetic, expressive, uses exclamation marks and emojis",
        values=["User experience", "Creativity", "Collaboration"],
        behavioral_rules=["Be enthusiastic and encouraging", "Use vivid metaphors and examples", "Ask follow-up questions to engage"],
    ),
}

DEFAULT_QUESTIONS = [
    "What do you think about AI replacing human jobs?",
    "Our project deadline got moved up by a week. How should we handle this?",
    "I made a mistake in production. What should I do?",
]


def print_header(text: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_agent(brain: Brain, response: str) -> None:
    p = brain.persona
    o = p.ocean
    tag = f"{p.name} | {p.l0_innate.traits.get('occupation', '')} | O={o.openness} C={o.conscientiousness} E={o.extraversion} A={o.agreeableness} N={o.neuroticism}"
    emo = brain.emotion
    print(f"\n  [{tag}]")
    print(f"  Emotion: {emo.closest_emotion()} (P={emo.pleasure:+.2f}, A={emo.arousal:+.2f}, D={emo.dominance:+.2f})")
    print(f"  >>> {response}\n")


async def run_comparison(model: str, questions: list[str]) -> None:
    llm = OpenAIAdapter(model=model)
    brains = {name: Brain(persona=spec, llm=llm) for name, spec in AGENTS.items()}

    print_header("PERSONA COMPARISON TEST")
    print(f"  Model: {model}")
    print(f"  Agents: {', '.join(f'{s.name} (E={s.ocean.extraversion}, N={s.ocean.neuroticism})' for s in AGENTS.values())}")

    # -- Q&A comparison --
    for q in questions:
        print_header(f"Q: {q}")
        for brain in brains.values():
            r = await brain.chat(q)
            print_agent(brain, r)

    # -- Emotional event --
    event_pad = (-0.6, 0.5, -0.4)
    print_header(f"EMOTIONAL EVENT: harsh criticism  PAD={event_pad}")

    for name, brain in brains.items():
        before = brain.emotion.closest_emotion()
        brain.apply_event_emotion(event_pad)
        after = brain.emotion.closest_emotion()
        n = brain.persona.ocean.neuroticism
        print(f"  {brain.persona.name} (N={n}): {before} -> {after}  {brain.emotion.to_prompt()}")

    # -- Post-event response --
    q = "How are you feeling right now?"
    print_header(f"Q (post-event): {q}")
    for brain in brains.values():
        r = await brain.chat(q)
        print_agent(brain, r)

    # -- Emotion decay --
    print_header("EMOTION DECAY (10 steps, rate=0.2)")
    for brain in brains.values():
        before = f"{brain.emotion.closest_emotion()} P={brain.emotion.pleasure:+.2f}"
        for _ in range(10):
            brain.decay_emotion(rate=0.2)
        after = f"{brain.emotion.closest_emotion()} P={brain.emotion.pleasure:+.2f}"
        print(f"  {brain.persona.name}: {before}  ->  {after}")


async def run_interactive(model: str, agent_name: str) -> None:
    if agent_name not in AGENTS:
        print(f"Unknown agent: {agent_name}. Available: {', '.join(AGENTS.keys())}")
        return

    llm = OpenAIAdapter(model=model)
    brain = Brain(persona=AGENTS[agent_name], llm=llm)

    p = brain.persona
    o = p.ocean
    print_header(f"INTERACTIVE MODE — {p.name}")
    print(f"  {p.l0_innate.traits.get('occupation', '')} | O={o.openness} C={o.conscientiousness} E={o.extraversion} A={o.agreeableness} N={o.neuroticism}")
    print(f"  Tone: {p.tone}")
    print(f"  Emotion: {brain.emotion.closest_emotion()} {brain.emotion.to_prompt()}")
    print(f"\n  Commands:  :q = quit  :emo P A D = apply emotion  :decay = decay emotion\n")

    while True:
        try:
            user_input = input(f"You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input == ":q":
            break
        if user_input.startswith(":emo "):
            try:
                parts = user_input.split()
                pad = (float(parts[1]), float(parts[2]), float(parts[3]))
                brain.apply_event_emotion(pad)
                print(f"  [Emotion shifted] {brain.emotion.closest_emotion()} {brain.emotion.to_prompt()}")
            except (IndexError, ValueError):
                print("  Usage: :emo <P> <A> <D>  (e.g. :emo -0.5 0.4 -0.3)")
            continue
        if user_input == ":decay":
            brain.decay_emotion(rate=0.2)
            print(f"  [Emotion decayed] {brain.emotion.closest_emotion()} {brain.emotion.to_prompt()}")
            continue

        response = await brain.chat(user_input)
        emo = brain.emotion
        print(f"\n{p.name} [{emo.closest_emotion()}] > {response}\n")


async def main():
    parser = argparse.ArgumentParser(description="Agethos persona demo")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("compare", help="Compare two agents side-by-side")

    chat_parser = sub.add_parser("chat", help="Interactive chat with an agent")
    chat_parser.add_argument("agent", choices=list(AGENTS.keys()), help="Agent to chat with")

    args = parser.parse_args()

    if args.command == "chat":
        await run_interactive(args.model, args.agent)
    else:
        await run_comparison(args.model, DEFAULT_QUESTIONS)


if __name__ == "__main__":
    asyncio.run(main())
