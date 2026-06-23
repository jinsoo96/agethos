"""agethos.eval — in-box harness: persona/retrieval/transplant metrics + LoCoMo & Sotopia adapters."""
from agethos.eval.locomo import evaluate_recall, ingest_conversation
from agethos.eval.metrics import (
    ocean_similarity,
    persona_consistency,
    persona_drift_curve,
    retrieval_metrics,
    text_similarity,
    transplant_fidelity,
)
from agethos.eval.sotopia import persona_to_sotopia_profile, score_episode

__all__ = [
    "text_similarity", "ocean_similarity", "persona_consistency", "persona_drift_curve",
    "retrieval_metrics", "transplant_fidelity",
    "ingest_conversation", "evaluate_recall",
    "persona_to_sotopia_profile", "score_episode",
]
