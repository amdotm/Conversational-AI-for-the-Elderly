from __future__ import annotations
from typing import Literal
from utterance_features import UtteranceFeatures

NonFluencyLabel = Literal["FLUENT", "HESITANT", "ABANDONED", "SELF_REPAIR", "INTERRUPTED"]

def classify_nonfluency(
    features: UtteranceFeatures,
    *,
    repair_action: str,
    repeat_request: bool,
    was_interrupt: bool,
) -> NonFluencyLabel:
    """
    Rule-based classification (thesis-friendly and explainable).
    Uses local signals only (no extra LLM calls).
    """
    if was_interrupt:
        return "INTERRUPTED"

    if repair_action in {"NO_SPEECH"}:
        return "HESITANT"

    if repeat_request:
        return "INTERRUPTED"

    if features.word_count <= 2 and features.has_trailing_conjunction:
        return "ABANDONED"

    if features.has_repair_marker or features.repetition_score >= 0.25:
        return "SELF_REPAIR"

    if features.word_count <= 3 and not features.ends_with_punctuation:
        return "HESITANT"

    return "FLUENT"
