from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from memory import MemoryState, question_pressure

DialogueAct = Literal[
    "COMMENT",
    "ELABORATE",
    "SUMMARIZE",
    "CLARIFY",
    "ANCHOR_AND_RESUME",
    "REPAIR_GENTLE",
    "NUDGE",
    "PROGRESS_TOPIC", 
]

@dataclass
class PolicyDecision:
    act: DialogueAct
    question_budget: int
    pause_tier: Literal["FAST", "MEDIUM", "SLOW"]
    max_sentences: int

def _needs_clarification(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return True
    low = t.lower()
    # Ambiguity signals
    if low in {"i'm", "it was", "i was", "maybe", "not sure"}:
        return True
    if t.endswith("..."):
        return True
    # Too many pronouns without content
    if len(t.split()) <= 3 and any(p in low for p in ["it", "that", "this", "he", "she", "they"]):
        return True
    return False

def decide_policy(
    mem: MemoryState,
    *,
    user_text: str,
    repair_action: str,
    nonfluency_label: str,
    repeat_request: bool,
    was_interrupt: bool,
) -> PolicyDecision:
    """
    Stable policy with two key goals:
    1) Avoid "interrogation mode" (question after every turn)
    2) Reduce parroting by preferring COMMENT and topic progression over summarization

    Practical changes for the second test:
    - After ~4 turns, fluent users should receive fewer questions by default.
    """
    # Pause tier 
    pause_tier: Literal["FAST", "MEDIUM", "SLOW"] = "MEDIUM"
    if nonfluency_label in {"HESITANT", "ABANDONED", "SELF_REPAIR", "INTERRUPTED"}:
        pause_tier = "SLOW"

    # Only use FAST after sustained fluency; otherwise MEDIUM
    if mem.last_nonfluency == "FLUENT" and mem.counts.get("FLUENT", 0) >= 4:
        pause_tier = "FAST"
    else:
        pause_tier = "MEDIUM"

    # Question budget control
    pressure = question_pressure(mem)

    # Opening phase:
    if mem.turn_index < 3:
        base_budget = 1
    else:
        
        base_budget = 0 if pressure >= 1 else 1

    # Repair-driven acts
    if repair_action == "NO_SPEECH":
        return PolicyDecision(act="NUDGE", question_budget=base_budget, pause_tier=pause_tier, max_sentences=1)

    if repair_action == "VERY_SHORT":
        return PolicyDecision(act="REPAIR_GENTLE", question_budget=1, pause_tier="SLOW", max_sentences=1)

    if repeat_request or was_interrupt or nonfluency_label == "INTERRUPTED":
        qb = 0 if pressure >= 1 else 1
        return PolicyDecision(act="ANCHOR_AND_RESUME", question_budget=qb, pause_tier="SLOW", max_sentences=2)

    # Normal flow
    if _needs_clarification(user_text):
        return PolicyDecision(act="CLARIFY", question_budget=1, pause_tier="SLOW", max_sentences=1)

    # If user is hesitant or self-repairing, keep low-pressure and do not question.
    if nonfluency_label in {"HESITANT", "SELF_REPAIR"}:
        return PolicyDecision(act="COMMENT", question_budget=0, pause_tier="SLOW", max_sentences=2)

    if mem.turn_index >= 4 and nonfluency_label == "FLUENT":
        
        if mem.turn_index % 3 == 0:  # Every 3rd turn, try to progress
            return PolicyDecision(act="PROGRESS_TOPIC", question_budget=0, pause_tier=pause_tier, max_sentences=2)
        return PolicyDecision(act="COMMENT", question_budget=0, pause_tier=pause_tier, max_sentences=2)

    
    if pressure >= 2:
        return PolicyDecision(act="COMMENT", question_budget=0, pause_tier=pause_tier, max_sentences=2)

    
    return PolicyDecision(act="ELABORATE", question_budget=base_budget, pause_tier=pause_tier, max_sentences=2)
