from __future__ import annotations

from enum import Enum, auto
import re
from typing import List



# Repeat intent classification

class RepeatIntent(Enum):
    NONE = auto()
    REPEAT_QUESTION = auto()
    REPEAT_LAST_UTTERANCE = auto()
    CLARIFY_OR_REPHRASE = auto()


# Triggers that suggest the user did not hear / wants repetition.
_REPEAT_TRIGGERS: List[str] = [
    "repeat",
    "say again",
    "again please",
    "once more",
    "didn't hear",
    "did not hear",
    "didn't catch",
    "did not catch",
    "pardon",
    "sorry i didn't hear",
    "could you repeat",
]

# Signals that the user is complaining about repetition, not requesting it.
_NEGATIVE_REPEAT_SIGNALS: List[str] = [
    # Direct complaints
    "stop repeating",
    "don't repeat",
    "do not repeat",
    "keep repeating",
    "going in circles",
    "circles",
    "you repeating me",
    "you repeating",
    "were repeating me",
    "were repeating",
    "are repeating me",
    "are repeating",
    "why repeat",
    "why are you repeating",
    "why were you repeating",
    "why you repeating",
    "you don't have to repeat",
    "don't have to repeat",
    "don't need to repeat",
    "no need to repeat",
    "necessarily have to repeat",
    "move on from repeating",
    "move on with",
    "from repeating",
    "that's what i said",
    "what i said",
    "i just said",
    "i already said",
    "already told you",
    "just told you",
]


def classify_repeat_intent(text: str) -> RepeatIntent:
    """
    Conservative repeat-intent classifier.

    Key design choice:
    - Do NOT treat any 'repeat-like' token as an instruction to re-say a long
      previous response. When ambiguous, we prefer to clarify.
    - IMPORTANT: Check for complaints FIRST before checking for repeat triggers.
    """
    if not text:
        return RepeatIntent.NONE

    low = text.lower().strip()

    # FIRST: Check if user is complaining about repetition 
    if any(sig in low for sig in _NEGATIVE_REPEAT_SIGNALS):
        return RepeatIntent.NONE  

    # Not a repeat signal
    if not any(trg in low for trg in _REPEAT_TRIGGERS):
        return RepeatIntent.NONE

    # User likely missed the question / end of sentence
    if (
        "question" in low
        or "last part" in low
        or "end of the sentence" in low
        or "end of your sentence" in low
        or "what did you say" in low
        or "what you said" in low
    ):
        return RepeatIntent.REPEAT_QUESTION

    # Ambiguous: clarify rather than repeating a paragraph
    return RepeatIntent.CLARIFY_OR_REPHRASE


# Filler handling (two pipelines)

_FILLERS_ANALYSIS = [
    r"\bum\b", r"\buh\b", r"\berm\b", r"\beh\b", r"\bah\b",
    r"\bhmm\b", r"\bumm\b",
    r"\byou know\b", r"\bi mean\b", r"\blike\b",
    r"\bsort of\b", r"\bkinda\b", r"\bkind of\b", r"\bwell\b",
]

# B) LLM fillers: minimal acoustic fillers only (low-information, non-lexical).
_FILLERS_LLM = [
    r"\bum\b", r"\buh\b", r"\berm\b", r"\beh\b", r"\bah\b", r"\bhmm\b", r"\bumm\b"
]

_WS = re.compile(r"\s+")


def _remove_patterns(text: str, patterns: List[str]) -> str:
    out = text
    for p in patterns:
        out = re.sub(p, "", out, flags=re.IGNORECASE)
    out = _WS.sub(" ", out)
    return out.strip()


def clean_for_llm(text: str) -> str:
    """
    Minimal cleaning before LLM:
    - Removes only acoustic fillers (um/uh/erm/etc.)
    - Preserves discourse markers like "well", "you know", "I mean", "like"
      because they can encode conversational intent and narrative pacing.
    """
    if not text:
        return ""
    return _remove_patterns(text, _FILLERS_LLM)


def clean_for_analysis(text: str) -> str:
    """
    Optional cleaning for analysis:
    - Removes both acoustic fillers and pragmatic discourse markers.
    - Use only if you explicitly want a normalized variant for some metric.

    Note: For nonfluency research, you often want the *raw* text, so treat this
    as optional and do not apply it globally.
    """
    if not text:
        return ""
    return _remove_patterns(text, _FILLERS_ANALYSIS)