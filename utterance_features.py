from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

_TRAILING_CONJ = re.compile(r"\b(and|because|so|but|then|or)\b[.!\s]*$", flags=re.IGNORECASE)
_REPAIR_MARKERS = re.compile(r"\b(i mean|sorry|no\s+wait|no,\s*wait|let me)\b", flags=re.IGNORECASE)

@dataclass
class UtteranceFeatures:
    word_count: int
    has_trailing_conjunction: bool
    has_repair_marker: bool
    repetition_score: float
    ends_with_punctuation: bool

def _repetition_score(words: List[str]) -> float:
    if not words:
        return 0.0
    # Simple repetition
    tail = words[-10:]
    unique = len(set(tail))
    return float(max(0, len(tail) - unique)) / float(len(tail))

def extract_features(text: str) -> UtteranceFeatures:
    t = (text or "").strip()
    words = [w.lower() for w in t.split() if w.strip()]
    wc = len(words)

    has_trailing = bool(_TRAILING_CONJ.search(t))
    has_marker = bool(_REPAIR_MARKERS.search(t))
    rep = _repetition_score(words)

    ends_punct = t.endswith((".", "!", "?")) if t else False

    return UtteranceFeatures(
        word_count=wc,
        has_trailing_conjunction=has_trailing,
        has_repair_marker=has_marker,
        repetition_score=rep,
        ends_with_punctuation=ends_punct,
    )
