from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
import numpy as np

from config import (
    REPAIR_FRAME_MS,
    REPAIR_AMP_THRESH,
    REPAIR_RATIO_NO_SPEECH,
    REPAIR_MIN_DUR_NO_SPEECH,
    REPAIR_MIN_WORDS_TRUST_ASR,
    REPAIR_VERY_SHORT_WORDS,
)

class RepairAction(Enum):
    OK = auto()
    NO_SPEECH = auto()
    VERY_SHORT = auto()
    EXIT = auto()
    AFFIRMATION = auto()  

@dataclass
class RepairDecision:
    action: RepairAction
    reason: str
    speech_ratio: float
    duration_sec: float

def _pcm16_mono_to_np(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(audio_bytes, dtype=np.int16)

def _speech_ratio(arr: np.ndarray, sr: int, frame_ms: int, amp_thresh: int) -> Tuple[float, float]:
    if arr.size == 0 or sr <= 0:
        return 0.0, 0.0
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, arr.size // frame_len)
    frames = arr[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.mean(np.abs(frames.astype(np.int32)), axis=1)
    voiced = (energy > amp_thresh).astype(np.float32)
    ratio = float(voiced.mean()) if n_frames > 0 else 0.0
    duration = arr.size / float(sr)
    return ratio, duration


# Affirmation words 
_AFFIRMATIONS = {
    "yes", "yeah", "yep", "yup", "uh huh", "uhuh", "mhm", "mmhm", "mm",
    "okay", "ok", "sure", "right", "true", "exactly", "indeed", "absolutely",
    "of course", "definitely", "certainly",
}

# Short acknowledgments that aren't really affirmations
_SHORT_ACKNOWLEDGMENTS = {
    "i see", "i know", "got it", "hmm", "hm", "ah", "oh",
}


def _is_affirmation(text: str) -> bool:
    """
    Check if user's response is a simple affirmation like yes/yeah/okay.
    These need special handling - bot should continue naturally, not give a nudge.
    """
    if not text:
        return False
    low = text.strip().lower()
    
    if low in _AFFIRMATIONS:
        return True
    
    for aff in _AFFIRMATIONS:
        if low == aff or low == aff + "." or low == aff + "!":
            return True
    
    # "Yeah, [something short]" patterns
    if low.startswith(("yes ", "yeah ", "yep ", "okay ", "ok ")):
        if len(low.split()) <= 4:
            return True
    
    return False


def _is_valid_short_answer(text: str) -> bool:
    """
    Elderly users often answer with 1–2 words meaningfully:
      - "Denmark", "Copenhagen"
      - "Yes", "No"
      - "Sometimes", "Maybe"
      - "Tired", "Fine"
    """
    if not text:
        return False
    low = text.strip().lower()

    if low in {"yes", "no", "yeah", "nope", "yep", "maybe", "sometimes", "often", "rarely"}:
        return True
    if low in {"fine", "good", "okay", "tired", "happy", "sad"}:
        return True

    if len(low.split()) <= 2 and low.replace(" ", "").isalpha():
        return True

    return False

def decide_repair(user_text: str, audio_bytes: bytes, sr: int) -> RepairDecision:
    text = (user_text or "").strip()
    low = text.lower()
    arr = _pcm16_mono_to_np(audio_bytes)
    ratio, dur = _speech_ratio(arr, sr, frame_ms=REPAIR_FRAME_MS, amp_thresh=REPAIR_AMP_THRESH)

    # Exit intents
    exit_keywords = {"exit", "quit", "goodbye", "bye", "good bye", "good night", "stop"}
    exit_phrases = [
        "that's it for today", "that is it for today", "that's all for today",
        "we are done for today", "we're done for today", "we are done now", "we're done now",
        "let's stop here", "let us stop here", "i want to stop", "i want to finish",
        "i think that's it", "we can stop now", "we can finish now",
        "talk to you tomorrow", "we talk tomorrow", "we'll talk tomorrow", "we will talk tomorrow",
        "see you tomorrow",
    ]

    if low in exit_keywords:
        return RepairDecision(RepairAction.EXIT, "exit keyword", ratio, dur)
    if any(p in low for p in exit_phrases):
        return RepairDecision(RepairAction.EXIT, "exit phrase", ratio, dur)

    if _is_affirmation(text):
        return RepairDecision(RepairAction.AFFIRMATION, "user affirmation", ratio, dur)

    words = text.split()
    if len(words) >= REPAIR_MIN_WORDS_TRUST_ASR:
        return RepairDecision(RepairAction.OK, "enough text; trust ASR", ratio, dur)

    if _is_valid_short_answer(text):
        return RepairDecision(RepairAction.OK, "valid short answer", ratio, dur)

    # No speech
    if ratio < REPAIR_RATIO_NO_SPEECH or dur < REPAIR_MIN_DUR_NO_SPEECH:
        return RepairDecision(RepairAction.NO_SPEECH, "low speech activity", ratio, dur)

    # Very short
    if len(words) <= REPAIR_VERY_SHORT_WORDS:
        return RepairDecision(RepairAction.VERY_SHORT, "utterance too short", ratio, dur)

    return RepairDecision(RepairAction.OK, "sufficient input", ratio, dur)