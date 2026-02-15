# memory.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Set

def _count_questions(text: str) -> int:
    if not text:
        return 0
    return text.count("?")


def _extract_asked_topics(bot_text: str) -> Set[str]:
    """
    Extract key phrases from bot questions to track what's been asked.
    This helps avoid repetitive questions like "favorite spot" over and over.
    """
    topics = set()
    if not bot_text:
        return topics
    
    low = bot_text.lower()
    
    # Track common question patterns
    if "favorite" in low:
        topics.add("favorite")
    if "spot" in low:
        topics.add("spot")
    if "place" in low:
        topics.add("place")
    if "memory" in low or "memories" in low:
        topics.add("memory")
    if "remember" in low:
        topics.add("remember")
    if "like to do" in low or "do you do" in low:
        topics.add("activities")
    if "children" in low or "kids" in low:
        topics.add("children")
    if "wife" in low or "husband" in low or "spouse" in low:
        topics.add("spouse")
    if "family" in low:
        topics.add("family")
    
    # Track common conversational topics
    if "tea" in low:
        topics.add("tea")
    if "coffee" in low:
        topics.add("coffee")
    if "weather" in low:
        topics.add("weather")
    if "cold" in low or "warm" in low or "hot" in low:
        topics.add("weather")
    if "cup" in low or "mug" in low:
        topics.add("tea")
    if "quiet" in low or "silence" in low or "peaceful" in low:
        topics.add("quiet")
    if "calm" in low or "relax" in low:
        topics.add("relaxation")
    
    # Track neighborhood/change themes (common repetition trap)
    if "neighborhood" in low or "neighbour" in low:
        topics.add("neighborhood")
    if "change" in low or "changes" in low or "changed" in low:
        topics.add("changes")
    if "years" in low and ("living" in low or "lived" in low):
        topics.add("living_duration")
    if "adjust" in low or "adjustment" in low:
        topics.add("adjustment")
    if "new people" in low or "new neighbors" in low or "more people" in low:
        topics.add("new_neighbors")
    
    return topics


@dataclass
class MemoryState:
    summary_text: str = ""

    # Turn counter
    turn_index: int = 0

    # Dialogue control
    questions_last_3: Deque[int] = field(default_factory=lambda: deque(maxlen=3))
    last_bot_act: str = ""
    last_nonfluency: str = "FLUENT"
    pause_tier: str = "MEDIUM"

    # Topic tracking
    last_user_topic_hint: str = ""

    asked_topics: Set[str] = field(default_factory=set)
    
    banned_topics: Set[str] = field(default_factory=set)
    
    recent_bot_questions: Deque[str] = field(default_factory=lambda: deque(maxlen=3))
    
    turns_on_current_topic: int = 0
    current_topic: str = ""

    # Non-fluency stats
    counts: Dict[str, int] = field(default_factory=lambda: {
        "FLUENT": 0, "HESITANT": 0, "ABANDONED": 0, "SELF_REPAIR": 0, "INTERRUPTED": 0
    })


def init_memory() -> MemoryState:
    return MemoryState()


def update_memory(
    mem: MemoryState,
    user_text: str,
    bot_text: str,
    bot_act: str,
    nonfluency_label: str,
    max_summary_chars: int = 900,
) -> MemoryState:
    mem.turn_index += 1

    # Update counters
    mem.last_bot_act = bot_act
    mem.last_nonfluency = nonfluency_label
    if nonfluency_label in mem.counts:
        mem.counts[nonfluency_label] += 1

    # Track question pressure
    mem.questions_last_3.append(_count_questions(bot_text))
    new_topics = _extract_asked_topics(bot_text)
    mem.asked_topics.update(new_topics)
    if bot_text and "?" in bot_text:
        mem.recent_bot_questions.append(bot_text.strip())
    user_low = (user_text or "").lower()
    # Detect if user is frustrated or wants to move on
    if any(phrase in user_low for phrase in ["i told you", "already told", "stop asking", "move on"]):
        mem.turns_on_current_topic = 99  # Force topic change
    else:
        mem.turns_on_current_topic += 1

    # Topic hint
    if user_text:
        words = user_text.strip().split()
        mem.last_user_topic_hint = " ".join(words[-8:]) if words else ""

    # Update summary
    add = ""
    if user_text:
        add += f"User mentioned: {user_text.strip()}\n"

    s = (mem.summary_text + "\n" + add).strip()
    mem.summary_text = s[-max_summary_chars:]

    return mem


def question_pressure(mem: MemoryState) -> int:
    """Total questions asked across last ~3 bot turns."""
    return int(sum(mem.questions_last_3)) if mem.questions_last_3 else 0


def get_avoided_topics(mem: MemoryState) -> str:
    """
    Return a string of topics to avoid asking about again.
    Used to inject into the prompt.
    """
    parts = []
    
    # Banned topics 
    if mem.banned_topics:
        parts.append(f"BANNED TOPICS (user complained - NEVER mention these): {', '.join(mem.banned_topics)}")
    
    # Already discussed
    if mem.asked_topics:
        parts.append(f"Already discussed (don't repeat): {', '.join(mem.asked_topics)}")
    
    return "\n".join(parts) if parts else ""


def add_banned_topic(mem: MemoryState, topic: str) -> MemoryState:
    """Add a topic to the banned list after user complained about it."""
    mem.banned_topics.add(topic.lower())
    return mem


def should_change_topic(mem: MemoryState) -> bool:
    """
    Returns True if we've been on the same topic too long (3+ turns)
    or if user expressed frustration.
    """
    return mem.turns_on_current_topic >= 3


def reset_topic_counter(mem: MemoryState) -> MemoryState:
    """Call this when changing to a new topic."""
    mem.turns_on_current_topic = 0
    return mem