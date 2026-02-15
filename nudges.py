# nudges.py
"""
Nudges for when user is silent or has nothing to say.
Based on P3's feedback: Instead of passive reassurances ("No rush"), 
offer concrete topics to help the user.
"""
import random

# Topic-offering nudges
TOPIC_NUDGES = [
    # Family
    "Tell me about your family. Do you have children or grandchildren?",
    "What was your family like growing up?",
    "Where did your family come from originally?",
    
    # Work/Career
    "What kind of work did you do? Did you enjoy it?",
    "What was your working life like?",
    "Did you have a job you particularly enjoyed?",
    
    # Childhood/Past
    "What was your childhood like? Where did you grow up?",
    "Do you have any favorite memories from when you were young?",
    "What was school like for you?",
    
    # Hobbies/Interests
    "Do you have any hobbies or things you enjoy doing?",
    "What do you like to do in your free time?",
    "Is there something you've always enjoyed doing?",
    
    # Daily life
    "What does a typical day look like for you?",
    "What have you been up to lately?",
    "Is there anything nice that happened recently?",
    
    # Places
    "Have you traveled anywhere interesting in your life?",
    "Is there a place that's special to you?",
    "Where's your favorite place you've ever been?",
    
    # General open
    "Is there anything you'd like to talk about?",
    "What's on your mind today?",
]

# For when conversation has been going in circles on same topic
TOPIC_CHANGE_NUDGES = [
    "Let's talk about something different. What was your working life like?",
    "How about we switch topics? Tell me about your family.",
    "I'm curious about something else — do you have any hobbies you enjoy?",
    "Let me ask you something different. Where did you grow up?",
    "I'd love to hear about a different part of your life. What do you like to do for fun?",
]

# Track which nudges have been used to avoid repetition
_used_nudges = set()


def next_nudge(question_budget: int = 1, topic_change: bool = False) -> str:
    """
    Get a nudge that offers a concrete topic for discussion.
    
    Args:
        question_budget: If 0, use a statement form. If 1, can ask a question.
        topic_change: If True, explicitly change to a new topic.
    """
    global _used_nudges
    
    if topic_change:
        available = [n for n in TOPIC_CHANGE_NUDGES if n not in _used_nudges]
        if not available:
            _used_nudges.clear()
            available = TOPIC_CHANGE_NUDGES
        choice = random.choice(available)
        _used_nudges.add(choice)
        return choice
    
    available = [n for n in TOPIC_NUDGES if n not in _used_nudges]
    if not available:
        _used_nudges.clear()
        available = TOPIC_NUDGES
    
    choice = random.choice(available)
    _used_nudges.add(choice)
    return choice


def reset_nudges():
    """Reset the used nudges tracker (call at start of session)."""
    global _used_nudges
    _used_nudges.clear()