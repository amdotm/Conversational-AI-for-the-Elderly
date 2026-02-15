"""
LLM-based guards for:
1. Classifying user intent (repeat request vs complaint vs normal)
2. Checking and fixing parroting in bot responses

These replace brittle pattern-matching with dynamic LLM classification.
"""

from __future__ import annotations
from typing import Tuple, Literal
from llm_provider import generate

# Intent Classification
UserIntent = Literal["REPEAT_REQUEST", "COMPLAINT_ABOUT_REPETITION", "NORMAL"]

_INTENT_CLASSIFICATION_PROMPT = """Analyze the user's message and classify their intent into exactly ONE category:

1. REPEAT_REQUEST - User wants the bot to repeat something they didn't hear
   Examples: "Can you say that again?", "I didn't hear you", "Please repeat", "What did you say?", "Pardon?"

2. COMPLAINT_ABOUT_REPETITION - User is frustrated that the bot keeps repeating/restating things
   Examples: "Stop repeating me", "Why do you keep saying what I said?", "Don't repeat everything", "You're going in circles", "Please do not repeat"

3. NORMAL - Any other message (user is just talking normally, even if they mention the word "repeat" in another context)

User message: "{user_text}"

Respond with ONLY the category name (REPEAT_REQUEST, COMPLAINT_ABOUT_REPETITION, or NORMAL):"""


def classify_user_intent(user_text: str) -> Tuple[UserIntent, dict]:
    """
    Use LLM to classify whether the user is:
    - Requesting repetition
    - Complaining about repetition  
    - Neither (normal conversation)
    
    Returns: (intent, usage_dict)
    """
    if not user_text or not user_text.strip():
        return "NORMAL", {}
    
    prompt = _INTENT_CLASSIFICATION_PROMPT.format(user_text=user_text)
    
    response, usage = generate(
        sys_prompt="You are a precise intent classifier. Respond with only the category name.",
        user_text=prompt,
        max_tokens=20,
        temperature=0.0,  
    )
    
    response = response.strip().upper()
    
    # Map response to valid intent
    if "REPEAT_REQUEST" in response:
        return "REPEAT_REQUEST", usage
    elif "COMPLAINT" in response:
        return "COMPLAINT_ABOUT_REPETITION", usage
    else:
        return "NORMAL", usage


# Anti-Parroting Check & Fix

_PARROTING_CHECK_PROMPT = """You are a conversation quality checker.

User said: "{user_text}"
Bot responded: "{bot_text}"

Is the bot PARROTING (restating/echoing what the user said)?

Parroting includes:
- Starting with "It sounds like you..." or "That sounds like..."
- Repeating the user's words back to them
- Summarizing what the user just said instead of adding something new
- Saying "It must be nice/wonderful/lovely that..." followed by user's content

Answer YES if parroting, NO if the bot adds something genuinely new.

Answer (YES or NO):"""

_REWRITE_PROMPT = """You are Olivia, a warm conversational companion for older adults.

User said: "{user_text}"
Your original response: "{bot_text}"

This response parrots/restates what the user said. Rewrite it to:
1. NOT repeat or paraphrase what the user said
2. Add something NEW - a thought, observation, or gentle question
3. Move the conversation forward
4. Keep it to 1-2 short sentences
5. Be warm but not sycophantic

FORBIDDEN phrases: "It sounds like...", "That sounds like...", "It must be...", "That must be...", "How lovely that..."

Rewritten response:"""


def check_and_fix_parroting(user_text: str, bot_text: str) -> Tuple[str, bool, dict]:
    """
    Check if bot response is parroting the user, and fix it if so.
    
    Returns: (final_response, was_fixed, combined_usage)
    """
    if not bot_text or not user_text:
        return bot_text, False, {}
    
    # Step 1: Check if parroting
    check_prompt = _PARROTING_CHECK_PROMPT.format(
        user_text=user_text,
        bot_text=bot_text
    )
    
    check_response, check_usage = generate(
        sys_prompt="You are a precise classifier. Answer only YES or NO.",
        user_text=check_prompt,
        max_tokens=10,
        temperature=0.0,
    )
    
    is_parroting = "YES" in check_response.strip().upper()
    
    if not is_parroting:
        return bot_text, False, check_usage
    
    # Step 2: Rewrite if parroting detected
    rewrite_prompt = _REWRITE_PROMPT.format(
        user_text=user_text,
        bot_text=bot_text
    )
    
    new_response, rewrite_usage = generate(
        sys_prompt="You are Olivia, a warm companion. Rewrite the response without parroting.",
        user_text=rewrite_prompt,
        max_tokens=150,
        temperature=0.7,
    )
    
    # Combine usage stats
    combined_usage = {
        "check_tokens": check_usage.get("total_tokens", 0),
        "rewrite_tokens": rewrite_usage.get("total_tokens", 0),
        "total_tokens": check_usage.get("total_tokens", 0) + rewrite_usage.get("total_tokens", 0),
    }
    
    return new_response.strip(), True, combined_usage