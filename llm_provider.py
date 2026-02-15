import os
import re
from typing import Tuple, Dict, Optional
from config import PROVIDER, MODEL_NAME, TEMPERATURE, MAX_TOKENS_DEFAULT, PRICING



_BANNED_STARTS = [
    r"^it sounds like\b",
    r"^that sounds like\b", 
    r"^it seems like\b",
    r"^that seems like\b",
    r"^it must be\b",
    r"^that must be\b",
    r"^it appears that\b",
]

_BANNED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _BANNED_STARTS]


def _filter_parroting(response: str) -> str:
    """
    Remove common parroting prefixes from LLM output.
    These phrases make the bot sound robotic and repetitive.
    """
    text = response.strip()
    
    for pattern in _BANNED_PATTERNS:
        match = pattern.match(text)
        if match:
            # Remove the banned prefix
            text = text[match.end():].strip()
            if text:
                text = text.lstrip(" ,;:-")
                if text:
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            break
    
    return text


def _openai_generate(
    sys_prompt: str,
    user_text: str,
    *,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    text = resp.choices[0].message.content or ""
    usage = {
        "model": resp.model,
        "provider": "openai",
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    if MODEL_NAME in PRICING:
        p = PRICING[MODEL_NAME]
        usage["usd_estimate"] = (
            (usage["prompt_tokens"] * p["in"] + usage["completion_tokens"] * p["out"]) / 1000
        )
    return text.strip(), usage


def generate(
    sys_prompt: str,
    user_text: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, Dict]:
    mt = MAX_TOKENS_DEFAULT if max_tokens is None else int(max_tokens)
    temp = TEMPERATURE if temperature is None else float(temperature)

    if PROVIDER == "openai":
        raw_text, usage = _openai_generate(sys_prompt, user_text, max_tokens=mt, temperature=temp)
        filtered_text = _filter_parroting(raw_text)
        return filtered_text, usage
    
    raise ValueError(f"Unknown provider: {PROVIDER}")