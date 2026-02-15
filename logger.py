import json
import time
from typing import Any, Dict, Optional

def log_turn(
    log_path: str,
    user_text: str,
    bot_text: str,
    stt_ms: int,
    llm_ms: int,
    tts_ms: int,
    note: str = "",
    usage: Optional[Dict[str, Any]] = None,
    *,
    dialogue_act: str = "",
    nonfluency_label: str = "",
    pause_tier: str = "",
    question_budget: int = -1,
    repair_action: str = "",
    repair_reason: str = "",
) -> None:
    entry = {
        "ts": time.time(),
        "user": user_text,
        "bot": bot_text,
        "stt_ms": stt_ms,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
        "note": note,
        "dialogue_act": dialogue_act,
        "nonfluency_label": nonfluency_label,
        "pause_tier": pause_tier,
        "question_budget": question_budget,
        "repair_action": repair_action,
        "repair_reason": repair_reason,
    }
    if usage:
        entry["llm_usage"] = usage

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
