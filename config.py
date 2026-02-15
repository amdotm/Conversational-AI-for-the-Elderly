from __future__ import annotations

PROVIDER = "openai"
MODEL_NAME = "gpt-4.1"   
TEMPERATURE = 0.6

# Default generation limit (can be overridden per dialogue act)
MAX_TOKENS_DEFAULT = 220


TTS_LANG = "en-GB"                 
TTS_VOICE = "en-GB-Neural2-F"      
TTS_SPEAKING_RATE = 0.90
TTS_PITCH = -1.5
TTS_GAIN_DB = -1.0


# Per-act max tokens 
MAX_TOKENS_BY_ACT = {
    "COMMENT": 120,
    "ELABORATE": 180,
    "SUMMARIZE": 170,
    "CLARIFY": 110,
    "ANCHOR_AND_RESUME": 160,
    "REPAIR_GENTLE": 130,
    "NUDGE": 110,
    "PROGRESS_TOPIC": 140,  
}

# Pricing in USD per 1K tokens (estimates)
PRICING = {
    "gpt-4o-mini": {"in": 0.00015, "out": 0.00060},
    "gpt-4.1": {"in": 0.002, "out": 0.008},
    "o4-mini": {"in": 0.0011, "out": 0.0044},
}


# Turn-taking / Pause tiering

PAUSE_TIERS_SEC = {
    "FAST": 2.5,     
    "MEDIUM": 3.5,   # default
    "SLOW": 5.5,     
}

# Energy threshold for VAD-lite in streaming loop
STT_ENERGY_THRESHOLD = 30.0

# Minimum listen seconds before allowing long-pause termination
MIN_LISTEN_SEC = 3.0

# Mic warmup discard time
MIC_WARMUP_SEC = 0.4

PREROLL_SEC = 0.3 

# Repair thresholds 
REPAIR_FRAME_MS = 30
REPAIR_AMP_THRESH = 700
REPAIR_RATIO_NO_SPEECH = 0.03
REPAIR_MIN_DUR_NO_SPEECH = 0.5
REPAIR_MIN_WORDS_TRUST_ASR = 3
REPAIR_VERY_SHORT_WORDS = 2
