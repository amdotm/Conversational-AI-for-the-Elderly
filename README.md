# Elderly Voice Chatbot (Master's Thesis Project)

This repository contains the code for a Master's thesis project at DTU.

**Conversational AI for Elderly: Designing voice based dialogue systems to accommodate non fluent speech.**

The project develops and evaluates a voice based conversational AI prototype designed for elderly users, with a specific focus on:

- Non fluent speech (pauses, hesitations, self corrections)
- Interruptions and turn taking robustness
- Adaptive dialogue behaviour (pacing, confirmation, repair strategies)

## File Structure

| File | Description |
|------|-------------|
| `main_live.py` | Main application loop, audio capture, STT/TTS integration, turn management |
| `config.py` | All configurable parameters (pause tiers, model settings, TTS voice, pricing) |
| `dialogue_policy.py` | Deterministic dialogue policy engine (dialogue act selection, question budget) |
| `prompts.py` | System prompt templates for GPT 4.1 |
| `llm_provider.py` | OpenAI API wrapper for language generation |
| `llm_guards.py` | Intent classification and anti parroting guard |
| `stt_gcloud.py` | Google Cloud Speech to Text streaming configuration |
| `tts_gcloud.py` | Google Cloud Text to Speech synthesis and playback |
| `nonfluency_classifier.py` | Rule based classifier for speech fluency labels (FLUENT, HESITANT, etc.) |
| `disfluency.py` | Disfluency cleaning module (filler removal, repetition collapsing) |
| `repair.py` | Repair handler (NO_SPEECH, VERY_SHORT, AFFIRMATION, EXIT detection) |
| `memory.py` | Topic tracking, banned topics, conversation state management |
| `nudges.py` | Active nudge topic repertoire for silent or stuck users |
| `utterance_features.py` | Feature extraction from transcripts (word count, fillers, trailing conjunctions) |
| `profile.py` | User profile placeholder |
| `starter_questions.py` | Onboarding question pool |
| `logger.py` | JSONL session logger |
| `audio_play.py` | Audio playback utility |
| `list_voices.py` | Helper to list available Google TTS voices |
| `start_project.sh` | Launch script |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API credentials

The system requires two API credentials:

**Google Cloud** (for Speech to Text and Text to Speech):
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
```

**OpenAI** (for GPT 4.1):
```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Run

```bash
./start_project.sh
```

Or directly:
```bash
python main_live.py
```

## System Requirements

- Python 3.10+
- Linux (tested on Ubuntu via WSL2)
- Working microphone
- Internet connection (for Google Cloud and OpenAI APIs)

## Ethics and Privacy

This repository does not contain participant audio, transcripts, or identifiable data. Session logs used during testing were stored locally and handled according to the study protocol described in the thesis.

## License

MIT
