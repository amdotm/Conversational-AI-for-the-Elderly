# Elderly Voice Chatbot (Master’s Thesis Project)

This repository contains the code and documentation for a Master’s thesis project at DTU:

**Conversational AI for Elderly: Oral interaction tuning to accommodate non fluent speech, interruptions etc.**

The project develops and evaluates a voice based conversational AI prototype designed for elderly users, with a specific focus on:
- non fluent speech (pauses, hesitations, self corrections)
- interruptions and turn taking robustness
- adaptive dialogue behaviour (pacing, confirmation, repair strategies)

## Repository structure

- `src/` Core implementation (ASR, dialogue, TTS, evaluation utilities)
- `scripts/` Helper scripts to run the prototype and evaluation
- `docs/` Thesis aligned documentation (architecture, protocol, ethics)
- `data/` Local datasets and recordings (not committed to Git)
- `results/` Local outputs and metrics (not committed to Git)

## Ethics and privacy

This repository does not store participant audio, transcripts, or identifiable data.
Any evaluation artefacts must be stored locally and handled according to the study protocol.
See `docs/ethics_privacy.md`.

## Quick start

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
