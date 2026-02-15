#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
# set your creds path if you use a service account file:
# export GOOGLE_APPLICATION_CREDENTIALS="$HOME/keys/gcp-speech.json"
# export OpenAI key if using API
export OPENAI_API_KEY="$(cat $HOME/.secrets/openai.key 2>/dev/null || echo "")"
python main_live.py
