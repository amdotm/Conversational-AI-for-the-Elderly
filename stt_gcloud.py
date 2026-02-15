from __future__ import annotations

from typing import Iterable, Optional
from google.cloud import speech_v1p1beta1 as speech

_SPEECH_CLIENT: Optional[speech.SpeechClient] = None


def get_speech_client() -> speech.SpeechClient:
    """
    Lazily create and then reuse the Google Speech client.
    Creating a new SpeechClient per turn can add several seconds of latency due to
    gRPC channel setup, auth, and DNS resolution.
    """
    global _SPEECH_CLIENT
    if _SPEECH_CLIENT is None:
        _SPEECH_CLIENT = speech.SpeechClient()
    return _SPEECH_CLIENT


def make_streaming_config(sample_rate: int, language_code: str) -> speech.StreamingRecognitionConfig:
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    return speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False,
    )


def streaming_recognize(
    requests: Iterable[speech.StreamingRecognizeRequest],
    streaming_config: speech.StreamingRecognitionConfig,
):
    client = get_speech_client()
    return client.streaming_recognize(config=streaming_config, requests=requests)


def transcribe_bytes(audio_bytes: bytes, sample_rate: int, language_code: str) -> str:
    """
    Non-streaming transcription for short audio clips (e.g., onboarding).
    Uses the same cached SpeechClient to avoid repeated setup latency.
    """
    if not audio_bytes:
        return ""

    client = get_speech_client()

    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_automatic_punctuation=True,
        model="latest_long",
    )

    resp = client.recognize(config=config, audio=audio)

    parts = []
    for r in resp.results:
        if r.alternatives:
            parts.append((r.alternatives[0].transcript or "").strip())

    return " ".join([p for p in parts if p]).strip()
