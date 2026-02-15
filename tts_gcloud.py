from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave
from typing import Optional

from google.cloud import texttospeech

from config import (
    TTS_LANG,
    TTS_VOICE,
    TTS_SPEAKING_RATE,
    TTS_PITCH,
    TTS_GAIN_DB,
)

_TTS_CLIENT: Optional[texttospeech.TextToSpeechClient] = None


def _get_client() -> texttospeech.TextToSpeechClient:
    global _TTS_CLIENT
    if _TTS_CLIENT is None:
        _TTS_CLIENT = texttospeech.TextToSpeechClient()
    return _TTS_CLIENT


def _write_wav(path: str, pcm_bytes: bytes, sample_rate_hz: int) -> None:
    # LINEAR16 = 16-bit signed PCM, mono
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate_hz)
        w.writeframes(pcm_bytes)


def _wav_duration_seconds(path: str) -> float:
    """
    Compute duration from WAV header (frames / framerate).
    This is used as a fallback if the audio player returns before playback ends.
    """
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    if rate <= 0:
        return 0.0
    return float(frames) / float(rate)


def _play_wav_paplay_strict(path: str, *, extra_tail_sec: float = 0.15) -> None:
    """
    Play WAV via paplay. In some PulseAudio setups, paplay can return before the
    sound is fully rendered to the speaker. We enforce wall-clock blocking by
    waiting for the WAV duration if needed.
    """
    dur = _wav_duration_seconds(path)
    t0 = time.time()

    subprocess.run(["paplay", path], check=True)

    elapsed = time.time() - t0
    remaining = (dur - elapsed)
    if remaining > 0:
        time.sleep(remaining)
    if extra_tail_sec > 0:
        time.sleep(extra_tail_sec)


def speak(
    text: str,
    *,
    language_code: str = TTS_LANG,
    voice_name: str = TTS_VOICE,
    speaking_rate: float = TTS_SPEAKING_RATE,
    pitch: float = TTS_PITCH,
    volume_gain_db: float = TTS_GAIN_DB,
    sample_rate_hz: int = 48000,
) -> None:
    if not text.strip():
        return

    client = _get_client()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db,
        sample_rate_hertz=sample_rate_hz,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    fd, wav_path = tempfile.mkstemp(prefix="olivia_tts_", suffix=".wav")
    os.close(fd)
    try:
        _write_wav(wav_path, response.audio_content, sample_rate_hz)
        _play_wav_paplay_strict(wav_path)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
