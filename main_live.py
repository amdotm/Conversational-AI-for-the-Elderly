from __future__ import annotations
import re
import time
import numpy as np
import sounddevice as sd
from google.cloud import speech_v1p1beta1 as speech
from config import (
    MODEL_NAME,
    PROVIDER,
    PAUSE_TIERS_SEC,
    STT_ENERGY_THRESHOLD,
    MIN_LISTEN_SEC,
    MIC_WARMUP_SEC,
    PREROLL_SEC,
    MAX_TOKENS_BY_ACT,
)
from tts_gcloud import speak
from stt_gcloud import make_streaming_config, streaming_recognize, transcribe_bytes
from prompts import get_system_prompt, get_onboarding_prompt,get_human_style_injection
from starter_questions import get_intro_text, extract_location_hint
from llm_provider import generate
from logger import log_turn
from disfluency import clean_for_llm
from repair import decide_repair, RepairAction
from nudges import next_nudge

from memory import init_memory, update_memory, get_avoided_topics, should_change_topic, reset_topic_counter, add_banned_topic, _extract_asked_topics
from utterance_features import extract_features
from nonfluency_classifier import classify_nonfluency
from dialogue_policy import decide_policy


from llm_guards import classify_user_intent, check_and_fix_parroting


# -----------------------------
# Constants / Configuration
# -----------------------------
SAMPLE_RATE = 16000
LOG_PATH = "session.jsonl"

sd.default.device = (None, 1)  
sd.default.latency = "high"
sd.default.blocksize = 2048

# English only 
STT_LANG = "en-US"
TTS_LANG = "en-GB"

# Turn-taking barrier: after TTS playback, wait a bit before opening mic
TTS_SETTLE_SEC = 0.2


# Helpers
_QUESTION_RE = re.compile(r"([^?.!]*\?)")

def extract_last_question(text: str) -> str:
    """Extract the last question sentence from bot text (for repeat handling)."""
    if not text:
        return ""
    qs = _QUESTION_RE.findall(text)
    return qs[-1].strip() if qs else ""


def speaking_rate_for_pause_tier(pause_tier: str) -> float:
    """Slightly speed up TTS for fast-paced users to reduce perceived latency."""
    if pause_tier == "FAST":
        return 1.05
    if pause_tier == "SLOW":
        return 0.94
    return 0.98


def speak_blocking(text: str, language_code: str = TTS_LANG, *, speaking_rate: float = 0.95) -> None:
    """
    Speak via Google TTS + paplay, then wait briefly so mic doesn't capture tail/echo.
    """
    speak(text, language_code=language_code, speaking_rate=speaking_rate)
    time.sleep(TTS_SETTLE_SEC)


def record_until_silence(
    max_silence_sec: float = 4.5,
    sr: int = SAMPLE_RATE,
    frame_ms: int = 30,
    amp_thresh: int = 50,
    max_duration_sec: float = 20.0,
):
    print(f"\n[🎙️] Listening… ")
    frame_len = max(1, int(sr * frame_ms / 1000))
    silence = 0.0
    chunks = []
    start = time.time()

    with sd.InputStream(samplerate=sr, channels=1, dtype="int16") as stream:
        while True:
            frames, _ = stream.read(frame_len)
            chunks.append(frames.copy())

            energy = float(np.abs(frames.astype(np.int32)).mean())
            if energy < amp_thresh:
                silence += frame_ms / 1000.0
            else:
                silence = 0.0

            if silence >= max_silence_sec:
                break
            if (time.time() - start) >= max_duration_sec:
                break

    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.int16)
    return audio.tobytes(), sr


def listen_and_transcribe_google_streaming(
    *,
    sr: int,
    lang_code: str,
    max_total_seconds: float,
    long_pause_sec: float,
    energy_threshold: float,
):
    """
    Google StreamingRecognize with mic in real time.
    Returns: (full_text, audio_bytes)
    """
    frames_all = []
    full_text_fragments = []

    streaming_config = make_streaming_config(sample_rate=sr, language_code=lang_code)

    frame_ms = 60
    block_size = int(sr * frame_ms / 1000)

    # Safety timeout: max time to wait after hearing voice but getting no transcript
    MAX_WAIT_NO_TRANSCRIPT = 20.0  # seconds


    def audio_request_generator():
        nonlocal frames_all

        start_time = time.time()
        last_voice_time = start_time
        heard_any_voice = False
        first_voice_time = None
        got_any_transcript = False
        preroll_frames = []

        print("[🎙️] Streaming to Google STT…")

        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="int16") as stream:
                # Phase 1: Collect pre-roll buffer
                preroll_end = time.time() + PREROLL_SEC
                while time.time() < preroll_end:
                    frames, _ = stream.read(block_size)
                    if frames is not None and frames.size > 0:
                        preroll_frames.append(frames.copy())

                # Send all pre-roll frames immediately to Google
                for pf in preroll_frames:
                    frames_all.append(pf.copy())
                    yield speech.StreamingRecognizeRequest(audio_content=pf.tobytes())

                # Phase 2: Normal streaming
                while True:
                    frames, _ = stream.read(block_size)
                    now = time.time()

                    # Hard cap
                    if (now - start_time) > max_total_seconds:
                        print("Max total listening time reached")
                        return

                    if frames is None or frames.size == 0:
                        continue

                    frames_all.append(frames.copy())

                    # Energy-based VAD-lite
                    data = frames.astype(np.int16).flatten().astype(np.float32)
                    energy = float(np.mean(np.abs(data)))
                    is_voice = energy > energy_threshold

                    if is_voice:
                        if not heard_any_voice:
                            first_voice_time = now
                        heard_any_voice = True
                        last_voice_time = now
                    else:
                        # Check for end of turn after user has spoken
                        if heard_any_voice:
                            silence_dur = now - last_voice_time
                            total_listen = now - start_time
                            if total_listen >= MIN_LISTEN_SEC and silence_dur >= long_pause_sec:
                                print(f"Long pause ({long_pause_sec:.1f}s) → end of turn (silence)")
                                yield speech.StreamingRecognizeRequest(audio_content=frames.tobytes())
                                return

                    # Safety timeout for background noise
                    if heard_any_voice and first_voice_time:
                        time_since_first_voice = now - first_voice_time
                        if time_since_first_voice > MAX_WAIT_NO_TRANSCRIPT and not got_any_transcript:
                            print(f"[Safety] Heard audio for {time_since_first_voice:.1f}s but no transcript — likely background noise")
                            return

                    yield speech.StreamingRecognizeRequest(audio_content=frames.tobytes())

        except Exception as e:
            print(f"[STT generator error] {repr(e)}")
            return
    
    got_any_transcript = False
    for resp in streaming_recognize(audio_request_generator(), streaming_config):
        for result in resp.results:
            if result.is_final and result.alternatives:
                transcript = result.alternatives[0].transcript.strip()
                if transcript:
                    print(f"[STT] Final chunk: {transcript}")
                    full_text_fragments.append(transcript)
                    got_any_transcript = True  

    full_text = " ".join(full_text_fragments).strip()

    if frames_all:
        full_audio_np = np.concatenate(frames_all, axis=0)
        audio_bytes = full_audio_np.tobytes()
    else:
        audio_bytes = b""

    return full_text, audio_bytes


def _inject_dialogue_directives(sys_prompt: str, act: str, question_budget: int, max_sentences: int) -> str:
    return (
        sys_prompt
        + f"\nDIALOGUE_ACT: {act}\n"
        + f"QUESTION_BUDGET: {question_budget}\n"
        + f"MAX_SENTENCES: {max_sentences}\n"
    )


def play_onboarding(mem) -> str:
    """
    Onboarding:
      1) Olivia intro monologue
      2) Ask location once
      3) User responds
      4) GPT responds with onboarding prompt (no forced question)
    """
    intro_text = get_intro_text()

    speak_blocking(
        intro_text,
        language_code=TTS_LANG,
        speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
    )
    log_turn(LOG_PATH, "[system_intro]", intro_text, 0, 0, 0, note="onboarding_intro")

    audio, sr = record_until_silence(
        max_silence_sec=2.5,
        sr=SAMPLE_RATE,
        frame_ms=30,
        amp_thresh=50,
        max_duration_sec=20.0,
    )
    user_first = (transcribe_bytes(audio, sample_rate=sr, language_code=STT_LANG) or "").strip()
    print(f"You (onboarding): {user_first or '[…no speech…]'}")

    sys_prompt = get_onboarding_prompt()
    hint = extract_location_hint(user_first)
    if hint:
        sys_prompt += f"\nContext hint (may be wrong): user may be from: {hint}\n"

    act = "COMMENT" if user_first else "NUDGE"
    question_budget = 1 if user_first else 0
    sys_prompt = _inject_dialogue_directives(sys_prompt, act, question_budget, max_sentences=2)

    llm_input = user_first if user_first else "The user was silent."
    t2 = time.time()
    bot_text, usage = generate(sys_prompt, llm_input, max_tokens=MAX_TOKENS_BY_ACT.get(act, 150))
    
    # Apply LLM-based anti-parroting check
    if user_first:
        bot_text, was_fixed, guard_usage = check_and_fix_parroting(user_first, bot_text)
        if was_fixed:
            print(f"[Guard] Parroting detected and fixed")
    
    t3 = time.time()

    print(f"Bot (onboarding): {bot_text}")
    speak_blocking(
        bot_text,
        language_code=TTS_LANG,
        speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
    )
    t4 = time.time()

    nonflu = "HESITANT" if not user_first else "FLUENT"
    mem = update_memory(mem, user_first, bot_text, act, nonflu)

    log_turn(
        LOG_PATH,
        user_first,
        bot_text,
        0,
        int(1000 * (t3 - t2)),
        int(1000 * (t4 - t3)),
        note=f"onboarding;provider={PROVIDER};model={MODEL_NAME}",
        usage=usage,
        dialogue_act=act,
        nonfluency_label=nonflu,
        pause_tier=mem.pause_tier,
        question_budget=question_budget,
        repair_action="ONBOARD",
        repair_reason="",
    )

    return bot_text


if __name__ == "__main__":
    print("Olivia- Conversational AI for the Elderly")
    print(f"Using model: {MODEL_NAME} via {PROVIDER}")

    mem = init_memory()

    last_bot_text = play_onboarding(mem)
    last_bot_question = extract_last_question(last_bot_text)

    while True:
        long_pause = PAUSE_TIERS_SEC.get(mem.pause_tier, PAUSE_TIERS_SEC["MEDIUM"])

        # 1) Listen (streaming STT)
        t0 = time.time()
        user_text, audio_bytes = listen_and_transcribe_google_streaming(
            sr=SAMPLE_RATE,
            lang_code=STT_LANG,
            max_total_seconds=300.0,
            long_pause_sec=long_pause,
            energy_threshold=STT_ENERGY_THRESHOLD,
        )
        t1 = time.time()

        user_text = (user_text or "").strip()
        print(f"You: {user_text or '[…no speech…]'}")

        # Repair decision (audio-based)
        decision = decide_repair(user_text, audio_bytes, SAMPLE_RATE)

        if decision.action == RepairAction.EXIT:
            say = "Okay, I'll stop here. Talk to you later."
            speak_blocking(
                say,
                language_code=TTS_LANG,
                speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
            )
            log_turn(
                LOG_PATH,
                user_text,
                say,
                int(1000 * (t1 - t0)),
                0,
                0,
                note="exit",
                dialogue_act="COMMENT",
                nonfluency_label=mem.last_nonfluency,
                pause_tier=mem.pause_tier,
                question_budget=0,
                repair_action=decision.action.name,
                repair_reason=decision.reason,
            )
            break

        # LLM-based intent classification (replaces pattern matching)
        user_intent, intent_usage = classify_user_intent(user_text)
        print(f"[Intent] {user_intent}")
        
        if user_intent == "REPEAT_REQUEST":
            # User wants repetition
            if last_bot_question:
                say = f"Sure. {last_bot_question}"
            else:
                say = "Would you like me to rephrase what I said?"
            
            speak_blocking(
                say,
                language_code=TTS_LANG,
                speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
            )
            log_turn(
                LOG_PATH,
                user_text,
                say,
                int(1000 * (t1 - t0)),
                0,
                0,
                note="repeat_request;llm_classified",
                dialogue_act="ANCHOR_AND_RESUME",
                nonfluency_label="INTERRUPTED",
                pause_tier="SLOW",
                question_budget=0,
                repair_action="REPEAT_REQUEST",
                repair_reason="llm_intent_classification",
            )
            continue
        
        elif user_intent == "COMPLAINT_ABOUT_REPETITION":
            # User is frustrated about repetition - apologize and OFFER a new topic
            from nudges import next_nudge
            new_topic = next_nudge(question_budget=1, topic_change=True)
            say = f"I'm sorry about that. {new_topic}"
            
            # Clear memory to force fresh conversation
            mem.summary_text = ""
            mem.turns_on_current_topic = 0
            
            speak_blocking(
                say,
                language_code=TTS_LANG,
                speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
            )
            log_turn(
                LOG_PATH,
                user_text,
                say,
                int(1000 * (t1 - t0)),
                0,
                0,
                note="complaint_about_repetition;offered_new_topic",
                dialogue_act="PROGRESS_TOPIC",
                nonfluency_label="FLUENT",
                pause_tier=mem.pause_tier,
                question_budget=1,
                repair_action="COMPLAINT_HANDLED",
                repair_reason="llm_intent_classification",
            )
            last_bot_text = say
            last_bot_question = extract_last_question(say)
            continue

        # 3) Feature extraction + nonfluency label
        feats = extract_features(user_text)
        was_interrupt = False 
        nonflu = classify_nonfluency(
            feats,
            repair_action=decision.action.name,
            repeat_request=False,
            was_interrupt=was_interrupt,
        )

        # 4) Dialogue policy (act + question budget + pause tier)
        pol = decide_policy(
            mem,
            user_text=user_text,
            repair_action=decision.action.name,
            nonfluency_label=nonflu,
            repeat_request=False,
            was_interrupt=was_interrupt,
        )
        mem.pause_tier = pol.pause_tier

        # Handle NO_SPEECH via nudges
        if decision.action == RepairAction.NO_SPEECH:
            # Check if conversation is stuck on same topic
            topic_change = should_change_topic(mem)
            say = next_nudge(question_budget=pol.question_budget, topic_change=topic_change)
            
            if topic_change:
                mem = reset_topic_counter(mem)
            
            speak_blocking(
                say,
                language_code=TTS_LANG,
                speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
            )
            last_bot_text = say
            last_bot_question = extract_last_question(say)

            mem = update_memory(mem, user_text, say, "NUDGE", "HESITANT")
            log_turn(
                LOG_PATH,
                user_text,
                say,
                int(1000 * (t1 - t0)),
                0,
                0,
                note=f"topic_nudge;topic_change={topic_change}",
                dialogue_act="NUDGE",
                nonfluency_label="HESITANT",
                pause_tier=mem.pause_tier,
                question_budget=pol.question_budget,
                repair_action=decision.action.name,
                repair_reason=decision.reason,
            )
            continue

        # Handle VERY_SHORT
        if decision.action == RepairAction.AFFIRMATION:
            sys_prompt = get_system_prompt()
            sys_prompt = _inject_dialogue_directives(sys_prompt, "ELABORATE", 1, max_sentences=2)
            
            if mem.summary_text:
                sys_prompt += f"\nConversation context (brief):\n{mem.summary_text}\n"
            
            # Special instruction for affirmations
            sys_prompt += """
        The user just said a short affirmation like "yes", "yeah", or "okay".
        This means they agree or are listening. Continue the conversation naturally:
        - Share a follow-up thought or gentle question about what you were discussing
        - Don't say "No rush" or "I'm here" - that's not needed
        - Don't ask them to share more - just continue the flow
        - Keep it to 1-2 sentences
        """
            
            llm_text = f"User said: {user_text}"
            t2 = time.time()
            bot_text, usage = generate(sys_prompt, llm_text, max_tokens=120)
            
            # Apply anti-parroting check
            bot_text, was_fixed, guard_usage = check_and_fix_parroting(user_text, bot_text)
            
            t3 = time.time()
            
            print(f"Bot: {bot_text}")
            speak_blocking(
                bot_text,
                language_code=TTS_LANG,
                speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
            )
            t4 = time.time()
            
            last_bot_text = bot_text
            last_bot_question = extract_last_question(bot_text)
            mem = update_memory(mem, user_text, bot_text, "ELABORATE", "FLUENT")
            
            log_turn(
                LOG_PATH,
                user_text,
                bot_text,
                int(1000 * (t1 - t0)),
                int(1000 * (t3 - t2)),
                int(1000 * (t4 - t3)),
                note=f"affirmation;provider={PROVIDER};model={MODEL_NAME}",
                usage=usage,
                dialogue_act="ELABORATE",
                nonfluency_label="FLUENT",
                pause_tier=mem.pause_tier,
                question_budget=1,
                repair_action=decision.action.name,
                repair_reason=decision.reason,
            )
            continue

        #LLM response (single call)
        llm_text = clean_for_llm(user_text)
        sys_prompt = get_system_prompt()
        sys_prompt = _inject_dialogue_directives(sys_prompt, pol.act, pol.question_budget, pol.max_sentences)

        if mem.summary_text:
            sys_prompt += f"\nConversation context (brief):\n{mem.summary_text}\n"
        if mem.last_user_topic_hint:
            sys_prompt += f"\nLast user topic hint: {mem.last_user_topic_hint}\n"
        
        # Add topics to avoid 
        avoided = get_avoided_topics(mem)
        if avoided:
            sys_prompt += f"\n{avoided}\n"
        
        # Check if we should change topic
        if should_change_topic(mem):
            sys_prompt += """
                    CRITICAL: We've been discussing the same topic for too long. You MUST change to a completely different subject.
                    DO NOT mention: neighborhood, changes, living there, adjusting, new neighbors, or anything related to the current topic.
                    Instead, ask about something totally different like: their family, their work, their hobbies, or their childhood.
                    """
            mem = reset_topic_counter(mem)

        # Add human style
        sys_prompt += get_human_style_injection()

        if pol.act == "COMMENT":
            sys_prompt += "\nAct guidance: Make a warm, brief comment. You don't need to ask a question - a simple acknowledgment is fine.\n"
        elif pol.act == "ELABORATE":
            sys_prompt += "\nAct guidance: Share a thought or ask ONE question (not both). Keep it simple.\n"
        elif pol.act == "PROGRESS_TOPIC":
            sys_prompt += "\nAct guidance: Gently shift to something new. Don't keep asking about the same thing.\n"
        elif pol.act == "CLARIFY":
            sys_prompt += "\nAct guidance: Ask ONE short clarification. Don't add extra questions.\n"
        elif pol.act == "ANCHOR_AND_RESUME":
            sys_prompt += "\nAct guidance: Acknowledge briefly and move on. Don't repeat what they said.\n"

        t2 = time.time()
        bot_text, usage = generate(sys_prompt, llm_text, max_tokens=MAX_TOKENS_BY_ACT.get(pol.act, 180))
        
        # Apply LLM-based anti-parroting check
        bot_text, was_fixed, guard_usage = check_and_fix_parroting(user_text, bot_text)
        if was_fixed:
            print(f"[Guard] Parroting detected and fixed")
        
        t3 = time.time()

        print(f"Bot: {bot_text}")

        #Speak
        speak_blocking(
            bot_text,
            language_code=TTS_LANG,
            speaking_rate=speaking_rate_for_pause_tier(mem.pause_tier),
        )
        t4 = time.time()

        #Update memory/logging
        last_bot_text = bot_text
        last_bot_question = extract_last_question(bot_text)
        mem = update_memory(mem, user_text, bot_text, pol.act, nonflu)

        log_turn(
            LOG_PATH,
            user_text,
            bot_text,
            int(1000 * (t1 - t0)),
            int(1000 * (t3 - t2)),
            int(1000 * (t4 - t3)),
            note=f"provider={PROVIDER};model={MODEL_NAME};parroting_fixed={was_fixed}",
            usage=usage,
            dialogue_act=pol.act,
            nonfluency_label=nonflu,
            pause_tier=mem.pause_tier,
            question_budget=pol.question_budget,
            repair_action=decision.action.name,
            repair_reason=decision.reason,
        )