import base64
import io
import json
import os
import threading
import time
import warnings
import wave
from pathlib import Path
import numpy as np
import sounddevice as sd
from llama_cpp import Llama
from kokoro import KPipeline
from bridge import get_output_mode, reset_display, send_text
from servo_controller import close_mouth, play_speech

os.environ["HF_HUB_OFFLINE"] = "1"
#os.environ.setdefault("HF_HOME", r"D:\models\tts")
warnings.filterwarnings("ignore")

LLM_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
VOICE = "af_heart"
SAMPLE_RATE = 24000
AUDIO_LEAD_MS = 200
MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are a friendly guide robot at NSPU, Naypyitaw State Polytechnic "
    "University in Myanmar. Answer visitors' questions clearly and concisely "
    "in English. Keep answers to 2 or 3 short sentences when possible."
)

gpu_lock = threading.Lock()
_llm = None
_pipeline = None
conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

def _load_models():
    global _llm, _pipeline
    if _llm is None:
        _llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=4096, n_gpu_layers=24, verbose=False)
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", device="cuda")

def stream_event(event_type, data=None):
    return json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n"

def _synthesize(text):
    generator = _pipeline(text=text, voice=VOICE, speed=1.0, split_pattern=r"\n+")
    wav = None
    for result in generator:
        wav = result.audio
    if wav is None:
        raise RuntimeError("Kokoro returned no audio")
    return np.array(wav, dtype=np.float32)

def _audio_to_b64(audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _play_with_servo(audio):
    audio_list = audio.tolist()
    if get_output_mode() == "phone":
        yield stream_event("audio", {"b64": _audio_to_b64(audio), "sampleRate": SAMPLE_RATE})
        time.sleep(len(audio_list) / SAMPLE_RATE + 1.2)
    else:
        sd.play(audio, samplerate=SAMPLE_RATE)
        time.sleep(AUDIO_LEAD_MS / 1000)
        play_speech(audio_list, SAMPLE_RATE)
        sd.wait()
    close_mouth()

def run_local_ai_stream(prompt, temperature=0.7):
    yield stream_event("reset")
    reset_display()

    with gpu_lock:
        _load_models()
        conversation.append({"role": "user", "content": prompt})
        if len(conversation) > MAX_TURNS * 2 + 1:
            conversation[:] = [conversation[0]] + conversation[-(MAX_TURNS * 2):]

        stream = _llm.create_chat_completion(
            messages=conversation,
            stream=True,
            temperature=temperature,
        )

        response_text = ""
        sentence_buffer = ""
        sentence_endings = {".", "!", "?", "\n"}

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if not token:
                continue

            print(token, end="", flush=True)
            send_text(token)
            yield stream_event("text", token)
            response_text += token
            sentence_buffer += token

            if any(sentence_buffer.rstrip().endswith(p) for p in sentence_endings) and len(sentence_buffer.strip()) > 5:
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                yield stream_event("status", "speaking")
                try:
                    yield from _play_with_servo(_synthesize(sentence))
                except Exception as exc:
                    send_text(f"\n[tts error: {exc}]")

        if sentence_buffer.strip():
            yield stream_event("status", "speaking")
            try:
                yield from _play_with_servo(_synthesize(sentence_buffer.strip()))
            except Exception as exc:
                send_text(f"\n[tts error: {exc}]")

        print("\n")
        conversation.append({"role": "assistant", "content": response_text.strip()})

    yield stream_event("done")
