import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd

from bridge import reset_display, send_event, send_text, show_route
from memory_store import retrieve_context, store_conversation


KAGGLE_AGENT_URL = os.getenv("KAGGLE_AGENT_URL", "").rstrip("/")
KAGGLE_AGENT_TOKEN = os.getenv("KAGGLE_AGENT_TOKEN", "")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("KAGGLE_AGENT_TIMEOUT", "600"))


def get_config():
    return {
        "url": KAGGLE_AGENT_URL,
        "hasToken": bool(KAGGLE_AGENT_TOKEN),
    }


def set_config(url=None, token=None):
    global KAGGLE_AGENT_URL, KAGGLE_AGENT_TOKEN

    if url is not None:
        KAGGLE_AGENT_URL = url.strip().rstrip("/")

    if token is not None:
        KAGGLE_AGENT_TOKEN = token.strip()


def _headers():
    headers = {"Content-Type": "application/json"}
    if KAGGLE_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {KAGGLE_AGENT_TOKEN}"
    return headers


def _play_audio(event):
    audio_b64 = event.get("data")
    sample_rate = int(event.get("sampleRate", 16000))
    dtype = event.get("dtype", "float32")

    if not audio_b64:
        return

    audio_bytes = base64.b64decode(audio_b64)
    audio = np.frombuffer(audio_bytes, dtype=np.dtype(dtype))
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def _forward_event(event):
    event_type = event.get("type")

    if event_type == "reset":
        reset_display()
    elif event_type == "text":
        send_text(event.get("data", ""))
    elif event_type == "route":
        show_route(event.get("data"))
    elif event_type == "audio":
        _play_audio(event)
    elif event_type:
        send_event(event_type, event.get("data"))


def translate_prompt(prompt, language):
    if language != "mm":
        return prompt

    payload = json.dumps({"text": prompt, "source": "mya_Mymr", "target": "eng_Latn"}).encode("utf-8")
    request = Request(
        f"{KAGGLE_AGENT_URL}/translate",
        data=payload,
        headers=_headers(),
        method="POST",
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("translatedText", prompt)


def handle_prompt(prompt, language):
    if not KAGGLE_AGENT_URL:
        send_text("KAGGLE_AGENT_URL is not set on the laptop.")
        return

    english_prompt = prompt
    context = {"uni_context": [], "memory_context": []}

    try:
        english_prompt = translate_prompt(prompt, language)
        context = retrieve_context(english_prompt)
    except Exception as exc:
        send_text(f"Local RAG unavailable: {exc}\n")

    payload = json.dumps(
        {
            "prompt": prompt,
            "language": language,
            "english_prompt": english_prompt,
            "uni_context": context["uni_context"],
            "memory_context": context["memory_context"],
        }
    ).encode("utf-8")
    request = Request(
        f"{KAGGLE_AGENT_URL}/chat/stream",
        data=payload,
        headers=_headers(),
        method="POST",
    )

    try:
        memory_payload = None
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    send_text(line)
                    continue

                if event.get("type") == "memory":
                    memory_payload = event.get("data") or {}
                    continue

                _forward_event(event)

        if memory_payload:
            store_conversation(
                language,
                prompt,
                memory_payload.get("english_prompt", english_prompt),
                memory_payload.get("english_response", ""),
            )
    except HTTPError as exc:
        send_text(f"Kaggle agent HTTP error: {exc.code}")
    except URLError as exc:
        send_text(f"Could not reach Kaggle agent: {exc.reason}")
    except Exception as exc:
        send_text(f"Kaggle agent error: {exc}")
