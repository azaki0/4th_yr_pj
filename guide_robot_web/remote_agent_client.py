import base64
from io import BytesIO
import json
import logging
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import numpy as np
import sounddevice as sd
from bridge import reset_display, send_event, send_text, show_route
from hand_controller import start_hand, stop_hand
from latency_tracker import TimerContext, record_first_response, set_request_info, finalize_request
from memory_store import retrieve_context, store_conversation
from navigation import route_to_place
from servo_controller import close_mouth, play_speech

AUDIO_LEAD_MS = 200
logger = logging.getLogger(__name__)

KAGGLE_AGENT_URL = os.getenv("KAGGLE_AGENT_URL", "").rstrip("/")
KAGGLE_AGENT_TOKEN = os.getenv("KAGGLE_AGENT_TOKEN", "")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("KAGGLE_AGENT_TIMEOUT", "600"))
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "auto").lower()
KOKORO_SAMPLE_RATE = 24000

TTS_MODEL_PATH = "D:/codes/Vits_mms_finetune/finetune-hf-vits/mms-tts-mya-female-v3"

_tts_lock = threading.Lock()
_kokoro_pipeline = None

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

def _resolve_kokoro_device():
    if KOKORO_DEVICE != "auto":
        return KOKORO_DEVICE

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _get_kokoro():
    global _kokoro_pipeline

    if _kokoro_pipeline is None:
        from kokoro import KPipeline

        device = _resolve_kokoro_device()
        try:
            _kokoro_pipeline = KPipeline(lang_code="a", device=device)
        except Exception:
            if device == "cuda":
                _kokoro_pipeline = KPipeline(lang_code="a", device="cpu")
            else:
                raise

    return _kokoro_pipeline

def _kokoro_tts(text):
    pipeline = _get_kokoro()
    wav = None

    for _, _, audio in pipeline(text=text,voice="af_heart",speed=1.0,split_pattern=r"\n+",):
        wav = audio

    if wav is None:
        raise RuntimeError("Kokoro returned no audio")

    return np.asarray(wav, dtype=np.float32), KOKORO_SAMPLE_RATE

def _gtts_tts(text):
    from gtts import gTTS
    import torch
    import torchaudio

    mp3_buffer = BytesIO()
    gTTS(text=text, lang="my").write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    waveform, sample_rate = torchaudio.load(mp3_buffer, format="mp3")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0).to(torch.float32).cpu().numpy().astype(np.float32), sample_rate

""" def _mms_tts(text):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel
    import torch

    MMS_TTS_MODEL = VitsModel.from_pretrained(TTS_MODEL_PATH)
    MMS_TTS_TOKENIZER = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)

    inputs = MMS_TTS_TOKENIZER(text, return_tensors="pt")
    with torch.no_grad():
        output = MMS_TTS_MODEL(**inputs).waveform
    return output.squeeze().cpu().numpy().astype(np.float32), sample_rate """

def _local_tts(text, language):
    with _tts_lock:
        if language == "mm":
            return _gtts_tts(text)
        return _kokoro_tts(text)

def _check_gtts_available():
    import gtts
    import torchaudio

    if not hasattr(torchaudio, "load"):
        raise RuntimeError("torchaudio.load is not available")
    try:
        from torchaudio.utils import ffmpeg_utils

        decoders = ffmpeg_utils.get_audio_decoders()
        if decoders and "mp3" not in decoders:
            raise RuntimeError("torchaudio MP3 decoder is not available")
    except ImportError:
        pass
    return gtts.__version__

def preload_local_models():
    loaded = []
    errors = {}

    with _tts_lock:
        try:
            _get_kokoro()
            loaded.append(f"kokoro:{_resolve_kokoro_device()}")
        except Exception as exc:
            errors["kokoro"] = str(exc)

        try:
            _check_gtts_available()
            loaded.append("gtts")
        except Exception as exc:
            errors["gtts"] = str(exc)

    return {
        "ok": not errors,
        "loaded": loaded,
        "errors": errors,
    }

def preload_remote_models():
    if not KAGGLE_AGENT_URL:
        return {"ok": False, "error": "KAGGLE_AGENT_URL is not set"}

    request = Request(
        f"{KAGGLE_AGENT_URL}/models/load",
        data=json.dumps({}).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))

def preload_models():
    result = {"local": preload_local_models()}

    try:
        result["kaggle"] = preload_remote_models()
    except Exception as exc:
        result["kaggle"] = {"ok": False, "error": str(exc)}

    result["ok"] = bool(result["local"].get("ok")) and bool(result["kaggle"].get("ok"))
    return result

def _stream_audio_to_web(audio, sample_rate):
    with TimerContext("stream_audio_to_web"):
        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
            send_event("audio", {
                "data": audio_b64,
                "sampleRate": sample_rate,
                "dtype": "int16",
            })
        except Exception as exc:
            logger.warning("Failed to stream audio to web: %s", exc)

def _play_local_speech(text, language):
    if not text:
        return

    with TimerContext("local_tts"):
        try:
            audio, sample_rate = _local_tts(text, language)
        except Exception as exc:
            send_text(f"Local TTS unavailable: {exc}\n")
            return

    with TimerContext("stream_audio_to_web"):
        _stream_audio_to_web(audio, sample_rate)

    with TimerContext("audio_playback"):
        start_hand()
        sd.play(audio, samplerate=sample_rate)
        time.sleep(AUDIO_LEAD_MS / 1000)
        with TimerContext("play_speech_servo"):
            play_speech(audio, sample_rate)
        sd.wait()
        stop_hand()
        with TimerContext("close_mouth"):
            close_mouth()

def _play_audio(event):
    with TimerContext("remote_audio_playback"):
        audio_b64 = event.get("data")
        sample_rate = int(event.get("sampleRate", 16000))
        dtype = event.get("dtype", "float32")

        if not audio_b64:
            return

        audio_bytes = base64.b64decode(audio_b64)
        audio = np.frombuffer(audio_bytes, dtype=np.dtype(dtype))

        start_hand()
        sd.play(audio, samplerate=sample_rate)
        time.sleep(AUDIO_LEAD_MS / 1000)
        with TimerContext("play_speech_servo"):
            play_speech(audio, sample_rate)
        sd.wait()
        stop_hand()
        with TimerContext("close_mouth"):
            close_mouth()

def _forward_event(event):
    record_first_response()
    event_type = event.get("type")

    if event_type == "reset":
        reset_display()
    elif event_type == "text":
        send_text(event.get("data", ""))
    elif event_type == "route":
        show_route(event.get("data"))
    elif event_type == "audio":
        _play_audio(event)
    elif event_type == "speech":
        _play_local_speech(event.get("data", ""), event.get("language", "en"))
    elif event_type:
        send_event(event_type, event.get("data"))

def translate_prompt(prompt, language):
    if language != "mm":
        return prompt

    with TimerContext("translate_prompt"):
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
    set_request_info(prompt, language)
    try:
        with TimerContext("handle_prompt_total") as total_timer:
            if not KAGGLE_AGENT_URL:
                send_text("KAGGLE_AGENT_URL is not set on the laptop.")
                record_first_response()
                return

            english_prompt = prompt
            context = {"uni_context": [], "memory_context": []}

            with TimerContext("translate_prompt"):
                try:
                    english_prompt = translate_prompt(prompt, language)
                except Exception as exc:
                    logger.warning("Prompt translation unavailable: %s", exc)
                    if language == "mm":
                        send_text("ဘာသာပြန်စနစ် မရရှိသေးပါ။ ခဏနောက်မှ ပြန်စမ်းပါ။\n")
                        record_first_response()
                        send_event("done")
                        return

            with TimerContext("route_check"):
                try:
                    if route_to_place(english_prompt):
                        context = {"uni_context": [], "memory_context": []}
                    else:
                        with TimerContext("retrieve_context"):
                            context = retrieve_context(english_prompt, uni_results=4, memory_results=2)
                except Exception as exc:
                    logger.warning("Local RAG unavailable: %s", exc)
                    context = {"uni_context": [], "memory_context": []}

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

            with TimerContext("kaggle_chat_stream") as kaggle_timer:
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
                        with TimerContext("store_conversation"):
                            store_conversation(
                                language,
                                prompt,
                                memory_payload.get("english_prompt", english_prompt),
                                memory_payload.get("english_response", ""),
                            )
                    send_event("done")
                except HTTPError as exc:
                    send_text(f"Kaggle agent HTTP error: {exc.code}")
                    send_event("done")
                except URLError as exc:
                    send_text(f"Could not reach Kaggle agent: {exc.reason}")
                    send_event("done")
                except Exception as exc:
                    send_text(f"Kaggle agent error: {exc}")
                    send_event("done")
    except Exception as exc:
        logger.exception("handle_prompt failed: %s", exc)
        send_text(f"Internal error: {exc}")
        send_event("done")
    finally:
        finalize_request()
