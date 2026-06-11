import base64
import json
import os
import re
import threading
import warnings

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from huggingface_hub import HfApi, hf_hub_download
from kokoro import KPipeline
from llama_cpp import Llama
from num2words import num2words
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

from navigation import route_to_place


app = Flask(__name__)
warnings.filterwarnings("ignore")

API_TOKEN = os.getenv("KAGGLE_AGENT_TOKEN", "")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/kaggle/working/model_cache")
QWEN_MODEL_PATH_EN = os.getenv("QWEN_MODEL_PATH_EN", "")
QWEN_MODEL_PATH_MM = os.getenv("QWEN_MODEL_PATH_MM", "")
QWEN_MODEL_REPO_EN = os.getenv("QWEN_MODEL_REPO_EN", "Qwen/Qwen2.5-7B-Instruct-GGUF")
QWEN_MODEL_REPO_MM = os.getenv("QWEN_MODEL_REPO_MM", QWEN_MODEL_REPO_EN)
QWEN_MODEL_FILE_EN = os.getenv("QWEN_MODEL_FILE_EN", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
QWEN_MODEL_FILE_MM = os.getenv("QWEN_MODEL_FILE_MM", QWEN_MODEL_FILE_EN)
TRANSLATER_MODEL_PATH = os.getenv("TRANSLATER_MODEL_PATH", "")
TRANSLATER_MODEL_ID = os.getenv("TRANSLATER_MODEL_ID", "facebook/nllb-200-distilled-1.3B")
VITS_MODEL_PATH = os.getenv("VITS_MODEL_PATH", "")
VITS_MODEL_ID = os.getenv("VITS_MODEL_ID", "facebook/mms-tts-mya")
N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "32"))

MYANMAR_PLACE_NAMES = {
    "Main Entrance": "ပင်မဝင်ပေါက်",
    "Entrance B": "ဝင်ပေါက် ဘီ",
    "Main Building": "ပင်မဆောင်",
    "Workshop Left Entrance": "ဝပ်ရှော့ဘယ်ဘက်ဝင်ပေါက်",
    "Workshop Right Entrance": "ဝပ်ရှော့ညာဘက်ဝင်ပေါက်",
    "Teacher Dormitory A": "ဆရာ/ဆရာမ အိပ်ဆောင် အေ",
    "Teacher Dormitory B": "ဆရာ/ဆရာမ အိပ်ဆောင် ဘီ",
    "Teacher Dormitory C": "ဆရာ/ဆရာမ အိပ်ဆောင် စီ",
    "Teacher Dormitory D": "ဆရာ/ဆရာမ အိပ်ဆောင် ဒီ",
    "View Point": "ရှုခင်းကြည့်နေရာ",
    "Boys Dormitory A": "ကျောင်းသား အိပ်ဆောင် အေ",
    "Boys Dormitory B": "ကျောင်းသား အိပ်ဆောင် ဘီ",
    "Canteen": "စားသောက်ဆိုင်",
    "Girls Dormitory": "ကျောင်းသူ အိပ်ဆောင်",
    "Stadium": "အားကစားကွင်း",
}

MYANMAR_DIGITS = str.maketrans("0123456789.", "၀၁၂၃၄၅၆၇၈၉.")
number_pattern = re.compile(r"\d+(\.\d+)?(?:,\d{3})*")
model_lock = threading.Lock()

system_prompt = (
    "You are a friendly university guide robot. "
    "Give concise spoken answers. "
    "When route information has already been shown on the robot display, explain the route naturally. "
    "Do not make up classroom or facility locations that are not provided."
)

conversations = {
    "en": [{"role": "system", "content": system_prompt}],
    "mm": [{"role": "system", "content": system_prompt}],
}

print("Kaggle guide robot server starting. Models load on first use.")
llm_en = None
translator_tokenizer = None
translator_model = None
vits_model = None
vits_tokenizer = None
kokoro_pipeline = None
vits_sample_rate = 16000


def resolve_hf_model(local_path, model_id):
    return local_path if local_path and os.path.exists(local_path) else model_id


def resolve_gguf_path(local_path, repo_id, filename):
    if local_path and os.path.exists(local_path):
        return local_path

    if local_path:
        print(f"Model path not found, downloading instead: {local_path}")

    api = HfApi()
    repo_files = api.list_repo_files(repo_id)

    split_match = re.match(r"(.+)-\d{5}-of-\d{5}\.gguf$", filename)
    split_prefix = split_match.group(1) if split_match else filename.removesuffix(".gguf")
    split_files = sorted(
        file
        for file in repo_files
        if re.match(rf"{re.escape(split_prefix)}-\d{{5}}-of-\d{{5}}\.gguf$", file)
    )

    if split_files:
        print(f"Downloading split GGUF for {repo_id}: {', '.join(split_files)}")
        downloaded = [
            hf_hub_download(repo_id=repo_id, filename=file, cache_dir=MODEL_CACHE_DIR)
            for file in split_files
        ]
        return downloaded[0]

    if filename in repo_files:
        print(f"Downloading {repo_id}/{filename} to {MODEL_CACHE_DIR}")
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=MODEL_CACHE_DIR)

    quant_name = filename.removesuffix(".gguf").lower()
    candidates = sorted(
        file
        for file in repo_files
        if file.lower().endswith(".gguf") and quant_name in file.lower()
    )

    if candidates:
        split_candidates = [
            file for file in candidates if re.match(r".+-\d{5}-of-\d{5}\.gguf$", file)
        ]
        if split_candidates:
            first_split = sorted(split_candidates)[0]
            return resolve_gguf_path("", repo_id, first_split)

        selected = candidates[0]
        print(f"Exact GGUF not found. Downloading closest match: {repo_id}/{selected}")
        return hf_hub_download(repo_id=repo_id, filename=selected, cache_dir=MODEL_CACHE_DIR)

    raise FileNotFoundError(f"Could not find GGUF file '{filename}' in {repo_id}")


def get_llm(language):
    global llm_en

    if llm_en is None:
        local_path = QWEN_MODEL_PATH_MM if language == "mm" and QWEN_MODEL_PATH_MM else QWEN_MODEL_PATH_EN
        repo_id = QWEN_MODEL_REPO_MM if language == "mm" else QWEN_MODEL_REPO_EN
        filename = QWEN_MODEL_FILE_MM if language == "mm" else QWEN_MODEL_FILE_EN
        model_path = resolve_gguf_path(local_path, repo_id, filename)
        llm_en = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=N_GPU_LAYERS, verbose=False)
    return llm_en


def get_translator():
    global translator_tokenizer, translator_model

    if translator_tokenizer is None or translator_model is None:
        model_name = resolve_hf_model(TRANSLATER_MODEL_PATH, TRANSLATER_MODEL_ID)
        translator_tokenizer = AutoTokenizer.from_pretrained(model_name)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return translator_tokenizer, translator_model


def get_vits():
    global vits_model, vits_tokenizer, vits_sample_rate

    if vits_model is None or vits_tokenizer is None:
        model_name = resolve_hf_model(VITS_MODEL_PATH, VITS_MODEL_ID)
        vits_model = VitsModel.from_pretrained(model_name)
        vits_tokenizer = AutoTokenizer.from_pretrained(model_name)
        vits_sample_rate = getattr(vits_model.config, "sampling_rate", 16000)

    return vits_model, vits_tokenizer, vits_sample_rate


def get_kokoro():
    global kokoro_pipeline

    if kokoro_pipeline is None:
        kokoro_pipeline = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu")

    return kokoro_pipeline

tokenizer, model = get_translator()
pipeline = get_kokoro()
vit_model, vit_tokenizer, sample_rate = get_vits()
llm = get_llm("en")

def authorized():
    if not API_TOKEN:
        return True
    auth_header = request.headers.get("Authorization", "")
    return auth_header == f"Bearer {API_TOKEN}"


def event(event_type, data=None, **extra):
    payload = {"type": event_type, "data": data}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False) + "\n"


def normalize_numwords(text):
    return text.replace(",", "").replace(" and ", " ")


def numbers_to_words(text):
    def repl(match):
        token = match.group(0).replace(",", "")
        if "." in token:
            whole, frac = token.split(".", 1)
            whole_words = num2words(int(whole), lang="en")
            frac_words = " ".join(num2words(int(d), lang="en") for d in frac)
            return normalize_numwords(f"{whole_words} point {frac_words}")

        try:
            return normalize_numwords(num2words(int(token), lang="en"))
        except Exception:
            return match.group(0)

    return number_pattern.sub(repl, text)


def to_myanmar_number(value):
    return str(value).translate(MYANMAR_DIGITS)


def format_walking_time_mm(seconds):
    if seconds < 60:
        return f"{to_myanmar_number(seconds)} စက္ကန့်"

    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        return f"{to_myanmar_number(minutes)} မိနစ်"

    return f"{to_myanmar_number(minutes)} မိနစ် {to_myanmar_number(remaining_seconds)} စက္ကန့်"


def translate(text, src_lang, target_lang):
    
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=512,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def text_chunker(token_stream):
    buffer = ""
    sentence_endings = {".", "!", "?", "\n"}

    for token in token_stream:
        buffer += token
        if any(buffer.endswith(p) for p in sentence_endings) and len(buffer.strip()) > 5:
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()


def kokoro_tts(text):
    wav = None
    
    for _, _, audio in pipeline(text=text, voice="af_heart", speed=1.0, split_pattern=r"\n+"):
        wav = audio
    if wav is None:
        return None
    return np.asarray(wav, dtype=np.float32), 24000


def vits_tts(text):
    
    inputs = vit_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = vit_model(**inputs).waveform
    return output.squeeze().cpu().numpy().astype(np.float32), sample_rate


def audio_event(audio, sample_rate):
    return event(
        "audio",
        base64.b64encode(audio.tobytes()).decode("ascii"),
        sampleRate=sample_rate,
        dtype=str(audio.dtype),
    )


def localize_route(route, language):
    if language != "mm":
        return route

    route["displayLanguage"] = "mm"
    route["startNameLocalized"] = MYANMAR_PLACE_NAMES.get(route["startName"], route["startName"])
    route["destinationNameLocalized"] = MYANMAR_PLACE_NAMES.get(route["destinationName"], route["destinationName"])
    route["distanceLocalized"] = to_myanmar_number(route["distance"])
    route["distanceUnitLocalized"] = "ပေ"
    route["walkingTimeTextLocalized"] = format_walking_time_mm(route["walkingTimeSeconds"])
    return route


def build_route_context(english_prompt, language):
    route = route_to_place(english_prompt)
    if route is None:
        return None, None

    route = localize_route(route, language)
    route_sentence = (
        "The direction is shown on the display. "
        f"From {route['startName']} to {route['destinationName']}, "
        f"the distance is {numbers_to_words(str(route['distance']))} feet "
        f"and the estimated walking time is {numbers_to_words(route['walkingTimeText'])}."
    )
    context = (
        f"Route already shown to the user: {route_sentence} "
        "Reply with exactly that sentence in natural English. "
        "Do not include labels like start=, destination=, distance=, or any internal node names."
    )
    return route, context


def stream_completion(prompt, language):
    
    convo = conversations[language]
    convo.append({"role": "user", "content": prompt})
    if len(convo) > 7:
        conversations[language] = [convo[0]] + convo[-5:]
        convo = conversations[language]

    with model_lock:
        stream = llm.create_chat_completion(messages=convo, stream=True, temperature=0.7)

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]


def format_context(title, chunks):
    clean_chunks = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    if not clean_chunks:
        return ""
    joined = "\n\n".join(f"- {chunk}" for chunk in clean_chunks)
    return f"{title}:\n{joined}\n\n"


def run_chat(prompt, language, english_prompt=None, uni_context=None, memory_context=None):
    yield event("reset")

    english_prompt = english_prompt or (translate(prompt, "mya_Mymr", "eng_Latn") if language == "mm" else prompt)
    route, route_context = build_route_context(english_prompt, language)
    if route:
        yield event("route", route)

    context_prefix = (
        format_context("UNIVERSITY INFORMATION", uni_context or [])
        + format_context("RELEVANT CONVERSATION MEMORY", memory_context or [])
    )

    if route_context:
        prompt_for_llm = f"{context_prefix}{route_context}\nUSER PROMPT: {english_prompt}"
    else:
        prompt_for_llm = f"{context_prefix}USER PROMPT: {english_prompt}" if context_prefix else english_prompt

    response_text = ""

    for sentence in text_chunker(stream_completion(prompt_for_llm, language)):
        response_text += sentence + " "

        if language == "mm":
            display_sentence = translate(numbers_to_words(sentence), "eng_Latn", "mya_Mymr")
            yield event("text", display_sentence + " ")
            audio, sample_rate = vits_tts(display_sentence)
        else:
            yield event("text", sentence + " ")
            audio, sample_rate = kokoro_tts(sentence)

        if audio is not None:
            yield audio_event(audio, sample_rate)

    english_response = response_text.strip()
    conversations[language].append({"role": "assistant", "content": english_response})
    yield event(
        "memory",
        {
            "english_prompt": english_prompt,
            "english_response": english_response,
        },
    )


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    language = data.get("language", "en").lower()
    english_prompt = data.get("english_prompt") or None
    uni_context = data.get("uni_context") or []
    memory_context = data.get("memory_context") or []

    if language not in {"en", "mm"}:
        return jsonify({"error": "language must be en or mm"}), 400
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    return Response(
        stream_with_context(run_chat(prompt, language, english_prompt, uni_context, memory_context)),
        mimetype="text/plain",
    )


@app.route("/translate", methods=["POST"])
def translate_api():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    source = data.get("source", "mya_Mymr")
    target = data.get("target", "eng_Latn")

    if not text:
        return jsonify({"error": "text is required"}), 400

    return jsonify({"translatedText": translate(text, source, target)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False, threaded=True)
