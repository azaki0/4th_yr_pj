from io import BytesIO
import numpy as np
import torch
import torchaudio
import sounddevice as sd
import time

TTS_MODEL_PATH = "D:/codes/Vits_mms_finetune/finetune-hf-vits/mms-tts-mya-female-v3"

def text_chunker(token_stream):
    buffer = ""
    sentence_endings = {"။", "၊", "\n"}

    for token in token_stream:
        buffer += token
        if any(buffer.endswith(p) for p in sentence_endings) and len(buffer.strip()) > 5:
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()

def _gtts_tts(text):
    from gtts import gTTS

    mp3_buffer = BytesIO()
    gTTS(text=text, lang="my").write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    waveform, sample_rate = torchaudio.load(mp3_buffer, format="mp3")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0).to(torch.float32).cpu().numpy().astype(np.float32), sample_rate

def _mms_tts(text):
    from transformers import AutoTokenizer, VitsModel

    MMS_TTS_MODEL = VitsModel.from_pretrained(TTS_MODEL_PATH)
    MMS_TTS_TOKENIZER = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)

    inputs = MMS_TTS_TOKENIZER(text, return_tensors="pt")
    with torch.no_grad():
        output = MMS_TTS_MODEL(**inputs).waveform
    return output.squeeze().cpu().numpy().astype(np.float32)

""" start_time = time.perf_counter()
audio1 = _gtts_tts("မင်္ဂလာပါ။ ကျွန်တော်က ဂိုက်ရိုဘော့ပါ။")
end_time = time.perf_counter()
print(f"gTTS TTS generation time: {end_time - start_time:.2f} seconds")

for text in text_chunker("မင်္ဂလာပါ။ ကျွန်တော်က ဂိုက်ရိုဘော့ပါ။"):
    start_time = time.perf_counter()
    audio2 = _mms_tts(text)
    end_time = time.perf_counter()
    print(f"MMS TTS generation time: {end_time - start_time:.2f} seconds") """

start_time = time.perf_counter()
audio1 = _gtts_tts("မင်္ဂလာပါ။ ကျွန်တော်က ဂိုက်ရိုဘော့ပါ။")
end_time = time.perf_counter()
print(f"TTS generation time: {end_time - start_time:.2f} seconds")

sd.play(audio1[0], samplerate=24000)
sd.wait()
"""time.sleep(3)
sd.play(audio2, samplerate=24000)
sd.wait() """

