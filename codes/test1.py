import torch
import numpy as np
import sounddevice as sd
import wave
from collections import deque
from transformers import AutoProcessor, AutoConfig, AutoModelForCTC

W2V2_MODEL_PATH = "D:/models/burmese_w2v2_asr_8bit"

SAMPLE_RATE = 16000
MIC_BLOCK_SECONDS = 0.25
MIC_SILENCE_THRESHOLD = 0.01
MIC_START_THRESHOLD = 0.02
MIC_END_SILENCE_SECONDS = 0.8
MIC_MIN_SPEECH_SECONDS = 0.6
MIC_MAX_SPEECH_SECONDS = 12.0
MIC_PREROLL_SECONDS = 0.5
MIC_CALIBRATION_SECONDS = 1.0
DEBUG_SAVE_MIC_AUDIO = True
MIC_DEVICE = None
TEST_AUDIO_FILE = None

asr_processor = AutoProcessor.from_pretrained(W2V2_MODEL_PATH)
asr_config = AutoConfig.from_pretrained(W2V2_MODEL_PATH)
if hasattr(asr_config, "quantization_config"):
    delattr(asr_config, "quantization_config")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    asr_model = AutoModelForCTC.from_pretrained(
        W2V2_MODEL_PATH,
        config=asr_config,
        low_cpu_mem_usage=True,
    )
    asr_model.to("cuda")
else:
    asr_model = AutoModelForCTC.from_pretrained(
        W2V2_MODEL_PATH,
        config=asr_config,
        low_cpu_mem_usage=True,
    )

print("Quantized Burmese Wav2Vec2 model loaded successfully!")

def transcribe_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    print(f"Audio stats: peak={peak:.5f}, rms={rms:.5f}, samples={len(audio)}")
    if peak < 1e-4:
        return ""

    inputs = asr_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    asr_device = next(asr_model.parameters()).device
    inputs = {key: value.to(asr_device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = asr_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

def save_debug_wav(audio, path, normalize=False):
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if normalize and peak > 0:
        audio = audio / peak

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

def load_wav(path):
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        audio = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav files are supported here, got sample width {sample_width}.")

    audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, but this quick test expects {SAMPLE_RATE} Hz.")

    return audio

def record_utterance():
    block_size = int(SAMPLE_RATE * MIC_BLOCK_SECONDS)
    silence_blocks_needed = max(1, int(MIC_END_SILENCE_SECONDS / MIC_BLOCK_SECONDS))
    max_blocks = max(1, int(MIC_MAX_SPEECH_SECONDS / MIC_BLOCK_SECONDS))
    min_samples = int(MIC_MIN_SPEECH_SECONDS * SAMPLE_RATE)
    calibration_blocks = max(1, int(MIC_CALIBRATION_SECONDS / MIC_BLOCK_SECONDS))
    preroll_blocks = max(1, int(MIC_PREROLL_SECONDS / MIC_BLOCK_SECONDS))

    chunks = []
    preroll = deque(maxlen=preroll_blocks)
    speech_started = False
    silent_blocks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=block_size,
        device=MIC_DEVICE,
    ) as stream:
        print("Calibrating mic noise. Stay quiet...")
        noise_floor = 0.0
        for _ in range(calibration_blocks):
            block, _ = stream.read(block_size)
            block = block.reshape(-1)
            noise_floor += float(np.sqrt(np.mean(np.square(block))))

        noise_floor /= calibration_blocks
        start_threshold = max(MIC_START_THRESHOLD, noise_floor * 3.0)
        silence_threshold = max(MIC_SILENCE_THRESHOLD, noise_floor * 1.5)
        print(f"Listening... noise={noise_floor:.5f}, start={start_threshold:.5f}, silence={silence_threshold:.5f}")

        while True:
            block, _ = stream.read(block_size)
            block = block.reshape(-1)
            volume = float(np.sqrt(np.mean(np.square(block))))

            if not speech_started:
                preroll.append(block.copy())
                if volume >= start_threshold:
                    speech_started = True
                    chunks.extend(preroll)
                    chunks.append(block.copy())
                continue

            chunks.append(block.copy())

            if volume < silence_threshold:
                silent_blocks += 1
            else:
                silent_blocks = 0

            enough_audio = sum(len(chunk) for chunk in chunks) >= min_samples
            reached_silence = silent_blocks >= silence_blocks_needed
            reached_limit = len(chunks) >= max_blocks

            if (enough_audio and reached_silence) or reached_limit:
                audio = np.concatenate(chunks)
                return audio

if __name__ == "__main__":
    print(sd.query_devices())
    if TEST_AUDIO_FILE:
        audio = load_wav(TEST_AUDIO_FILE)
        burmese_prompt = transcribe_audio(audio)
        print(f"File recognized: {burmese_prompt}")
        raise SystemExit

    while True:
        audio = record_utterance()
        if DEBUG_SAVE_MIC_AUDIO:
            save_debug_wav(audio, "mic_debug_raw.wav", normalize=False)
            save_debug_wav(audio, "mic_debug.wav", normalize=True)
        burmese_prompt = transcribe_audio(audio)
        duration = len(audio) / SAMPLE_RATE
        print(f"Recognized ({duration:.2f}s): {burmese_prompt}")
        with open("response.txt", "a", encoding="utf-8") as f:
            f.write(burmese_prompt + "\n")
