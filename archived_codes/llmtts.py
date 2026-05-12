import threading
import queue
import numpy as np
import sounddevice as sd
import sys
import datetime
from llama_cpp import Llama
from TTS.api import TTS

SPEAKER_WAV_PATH = "ina.wav" 
QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
TTS_MODEL_PATH = "D:/models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
TTS_CONFIG_PATH = "D:/models/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
SAMPLE_RATE = 24000

class MetricsLogger:
    def __init__(self, filename="logs/qwen7b_q4_4096_32.log"):
        self.filename = filename
        self.terminal = sys.stdout #keep reference to the real terminal

    def write(self, message):
        self.terminal.write(message)
        
        if "['" in message:
            with open(self.filename, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] > {message.strip()}, ")

        if "Processing time" in message:
            with open(self.filename, "a") as f:
                f.write(f"{message.strip()}, ")

        if "Real-time factor" in message:
            with open(self.filename, "a") as f:
                f.write(f"{message.strip()}\n")

    def flush(self):
        self.terminal.flush()

sys.stdout = MetricsLogger()

print("Loading Models...")
llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=32,
    verbose=False
)

tts = TTS(
    model_path=TTS_MODEL_PATH,
    config_path=TTS_CONFIG_PATH,
    progress_bar=False,
).to("cuda")

text_queue = queue.Queue()
audio_queue = queue.Queue()


def tts_generation_worker():
    """Consumes text, generates audio, pushes to audio_queue."""
    while True:
        text_chunk = text_queue.get()
        if text_chunk is None:
            audio_queue.put(None)
            break
        
        try:
            wav = tts.tts(
                text=text_chunk, 
                speaker_wav=SPEAKER_WAV_PATH, 
                language="en"
            )
            wav_np = np.array(wav, dtype=np.float32)
            audio_queue.put(wav_np)
        except Exception as e:
            print(f"\n TTS Error: {e}")
        finally:
            text_queue.task_done()

def audio_playback_worker():
    """Consumes audio arrays and plays them sequentially."""
    while True:
        wav_chunk = audio_queue.get()
        if wav_chunk is None: 
            break
        
        sd.play(wav_chunk, samplerate=SAMPLE_RATE)
        sd.wait() 
        audio_queue.task_done()

threading.Thread(target=tts_generation_worker, daemon=True).start()
threading.Thread(target=audio_playback_worker, daemon=True).start()

print("System Ready.")

def stream_qwen(prompt):
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Keep answers concise and conversational."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.7,
    )
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            yield chunk["choices"][0]["delta"]["content"]

def text_chunker(token_stream):
    """Yields complete sentences for smoother TTS."""
    buffer = ""
    sentence_endings = {'!', '?', '\n'}
    
    for token in token_stream:
        buffer += token
        if any(buffer.endswith(p) for p in sentence_endings) and len(buffer.strip()) > 5:
            yield buffer.strip()
            buffer = ""
            
    if buffer.strip():
        yield buffer.strip()


while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ["q", "exit"]:
            break
        
        print("AI: ", end="", flush=True)
        
        token_generator = stream_qwen(user_input)
        
        for sentence in text_chunker(token_generator):
            print(sentence + " ", end="", flush=True)
            text_queue.put(sentence)
            
    except KeyboardInterrupt:
        break

text_queue.put(None)