import threading,queue,time, os
import numpy as np
import sounddevice as sd
import warnings
from llama_cpp import Llama
from kokoro import KPipeline
from silero import VoiceTranscriber

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['PYTHONUTF8'] = '1' #UTF8 py for emojis
warnings.filterwarnings("ignore")

#WHISPER_PATH = r"C:\Users\Kyouk\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots\536b0662742c02347bc0e980a01041f333bce120"
WHISPER_PATH = r"D:\models\models--Systran--faster-distil-whisper-small.en\snapshots\ef77d90526ccd62cde3808ee70626a01e5cf83e4"
VAD_PATH = r"txt_files\silero_vad.jit"
QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
SAMPLE_RATE = 24000
log_filename = "logs/junks.log"

print("Loading Models")
llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=32,
    verbose=False
)

pipeline = KPipeline(lang_code='a', device='cuda')

listener = VoiceTranscriber(WHISPER_PATH, VAD_PATH)
listener.start_listening()

print("Listening")

text_queue = queue.Queue()
audio_queue = queue.Queue()

def tts_generation_worker(): #take in text, generates audio, add to audio_queue
    while True:
        text_chunk = text_queue.get()
        if text_chunk is None:
            audio_queue.put(None)
            break
        
        try:
            start_time = time.perf_counter() #benchmark_counter

            generator = pipeline(
                text=text_chunk, voice='af_heart', 
                speed=1, split_pattern=r'\n+'
            )
            for i, (gs, ps, audio) in enumerate(generator):
                wav = audio
            wav_np = np.array(wav, dtype=np.float32)

            end_time = time.perf_counter()

            proc_time = end_time - start_time
            audio_duration = len(wav_np) / 24000 
            rtf = proc_time / audio_duration if audio_duration > 0 else 0

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            with open(log_filename,"a") as f:
                f.write(f"[{timestamp}], ['{text_chunk}'], Processing time: {proc_time:.4f}, Real-time factor: {rtf:.4f}\n")

            audio_queue.put(wav_np)
        except Exception as e:
            print(f"\n TTS Error: {e}")
        finally:
            text_queue.task_done()

def audio_playback_worker(): #take audio arrays and plays them in order
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

def text_chunker(token_stream): #yield complete sentences
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
        #user_input = listener.get_next_text()
        user_input = input("Enter: ")
        #print(user_input)
        if user_input.lower() in ["q.", "exit.", "bye.", "see you later."]:
            break
        
        print("\nAI: ", end="", flush=True)
        
        token_generator = stream_qwen(user_input)
        
        for sentence in text_chunker(token_generator):
            print(sentence + " ", end="", flush=True)
            text_queue.put(sentence)
            
    except KeyboardInterrupt:
        break

text_queue.put(None)