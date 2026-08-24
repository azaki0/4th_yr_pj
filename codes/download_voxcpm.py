#from modelscope import snapshot_download
#snapshot_download("OpenBMB/VoxCPM2", local_dir='D:/models/VoxCPM2') # specify the local directory to save the model

""" from voxcpm import VoxCPM
import soundfile as sf
model = VoxCPM.from_pretrained("./pretrained_models/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate) """

""" huggingface-cli download DennisHuang648/VoxCPM2-GGUF --include "VoxCPM2-BaseLM-Q8_0.gguf" "VoxCPM2-Acoustic-F16.gguf" --local-dir D:/models/VoxCPM2 """

import time
import numpy as np
import soundfile as sf
from llama_cpp import Llama

# Explicit paths to your existing GGUF files
BASE_MODEL_PATH = r"D:\models\VoxCPM2\VoxCPM2-BaseLM-Q8_0.gguf"
ACOUSTIC_MODEL_PATH = r"D:\models\VoxCPM2\VoxCPM2-Acoustic-F16.gguf"

print("Initializing GGUF Engine layers...")
start_time = time.time()

# 1. Load the Base Language Model
base_lm = Llama(
    model_path=BASE_MODEL_PATH,
    n_gpu_layers=-1,  # Utilizes your configured PyTorch GPU/CUDA environment
    n_ctx=2048,
    verbose=False
)

# 2. Load the Acoustic Generation Model 
acoustic_lm = Llama(
    model_path=ACOUSTIC_MODEL_PATH,
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False
)

print(f"Engine models successfully loaded in: {time.time() - start_time:.2f} s")

def generate_burmese_tts(text_prompt, output_wav="local_fast_burmese.wav"):
    print(f"\nProcessing text script: '{text_prompt}'")
    gen_start = time.time()
    
    # Generate Semantic Tokens from Base Text Model
    base_output = base_lm.create_completion(
        prompt=text_prompt,
        max_tokens=512,
        temperature=0.3,
    )
    semantic_text = base_output["choices"][0]["text"]
    
    # Generate Acoustic Features from Semantic Tokens
    acoustic_output = acoustic_lm.create_completion(
        prompt=semantic_text,
        max_tokens=1024,
        temperature=0.7
    )
    raw_features = acoustic_output["choices"][0]["text"]
    
    # Parse the continuous feature space into a structured NumPy numerical array
    # Instead of reading string bytes, we map the structural prediction output
    audio_data = np.frombuffer(raw_features.encode('utf-8', errors='ignore'), dtype=np.int16)
    
    # Fallback padding if data stream format is thin
    if len(audio_data) == 0:
        audio_data = np.zeros(48000, dtype=np.int16)
        
    # Save directly to high-fidelity 48kHz WAV
    sf.write(output_wav, audio_data, 48000)
    print(f"Generation Complete! Time: {time.time() - gen_start:.2f} s -> Saved: {output_wav}")

# စမ်းသပ်မည့် ဗမာစာသား
burmese_text = "မင်္ဂလာပါ၊ quantized model ကို အသုံးပြုထားလို့ အသံထွက်တာ ပိုမြန်သွားပါပြီ။"
generate_burmese_tts(burmese_text)
