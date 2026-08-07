import json
import os
import threading
import warnings
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
from llama_cpp import Llama


DB_PARAMS = {
    "dbname": "misaki_en",
    "user": "azaki",
    "password": "mackenziefoy",
    "host": "localhost",
    "port": "5432",
}

QWEN_MODEL_PATH = r"D:\models\qwen\qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = "D:/models/nomic-embed-text-v1.5.Q6_K.gguf"

SAMPLE_RATE = 24000

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LLAMA_CPP_LIB"] = "0"
warnings.filterwarnings("ignore")

gpu_lock = threading.Lock()

print("Loading models.")

llm = Llama(
    model_path=QWEN_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=24,
    verbose=False,
)

pipeline = KPipeline(lang_code="a", device="cuda")

