#from transformers import AutoProcessor, AutoModelForCTC

#processor = AutoProcessor.from_pretrained("YonaKhine/finetuned-w2v2-bert-burmese-asr")
#model = AutoModelForCTC.from_pretrained("YonaKhine/finetuned-w2v2-bert-burmese-asr")

from huggingface_hub import snapshot_download
#MODEL_ID = "Systran/faster-distil-whisper-medium.en"
MODEL_ID = "dropbox-dash/faster-whisper-large-v3-turbo"
LOCAL_DIR = "D:\\models\\whisper\\faster-whisper-large-v3-turbo"

model_path = snapshot_download(repo_id=MODEL_ID,local_dir=LOCAL_DIR,max_workers=12)

print(f"Files saved to: {model_path}")
