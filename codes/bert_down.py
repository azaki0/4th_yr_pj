from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("YonaKhine/finetuned-w2v2-bert-burmese-asr")
model = AutoModelForCTC.from_pretrained("YonaKhine/finetuned-w2v2-bert-burmese-asr")