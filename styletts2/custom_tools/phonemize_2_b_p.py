import pandas as pd

df = pd.read_csv("kokoro_tts_finetune\\datasets\\my_mm_female\\metadata_original.csv",)

sentences = df["transcription"]

with open("kokoro_tts_finetune\\txt_files\\burmese_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence)
        f.write("\n")