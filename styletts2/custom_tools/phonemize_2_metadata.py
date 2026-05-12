import pandas as pd

df = pd.read_csv("kokoro_tts_finetune\\datasets\\my_mm_female\\metadata_original.csv",)
df1 = pd.read_csv("kokoro_tts_finetune\\txt_files\\burmese_phonemes.txt", header=None, names=['transcription'])

df["transcription"] = df1["transcription"]

df.to_csv("kokoro_tts_finetune\\datasets\\my_mm_female\\metadata.csv", index=False)