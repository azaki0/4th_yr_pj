import re


with open("kokoro_tts_finetune\\datasets\\my_mm_female_24k\\metadata.csv", "r", encoding='utf-8') as f:
    sentence = f.read()
    result = re.sub(r',(?=(?:(?:[^"]*"){2})*[^"]*$)', '|', sentence)
    
with open("kokoro_tts_finetune\\datasets\\my_mm_female_24k\\metadata1.csv", "w", encoding='utf-8') as f:
    f.write(result)
