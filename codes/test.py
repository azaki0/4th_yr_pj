from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from burmese_syntax_corrector import correct_english_to_burmese_blueprint

TRANSLATER_MODEL_PATH = r"D:\models\nllb-200-distilled-600M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#tokenizer = AutoTokenizer.from_pretrained(TRANSLATER_MODEL_PATH)
#model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATER_MODEL_PATH).to(device)

text = "NVIDIA is one of the most important technology companies in the world, best known for designing powerful graphics processing units, or GPUs, that are used in gaming, creative work, scientific computing, artificial intelligence, robotics, and data centers. The company started by focusing on computer graphics, but over time its GPUs became useful for much more than rendering images because they can process many calculations in parallel. This made NVIDIA hardware especially valuable for training and running modern AI models, including large language models, computer vision systems, speech recognition tools, and autonomous machine applications. NVIDIA also provides software platforms such as CUDA, which allows developers and researchers to write programs that use GPU acceleration efficiently. Today, NVIDIA plays a major role in the AI industry because many companies, universities, and research labs rely on its chips and tools to build faster, more capable intelligent systems."


def translate(text, src_lang, target_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=512,
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


syntax_result = correct_english_to_burmese_blueprint(text)
preprocessed_text = syntax_result.blueprint

with open("translated_text.txt", "a", encoding="utf-8") as f:
    f.write(f"source: {text}\n")
    f.write(f"no_preprocess: {translate(text, 'eng_Latn', 'mya_Mymr')}\n")
    #f.write(f"blueprint_debug: {syntax_result.blueprint}\n")
    f.write(f"nllb_safe_text: {syntax_result.nllb_text}\n")
    f.write(f"translated_safe: {translate(syntax_result.nllb_text, 'eng_Latn', 'mya_Mymr')}\n\n")