from pathlib import Path

G2P_DICT = Path("kokoro_tts_finetune\\txt_files\\myg2p.ver2.0.txt")
INPUT_TEXT = Path("kokoro_tts_finetune\\txt_files\\burmese_sentences.txt")
OUTPUT_TEXT = Path("kokoro_tts_finetune\\txt_files\\burmese_phonemes.txt")
USE_IPA = True

def load_g2p():
    g2p = {}

    with G2P_DICT.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            _, word, syllables, pronunciation, ipa = parts[:5]
            output = ipa if USE_IPA else pronunciation
            g2p.setdefault(word, output)

            syl_parts = syllables.split()
            out_parts = output.split()
            if len(syl_parts) == len(out_parts):
                for syl, out in zip(syl_parts, out_parts):
                    g2p.setdefault(syl, out)

    return g2p


def convert_sentence(text, g2p):
    result = []
    max_len = max(len(word) for word in g2p)
    i = 0

    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        match = None
        for end in range(min(len(text), i + max_len), i, -1):
            chunk = text[i:end]
            if chunk in g2p:
                match = chunk
                break

        if match:
            result.append(g2p[match])
            i += len(match)
        else:
            result.append(text[i])
            i += 1

    return " ".join(result)


def main():
    g2p = load_g2p()
    lines = INPUT_TEXT.read_text(encoding="utf-8").splitlines()

    phoneme_lines = []
    for line in lines:
        line = line.strip()
        if line:
            phoneme_lines.append(convert_sentence(line, g2p))

    OUTPUT_TEXT.write_text("\n".join(phoneme_lines), encoding="utf-8")
    print(f"Wrote {len(phoneme_lines)} lines to {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()
