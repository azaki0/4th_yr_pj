import sys
from pathlib import Path


REPORT_TEXT = Path("burmese_left_report.txt")


def has_burmese(char):
    code = ord(char)
    return 0x1000 <= code <= 0x109F or 0xAA60 <= code <= 0xAA7F or 0xA9E0 <= code <= 0xA9FF


def main():
    phoneme_text = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kokoro_tts_finetune\\txt_files\\burmese_phonemes.txt")

    if not phoneme_text.exists():
        print(f"File not found: {phoneme_text}")
        print("Usage: python check_burmese_left.py your_phoneme_file.txt")
        return

    lines = phoneme_text.read_text(encoding="utf-8").splitlines()
    report = []

    for line_no, line in enumerate(lines, start=1):
        leftovers = sorted(set(char for char in line if has_burmese(char)))
        if leftovers:
            report.append(f"Line {line_no}: {' '.join(leftovers)}")
            report.append(line)
            report.append("")

    if report:
        REPORT_TEXT.write_text("\n".join(report), encoding="utf-8")
        print(f"Found Burmese letters in {len(report) // 3} lines.")
        print(f"Report written to {REPORT_TEXT}")
    else:
        REPORT_TEXT.write_text("No Burmese letters found.\n", encoding="utf-8")
        print("No Burmese letters found.")


if __name__ == "__main__":
    main()
