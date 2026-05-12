from pathlib import Path
import subprocess


INPUT_FOLDER = Path(r"kokoro_tts_finetune\datasets\my_mm_female")
OUTPUT_FOLDER = Path(r"kokoro_tts_finetune\datasets\my_mm_female_24k")

TARGET_SAMPLE_RATE = 24000

def convert_one(input_wav, output_wav):
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(output_wav),
        ],
        check=True,
    )


def main():
    wav_files = sorted(INPUT_FOLDER.rglob("*.wav"))

    print(f"Found {len(wav_files)} wav files.")

    for index, input_wav in enumerate(wav_files, start=1):
        output_wav = OUTPUT_FOLDER / input_wav.relative_to(INPUT_FOLDER)
        convert_one(input_wav, output_wav)
        print(f"{index}/{len(wav_files)} converted: {input_wav.name}")

    print("Done.")
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
