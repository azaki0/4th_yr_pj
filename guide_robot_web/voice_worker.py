import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHISPER_PATH = r"D:\models\whisper\models--Systran--faster-whisper-medium\snapshots\08e178d48790749d25932bbc082711ddcfdfbc4f"
VAD_PATH = str(PROJECT_ROOT / "txt_files" / "silero_vad.jit")

from silero import VoiceTranscriber

transcriber = None


def ensure_transcriber():
    global transcriber
    if transcriber is None:
        transcriber = VoiceTranscriber(WHISPER_PATH, VAD_PATH)
        transcriber.start_listening()
    return transcriber


def write_response(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        try:
            command = json.loads(line)
            action = command.get("action")

            if action == "preload":
                ensure_transcriber()
                write_response({"ok": True, "loaded": ["whisper", "silero_vad"]})
            elif action == "capture":
                timeout = float(command.get("timeout", 30))
                transcriber = ensure_transcriber()
                transcriber.reset_capture()
                text = transcriber.get_next_text(timeout=timeout)
                write_response({"ok": True, "text": text})
            else:
                write_response({"ok": False, "error": "unknown action"})
        except Exception as exc:
            write_response({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
