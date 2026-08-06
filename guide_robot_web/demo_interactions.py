import time
import wave
import winsound
from array import array
from pathlib import Path
from bridge import reset_display, send_event, send_text
from servo_controller import close_mouth, play_speech

AUDIO_DIR = Path(__file__).resolve().parent / "static" / "audio"
TEXT_CHUNK_DELAY_SECONDS = 0.18
AUDIO_LEAD_MS = 200

INTERACTIONS = [
    {
        "id": "welcome",
        "label": "Welcome",
        "text": "မဂ်လာပါ။ ကျွန်မကတော့ နေပြည်တော်နည်းပညာတက္ကသိုလ် Naypyitaw State Polytechnic University စက်မှုဌာနမှ ကျောင်းသားများမှ ဖန်တီးထားသော AI စက်ရုပ်တစ်ခုဖြစ်ပါတယ်။",
        "audio": "welcome.wav",
    },
    {
        "id": "overview",
        "label": "NSPU Overview",
        "text": "NayPyiTaw State Polytechnic တက္ကသိုလ်သည် ဇမ္ဗူသီရိ မြို့နယ်တွင် တည်ရှိသော နည်းပညာတက္ကသိုလ် တစ်ခု ဖြစ်ပါသည်။ ဤတက္ကသိုလ်တွင် မြို့ပြ၊ စက်မှု၊ အီလက်ထရွန်းနစ်၊ လျှပ်စစ်စွမ်းအား၊ ကွန်ပျူတာ အင်ဂျင်နီယာနှင့် ဗိသုကာ ဘာသာရပ်များကို သင်ကြားပေးလျက် ရှိပါသည်။",
        "audio": "nspu_overview.wav",
    },
    {
        "id": "help",
        "label": "Demo",
        "text": "အင်တာနက် ချိတ်ဆက်မှု မရရှိသည့်အတွက် ကျွန်မရဲ့ စွမ်းဆောင်နိုင်စွမ်း အစစ်အမှန်ကို မပြသနိုင်သော်လည်း ယခု Demo Version မှတစ်ဆင့် Screen ဖြင့်  Interactive ဖြစ်အောင် ပြုလုပ်ထားပါတယ်ရှင့်။",
        "audio": "saying_demo.wav",
    },
    {
        "id": "goodbye",
        "label": "Goodbye",
        "text": "အခုလို မိတ်ဆက်ခွင့်ရရှိသည့်အတွက်ကျေးဇူးတင်ပါတယ်ရှင်။ ဒီမှာတင်နှုတ်ဆက်လိုက်ပါတယ်ရှင်။",
        "audio": "goodbye.wav",
    },
]

def _read_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        mono = [(sample - 128) / 128 for sample in frames]
    elif sample_width == 2:
        samples = array("h")
        samples.frombytes(frames)
        mono = [sample / 32768 for sample in samples]
    elif sample_width == 4:
        samples = array("i")
        samples.frombytes(frames)
        mono = [sample / 2147483648 for sample in samples]
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        mono = [
            sum(mono[index:index + channels]) / channels
            for index in range(0, len(mono), channels)
        ]

    return mono, sample_rate

def _play_audio_file(filename):
    path = AUDIO_DIR / filename
    if not path.exists():
        send_text(f"\n\nAudio placeholder missing: {filename}")
        return

    audio, sample_rate = _read_wav(path)
    duration_seconds = len(audio) / sample_rate if sample_rate else 0

    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    time.sleep(AUDIO_LEAD_MS / 1000)
    play_speech(audio, sample_rate)
    time.sleep(max(0, duration_seconds - (AUDIO_LEAD_MS / 1000)))
    winsound.PlaySound(None, winsound.SND_PURGE)
    close_mouth()

def _stream_text(text):
    for sentence in text.split(". "):
        clean = sentence.strip()
        if not clean:
            continue
        suffix = "" if clean.endswith((".", "!", "?")) else "."
        send_text(clean + suffix + " ")
        time.sleep(TEXT_CHUNK_DELAY_SECONDS)

def run_interaction(interaction):
    reset_display()
    send_event("interaction", {"id": interaction["id"], "label": interaction["label"]})
    _stream_text(interaction["text"])
    _play_audio_file(interaction["audio"])
    send_event("done")
