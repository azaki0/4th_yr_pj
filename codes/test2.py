from guide_robot_web.silero import VoiceTranscriber

WHISPER_PATH = r"D:\models\models--Systran--faster-distil-whisper-small.en\snapshots\ef77d90526ccd62cde3808ee70626a01e5cf83e4"
VAD_PATH = r"txt_files\silero_vad.jit"
SAMPLE_RATE = 24000

listener = VoiceTranscriber(WHISPER_PATH, VAD_PATH)
listener.start_listening()

print("Listening")

while True:
    prompt = listener.get_next_text()
    print(prompt)