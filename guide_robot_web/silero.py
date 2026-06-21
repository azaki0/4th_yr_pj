import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
import queue


class VoiceTranscriber:
    def __init__(self, whisper_model_path, vad_model_path):
        self.device = self._pick_device()
        try:
            self.whisper = WhisperModel(whisper_model_path, device=self.device, compute_type="int8_float16" if self.device == "cuda" else "int8")
            self.vad_model = torch.jit.load(vad_model_path).eval()
        except Exception:
            if self.device == "cuda":
                self.device = "cpu"
                self.whisper = WhisperModel(whisper_model_path, device="cpu", compute_type="int8")
                self.vad_model = torch.jit.load(vad_model_path, map_location="cpu").eval()
            else:
                raise

        self.text_queue = queue.Queue()
        self.sample_rate = 16000
        self.block_size = 1024
        self.vad_chunk_size = 512
        self.silence_limit = 15
        self.max_record_seconds = 8

    @staticmethod
    def _pick_device():
        if not torch.cuda.is_available():
            return "cpu"
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            return "cpu"

    def _vad_check(self, chunk):
        x = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            prob = self.vad_model(x, self.sample_rate).item()
        return prob > 0.5

    def _transcribe(self, audio_data):
        if len(audio_data) < self.sample_rate * 0.4: return
        
        segments, _ = self.whisper.transcribe(
            audio_data,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            initial_prompt="NSPU means Naypyitaw State Polytechnic University. Campus places include canteen, main building, workshop, dormitory, stadium, and entrance.",
            no_speech_threshold=0.35,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.6,
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        
        if text:
            self.text_queue.put(text)

    def start_listening(self):
        return

    def get_next_text(self, timeout=None):
        audio_data = self.record_until_silence(timeout or self.max_record_seconds)
        if audio_data is None:
            return None
        self._transcribe(audio_data)
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None

    def reset_capture(self):
        while True:
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

    def record_until_silence(self, timeout):
        recorded = []
        buffer = np.zeros(0, dtype=np.float32)
        silence_frames = 0
        speaking = False
        max_samples = int(self.sample_rate * min(timeout, self.max_record_seconds))

        with sd.InputStream(samplerate=self.sample_rate, blocksize=self.block_size, channels=1, dtype="float32") as stream:
            while sum(len(chunk) for chunk in recorded) < max_samples:
                indata, _ = stream.read(self.block_size)
                buffer = np.concatenate((buffer, indata[:, 0].copy()))

                while len(buffer) >= self.vad_chunk_size:
                    chunk = buffer[:self.vad_chunk_size]
                    buffer = buffer[self.vad_chunk_size:]
                    is_voice = self._vad_check(chunk)

                    if is_voice:
                        speaking = True
                        silence_frames = 0
                        recorded.append(chunk)
                    elif speaking:
                        silence_frames += 1
                        recorded.append(chunk)
                        if silence_frames > self.silence_limit:
                            audio = np.concatenate(recorded).astype(np.float32)
                            return audio if len(audio) >= self.sample_rate * 0.35 else None

        if recorded:
            audio = np.concatenate(recorded).astype(np.float32)
            return audio if len(audio) >= self.sample_rate * 0.35 else None
        return None
        
        #ct2-transformers-converter \ --model C:\\Users\\Kyouk\\.cache\\huggingface\\hub\\models--Chonlasitk--whisper-burmese\\snapshots\\adef53627279bcbe68354c0629477e4b9a2b49b2 \ --D:\\models\\whisper-burmese \ --quantization float16
