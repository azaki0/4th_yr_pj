import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
import queue
import threading

class VoiceTranscriber:
    def __init__(self, whisper_model_path, vad_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper = WhisperModel(whisper_model_path, device=self.device, compute_type="int8")
        self.vad_model = torch.jit.load(vad_model_path).eval()
    
        self.text_queue = queue.Queue()
        
        self.sample_rate = 16000
        self.vad_chunk_size = 512
        self.silence_limit = 15
        
        self.buffer = np.zeros(0, dtype=np.float32)
        self.speech_accum = []
        self.silence_frames = 0
        self.is_speaking = False

    def _vad_check(self, chunk):
        x = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            prob = self.vad_model(x, self.sample_rate).item()
        return prob > 0.5

    def _audio_callback(self, indata, frames, time, status):
        self.buffer = np.concatenate((self.buffer, indata[:, 0]))
        
        while len(self.buffer) >= self.vad_chunk_size:
            chunk = self.buffer[:self.vad_chunk_size]
            self.buffer = self.buffer[self.vad_chunk_size:]
            
            if self._vad_check(chunk):
                if not self.is_speaking:
                    self.is_speaking = True
                self.silence_frames = 0
                self.speech_accum.extend(chunk)
            elif self.is_speaking:
                self.silence_frames += 1
                self.speech_accum.extend(chunk)
                
                if self.silence_frames > self.silence_limit:
                    self.is_speaking = False
                    audio_data = np.array(self.speech_accum, dtype=np.float32)
                    self.speech_accum = []
                    
                    threading.Thread(target=self._transcribe, args=(audio_data,), daemon=True).start()

    def _transcribe(self, audio_data):
        if len(audio_data) < self.sample_rate * 0.4: return
        
        segments, _ = self.whisper.transcribe(audio_data, beam_size=5, language="en")
        text = " ".join([s.text.strip() for s in segments]).strip()
        
        if text:
            self.text_queue.put(text)

    def start_listening(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.vad_chunk_size,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()

    def get_next_text(self, timeout=None):
        try:
            return self.text_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
        #ct2-transformers-converter \ --model C:\\Users\\Kyouk\\.cache\\huggingface\\hub\\models--Chonlasitk--whisper-burmese\\snapshots\\adef53627279bcbe68354c0629477e4b9a2b49b2 \ --D:\\models\\whisper-burmese \ --quantization float16
