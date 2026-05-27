from kittentts import KittenTTS
import sounddevice as sd
m = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = m.generate("This high quality TTS model works without a GPU", voice='Kiki' )

# available_voices : ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
# Save the audio

sd.play(audio, 24000)
sd.wait()
