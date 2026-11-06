import sounddevice as sd
import soundfile as sf
import os

duration = 5
sample_rate = 44100

os.makedirs("recordings", exist_ok=True)

print("Recording started... Speak now!")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    device=1
)

sd.wait()

sf.write("recordings/test.wav", audio, sample_rate)

print("Recording saved successfully!")