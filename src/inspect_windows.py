import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from pathlib import Path

audio_path = Path("dataset/scream/nonspeech7k_005247.flac")
output_dir = Path("inspection")
output_dir.mkdir(exist_ok=True)

audio, sample_rate = sf.read(audio_path)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

target_rate = 22050

if sample_rate != target_rate:
    common = gcd(sample_rate, target_rate)
    audio = resample_poly(
        audio,
        target_rate // common,
        sample_rate // common
    )

window_size = target_rate * 2

print("Original sample rate:", sample_rate)
print("Audio duration:", round(len(audio) / target_rate, 2), "seconds")

for i in range(2):
    start = i * window_size
    end = start + window_size
    window = audio[start:end]

    if len(window) == window_size:
        output_path = output_dir / f"window_{i + 1}.wav"
        sf.write(output_path, window, target_rate)
        print("Saved:", output_path)
