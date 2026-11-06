import librosa
import numpy as np

audio, sample_rate = librosa.load(
    "recordings/test.wav",
    sr=22050
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

mfcc_mean = np.mean(mfcc, axis=1)

print("MFCC Features:")
print(mfcc_mean)

print("Total Features:", len(mfcc_mean))