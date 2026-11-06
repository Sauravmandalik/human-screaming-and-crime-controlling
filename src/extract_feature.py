import librosa
import numpy as np

# Load recorded audio
audio, sample_rate = librosa.load(
    "recordings/test.wav",
    sr=22050
)

# Extract MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

# Calculate average of each MFCC
mfcc_mean = np.mean(mfcc, axis=1)

print("MFCC Features:")
print(mfcc_mean)

print("Total Features:", len(mfcc_mean))