
import numpy as np
import soundfile as sf
import joblib

from scipy.signal import resample_poly
from math import gcd
from python_speech_features import mfcc

print("Starting audio diagnosis...")

# Load recording
audio, sample_rate = sf.read("recordings/test.wav")

if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

# Resample audio
TARGET_RATE = 22050

if sample_rate != TARGET_RATE:
    common = gcd(sample_rate, TARGET_RATE)
    audio = resample_poly(
        audio,
        TARGET_RATE // common,
        sample_rate // common
    )

# Extract 40 MFCC features
mfcc_features = mfcc(
    audio.astype(np.float64),
    samplerate=TARGET_RATE,
    numcep=20,
    nfft=1024
)

new_features = np.concatenate([
    np.mean(mfcc_features, axis=0),
    np.std(mfcc_features, axis=0)
])

new_input = new_features.reshape(1, -1)

# Load training data and model
X = np.load("data/features.npy")
y = np.load("data/labels.npy")
model = joblib.load("models/scream_detector.pkl")

# Predict
prediction = model.predict(new_input)[0]
score = model.decision_function(new_input)[0]

print("\n========== MODEL RESULT ==========")
print("Prediction:", "SCREAM" if prediction == 1 else "NON-SCREAM")
print("Decision score:", round(float(score), 4))

# Scale features using the fitted training scaler
scaler = model.steps[0][1]

X_scaled = scaler.transform(X)
new_scaled = scaler.transform(new_input)

# Calculate distances
distances = np.linalg.norm(
    X_scaled - new_scaled,
    axis=1
)

# Compare both classes
scream_distances = distances[y == 1]
non_scream_distances = distances[y == 0]

print("\n========== CLASS DISTANCE COMPARISON ==========")

print("Scream samples:", len(scream_distances))
print("Non-scream samples:", len(non_scream_distances))

print("Closest scream distance:",
      round(float(np.min(scream_distances)), 3))

print("Closest non-scream distance:",
      round(float(np.min(non_scream_distances)), 3))

print("Average scream distance:",
      round(float(np.mean(scream_distances)), 3))

print("Average non-scream distance:",
      round(float(np.mean(non_scream_distances)), 3))

print("\n========== DIAGNOSIS COMPLETE ==========")
