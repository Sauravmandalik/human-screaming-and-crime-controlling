
import numpy as np
import soundfile as sf
import joblib

from scipy.signal import resample_poly
from math import gcd
from python_speech_features import mfcc

# File paths
AUDIO_PATH = "recordings/test.wav"
FEATURES_PATH = "data/features_v2.npy"
LABELS_PATH = "data/labels_v2.npy"
MODEL_PATH = "models/scream_detector.pkl"

TARGET_RATE = 22050

# 1. Load the latest recording
audio, sample_rate = sf.read(AUDIO_PATH)

print("\n========== AUDIO INFORMATION ==========")
print("Original sample rate:", sample_rate)
print("Audio duration:", round(len(audio) / sample_rate, 2), "seconds")

# 2. Convert stereo audio to mono
if audio.ndim == 2:
    audio = np.mean(audio, axis=1)

# 3. Convert audio to 22050 Hz
if sample_rate != TARGET_RATE:
    common = gcd(sample_rate, TARGET_RATE)

    audio = resample_poly(
        audio,
        TARGET_RATE // common,
        sample_rate // common
    )

# 4. Extract MFCC features
mfcc_features = mfcc(
    audio.astype(np.float64),
    samplerate=TARGET_RATE,
    numcep=20,
    nfft=1024
)

# 5. Calculate mean and standard deviation
mfcc_mean = np.mean(mfcc_features, axis=0)
mfcc_std = np.std(mfcc_features, axis=0)

new_features = np.concatenate([mfcc_mean, mfcc_std])
new_input = new_features.reshape(1, -1)

print("\nExtracted feature shape:", new_input.shape)

# 6. Load training dataset and model
X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)
model = joblib.load(MODEL_PATH)

print("\n========== DATASET INFORMATION ==========")
print("Training features:", X.shape)
print("Training labels:", y.shape)
print("Class distribution:", np.unique(y, return_counts=True))
print("All features finite:", np.isfinite(X).all())
print("New features finite:", np.isfinite(new_input).all())

# 7. Check feature dimensions
if X.shape[1] != new_input.shape[1]:
    raise ValueError("Training and recording feature dimensions do not match.")

if not np.isfinite(X).all() or not np.isfinite(new_input).all():
    raise ValueError("NaN or infinite values found in features.")

# 8. Get model prediction and decision score
prediction = model.predict(new_input)[0]
score = model.decision_function(new_input)[0]

print("\n========== MODEL RESULT ==========")
print("SVM decision score:", round(float(score), 4))
print("Prediction:", "SCREAM" if prediction == 1 else "NON-SCREAM")

# 9. Apply the model's fitted StandardScaler
scaler = model.steps[0][1]

X_scaled = scaler.transform(X)
new_scaled = scaler.transform(new_input)

# 10. Find the 10 closest training examples
distances = np.linalg.norm(X_scaled - new_scaled, axis=1)
nearest_indices = np.argsort(distances)[:10]

print("\n========== 10 CLOSEST TRAINING SAMPLES ==========")

for rank, index in enumerate(nearest_indices, start=1):
    label = "SCREAM" if y[index] == 1 else "NON-SCREAM"

    print(
        f"{rank}. {label} | Distance: {distances[index]:.3f}"
    )

# 11. Compare SVM scores across the training dataset
training_scores = model.decision_function(X)

print("\n========== TRAINING SCORE COMPARISON ==========")

for label, name in [(1, "SCREAM"), (0, "NON-SCREAM")]:
    class_scores = training_scores[y == label]

    if len(class_scores) > 0:
        print(
            f"{name}: "
            f"Min={class_scores.min():.3f}, "
            f"Max={class_scores.max():.3f}, "
            f"Mean={class_scores.mean():.3f}"
        )

print("\n========== DIAGNOSTIC COMPLETED ==========")
