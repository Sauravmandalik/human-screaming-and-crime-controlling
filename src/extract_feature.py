
import os
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly
from math import gcd
from python_speech_features import mfcc

DATASET_PATH = "dataset"
TARGET_RATE = 22050

features = []
labels = []

categories = {
    "scream": 1,
    "non_scream": 0
}

for category, label in categories.items():

    folder_path = os.path.join(DATASET_PATH, category)

    print(f"\nProcessing {category}...")

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".wav"):

            file_path = os.path.join(folder_path, filename)

            try:
                audio, sample_rate = sf.read(file_path)

                if audio.ndim == 2:
                    audio = np.mean(audio, axis=1)

                if sample_rate != TARGET_RATE:
                    common = gcd(sample_rate, TARGET_RATE)
                    up = TARGET_RATE // common
                    down = sample_rate // common
                    audio = resample_poly(audio, up, down)

                if len(audio) == 0:
                    continue

                mfcc_features = mfcc(
                    audio.astype(np.float64),
                    samplerate=TARGET_RATE,
                    numcep=20,
                    nfft=1024
                
                )

                mfcc_mean = np.mean(mfcc_features, axis=0)
                mfcc_std = np.std(mfcc_features, axis=0)

                combined_features = np.concatenate(
                    [mfcc_mean, mfcc_std]
                )

                features.append(combined_features)
                labels.append(label)

            except Exception as error:
                print(f"Could not process {filename}: {error}")

X = np.array(features)
y = np.array(labels)

os.makedirs("data", exist_ok=True)

np.save("data/features.npy", X)
np.save("data/labels.npy", y)

print("\nFeature extraction completed!")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Total audio files processed:", len(y))
print("Scream samples:", np.sum(y == 1))
print("Non-scream samples:", np.sum(y == 0))
