
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
filenames = []

categories = {
    "scream": 1,
    "non_scream": 0,
    "normal_speech": 0,
    "laughing": 0,
    "shouting": 0
}
for category, label in categories.items():

    folder_path = os.path.join(DATASET_PATH, category)

    print(f"\nProcessing: {category}")

    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith((".wav", ".flac")):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            audio, sample_rate = sf.read(file_path)

            # Convert stereo audio to mono
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            # Resample audio to 22050 Hz
            if sample_rate != TARGET_RATE:
                common = gcd(sample_rate, TARGET_RATE)

                audio = resample_poly(
                    audio,
                    TARGET_RATE // common,
                    sample_rate // common
                )

            if len(audio) == 0:
                continue

            # Extract MFCC features
            mfcc_features = mfcc(
                audio.astype(np.float64),
                samplerate=TARGET_RATE,
                numcep=20,
                nfft=1024
            )

            # Calculate mean and standard deviation
            mfcc_mean = np.mean(mfcc_features, axis=0)
            mfcc_std = np.std(mfcc_features, axis=0)

            combined_features = np.concatenate(
                [mfcc_mean, mfcc_std]
            )

            features.append(combined_features)
            labels.append(label)
            filenames.append(os.path.join(category, filename))

        except Exception as error:
            print(f"Could not process {filename}: {error}")

X = np.array(features)
y = np.array(labels)
file_names = np.array(filenames)

os.makedirs("data", exist_ok=True)

np.save("data/features_v2.npy", X)
np.save("data/labels_v2.npy", y)
np.save("data/filenames_v2.npy", file_names)

print("\nFeature extraction completed!")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Filename count:", len(file_names))
print("Scream samples:", np.sum(y == 1))
print("Non-scream samples:", np.sum(y == 0))
print("All three arrays aligned:", len(X) == len(y) == len(file_names))
