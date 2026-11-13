
import os
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly
from math import gcd
from python_speech_features import mfcc

DATASET_PATH = "dataset"
TARGET_RATE = 22050

WINDOW_SECONDS = 2
WINDOW_SIZE = TARGET_RATE * WINDOW_SECONDS

features = []
labels = []
filenames = []
window_numbers = []

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

            # Resample to 22050 Hz
            if sample_rate != TARGET_RATE:
                common = gcd(sample_rate, TARGET_RATE)

                audio = resample_poly(
                    audio,
                    TARGET_RATE // common,
                    sample_rate // common
                )

            audio = audio.astype(np.float64)

            # Process complete 2-second windows only
            total_windows = len(audio) // WINDOW_SIZE

            if total_windows == 0:
                print(f"Skipped (shorter than 2 seconds): {filename}")
                continue

            for window_index in range(total_windows):

                start = window_index * WINDOW_SIZE
                end = start + WINDOW_SIZE

                audio_window = audio[start:end]

                mfcc_features = mfcc(
                    audio_window,
                    samplerate=TARGET_RATE,
                    numcep=20,
                    nfft=1024
                )

                mfcc_mean = np.mean(mfcc_features, axis=0)
                mfcc_std = np.std(mfcc_features, axis=0)

                combined_features = np.concatenate([
                    mfcc_mean,
                    mfcc_std
                ])

                features.append(combined_features)
                labels.append(label)

                filenames.append(
                    os.path.join(category, filename)
                )

                window_numbers.append(window_index)

        except Exception as error:
            print(f"Could not process {filename}: {error}")

X = np.array(features)
y = np.array(labels)
file_names = np.array(filenames)
window_ids = np.array(window_numbers)

os.makedirs("data", exist_ok=True)

np.save("data/features_v3.npy", X)
np.save("data/labels_v3.npy", y)
np.save("data/filenames_v3.npy", file_names)
np.save("data/window_numbers_v3.npy", window_ids)

print("\nFeature extraction completed!")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("Filename count:", len(file_names))
print("Window number count:", len(window_ids))
print("Scream windows:", np.sum(y == 1))
print("Non-scream windows:", np.sum(y == 0))
print("All arrays aligned:",
      len(X) == len(y) == len(file_names) == len(window_ids))
