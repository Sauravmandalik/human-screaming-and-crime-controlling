
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

DATASET_PATH = "dataset"
WINDOW_SECONDS = 2

FREQUENCY_BANDS = {
    "low_0_500": (0, 500),
    "low_mid_500_1000": (500, 1000),
    "mid_1000_2000": (1000, 2000),
    "high_2000_4000": (2000, 4000),
    "very_high_4000_8000": (4000, 8000),
    "upper_8000_11025": (8000, 11025)
}

results = []

for category in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(folder):
        continue

    label = "scream" if category.lower() == "scream" else "non_scream"

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".wav", ".flac")):
            continue

        path = os.path.join(folder, filename)

        try:
            audio, sr = sf.read(path)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            window_size = sr * WINDOW_SECONDS

            for i in range(len(audio) // window_size):

                window = audio[
                    i * window_size:(i + 1) * window_size
                ]

                frequencies, power = welch(
                    window,
                    fs=sr,
                    nperseg=min(2048, len(window))
                )

                total_power = np.trapezoid(power, frequencies)

                if total_power == 0:
                    continue

                row = {
                    "filename": filename,
                    "category": category,
                    "label": label,
                    "window": i + 1
                }

                for band, (low, high) in FREQUENCY_BANDS.items():

                    mask = (
                        (frequencies >= low) &
                        (frequencies < min(high, sr / 2))
                    )

                    band_power = np.trapezoid(
                        power[mask],
                        frequencies[mask]
                    ) if np.sum(mask) >= 2 else 0

                    row[band] = (
                        band_power / total_power
                    ) * 100

                row["dominant_frequency"] = frequencies[
                    np.argmax(power)
                ]

                results.append(row)

        except Exception as e:
            print("Error:", filename, e)

df = pd.DataFrame(results)

os.makedirs("analysis", exist_ok=True)

df.to_csv(
    "analysis/frequency_results.csv",
    index=False
)

print("\nFrequency analysis completed.")
print("Total windows:", len(df))
print("\nAverage frequency-band energy by class:")

columns = list(FREQUENCY_BANDS.keys())

print(
    df.groupby("label")[columns + ["dominant_frequency"]]
    .mean()
    .round(2)
)

print("\nSaved: analysis/frequency_results.csv")
