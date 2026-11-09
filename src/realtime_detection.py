import numpy as np
import sounddevice as sd
import joblib
from collections import deque

from python_speech_features import mfcc

MODEL_PATH = "models/scream_detector.pkl"
SAMPLE_RATE = 22050
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

model = joblib.load(MODEL_PATH)

prediction_history = deque(maxlen=3)

print("Real-time Scream Detection Started")
print("Press Ctrl + C to stop.")

try:
    while True:
        audio = sd.rec(
            WINDOW_SIZE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()
        audio = audio.flatten()

        mfcc_features = mfcc(
            audio.astype(np.float64),
            samplerate=SAMPLE_RATE,
            numcep=20,
            nfft=1024
        )

        features = np.concatenate([
            np.mean(mfcc_features, axis=0),
            np.std(mfcc_features, axis=0)
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        prediction_history.append(prediction == 1)

        raw_result = "SCREAM" if prediction == 1 else "NON-SCREAM"

        if len(prediction_history) < 3:
            final_result = "ANALYSING"
        elif sum(prediction_history) >= 2:
            final_result = "SCREAM DETECTED"
        else:
            final_result = "NO SCREAM"

        print(
            f"Raw: {raw_result} | "
            f"Score: {score:.3f} | "
            f"Final: {final_result}"
        )

except KeyboardInterrupt:
    print("\nDetection stopped.")
