
import os
import csv
import numpy as np
import soundfile as sf
import joblib

from datetime import datetime
from scipy.signal import resample_poly
from math import gcd

from python_speech_features import mfcc

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# File locations and settings
MODEL_PATH = "models/scream_detector.pkl"
TEST_PATH = "test_audio"
RESULT_PATH = "evaluation_results"
TARGET_RATE = 22050

features = []
labels = []
filenames = []

categories = {
    "scream": 1,
    "non_scream": 0
}

print("Starting independent audio evaluation...")

# Step 1: Read and process test audio
for category, label in categories.items():

    folder = os.path.join(TEST_PATH, category)

    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue

    print("\nProcessing:", category)

    for filename in sorted(os.listdir(folder)):

        if not filename.lower().endswith((".wav", ".flac")):
            continue

        path = os.path.join(folder, filename)

        try:
            audio, sample_rate = sf.read(path)

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            if sample_rate != TARGET_RATE:
                common = gcd(sample_rate, TARGET_RATE)

                audio = resample_poly(
                    audio,
                    TARGET_RATE // common,
                    sample_rate // common
                )

            if len(audio) == 0:
                print("Empty audio:", filename)
                continue

            mfcc_features = mfcc(
                audio.astype(np.float64),
                samplerate=TARGET_RATE,
                numcep=20,
                nfft=1024
            )

            mfcc_mean = np.mean(mfcc_features, axis=0)
            mfcc_std = np.std(mfcc_features, axis=0)

            combined = np.concatenate([
                mfcc_mean,
                mfcc_std
            ])

            features.append(combined)
            labels.append(label)
            filenames.append(filename)

        except Exception as error:
            print("Could not process", filename, ":", error)


# Step 2: Check test data
if len(features) == 0:
    print("\nNo test audio files found.")
    raise SystemExit


# Step 3: Convert to NumPy arrays
X_test = np.array(features)
y_test = np.array(labels)

print("\nTest recordings:", len(X_test))
print("Feature shape:", X_test.shape)


# Step 4: Load trained model
model = joblib.load(MODEL_PATH)


# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Calculate decision scores
scores = model.decision_function(X_test)

print("\nDecision Score Analysis:")
print("Minimum score:", round(np.min(scores), 3))
print("Maximum score:", round(np.max(scores), 3))
print("Average score:", round(np.mean(scores), 3))


# Step 7: Calculate metrics
accuracy = accuracy_score(y_test, predictions) * 100

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Non-scream", "Scream"],
    zero_division=0
)

false_positives = []
false_negatives = []

for filename, actual, predicted, score in zip(
    filenames, y_test, predictions, scores
):

    if actual == 0 and predicted == 1:
        false_positives.append((filename, score))

    elif actual == 1 and predicted == 0:
        false_negatives.append((filename, score))


# Step 8: Create evaluation output folder
os.makedirs(RESULT_PATH, exist_ok=True)

# Use one timestamp for both output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = os.path.join(
    RESULT_PATH,
    f"predictions_{timestamp}.csv"
)

report_path = os.path.join(
    RESULT_PATH,
    f"report_{timestamp}.txt"
)


# Step 9: Save individual predictions to CSV
with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Filename",
        "Actual",
        "Predicted",
        "Decision Score",
        "Result"
    ])

    for filename, actual, predicted, score in zip(
        filenames, y_test, predictions, scores
    ):

        actual_name = (
            "SCREAM" if actual == 1 else "NON-SCREAM"
        )

        predicted_name = (
            "SCREAM" if predicted == 1 else "NON-SCREAM"
        )

        status = (
            "CORRECT" if actual == predicted else "WRONG"
        )

        writer.writerow([
            filename,
            actual_name,
            predicted_name,
            round(float(score), 4),
            status
        ])


# Step 10: Prepare review folder information
review_folder = os.path.join(
    TEST_PATH,
    "review_non_scream"
)

review_files = []

if os.path.isdir(review_folder):
    review_files = [
        name for name in sorted(os.listdir(review_folder))
        if name.lower().endswith((".wav", ".flac"))
    ]


# Step 11: Save complete text report
with open(report_path, "w", encoding="utf-8") as file:

    file.write("HUMAN SCREAM DETECTION - EVALUATION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(
        "Evaluation date and time: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "\n"
    )

    file.write(f"Model: {MODEL_PATH}\n")
    file.write(f"Test folder: {TEST_PATH}\n")
    file.write(f"Total recordings: {len(X_test)}\n")
    file.write(f"Scream recordings: {np.sum(y_test == 1)}\n")
    file.write(f"Non-scream recordings: {np.sum(y_test == 0)}\n\n")

    file.write("FEATURE INFORMATION\n")
    file.write("-" * 50 + "\n")
    file.write("Sample rate: 22050 Hz\n")
    file.write("MFCC coefficients: 20\n")
    file.write("Features per recording: 40\n\n")

    file.write("DECISION SCORE ANALYSIS\n")
    file.write("-" * 50 + "\n")
    file.write(f"Minimum: {np.min(scores):.3f}\n")
    file.write(f"Maximum: {np.max(scores):.3f}\n")
    file.write(f"Average: {np.mean(scores):.3f}\n\n")

    file.write("EVALUATION RESULTS\n")
    file.write("-" * 50 + "\n")
    file.write(f"Accuracy: {accuracy:.2f}%\n")
    file.write(f"False positives: {len(false_positives)}\n")
    file.write(f"False negatives: {len(false_negatives)}\n\n")

    file.write("CONFUSION MATRIX\n")
    file.write("Rows = Actual [Non-scream, Scream]\n")
    file.write("Columns = Predicted [Non-scream, Scream]\n")
    file.write(str(matrix) + "\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(report + "\n")

    file.write("FALSE POSITIVES\n")
    file.write("-" * 50 + "\n")

    if false_positives:
        for name, score in false_positives:
            file.write(f"{name} | Score: {score:.3f}\n")
    else:
        file.write("No false positives.\n")

    file.write("\nFALSE NEGATIVES\n")
    file.write("-" * 50 + "\n")

    if false_negatives:
        for name, score in false_negatives:
            file.write(f"{name} | Score: {score:.3f}\n")
    else:
        file.write("No false negatives.\n")

    file.write("\nREVIEW FOLDER RECORDINGS\n")
    file.write("-" * 50 + "\n")
    file.write(
        "These recordings are excluded from evaluation "
        "because they are stored outside the test class folders.\n"
    )

    if review_files:
        for name in review_files:
            file.write(name + "\n")
    else:
        file.write("No review recordings found.\n")


# Step 12: Display results
print("\nEvaluation Results:")
print("Accuracy:", round(accuracy, 2), "%")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(report)

print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))

print("\nIndividual Predictions:")

for filename, actual, predicted, score in zip(
    filenames, y_test, predictions, scores
):

    actual_name = "SCREAM" if actual == 1 else "NON-SCREAM"
    predicted_name = "SCREAM" if predicted == 1 else "NON-SCREAM"

    status = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"{filename} | Actual: {actual_name} | "
        f"Predicted: {predicted_name} | "
        f"Score: {score:.3f} | {status}"
    )


print("\nEvaluation completed!")
print("\nSaved files:")
print(csv_path)
print(report_path)
