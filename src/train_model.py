
import os
import numpy as np
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load features, labels, and original filenames
X = np.load("data/features_v2.npy")
y = np.load("data/labels_v2.npy")
filenames = np.load("data/filenames_v2.npy")

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Create groups to prevent LibriSpeech speaker leakage
groups = []

for filename in filenames:

    filename = str(filename)
    basename = os.path.basename(filename)

    if basename.startswith("librispeech_"):
        speaker_id = basename.split("_")[1]
        groups.append("librispeech_speaker_" + speaker_id)
    else:
        # Keep other recordings as individual groups
        groups.append(filename)

groups = np.array(groups)

print("Unique groups:", len(np.unique(groups)))

# Split data into 5 speaker-aware folds
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class counts:")
print("Non-scream:", np.sum(y_train == 0))
print("Scream:", np.sum(y_train == 1))

print("\nTesting class counts:")
print("Non-scream:", np.sum(y_test == 0))
print("Scream:", np.sum(y_test == 1))

print("\nGroups appearing in both sets:",
      len(train_groups.intersection(test_groups)))

# Create SVM model
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    )
)

# Train model
print("\nTraining SVM model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

# Find false positives
test_filenames = filenames[test_indices]

false_positive_indices = np.where(
    (y_test == 0) & (y_pred == 1)
)[0]

print("\nFALSE POSITIVE FILES:")

for index in false_positive_indices:
    print(test_filenames[index])

# Find missed screams
false_negative_indices = np.where(
    (y_test == 1) & (y_pred == 0)
)[0]

print("\nFALSE NEGATIVE FILES:")

for index in false_negative_indices:
    print(test_filenames[index])

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Non-scream", "Scream"],
    zero_division=0
))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/scream_detector.pkl")

print("\nModel saved successfully!")
