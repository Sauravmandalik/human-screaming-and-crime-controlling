
import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load extracted features and labels
X = np.load("data/features.npy")
y = np.load("data/labels.npy")

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create SVM model
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    )
)

# Train the model
print("\nTraining SVM model...")
model.fit(X_train, y_train)

# Predict test results
y_pred = model.predict(X_test)

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

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/scream_detector.pkl")

print("\nModel saved successfully!")
