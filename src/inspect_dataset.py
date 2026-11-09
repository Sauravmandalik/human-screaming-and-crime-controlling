
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances

# Load training data and model
X = np.load("data/features.npy")
y = np.load("data/labels.npy")
model = joblib.load("models/scream_detector.pkl")

# Apply the same scaler used by the trained model
scaler = model.steps[0][1]
X_scaled = scaler.transform(X)

# Calculate distances between all training samples
distance_matrix = pairwise_distances(X_scaled)

# Ignore each sample's distance from itself
np.fill_diagonal(distance_matrix, np.inf)

# Find the nearest training sample for each recording
nearest_indices = np.argmin(distance_matrix, axis=1)
nearest_labels = y[nearest_indices]

# Analyse scream samples
scream_mask = y == 1
non_scream_mask = y == 0

scream_nearest_labels = nearest_labels[scream_mask]
scream_cross_count = np.sum(scream_nearest_labels == 0)

# Analyse non-scream samples
non_scream_nearest_labels = nearest_labels[non_scream_mask]
non_scream_cross_count = np.sum(non_scream_nearest_labels == 1)

print("\n========== DATASET OVERLAP REPORT ==========")

print("Total scream samples:", np.sum(scream_mask))
print("Scream samples nearest to non-scream:",
      scream_cross_count)

print("Total non-scream samples:", np.sum(non_scream_mask))
print("Non-scream samples nearest to scream:",
      non_scream_cross_count)

print("\nScream samples with non-scream nearest neighbour:",
      round(scream_cross_count / np.sum(scream_mask) * 100, 2), "%")

print("Non-scream samples with scream nearest neighbour:",
      round(non_scream_cross_count / np.sum(non_scream_mask) * 100, 2), "%")

print("\n============================================")
