# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Load feature data
# =========================
BASE_PATH = (
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/"
    "Introduction to Brain Computer Interfaces/Project Code/"
    "Extracted_Features"
)

N_SUBJECTS = 6

# -------------------------------------------------
# Load features and labels
# -------------------------------------------------

X_list = []
y_list = []

for i in range(1, N_SUBJECTS + 1):
    # Load features
    data = np.load(os.path.join(BASE_PATH, f"Extracted_features_S{i}.npz"))
    X_i = data["features"]

    # Load labels
    y_i = np.load(os.path.join(BASE_PATH, f"Labels_S{i}.npy"))

    # Safety checks
    assert X_i.shape[0] == y_i.shape[0], f"Mismatch in subject {i}"

    X_list.append(X_i)
    y_list.append(y_i)

# Concatenate across subjects (trials)
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

# =========================
# Cross-validation strategy
# =========================
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# =========================
# Pipeline (NO data leakage)
# =========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ))
])


# =========================
# Cross-validated accuracy
# =========================
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

for i, acc in enumerate(scores, start=1):
    print(f"Accuracy for Fold {i}: {acc:.2f}")

print(f"\nMean Cross-Validation Accuracy: {scores.mean():.4f}")


# =========================
# Cross-validated predictions
# =========================
y_pred = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

accuracy = accuracy_score(y, y_pred)
print(f"\nCross-Validated Accuracy: {accuracy:.4f}")


# =========================
# Classification report
# =========================
print("\nClassification Report:")
print(classification_report(y, y_pred))


# =========================
# Confusion matrix
# =========================
conf_matrix = confusion_matrix(y, y_pred)
class_labels = np.unique(y)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()
