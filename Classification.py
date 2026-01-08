# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Load feature data
# =========================
data_completed = []

for i in range(1, 36):
    try:
        with np.load(
            f'/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/Extracted_Features/Extracted_features_S{i}.npz'
        ) as f:
            data_completed.append(dict(f))
    except FileNotFoundError:
        print(f"File Extracted_features_S{i}.npz not found")

# Stack data: (subjects, epochs, features)
data = np.stack([d['features'] for d in data_completed])
n_subjects, n_epochs, n_features = data.shape

# Reshape to 2D: (samples, features)
X = data.reshape(n_subjects * n_epochs, n_features)

print("Data shape:", X.shape)


# =========================
# Load labels
# =========================
labels_read = loadmat(
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/SSVEP-BCI-Data/Freq_Phase.mat"
)

freq = labels_read['freqs'].ravel()  # shape: (n_epochs,)
labels = np.tile(freq, n_subjects*6)   # repeat for all subjects
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print("Labels shape:", labels.shape)


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
    ('svm', SVC(class_weight='balanced'))
])


# =========================
# Hyperparameter grid
# =========================
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10],
    'svm__gamma': [0.001, 0.01, 0.1],
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}


# =========================
# Grid Search
# =========================
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X, labels)

print("\nBest Parameters from GridSearchCV:")
print(grid.best_params_)

print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_


# =========================
# Fold-wise accuracy
# =========================
scores = cross_val_score(
    best_model,
    X,
    labels,
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
    best_model,
    X,
    labels,
    cv=cv,
    n_jobs=-1
)

accuracy = accuracy_score(labels, y_pred)
print(f"\nCross-Validated Accuracy: {accuracy:.4f}")


# =========================
# Classification report
# =========================
print("\nClassification Report:")
print(classification_report(labels, y_pred))


# =========================
# Confusion matrix
# =========================
conf_matrix = confusion_matrix(labels, y_pred)
class_labels = np.unique(labels)

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
