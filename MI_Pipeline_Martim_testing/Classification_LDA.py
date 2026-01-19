# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Load feature data
# =========================
BASE_PATH = (
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/"
    "Introduction to Brain Computer Interfaces/Project Code/"
    "Extracted_Features_CSP_RIEMANN"
)

N_SUBJECTS = 6

X_list = []
y_list = []
subject_ids = []

for i in range(1, N_SUBJECTS + 1):
    print(f"Loading Subject {i}")

    data = np.load(os.path.join(BASE_PATH, f"Features_S{i}.npz"))
    X_i = data["features"]
    y_i = np.load(os.path.join(BASE_PATH, f"Labels_S{i}.npy"))

    assert X_i.shape[0] == y_i.shape[0], f"Mismatch in subject {i}"

    X_list.append(X_i)
    y_list.append(y_i)
    subject_ids.extend([i] * len(y_i))

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
subject_ids = np.array(subject_ids)

print("\nTotal trials:", X.shape[0])
print("Feature dimension:", X.shape[1])


# =========================
# Leave-One-Subject-Out CV
# =========================
unique_subjects = np.unique(subject_ids)

all_preds = []
all_true = []
loso_scores = []

for test_subj in unique_subjects:
    print(f"\n==============================")
    print(f"Leaving out Subject {test_subj}")

    train_mask = subject_ids != test_subj
    test_mask  = subject_ids == test_subj

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver="svd"))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Test
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Subject {test_subj} Accuracy: {acc:.4f}")

    loso_scores.append(acc)
    all_preds.append(y_pred)
    all_true.append(y_test)


# =========================
# Aggregate results
# =========================
all_preds = np.concatenate(all_preds)
all_true  = np.concatenate(all_true)
loso_scores = np.array(loso_scores)

print("\n==============================")
print("LOSO Subject Accuracies:")
for s, a in zip(unique_subjects, loso_scores):
    print(f"Subject {s}: {a:.4f}")

print("\nMean LOSO Accuracy:", loso_scores.mean())
print("Std  LOSO Accuracy:", loso_scores.std())


# =========================
# Classification report
# =========================
print("\nClassification Report (LOSO):")
print(classification_report(all_true, all_preds))


# =========================
# Confusion Matrix
# =========================
conf_matrix = confusion_matrix(all_true, all_preds)
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
plt.title('LOSO Confusion Matrix')
plt.tight_layout()
plt.show()


#LOSO Subject Accuracies (DWT+LDA):
#Subject 1: 0.5162
#Subject 2: 0.5684
#Subject 3: 0.5129
#Subject 4: 0.7260
#Subject 5: 0.5046
#Subject 6: 0.5403

#Mean LOSO Accuracy: 0.5614087040524348
#Std  LOSO Accuracy: 0.07660567378970129



#LOSO Subject Accuracies (CSP+LDA):
#Subject 1: 0.5162
#Subject 2: 0.4938
#Subject 3: 0.5024
#Subject 4: 0.7082
#Subject 5: 0.4885
#Subject 6: 0.5014

#Mean LOSO Accuracy: 0.5350689113962258
#Std  LOSO Accuracy: 0.07788277766636406