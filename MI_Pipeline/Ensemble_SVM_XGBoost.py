# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# =========================
# Load feature data
# =========================
BASE_PATH = (
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/"
    "Introduction to Brain Computer Interfaces/Project Code/"
    "Extracted_Features_CSP_RIEMANN_BP_PLV"
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

    # ------------------------------------------------------
    # SVM branch (CSP + Riemann friendly)
    # ------------------------------------------------------
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])

    # ------------------------------------------------------
    # XGBoost branch (BP + PLV friendly)
    # ------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
        random_state=42,
        n_jobs=-1
    )


    # ------------------------------------------------------
    # Randorm Forest branch (BP + PLV friendly)
    # ------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )


    # ------------------------------------------------------
    # Soft-voting ensemble
    # ------------------------------------------------------
    ensemble = VotingClassifier(
    estimators=[
        ("svm", svm),
        ("xgb", xgb),
        ("rf", rf)
    ],
    voting="soft",
    weights=[1.5, 1.0, 1.2]  # SVM, XGB, RF
)


    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------
    ensemble.fit(X_train, y_train)

    # ------------------------------------------------------
    # Test
    # ------------------------------------------------------
    y_pred = ensemble.predict(X_test)
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
print("LOSO Subject Accuracies (SVM + XGB):")
for s, a in zip(unique_subjects, loso_scores):
    print(f"Subject {s}: {a:.4f}")

print("\nMean LOSO Accuracy:", loso_scores.mean())
print("Std  LOSO Accuracy:", loso_scores.std())


# =========================
# Classification report
# =========================
print("\nClassification Report (LOSO SVM + XGBoost):")
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
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LOSO Confusion Matrix (SVM + XGBoost)")
plt.tight_layout()
plt.show()



#LOSO Subject Accuracies (RF+ SVM + XGB):
#Subject 1: 0.6174
#Subject 2: 0.6474
#Subject 3: 0.6699
#Subject 4: 0.7270
#Subject 5: 0.6129
#Subject 6: 0.5526

#Mean LOSO Accuracy: 0.6378793335995877
#Std  LOSO Accuracy: 0.05382462995363811

#Classification Report (LOSO RF + SVM + XGBoost):
#              precision    recall  f1-score   support

#           0       0.64      0.68      0.66      3349
#           1       0.66      0.62      0.64      3365

#    accuracy                           0.65      6714
#   macro avg       0.65      0.65      0.65      6714
#weighted avg       0.65      0.65      0.65      6714

