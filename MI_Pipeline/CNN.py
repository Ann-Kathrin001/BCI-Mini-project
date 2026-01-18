import numpy as np
import mne
from scipy.io import loadmat
from mne_icalabel import label_components
import glob
from braindecode.models import EEGNetv4
import torch
from sklearn.model_selection import train_test_split

def zscore_per_channel(X_train, X_test, eps=1e-6):
    """
    X_train, X_test: (n_trials, n_channels, n_times)
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2), keepdims=True)

    X_train = (X_train - mean) / (std + eps)
    X_test  = (X_test  - mean) / (std + eps)

    return X_train, X_test


def preprocess_subject(mat_path):

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    s = data["subjectData"]

    fs = int(np.squeeze(s.fs))
    trials = s.trialsData
    labels = np.squeeze(s.trialsLabels).astype(int)

    # (n_trials, n_channels, n_samples)
    X = np.stack([t.T for t in trials], axis=0)
    y = labels.copy()



    info = mne.create_info(
        ch_names=[
            "F3","Fz","F4","FC3","FC1","FC2","FC4",
            "C3","Cz","C4","CP3","CP1","CP2","CP4","P3","P4"
        ],
        sfreq=fs,
        ch_types="eeg"
    )

    epochs = mne.EpochsArray(X, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False)

    # ----- Filtering -----
    epochs._data = mne.filter.notch_filter(
        epochs.get_data(),
        Fs=fs,
        freqs=50,
        method="iir"
    )
    epochs.filter(1, 70, method='fir')

    # ----- Bad channel detection -----
    data = epochs.get_data()
    variances = np.var(data, axis=(0, 2))
    z = (variances - np.mean(variances)) / np.std(variances)
    bad_idx = np.where(np.abs(z) > 5)[0]
    bads = [epochs.ch_names[i] for i in bad_idx]

    epochs.info['bads'] = bads
    epochs.interpolate_bads(reset_bads=True)
    epochs.set_eeg_reference('average')

    # ----- ICA -----
    #ica = mne.preprocessing.ICA(
    #    n_components=0.9,
    #    random_state=42,
    #    method='infomax',
    #    fit_params=dict(extended=True)
    #)
    #ica.fit(epochs)

    
    #ic_labels = label_components(epochs, ica, method="iclabel")

    #ica.exclude = [
    #    i for i, lab in enumerate(ic_labels["labels"])
    #    if lab not in ("brain", "other")
    #]

    #clean_epochs = ica.apply(epochs.copy())

    # ----- Extract final numpy data -----
    clean_epochs = epochs
    X_clean = clean_epochs.get_data()
    y_clean = y.copy()

    return X_clean, y_clean






base_path = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/MI_Pipeline/MI_BCI_Data"

files = [
    base_path + "/PAT013.mat",
    base_path + "/PAT015.mat",
    base_path + "/PAT021_A.mat",
    base_path + "/PATID15.mat",
    base_path + "/PATID16.mat",
    base_path + "/PATID26.mat"
]

X_all = []
y_all = []
subject_ids = []

for subj_idx, f in enumerate(files):
    print("Processing", f)

    X_subj, y_subj = preprocess_subject(f)

    X_all.append(X_subj)
    y_all.append(y_subj)

    subject_ids.extend([subj_idx] * len(y_subj))

# concatenate over subjects
X_all = np.concatenate(X_all, axis=0) # shape (n_total_trials, n_channels, n_samples)
y_all = np.concatenate(y_all, axis=0)
subject_ids = np.array(subject_ids)


n_total_trials, n_channels, n_samples = X_all.shape


model = EEGNetv4(
    n_chans=n_channels,
    n_outputs=2,
    n_times=n_samples,
    final_conv_length='auto'
)


unique_subjects = np.unique(subject_ids)
loso_accuracies = []

for test_subj in unique_subjects:
    print(f"\n==== Leaving out subject {test_subj} ====")

    # Split by subject
    train_mask = subject_ids != test_subj
    test_mask  = subject_ids == test_subj

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test  = X_all[test_mask]
    y_test  = y_all[test_mask]

    # Convert to torch
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()
    X_test  = torch.tensor(X_test).float()
    y_test  = torch.tensor(y_test).long()

    X_train, X_test = zscore_per_channel(X_train, X_test)

    # NEW MODEL FOR EACH FOLD
    model = EEGNetv4(
        n_chans=n_channels,
        n_outputs=2,
        n_times=n_samples,
        final_conv_length='auto'
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32
    n_epochs = 50

    # ---------- Training ----------
    for epoch in range(n_epochs):
        perm = torch.randperm(X_train.size(0))
        correct = 0
        total = 0
        loss_sum = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = preds.max(1)
            total += yb.size(0)
            correct += predicted.eq(yb).sum().item()

        train_acc = 100 * correct / total

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train acc {train_acc:.1f}%")

    # ---------- Testing ----------
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        _, predicted = preds.max(1)
        acc = (predicted == y_test).float().mean().item()

    print(f"Subject {test_subj} test accuracy: {acc*100:.2f}%")
    loso_accuracies.append(acc)

loso_accuracies = np.array(loso_accuracies)

print("\n==============================")
print("LOSO results:")
for s, a in zip(unique_subjects, loso_accuracies):
    print(f"Subject {s}: {a*100:.2f}%")

print("\nMean LOSO accuracy:", loso_accuracies.mean()*100)
print("Std LOSO accuracy:", loso_accuracies.std()*100)



#LOSO results :
#Subject 0: 60.93%
#Subject 1: 65.01%
#Subject 2: 56.96%
#Subject 3: 74.23%
#Subject 4: 55.76%
#Subject 5: 55.46%

#Mean LOSO accuracy: 61.39292120933533
#Std LOSO accuracy: 6.6425485218018565

