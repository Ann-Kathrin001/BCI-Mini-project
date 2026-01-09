import numpy as np
import mne
from scipy.io import loadmat
from mne_icalabel import label_components
import glob
from braindecode.models import EEGNetv4
import torch
from sklearn.model_selection import train_test_split

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
    epochs.filter(1, 70, method='iir')

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
    ica = mne.preprocessing.ICA(
        n_components=0.9,
        random_state=42,
        method='infomax',
        fit_params=dict(extended=True)
    )
    ica.fit(epochs)

    
    ic_labels = label_components(epochs, ica, method="iclabel")

    ica.exclude = [
        i for i, lab in enumerate(ic_labels["labels"])
        if lab not in ("brain", "other")
    ]

    clean_epochs = ica.apply(epochs.copy())

    # ----- Extract final numpy data -----
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


X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=0.2, stratify=y_all, random_state=42 ) 
X_train = torch.tensor(X_train).float()
X_test  = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test  = torch.tensor(y_test).long()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
batch_size = 32 
epochs = 50 

for epoch in range(epochs): 
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
        print(f"Epoch {epoch+1:03d} | Loss {loss_sum:.3f} | Train Acc {train_acc:.2f}%")

model.eval() 
with torch.no_grad(): 
    preds = model(X_test) 
    _, predicted = preds.max(1) 
    acc = (predicted == y_test).float().mean() 
print("Test accuracy:", acc.item() * 100, "%")