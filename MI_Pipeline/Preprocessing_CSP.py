from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_icalabel import label_components
from mne.decoding import CSP
import os
from mne.filter import filter_data

# ------------------------------------------------------------
# File list
# ------------------------------------------------------------

base_path = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/MI_Pipeline/MI_BCI_Data"

files = [
    base_path + "/PAT013.mat",
    base_path + "/PAT015.mat",
    base_path + "/PAT021_A.mat",
    base_path + "/PATID15.mat",
    base_path + "/PATID16.mat",
    base_path + "/PATID26.mat"
]


# ------------------------------------------------------------
# Save directory
# ------------------------------------------------------------

save_dir = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/Extracted_Features_CSP"
os.makedirs(save_dir, exist_ok=True)
i=0

# ------------------------------------------------------------
# Channel list
# ------------------------------------------------------------

channels = [
    "F3", "Fz", "F4",
    "FC3", "FC1", "FC2", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CP1", "CP2", "CP4",
    "P3", "P4"
]



# ------------------------------------------------------------
# Loop over subjects
# ------------------------------------------------------------

for location in files:

    print("\nProcessing:", location)

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    data = loadmat(location, squeeze_me=True, struct_as_record=False)
    s = data["subjectData"]

    subject_id = str(np.squeeze(s.subjectId))
    fs = int(np.squeeze(s.fs))
    init_delay = int(np.squeeze(s.INIT_DELAY))
    mi_duration = int(np.squeeze(s.MI_DURATION))
    trials = s.trialsData
    label = s.trialsLabels

    X = np.stack([t.T for t in trials], axis=0)
    y = np.squeeze(label).astype(int)

    # --------------------------------------------------------
    # Create Epochs
    # --------------------------------------------------------

    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(X, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False)

    # --------------------------------------------------------
    # Notch + Bandpass
    # --------------------------------------------------------

    epochs._data = mne.filter.notch_filter(
        epochs.get_data(),
        Fs=epochs.info["sfreq"],
        freqs=50,
        method="iir"
    )

    epochs.filter(l_freq=1, h_freq=70, method='fir')

    # --------------------------------------------------------
    # Remove Noisy Channels + Rereference
    # --------------------------------------------------------

    data_eeg = epochs.get_data(picks='eeg')
    variances = np.var(data_eeg, axis=(0, 2))
    z = (variances - np.mean(variances)) / np.std(variances)
    bad_idx = np.where(np.abs(z) > 5)[0]
    bad_chs = [epochs.ch_names[i] for i in bad_idx]

    epochs.info['bads'].extend(bad_chs)
    epochs.interpolate_bads(reset_bads=True)
    epochs.set_eeg_reference('average')
    


    # --------------------------------------------------------
    # ICA
    # --------------------------------------------------------

    ica = mne.preprocessing.ICA(
        n_components=0.9,
        random_state=42,
        method='infomax',
        fit_params=dict(extended=True),
        verbose=True
    )

    ica.fit(epochs)

    ica_labels = label_components(epochs, ica, method="iclabel")
   

    ica.exclude = [
        i for i, label in enumerate(ica_labels['labels'])
        if label != 'brain' and label != 'other'
    ]

    cleaned_epochs = ica.apply(epochs.copy())


    # --------------------------------------------------------
    # Z-score Normalization
    # --------------------------------------------------------
    
    X = cleaned_epochs.get_data()   # (n_epochs, n_channels, n_times)

    # compute on THIS subject (since you're doing subject-specific CSP)
    mu  = X.mean(axis=(0, 2), keepdims=True)    # (1, C, 1)
    std = X.std(axis=(0, 2), keepdims=True)

    X = (X - mu) / std
    X_clean = X


    

    # --------------------------------------------------------
    # Feature Extraction (FBCSP)
    # --------------------------------------------------------

    # ---------------- Filter bank ----------------
    bands = [(8,12),(12,16),(16,20),(20,24),(24,30),(30,35)]
    X_filt = []

    for low, high in bands:
        X_band = filter_data(X_clean, fs, low, high, method='fir', verbose=False)
        X_filt.append(X_band)

    # ---------------- CSP per band ----------------
    X_csp = []
    csp_models = []

    for X_band in X_filt:
        csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=True)
        X_band_csp = csp.fit(X_band, y).transform(X_band)

        X_csp.append(X_band_csp)
        csp_models.append(csp)

    # ---------------- Final FBCSP features ----------------
    features = np.concatenate(X_csp, axis=1)


    # --------------------------------------------------------
    # Save per subject
    # --------------------------------------------------------
    i=i+1
    feature_file = os.path.join(save_dir, f"Extracted_features_S{i}_CSP.npz")
    label_file = os.path.join(save_dir, f"Labels_S{i}_CSP.npy")

    np.savez_compressed(feature_file, features=features)
    np.save(label_file, y)

    print("Saved:", feature_file, label_file)
