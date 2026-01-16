from scipy.io import loadmat
import numpy as np
import mne
from mne_icalabel import label_components
from mne.decoding import CSP
import os
from mne.filter import filter_data
from scipy.signal import hilbert

# ---------------- Riemann ----------------
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ============================================================
# Log-bandpower (ERD)
# ============================================================
def log_bandpower(X):
    # X: (n_trials, n_channels, n_times)
    power = np.mean(X**2, axis=2)
    return np.log(power + 1e-10)


# ============================================================
# Phase Locking Value (PLV)
# ============================================================
def compute_plv(X):
    """
    X: (n_trials, n_channels, n_times)
    Returns: (n_trials, n_pairs)
    """
    n_trials, n_channels, _ = X.shape
    n_pairs = n_channels * (n_channels - 1) // 2

    plv_features = np.zeros((n_trials, n_pairs))

    for tr in range(n_trials):
        analytic = hilbert(X[tr], axis=1)
        phase = np.angle(analytic)

        k = 0
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_diff = phase[i] - phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_features[tr, k] = plv
                k += 1

    return plv_features


# ============================================================
# Files
# ============================================================
base_path = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/MI_Pipeline/MI_BCI_Data"

files = [
    base_path + "/PAT013.mat",
    base_path + "/PAT015.mat",
    base_path + "/PAT021_A.mat",
    base_path + "/PATID15.mat",
    base_path + "/PATID16.mat",
    base_path + "/PATID26.mat"
]

save_dir = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/Extracted_Features_CSP_RIEMANN_BP_PLV"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# Channels
# ============================================================
channels = [
    "F3", "Fz", "F4",
    "FC3", "FC1", "FC2", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CP1", "CP2", "CP4",
    "P3", "P4"
]

mi_channels = ["C3", "Cz", "C4", "CP3", "CP4"]
mi_idx = [channels.index(ch) for ch in mi_channels]


# ============================================================
# Loop over subjects
# ============================================================
subject_idx = 0

for location in files:

    subject_idx += 1
    print("\nProcessing:", location)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    data = loadmat(location, squeeze_me=True, struct_as_record=False)
    s = data["subjectData"]

    fs = int(np.squeeze(s.fs))
    trials = s.trialsData
    y = np.squeeze(s.trialsLabels).astype(int)

    X = np.stack([t.T for t in trials], axis=0)

    # --------------------------------------------------------
    # MNE Epochs
    # --------------------------------------------------------
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(X, info)

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, match_case=False)

    # --------------------------------------------------------
    # Notch + broadband filter
    # --------------------------------------------------------
    epochs._data = mne.filter.notch_filter(
        epochs.get_data(), Fs=fs, freqs=50, method="iir"
    )
    epochs.filter(1, 70, method="fir")

    # --------------------------------------------------------
    # Bad channel detection
    # --------------------------------------------------------
    data_eeg = epochs.get_data()
    var = np.var(data_eeg, axis=(0, 2))
    z = (var - var.mean()) / var.std()
    bads = [epochs.ch_names[i] for i in np.where(np.abs(z) > 5)[0]]

    epochs.info["bads"] = bads
    epochs.interpolate_bads(reset_bads=True)
    epochs.set_eeg_reference("average")

    # --------------------------------------------------------
    # ICA
    # --------------------------------------------------------
    ica = mne.preprocessing.ICA(
        n_components=0.9,
        random_state=42,
        method="infomax",
        fit_params=dict(extended=True),
    )
    ica.fit(epochs)

    labels = label_components(epochs, ica, method="iclabel")

    ica.exclude = [
        i for i, lab in enumerate(labels["labels"])
        if lab not in ["brain", "other"]
    ]

    epochs = ica.apply(epochs.copy())

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------
    X = epochs.get_data()
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)

    # --------------------------------------------------------
    # Keep MI channels
    # --------------------------------------------------------
    #X = X[:, mi_idx, :]

    # --------------------------------------------------------
    # Filter Bank
    # --------------------------------------------------------
    bands = [(8,12),(12,16),(16,20),(20,24),(24,30),(30,35)]

    X_filt = []
    X_bp = []
    X_plv = []

    for low, high in bands:
        Xb = filter_data(X, fs, low, high, verbose=False)
        X_filt.append(Xb)

        X_bp.append(log_bandpower(Xb))
        X_plv.append(compute_plv(Xb))

    # --------------------------------------------------------
    # CSP
    # --------------------------------------------------------
    X_csp = []
    for Xb in X_filt:
        csp = CSP(n_components=4, reg=0.1, log=True, norm_trace=True)
        X_csp.append(csp.fit_transform(Xb, y))

    features_csp = np.concatenate(X_csp, axis=1)

    # --------------------------------------------------------
    # Riemann
    # --------------------------------------------------------
    cov = Covariances("oas")
    ts = TangentSpace()

    X_r = []
    for Xb in X_filt:
        C = cov.fit_transform(Xb)
        X_r.append(ts.fit_transform(C))

    features_riemann = np.concatenate(X_r, axis=1)

    # --------------------------------------------------------
    # Bandpower
    # --------------------------------------------------------
    features_bp = np.concatenate(X_bp, axis=1)

    # --------------------------------------------------------
    # PLV
    # --------------------------------------------------------
    features_plv = np.concatenate(X_plv, axis=1)

    # --------------------------------------------------------
    # Hybrid feature vector
    # --------------------------------------------------------
    features = np.concatenate(
        [features_csp, features_riemann, features_bp, features_plv],
        axis=1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    np.savez_compressed(os.path.join(save_dir, f"Features_S{subject_idx}.npz"), features=features)
    np.save(os.path.join(save_dir, f"Labels_S{subject_idx}.npy"), y)

    print("Saved subject", subject_idx)
