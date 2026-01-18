# ============================================================
# Imports
# ============================================================
import os
import numpy as np
from scipy.io import loadmat
import mne
from mne_icalabel import label_components
import pywt
from skimage.transform import resize


# ============================================================
# Paths
# ============================================================
base_path = (
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/"
    "Introduction to Brain Computer Interfaces/Project Code/"
    "MI_Pipeline/MI_BCI_Data"
)

files = [
    base_path + "/PAT013.mat",
    base_path + "/PAT015.mat",
    base_path + "/PAT021_A.mat",
    base_path + "/PATID15.mat",
    base_path + "/PATID16.mat",
    base_path + "/PATID26.mat"
]

save_dir = (
    "/Users/ankadilfer/Desktop/Master DTU/Semester 2/"
    "Introduction to Brain Computer Interfaces/Project Code/"
    "Extracted_DWT_Scalograms"
)
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
# DWT / CWT scalogram function
# ============================================================
def compute_dwt_scalogram(
    signal,
    fs,
    freqs=np.linspace(8, 30, 32),
    wavelet="morl",
    out_size=(64, 128)
):
    """
    signal: (n_times,)
    returns: (freqs, times) scalogram
    """
    scales = fs / freqs
    coef, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    power = np.abs(coef) ** 2
    power = np.log(power + 1e-10)

    power = resize(
        power,
        out_size,
        mode="reflect",
        anti_aliasing=True
    )

    return power


# ============================================================
# Loop over subjects
# ============================================================
subject_idx = 0

for location in files:

    subject_idx += 1
    print("\nProcessing:", location)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    data = loadmat(location, squeeze_me=True, struct_as_record=False)
    s = data["subjectData"]

    fs = int(np.squeeze(s.fs))
    trials = s.trialsData
    y = np.squeeze(s.trialsLabels).astype(int)

    X = np.stack([t.T for t in trials], axis=0)
    print("Trials shape:", X.shape)

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
        epochs.get_data(),
        Fs=fs,
        freqs=50,
        method="iir"
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
        fit_params=dict(extended=True)
    )
    ica.fit(epochs)

    labels = label_components(epochs, ica, method="iclabel")

    ica.exclude = [
        i for i, lab in enumerate(labels["labels"])
        if lab not in ["brain", "other"]
    ]

    epochs = ica.apply(epochs.copy())

    # --------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------
    X = epochs.get_data()
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / \
        X.std(axis=(0, 2), keepdims=True)

    # --------------------------------------------------------
    # Keep MI channels only
    # --------------------------------------------------------
    X_mi = X[:, mi_idx, :]
    print("MI data shape:", X_mi.shape)

    # --------------------------------------------------------
    # Compute DWT scalograms
    # --------------------------------------------------------
    scalograms = []

    for trial in range(X_mi.shape[0]):
        trial_scales = []

        for ch in range(X_mi.shape[1]):
            scalo = compute_dwt_scalogram(
                X_mi[trial, ch],
                fs=fs,
                freqs=np.linspace(8, 30, 32),
                out_size=(64, 128)
            )
            trial_scales.append(scalo)

        trial_scales = np.stack(trial_scales, axis=0)
        scalograms.append(trial_scales)

    scalograms = np.stack(scalograms, axis=0)

    print("Scalograms shape:", scalograms.shape)
    # (n_trials, 5, 64, 128)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    np.save(
        os.path.join(save_dir, f"DWT_Scalograms_S{subject_idx}.npy"),
        scalograms
    )
    np.save(
        os.path.join(save_dir, f"Labels_S{subject_idx}.npy"),
        y
    )

    print(f"Saved Subject {subject_idx}")
