from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_icalabel import label_components
import pywt
import os

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

save_dir = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/Extracted_Features"
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
# DWT function
# ------------------------------------------------------------

def dwt_band_energy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=5)
    return [np.sum(c**2) for c in coeffs]

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

    #--------------------------------------------------------
    # Crop to MI period
    #--------------------------------------------------------

    tmin = float(s.INIT_DELAY)
    tmax = float(s.INIT_DELAY + s.MI_DURATION)
    epochs.crop(tmin, tmax)



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
    epochs.set_eeg_reference('average', projection=False)

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
    print(ica_labels)

    ica.exclude = [
        i for i, label in enumerate(ica_labels['labels'])
        if label != 'brain' and label != 'other'
    ]

    cleaned_epochs = ica.apply(epochs.copy())

    # --------------------------------------------------------
    # Feature Extraction (DWT)
    # --------------------------------------------------------

    cleaned_epochs = cleaned_epochs.get_data(picks='eeg')

    features = []
    for epoch in cleaned_epochs:
        epoch_features = []
        for channel in epoch:
            epoch_features.append(dwt_band_energy(channel))
        features.append(np.concatenate(epoch_features))

    features = np.array(features)

    # --------------------------------------------------------
    # Save per subject
    # --------------------------------------------------------
    i=i+1
    feature_file = os.path.join(save_dir, f"Extracted_features_S{i}.npz")
    label_file = os.path.join(save_dir, f"Labels_S{i}.npy")

    np.savez_compressed(feature_file, features=features)
    np.save(label_file, y)

    print("Saved:", feature_file, label_file)
