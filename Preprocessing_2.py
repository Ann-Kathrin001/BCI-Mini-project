#Packages
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_icalabel import label_components
import pywt
import os

#--------------------------------------------------------------------------------------------------------------------

#Constants
fs=250 # Sampling frequency
channel_names= [
  "FP1", "FPZ", "FP2", "AF3", "AF4",
  "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
  "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
  "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
  "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
  "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
  "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
  "CB1", "O1", "Oz", "O2", "CB2"
]


#--------------------------------------------------------------------------------------------------------------------
#Load Data and Create Epochs

## Load the .mat file for one subject
data = loadmat("/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/SSVEP-BCI-Data/S35.mat")

## Output the shape of the EEG data
eeg_data = data['data']  

## Reshape data 
X = np.transpose(eeg_data, (3, 2, 0, 1))   # (block, target, ch, time)
X = X.reshape(-1, 64, 1500)
n_blocks = 6
n_targets = 40
y = np.tile(np.arange(1, n_targets + 1), n_blocks)

##Create MNE Epochs
events = np.zeros((240, 3), dtype=int)
events[:, 0] = np.arange(240)          # fake sample index
events[:, 2] = y                       # event id
event_id = {f"target_{i}": i for i in range(1, 41)}
info = mne.create_info(
    ch_names=channel_names,   # length = 64
    sfreq=250,                # Hz (downsampled)
    ch_types="eeg"
)
epochs = mne.EpochsArray(
    X,
    info,
    events=events,
    event_id=event_id,
    tmin=-0.5        # 500 ms pre-stimulus
)



#--------------------------------------------------------------------------------------------------------------------
#Set Montage and Channel Types

montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_channel_types({
    'CB1': 'misc',
    'CB2': 'misc'
})
epochs.set_montage(montage, match_case=False)


#--------------------------------------------------------------------------------------------------------------------
#Bandpass Filter between 1-50 Hz

epochs.filter(
    l_freq=1,
    h_freq=50,
    method='iir'
)

#--------------------------------------------------------------------------------------------------------------------
#Remove Noisy Channels and Re-reference

## Remove noisy channels
data = epochs.get_data(picks='eeg')  # shape: (n_epochs, n_ch, n_times)
variances = np.var(data, axis=(0, 2))
z = (variances - np.mean(variances)) / np.std(variances)
bad_idx = np.where(np.abs(z) > 5)[0]
bad_chs=[epochs.ch_names[i] for i in bad_idx]

#Interpolate bad channels
epochs.info['bads'].extend(bad_chs)
epochs.interpolate_bads(reset_bads=True)

#Re-reference to average
epochs.set_eeg_reference('average', projection=False)


#--------------------------------------------------------------------------------------------------------------------
#ICA for Artifact Removal

ica = mne.preprocessing.ICA(
    n_components=0.9,
    random_state=42,
    method='infomax',
    fit_params=dict(extended=True),
    verbose=True
)
ica.fit(epochs)  

## Label components using ICLabel
ica_labels = label_components(epochs, ica, method="iclabel")
ica.exclude =[
    i for i, label in enumerate(ica_labels['labels'])
    if label != 'brain'
]

# Apply ICA to epochs
cleaned_epochs = ica.apply(epochs.copy())


#--------------------------------------------------------------------------------------------------------------------
#Feature Extraction using DWT


cleaned_epochs=epochs.get_data(picks='eeg')

def dwt_band_energy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=7)
    return [np.sum(c**2) for c in coeffs]

features = []
for epoch in cleaned_epochs:          # (n_channels, n_times)
    epoch_features = []
    for channel in epoch:             # (n_times,)
        epoch_features.append(dwt_band_energy(channel))
    features.append(np.concatenate(epoch_features))
features = np.array(features)


#--------------------------------------------------------------------------------------------------------------------
# Save Features per Subject

save_dir = "/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/Extracted_Features"
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, "Extracted_features_S35.npz")

np.savez_compressed(
    file_path,
    features=features
)