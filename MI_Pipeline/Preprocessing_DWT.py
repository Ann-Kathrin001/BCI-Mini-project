#Packages
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_icalabel import label_components
import pywt
import os

#--------------------------------------------------------------------------------------------------------------------
#Load Data and Create Epochs
# Load the .mat file for one subject

location="/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/MI_Pipeline/MI_BCI_Data/PAT013.mat"
data = loadmat(location, squeeze_me=True, struct_as_record=False)
s = data["subjectData"]   # or whatever the variable is called

subject_id = str(np.squeeze(s.subjectId))
fs = int(np.squeeze(s.fs))
init_delay = int(np.squeeze(s.INIT_DELAY))
mi_duration = int(np.squeeze(s.MI_DURATION))#
trials = s.trialsData
label = s.trialsLabels


# trials: length 434, each (1536, 16)
X = np.stack([t.T for t in trials], axis=0)   # (trials, channels, samples)
trials_num, channels, samples = X.shape
y = np.squeeze(label).astype(int)



events = np.zeros((len(y), 3), dtype=int)
events[:, 0] = np.arange(len(y))       # arbitrary sample index
events[:, 2] = y                       # class labels

event_id = {
    "left_hand": 0,
    "right_hand": 1
}


channels = [
    "F3", "Fz", "F4",
    "FC3", "FC1", "FC2", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CP1", "CP2", "CP4",
    "P3", "P4"
]


info = mne.create_info(
    ch_names=channels,   
    sfreq=fs,               
    ch_types="eeg"
)


epochs = mne.EpochsArray(
    X,
    info,
    events=events,
    event_id=event_id,
    tmin=0
)




#--------------------------------------------------------------------------------------------------------------------
#Set Montage and Channel Types

montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage, match_case=False)


#--------------------------------------------------------------------------------------------------------------------
#Notch Filter and Bandpass Filter between 1-70 Hz

epochs._data = mne.filter.notch_filter(
    epochs.get_data(),
    Fs=epochs.info["sfreq"],
    freqs=50,          
    method="iir"
)


epochs.filter(
    l_freq=1,
    h_freq=70,
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
print(ica_labels)
ica.exclude =[
    i for i, label in enumerate(ica_labels['labels'])
    if label != 'brain' and label!= 'other'
]

# Apply ICA to epochs
cleaned_epochs = ica.apply(epochs.copy())




#--------------------------------------------------------------------------------------------------------------------
#Feature Extraction using DWT


cleaned_epochs=epochs.get_data(picks='eeg')

def dwt_band_energy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=5)
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

file_path = os.path.join(save_dir, "Extracted_features_S1.npz")

np.savez_compressed(
    file_path,
    features=features
)

np.save(save_dir + f"/Labels_S1.npy", y)