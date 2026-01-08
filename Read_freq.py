from scipy.io import loadmat
import numpy as np

# Load the .mat file for one subject
data = loadmat("/Users/ankadilfer/Desktop/Master DTU/Semester 2/Introduction to Brain Computer Interfaces/Project Code/SSVEP-BCI-Data/Freq_Phase.mat")

print(data.keys())

phase=data['phases']  
freq=data['freqs']

print("Phase shape:", phase.shape)  
print("Frequency shape:", freq.shape)



# Flatten in case they are (40,1)
freq = freq.squeeze()
phase = phase.squeeze()

target_dict = {
    i: {
        "frequency": float(freq[i]),
        "phase": float(phase[i])
    }
    for i in range(len(freq))
}

print(target_dict[3])