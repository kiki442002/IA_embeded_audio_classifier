import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cmsisdsp as dsp



# Définir une classe Dataset personnalisée pour les données audio
class AudioDataset(Dataset):
    def __init__(self, csv_file, labels, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.label_mapping = labels
        self.window = librosa.filters.get_window('hann', 1024, fftbins=True)
        self.mel_filters = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=30,norm=1.0)

    def __len__(self):
        return len(self.data_frame)
    
    

    
    def __getitem__(self, idx):
        audio_path = "segmented_selected_data/"+self.data_frame.iloc[idx, 0]
        start_time = self.data_frame.iloc[idx, 1]
        end_time = self.data_frame.iloc[idx, 2]
        label = self.data_frame.iloc[idx, 3]

        # Charger et prétraiter l'audio
        #features = self.extraire_features(audio_path, start_time, end_time)
        features = self.extraire_features_old(audio_path, start_time, end_time)
        if self.transform:
            features = self.transform(features)

        # Ajouter une dimension pour le canal (1, car c'est un spectrogramme mono)
        features = np.expand_dims(features, axis=0)
        # Convertir le label en one-hot encodé
        label_one_hot = F.one_hot(torch.tensor(self.label_mapping[label]), num_classes=len(self.label_mapping)).float()
        return torch.tensor(features, dtype=torch.float32), label_one_hot






