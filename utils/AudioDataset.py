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



# Définir une classe Dataset personnalisée pour les données audio
class AudioDataset(Dataset):
    def __init__(self, csv_file, labels, selection_list = None, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.label_mapping = labels
        self.selection_list = selection_list
    def __len__(self):
        return len(self.selection_list) if self.selection_list != None else len(self.data_frame)
    
    def extraire_features(self,fichier_audio, start_time, end_time):
        # Calculer la durée à partir des temps de début et de fin
        duration = end_time - start_time
        # Charger le fichier audio entre les secondes spécifiées
        y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)

        # Calcule fenetre de hanning
        window = librosa.filters.get_window('hann', 1024, fftbins=True)
        # Obtenir les coefficients des filtres Mel
        mel_filters = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=30,norm=1.0)

        #separation des données en frames avec un overlap de 50%
        frames = librosa.util.frame(y, frame_length=1024, hop_length=512)[:, :60]

        # Application de la fenetre de hanning
        frames = frames * window[:, None]

        # calcul de la rfft des frames
        rfft = np.fft.rfft(frames, axis=0)

        # calcul la magnitude au carré de la rfft
        dsp = np.abs(rfft)**2/1024

        # calcul de la puissance des filtres mel
        mel_power = np.log(np.dot(mel_filters, dsp) + 1e-10)

        # z-score normalization
        z_score = ((mel_power - np.mean(mel_power)) /(np.std(mel_power)+1e-10))

        return z_score

    def __getitem__(self, idx):
        
        if(self.selection_list == None):
            data = self.data_frame.iloc[idx]
        else:
            data = self.data_frame.iloc[self.selection_list[idx]]
            

        audio_path = "segmented_selected_data/"+ data.iloc[0]
        start_time = data.iloc[1]
        end_time = data.iloc[2]
        label = data.iloc[3]

        # Charger et prétraiter l'audio
        features = self.extraire_features(audio_path, start_time, end_time)

        if self.transform:
            features = self.transform(features)

        # Ajouter une dimension pour le canal (1, car c'est un spectrogramme mono)
        features = np.expand_dims(features, axis=0)
        # Convertir le label en one-hot encodé
        label_one_hot = F.one_hot(torch.tensor(self.label_mapping[label]), num_classes=len(self.label_mapping)).float()
        return torch.tensor(features, dtype=torch.float32), label_one_hot






