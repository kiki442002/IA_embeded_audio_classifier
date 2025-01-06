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
    
    def extraire_features(self,fichier_audio, start_time, end_time):
        # Calculer la durée à partir des temps de début et de fin
        duration = end_time - start_time
        # Charger le fichier audio entre les secondes spécifiées
        y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
        

    
        #separation des données en frames avec un overlap de 50%
        frames = librosa.util.frame(y, frame_length=1024, hop_length=512)[:, :32]

        # Application de la fenetre de hanning
        frames = frames * self.window[:, None]
        frames = frames.T
    
        # Préparer un tableau pour stocker les résultats de la DSP
        dsp_output = np.zeros((frames.shape[0], 513), dtype=np.float32)

        # Initialiser la structure RFFT
        S = dsp.arm_rfft_fast_instance_f32()
        dsp.arm_rfft_fast_init_f32(S, 1024)
        
        for i,frame in enumerate(frames):
            frame_flat = frame.astype(np.float32)
            fft_result = np.zeros(1024, dtype=np.float32)
            fft_result=dsp.arm_rfft_fast_f32(S, frame_flat, 0)

            # Calculer la magnitude au carré des coefficients de la FFT pour obtenir la DSP
            dsp_result = np.zeros(513, dtype=np.float32)
            dsp_result[1:-1]=dsp.arm_cmplx_mag_squared_f32(fft_result[2:])
            dsp_result[0] = fft_result[0]**2
            dsp_result[-1] = fft_result[1]**2
            dsp_output[i] = dsp_result/1024
    

        # calcul de la puissance des filtres mel
        mel_power = np.log(np.dot(dsp_output, self.mel_filters.T) + 1e-10, dtype=np.float32)

        # z-score normalization
        for i,mel in enumerate(mel_power):
            mean = dsp.arm_mean_f32(mel)
            std = dsp.arm_std_f32(mel)
            mel_power[i] = (mel - mean) / std

        return mel_power.T

    def extraire_features_old(self,fichier_audio, start_time, end_time):
        # Calculer la durée à partir des temps de début et de fin
        duration = end_time - start_time
        # Charger le fichier audio entre les secondes spécifiées
        y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
        

        
        # # Génération d'une sinus de 1kHz de 2s en fonction de sr
        # t = np.linspace(0, 2, 2*sr, endpoint=False)
        # y = 80*np.sin(2*np.pi*1000*t)
        

        # Calcule fenetre de hanning
        window = self.window
        # Obtenir les coefficients des filtres Mel
        mel_filters = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=30,norm=1.0)

        #separation des données en frames avec un overlap de 50%
        frames = librosa.util.frame(y, frame_length=1024, hop_length=512)[:, :32]

        # Application de la fenetre de hanning
        frames = frames * window[:, None]

        # calcul de la rfft des frames
        rfft = np.fft.rfft(frames, axis=0)

        # calcul la magnitude au carré de la rfft
        dsp = np.abs(rfft)**2/1024

        # calcul de la puissance des filtres mel
        mel_power = np.log(np.dot(mel_filters, dsp) + 1e-10)

        # z-score normalization
        z_score = ((mel_power - np.mean(mel_power)) / np.std(mel_power))
        return z_score
    
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






