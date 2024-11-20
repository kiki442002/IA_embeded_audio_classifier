import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

def extraire_features(fichier_audio, start_time, end_time):
    # Calculer la durée à partir des temps de début et de fin
    duration = end_time - start_time
    # Charger le fichier audio entre les secondes spécifiées
    y, sr = librosa.load(fichier_audio, offset=start_time, duration=duration, sr=16000)
    
    # Calcule fenetre de hanning
    window = librosa.filters.get_window('hann', 1024, fftbins=True)
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
    z_score = (mel_power - np.mean(mel_power)) / np.std(mel_power)

    return z_score.T






 

# Exemple d'utilisation
fichier_audio = 'segmented_selected_data/cp005_1.wav'
start_time = 1  # seconde de début
end_time = 3   # seconde de fin

features = extraire_features(fichier_audio, start_time, end_time)



