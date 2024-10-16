import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Chemin du dossier contenant les fichiers audio
audio_folder = 'segmented_selected_data/'

# Charger les données de training et de test depuis le dossier segmented_selected_data
train_data = pd.read_csv(os.path.join(audio_folder, 'meta_train.csv'))
test_data = pd.read_csv(os.path.join(audio_folder, 'meta_test.csv'))

# Ajouter le chemin du dossier avant les noms de fichiers audio
train_data['audio'] = train_data['audio'].apply(lambda x: os.path.join(audio_folder, x))
test_data['audio'] = test_data['audio'].apply(lambda x: os.path.join(audio_folder, x))

# Fonction pour charger les fichiers audio et extraire les features
def extract_features(file_path):
    import librosa
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Extraire les features des fichiers audio
X_train = np.array([extract_features(file) for file in train_data['audio'].values])
y_train = train_data['label'].values
X_test = np.array([extract_features(file) for file in test_data['audio'].values])
y_test = test_data['label'].values

# Convertir les labels en format numérique si nécessaire
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Diviser les données de training pour la validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Construire le modèle
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Utiliser 'softmax' pour la classification multi-classes
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Utiliser 'sparse_categorical_crossentropy' pour multi-classes

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Évaluer le modèle sur les données de validation
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Sauvegarder le modèle au format H5
model.save('model.h5')