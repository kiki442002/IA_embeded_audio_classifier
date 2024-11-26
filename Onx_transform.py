import torch
import torch.onnx
from network import CNNNetwork  # Assurez-vous que CNNNetwork est défini dans network.py
import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf

# Charger le modèle PyTorch
model = CNNNetwork()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Exemple de données d'entrée
dummy_input = torch.randn(1, 1, 32, 30)  # Ajustez la taille en fonction de votre modèle
print(dummy_input.shape)

# Exporter le modèle en format ONNX
onnx_path = 'small_cnn_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])

# Charger le modèle ONNX
onnx_model = onnx.load(onnx_path)

# Nettoyer les noms des couches dans le modèle ONNX
for node in onnx_model.graph.node:
    node.name = node.name.replace('/', '_')
for tensor in onnx_model.graph.initializer:
    tensor.name = tensor.name.replace('/', '_')
for input_tensor in onnx_model.graph.input:
    input_tensor.name = input_tensor.name.replace('/', '_')
for output_tensor in onnx_model.graph.output:
    output_tensor.name = output_tensor.name.replace('/', '_')

print(onnx_model)

# Convertir le modèle ONNX en modèle Keras
k_model = onnx_to_keras(onnx_model, ['input'])

# Sauvegarder le modèle Keras
k_model.save('small_cnn_model.h5')

print("The model has been converted to Keras and saved as 'small_cnn_model.h5'")

# Charger le modèle Keras
model = tf.keras.models.load_model('small_cnn_model.h5')

# Convertir le modèle Keras en modèle TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle TFLite
with open('small_cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("The model has been converted to TFLite and saved as 'small_cnn_model.tflite'")