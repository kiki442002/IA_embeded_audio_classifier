import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle

def calculate_margin_of_error(samples, confidence_level=0.95):
    mean = np.mean(samples)
    std_dev = np.std(samples, ddof=1)  # ddof=1 pour un échantillon
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    n = len(samples)
    margin_of_error = z_score * (std_dev / np.sqrt(n))
    return mean, margin_of_error

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
random_list = load_data_from_pickle('random_list.pkl')
model_list = load_data_from_pickle('least_confidence_list.pkl')

# Calculer la moyenne et la marge d'erreur pour chaque pourcentage
means = []
margins_of_error = []
db_percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05 ,0.15, 0.25, 0.5]
db_percents_model = [0.0005,  0.002, 0.005, 0.01, 0.05 , 0.25, 0.5]
for accuracies in random_list:
    mean, margin_of_error = calculate_margin_of_error(accuracies)
    means.append(mean)
    margins_of_error.append(margin_of_error)

# Tracer les courbes
plt.figure(figsize=(10, 6))
plt.errorbar(db_percents, means, yerr=margins_of_error, fmt='-o', capsize=5, label='Random sampling')
plt.plot(db_percents_model, model_list, '-o', label='Least confidence sampling')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.title('Model Accuracy with Different Training Data Percentages')
plt.legend()
plt.grid(True)
plt.show()

    