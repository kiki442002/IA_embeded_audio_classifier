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


with open('random_list.pkl', 'rb') as f:
    random_list = pickle.load(f)

# Calculer la moyenne et la marge d'erreur pour chaque pourcentage
means = []
margins_of_error = []
db_percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05 ,0.15, 0.25, 0.5]
for accuracies in random_list:
    mean, margin_of_error = calculate_margin_of_error(accuracies)
    means.append(mean)
    margins_of_error.append(margin_of_error)

# Tracer les courbes
plt.figure(figsize=(10, 6))
plt.errorbar(db_percents, means, yerr=margins_of_error, fmt='-o', capsize=5, label='Accuracy with Margin of Error')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.title('Model Accuracy with Different Training Data Percentages')
plt.legend()
plt.grid(True)
plt.show()

    