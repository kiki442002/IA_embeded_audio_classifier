import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle


with open('random_list.pkl', 'rb') as f:
    loaded_list = pickle.load(f)

print(loaded_list)
# # Calculer la moyenne et la marge d'erreur pour chaque pourcentage
# means = []
# margins_of_error = []

# for accuracies in random_list:
#     mean, margin_of_error = calculate_margin_of_error(accuracies)
#     means.append(mean)
#     margins_of_error.append(margin_of_error)

# # Tracer les courbes
# plt.figure(figsize=(10, 6))
# plt.errorbar(db_percents, means, yerr=margins_of_error, fmt='-o', capsize=5, label='Accuracy with Margin of Error')
# plt.xlabel('Percentage of Training Data')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy with Different Training Data Percentages')
# plt.legend()
# plt.grid(True)
# plt.show()

    