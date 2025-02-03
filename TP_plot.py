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
random_list = load_data_from_pickle('active_learning_datas/random.pkl')
# model_list_1 = load_data_from_pickle('active_learning_datas/diversity_cluster_based_centroid.pkl')
# model_list_2 = load_data_from_pickle('active_learning_datas/diversity_cluster_based_outlier.pkl')
# model_list_3 = load_data_from_pickle('active_learning_datas/diversity_model_base_outlier.pkl')
# model_list_4 = load_data_from_pickle('active_learning_datas/uncertainty_least_confidence_sampling.pkl')
# model_list_5 = load_data_from_pickle('active_learning_datas/uncertainty_margin_confidence.pkl')
# model_list_6 = load_data_from_pickle('active_learning_datas/uncertainty_ratio_sampling.pkl')
#model_list_7 = load_data_from_pickle('active_learning_datas/model_combine_1.pkl')
#model_list_8 = load_data_from_pickle('active_learning_datas/model_combine_2.pkl')
#model_list_9 = load_data_from_pickle('active_learning_datas/model_combine_3.pkl')
#model_list_10 = load_data_from_pickle('active_learning_datas/ratio_bon.pkl')
model_list_11 = load_data_from_pickle('active_learning_datas/margin_bon.pkl')
model_list_12 = load_data_from_pickle('active_learning_datas/combine_methode_2.pkl')
model_list_13 = load_data_from_pickle('active_learning_datas/combine_methode_15%.pkl')
model_list_14 = load_data_from_pickle('active_learning_datas/best 50-50.pkl')
model_list_15 = load_data_from_pickle('active_learning_datas/combine_model_2_33%.pkl')
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
# plt.plot(db_percents_model, model_list_1, '-o', label='Diversity Cluster Based Centroid')
# plt.plot(db_percents_model, model_list_2, '-o', label='Diversity Cluster Based Outlier')
# plt.plot(db_percents_model, model_list_3, '-o', label='Diversity Model Base Outlier Sampling')
# plt.plot(db_percents_model, model_list_4, '-o', label='Uncertainty Least Confidence Sampling')
# plt.plot(db_percents_model, model_list_5, '-o', label='Uncertainty Margin Confidence Sampling')
# plt.plot(db_percents_model, model_list_6, '-o', label='Uncertainty Ratio Confidence Sampling')
#plt.plot(db_percents_model, model_list_7, '-o', label='Combine Method Sampling 1')
#plt.plot(db_percents_model, model_list_8, '-o', label='Combine Method Sampling 2')
#plt.plot(db_percents_model, model_list_9, '-o', label='Combine Method Sampling 3')
##plt.plot(db_percents_model, model_list_10, '-o', label='Uncertainty Ratio Confidence Sampling')
#plt.plot(db_percents_model, model_list_11, '-o', label='Uncertainty Margin Confidence Sampling')
#plt.plot(db_percents_model, model_list_12, '-o', label='Combine Methode 2 (25%,50%,25%)')
#plt.plot(db_percents_model, model_list_13, '-o', label='Combine Methode 2 (15%,65%,20%)')
#plt.plot(db_percents_model, model_list_14, '-o', label='Combine Methode 1 (50%,50%)')
plt.plot(db_percents_model, model_list_15, '-o', label='Combine Methode 2 (33%, 33%, 33%)')

plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.title('Model Accuracy with Different Training Data Percentages')
plt.legend()
plt.grid(True)
plt.show()

    