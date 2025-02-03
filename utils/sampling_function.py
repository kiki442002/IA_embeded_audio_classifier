
import torch as pt
import pandas as pd
import tqdm
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def uncertainty_least_confidence_sampling(model, dataloader, device):
    model.eval()
    df_confidence = pd.DataFrame({
    'confidence': pd.Series(dtype='float'),
    'index': pd.Series(dtype='int')
    })

    with tqdm.tqdm(total=len(dataloader), desc="Sampling Process", unit="iter") as pbar:
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            max_probs, _ = pt.max(outputs, dim=1)
            if df_confidence.empty:
                df_confidence = pd.DataFrame({'confidence': max_probs.detach().cpu().numpy() , 'index': i})
            else:
                df_confidence =  pd.concat([df_confidence, pd.DataFrame({'confidence': max_probs.detach().cpu().numpy() , 'index': i})])
            pbar.update(1)
         
    # Trier les échantillons par confiance croissante
    return df_confidence.sort_values(by='confidence', ascending=True)['index'].to_list()

def uncertainty_ratio_sampling(model, dataloader, device):
    model.eval()
    df_ratio = pd.DataFrame(columns=['ratio', 'index'])

    with tqdm.tqdm(total=len(dataloader), desc="Sampling Process", unit="iter") as pbar:
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Obtenir les deux plus grandes probabilités
            top2_probs, _ = pt.topk(outputs, 2, dim=1)
            max_probs = top2_probs[:, 0]
            second_max_probs = top2_probs[:, 1]
            ratio = (max_probs / (second_max_probs + 1e-10)).detach().cpu().numpy()  # Calculer le ratio et convertir en NumPy
            new_data = pd.DataFrame({'ratio': ratio, 'index': i})
            if df_ratio.empty:
                df_ratio = new_data
            else:
                df_ratio = pd.concat([df_ratio, new_data], ignore_index=True)

            pbar.update(1)

    # Trier les échantillons par ratio croissant
    return df_ratio.sort_values(by='ratio', ascending=True)['index'].to_list()

def uncertainty_margin_confidence(model, dataloader, device):
    model.eval()
    df_margin = pd.DataFrame(columns=['margin', 'index'])

    with tqdm.tqdm(total=len(dataloader), desc="Sampling Process", unit="iter") as pbar:
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Obtenir les deux plus grandes probabilités
            top2_probs, _ = pt.topk(outputs, 2, dim=1)
            max_probs = top2_probs[:, 0]
            second_max_probs = top2_probs[:, 1]
            margin = (max_probs - second_max_probs).detach().cpu().numpy()  # Calculer le ratio et convertir en NumPy
            new_data = pd.DataFrame({'margin': margin, 'index': i})
            if df_margin.empty:
                df_margin = new_data
            else:
                df_margin = pd.concat([df_margin, new_data], ignore_index=True)

            pbar.update(1)

    # Trier les échantillons par ratio croissant
    return df_margin.sort_values(by='margin', ascending=True)['index'].to_list()

def diversity_model_base_outlier_sampling(model, dataloader, device):
    model.eval()
    model.outlier = True
    outlier_scores = []

    with pt.no_grad():
        pbar = tqdm.tqdm(total=len(dataloader), desc="Outlier Sampling")
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Supposons que le modèle retourne des scores d'outlier
            scores = outputs.detach().cpu().numpy()
            outlier_scores.extend(scores)

            pbar.update(1)
        pbar.close()

    outlier_scores = np.array(outlier_scores)
    outlier_means = np.mean(outlier_scores, axis=0)
    outlier_dist = []
    for i, score in enumerate(outlier_scores):
        dist = np.linalg.norm(score - outlier_means)
        outlier_dist.append(dist)
    pd_dist = pd.DataFrame({'dist': outlier_dist, 'index': range(len(outlier_dist))})
    model.outlier = False
    return pd_dist.sort_values(by='dist', ascending=False)['index'].to_list()
    
def diversity_cluster_based_centroid(model, dataloader, device):
    model.eval()
    features = []
    with pt.no_grad():
        pbar = tqdm.tqdm(total=len(dataloader), desc="Cluster-based Sampling")
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Supposons que les sorties du modèle sont les caractéristiques
            features.append(outputs.detach().cpu().numpy())

            pbar.update(1)
        pbar.close()

    features = np.vstack(features)

    # Appliquer K-means pour regrouper les données en clusters
    n_clusters=5
    #determine_optimal_clusters(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_

    # Sélectionner l'échantillon le plus proche de chaque centre de cluster
    df_representative = pd.DataFrame(columns=['distance', 'index'])
    for center in cluster_centers:
        distances = pd.DataFrame({"distance":np.linalg.norm(features - center, axis=1),"index":range(len(features))})
        if df_representative.empty:
            df_representative = distances
        else:
            df_representative = pd.concat([df_representative, distances], ignore_index=True)
    
    return  df_representative.sort_values(by='distance', ascending=True).drop_duplicates(subset='index', keep='first')['index'].to_list()

def determine_optimal_clusters(features, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        inertias.append(kmeans.inertia_)
    
    # Calculer les coefficients directeurs des tangentes
    x = np.arange(1, max_clusters + 1)
    y = np.array(inertias)
    slopes = np.diff(y) / np.diff(x)
    
    # Afficher le graphique du coude
    plt.plot(x, y, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.axvline(x=5, linestyle='--', color='g', label=f'Optimal k = {5}')
    plt.legend()
    plt.show()
    
def diversity_cluster_based_outlier(model, dataloader, device):
    model.eval()
    features = []
    with pt.no_grad():
        pbar = tqdm.tqdm(total=len(dataloader), desc="Cluster-based Sampling")
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Supposons que les sorties du modèle sont les caractéristiques
            features.append(outputs.detach().cpu().numpy())

            pbar.update(1)
        pbar.close()

    features = np.vstack(features)

    # Appliquer K-means pour regrouper les données en clusters
    n_clusters=5
    #determine_optimal_clusters(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_

    # Sélectionner l'échantillon le plus proche de chaque centre de cluster
    df_representative = pd.DataFrame(columns=['distance', 'index'])
    for center in cluster_centers:
        distances = pd.DataFrame({"distance":np.linalg.norm(features - center, axis=1),"index":range(len(features))})
        if df_representative.empty:
            df_representative = distances
        else:
            df_representative = pd.concat([df_representative, distances], ignore_index=True)
    
    return  df_representative.sort_values(by='distance', ascending=False).drop_duplicates(subset='index', keep='first')['index'].to_list()

def uncertainty_diversity_sampling(model, dataloader, device):
    list1 = diversity_model_base_outlier_sampling(model, dataloader, device)
    list2 = uncertainty_ratio_sampling(model, dataloader, device)
    return (list1, list2)
def combine_method_sampling(model, dataloader, device):
    list1 = diversity_cluster_based_centroid(model, dataloader, device)
    list2 = diversity_model_base_outlier_sampling(model, dataloader, device)
    list3 = uncertainty_ratio_sampling(model, dataloader, device)
    return (list1, list2, list3)