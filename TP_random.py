import torch as pt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.AudioDataset import AudioDataset
from utils.TD_network import CNNNetwork
from IA_train import evaluate, train
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculate_margin_of_error(samples, confidence_level=0.95):
    mean = np.mean(samples)
    std_dev = np.std(samples, ddof=1)  # ddof=1 pour un échantillon
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    n = len(samples)
    margin_of_error = z_score * (std_dev / np.sqrt(n))
    return mean, margin_of_error



if __name__ == "__main__":
    ##########################
    ##### Hyperparamètres ####
    ##########################
    BATCH_SIZE = 1           # Taille du lot
    EPOCHS = 20              # Nombre d'époques
    PATIENCE = 50            # Nombre d'époques sans amélioration avant l'arrêt
    OUTSIZE = False          # True pour DA, False pour DB
    ##########################
    ##########################
    
    db_percents = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5]
    
    #################
    # Initilisation #
    #################
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Initialiser le modèle, la fonction de perte et l'optimiseur
    model = CNNNetwork().to(device)
    model.outsize = OUTSIZE

    print("Charger pour 4 labels" if OUTSIZE else "Charger pour 4 labels")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Initialisation des Labels et du dataloader de test
    labels = {"rain": 0, "walking":1, "wind": 2, "car_passing": 3}
    devData = AudioDataset("meta/bdd_B_dev.csv", labels)
    dev_loader = pt.utils.data.DataLoader(devData)

    testData = AudioDataset("meta/bdd_B_test.csv", labels)
    test_loader = pt.utils.data.DataLoader(testData)

    trainData = AudioDataset("meta/bdd_B_train.csv", labels)
    train_loader = pt.utils.data.DataLoader(trainData)

    df_db = pd.read_csv('meta/bdd_B_train.csv')

    random_list = []
    for percent in db_percents:
        accuracies = []
        for i in range(0,5):
            #Load model
            model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))

            # Choix des labels aléatoirement
            trainData.selection_list = df_db.sample(n=int(len(df_db)*percent)).index.to_list()
            train(model, train_loader, dev_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE)
            model.load_state_dict(pt.load('best_model.pth', weights_only=True))
            _ , accuracy = evaluate(model, test_loader, loss_fn, device, name="Final test")
            print(f"Accuracy for {percent} : {accuracy}")
            accuracies+=[accuracy]
        random_list+=[accuracies]

    # Calculer la moyenne et la marge d'erreur pour chaque pourcentage
    means = []
    margins_of_error = []

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

    


    

    

        

   

   


   

