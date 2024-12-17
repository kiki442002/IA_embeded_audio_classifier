import torch as pt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.AudioDataset import AudioDataset
import tqdm
from utils.TD_network import CNNNetwork
from IA_train import evaluate, train
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle




def least_confidence_sampling(model, dataloader, device):
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




if __name__ == "__main__":
    ##########################
    ##### Hyperparamètres ####
    ##########################
    BATCH_SIZE = 1           # Taille du lot
    EPOCHS = 20              # Nombre d'époques
    PATIENCE = 50            # Nombre d'époques sans amélioration avant l'arrêt
    OUTSIZE = False          # True pour DA, False pour DB
    NAME_LIST = "model1_list.pkl"
    ##########################
    ##########################
    
    db_percents = [0.0005,  0.002, 0.005, 0.01, 0.05 , 0.25, 0.5]

    #################
    # Initilisation #
    #################
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Initialiser le modèle, la fonction de perte et l'optimiseur
    model = CNNNetwork().to(device) 
    model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))
    model.outsize = OUTSIZE

    print("Charger pour 4 labels" if OUTSIZE else "Charger pour 4 labels")
    print(f"Nombre de paramètres : {model.count_parameters()}")
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

    sample_choice=least_confidence_sampling(model, train_loader, device)



    model_list = []
    for percent in db_percents:
        #Load model
        model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))

        # Choix des labels aléatoirement
        trainData.selection_list = sample_choice[:int(len(sample_choice)*percent)]
        train(model, train_loader, dev_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE)
        model.load_state_dict(pt.load('best_model.pth', weights_only=True))
        _ , accuracy = evaluate(model, test_loader, loss_fn, device, name="Final test")
        print(f"Accuracy for {percent} : {accuracy}")
        model_list+=[accuracy]
        with open(NAME_LIST, 'wb') as f:
            pickle.dump(model_list, f)




    

    

        

   

   


   

