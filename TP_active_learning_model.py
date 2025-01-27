import torch as pt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.AudioDataset import AudioDataset
from utils.TD_network import CNNNetwork
from IA_train import evaluate, train
import pickle
from utils.sampling_function import uncertainty_ratio_sampling, uncertainty_least_confidence_sampling, uncertainty_margin_confidence, diversity_model_base_outlier_sampling,diversity_cluster_based_centroid


if __name__ == "__main__":
    ##########################
    ##### Hyperparamètres ####
    ##########################
    BATCH_SIZE = 1           # Taille du lot
    EPOCHS = 20              # Nombre d'époques
    PATIENCE = 50            # Nombre d'époques sans amélioration avant l'arrêt
    OUTSIZE = False          # True pour DA, False pour DB
    NAME_LIST = "model1_list.pkl"
    SAMPLE_FUNCTION = uncertainty_ratio_sampling
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

    # Initialisation des Labels et dataloader
    labels = {"rain": 0, "walking":1, "wind": 2, "car_passing": 3}
    devData = AudioDataset("meta/bdd_B_dev.csv", labels)
    dev_loader = pt.utils.data.DataLoader(devData)

    testData = AudioDataset("meta/bdd_B_test.csv", labels)
    test_loader = pt.utils.data.DataLoader(testData)

    trainData = AudioDataset("meta/bdd_B_train.csv", labels)
    train_loader = pt.utils.data.DataLoader(trainData)

    sample_choice= SAMPLE_FUNCTION(model, train_loader, device)
    df_db = pd.read_csv('meta/bdd_B_train.csv')

    for percent in db_percents:
        trainData.selection_list = sample_choice[:int(len(sample_choice)*percent)]
        # Compter le nombre de labels par classe
        label_counts = df_db.iloc[trainData.selection_list]['label'].value_counts()
        print(f"Label counts for {percent * 100}% of the data:")
        print(label_counts)
        print("")

    model_list = []
    for percent in db_percents:
        #Load model
        model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))

        # Choix des labels
        trainData.selection_list = sample_choice[:int(len(sample_choice)*percent)]
        train(model, train_loader, dev_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE)
        model.load_state_dict(pt.load('best_model.pth', weights_only=True))
        _ , accuracy = evaluate(model, test_loader, loss_fn, device, name="Final test")
        print(f"Accuracy for {percent} : {accuracy}")
        model_list+=[accuracy]
        with open(NAME_LIST, 'wb') as f:
            pickle.dump(model_list, f)




    

    

        

   

   


   

