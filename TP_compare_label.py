import pandas as pd
from utils.sampling_function import uncertainty_least_confidence_sampling, uncertainty_ratio_sampling, uncertainty_margin_confidence
from utils.AudioDataset import AudioDataset
import torch as pt
from utils.TD_network import CNNNetwork


SAMPLE_FUNCTION = [uncertainty_margin_confidence,uncertainty_ratio_sampling,uncertainty_least_confidence_sampling]





db_percents = [0.0005,  0.002, 0.005, 0.01, 0.05 , 0.25, 0.5]

#################
# Initilisation #
#################
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

# Initialiser le modèle, la fonction de perte et l'optimiseur
model = CNNNetwork().to(device) 
model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))
model.outsize = False

labels = {"rain": 0, "walking":1, "wind": 2, "car_passing": 3}
trainData = AudioDataset("meta/bdd_B_train.csv", labels)
train_loader = pt.utils.data.DataLoader(trainData)

df_db = pd.read_csv('meta/bdd_B_train.csv')


for sample_choice in SAMPLE_FUNCTION:
    print("____________________________________________________________________________________")
    print(f"Sampling function : {sample_choice.__name__}")
    print()
    model.load_state_dict(pt.load('chosen_model.pth', weights_only=True))
    sample_choice = sample_choice(model, train_loader, device)
    for percent in db_percents:
        sample_list = sample_choice[:int(len(sample_choice)*percent)]
        # Compter le nombre de labels par classe
        label_counts = df_db.iloc[sample_list]['label'].value_counts()
        print(f"Label counts for {percent * 100}% of the data:")
        print(label_counts)
        print("")
    
