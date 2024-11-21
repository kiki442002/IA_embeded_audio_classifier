import torch as pt
import torch.nn as nn
import torch.optim as optim
import tqdm
from DataLoader import AudioDataset
from network import CNNNetwork

def train_single_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backpropagate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Loss: {running_loss / len(dataloader)}")

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with pt.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = pt.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, patience):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_single_epoch(model, train_loader, loss_fn, optimizer, device)
        
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            pt.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        print('-------------------------------------------')
    print('Finished Training')

if __name__ == "__main__":
    # Initialisation des paramètres et des objets nécessaires
    BATCH_SIZE = 128
    EPOCHS = 10
    PATIENCE = 3
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Charger les données
    trainData = AudioDataset("meta/bdd_A_train.csv")
    testData = AudioDataset("meta/bdd_A_test.csv")  # Assurez-vous d'avoir un ensemble de validation

    train_loader = pt.utils.data.DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = pt.utils.data.DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialiser le modèle, la fonction de perte et l'optimiseur
    model = CNNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Entraîner le modèle avec early stopping
    train(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE)