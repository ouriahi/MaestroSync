import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Charger les données générées par votre script de création de données
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Conversion des labels (qui sont des chaînes) en entiers
unique_labels = np.unique(labels)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels_int = np.array([label_to_int[label] for label in labels])

# Conversion des données et labels en tenseurs PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels_int, dtype=torch.long)

# Division en ensembles d'entraînement (80%) et de test (20%)
num_samples = data.shape[0]
indices = torch.randperm(num_samples)
split = int(0.8 * num_samples)
train_indices = indices[:split]
test_indices = indices[split:]

train_data = data_tensor[train_indices]
train_labels = labels_tensor[train_indices]
test_data = data_tensor[test_indices]
test_labels = labels_tensor[test_indices]

# Création des datasets et DataLoaders
batch_size = 64
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définition d'un réseau de neurones simple à une couche cachée
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Récupération des dimensions : nombre de caractéristiques et de classes
input_dim = data.shape[1]
num_classes = len(unique_labels)
model = Net(input_dim, num_classes)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()            # Réinitialisation des gradients
        outputs = model(inputs)            # Passage avant
        loss = criterion(outputs, targets) # Calcul de la perte
        loss.backward()                    # Rétropropagation
        optimizer.step()                   # Mise à jour des poids
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Évaluation du modèle sur l'ensemble de test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"{accuracy*100:.2f}% des échantillons ont été correctement classifiés!")

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), 'model.pth')
