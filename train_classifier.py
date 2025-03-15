import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle

# Charger les données
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convertir les données en tenseurs PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Mélanger les données et les étiquettes de manière synchrone
indices = torch.randperm(len(data_tensor))
data_tensor = data_tensor[indices]
labels_tensor = labels_tensor[indices]

# Définir la taille des ensembles d'entraînement et de test
test_size = int(0.2 * len(data_tensor))
train_size = len(data_tensor) - test_size

# Diviser les données en ensembles d'entraînement et de test
train_dataset, test_dataset = random_split(TensorDataset(data_tensor, labels_tensor), [train_size, test_size])

# Créer des DataLoaders pour les ensembles d'entraînement et de test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Définir le modèle
class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = data_tensor.shape[1]
hidden_size = 128
output_size = len(np.unique(labels))

model = ClassifierModel(input_size, hidden_size, output_size)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0  # Pour cumuler le loss de chaque batch
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    moyenne_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {moyenne_loss:.4f}")

# Évaluer le modèle
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    accuracy = correct / total

print(f'{accuracy * 100:.2f}% des échantillons ont été classifiés correctement!')

# Enregistrer le modèle
torch.save(model, 'model.pth')