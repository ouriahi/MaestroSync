import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Charger les données d'entraînement
X_train = np.load("data/dataset/X_data.npy", allow_pickle=True)
y_train = np.load("data/dataset/Y_data.npy")

# Déterminer la longueur maximale des séquences
max_length = max(len(seq) for seq in X_train)

# Fonction pour appliquer le padding aux séquences
def pad_sequences(sequences, max_length):
    padded_sequences = np.zeros((len(sequences), max_length, 21 * 3))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = np.array(seq).reshape(-1, 21 * 3)
    return padded_sequences

# Appliquer le padding aux données d'entraînement
X_train_padded = pad_sequences(X_train, max_length)
y_train = np.array(y_train)

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Définir le modèle
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureRecognitionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = max_length * 21 * 3
hidden_size = 128
output_size = 4  # 4 gestes à reconnaître

model = GestureRecognitionModel(input_size, hidden_size, output_size)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Enregistrer le modèle
torch.save(model, "data/gesture_model.pth")