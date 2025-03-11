import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Masking
from tensorflow.keras.optimizers import Adam
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

# Définir le modèle
model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_length, 21 * 3)),  # Masquer les valeurs de padding
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # 4 gestes à reconnaître
])

# Compiler le modèle
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Enregistrer le modèle
model.save("data/gesture_model.h5")