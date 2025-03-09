import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

# Charger les données d'entraînement
X_train = np.load("data/X_train.npy")
y_train = np.load("data/Y_train.npy")

# Définir le modèle
model = Sequential([
    Flatten(input_shape=(21 * 3,)),  # 21 landmarks, chaque landmark a 3 coordonnées (x, y, z)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # 4 gestes à reconnaître
])

# Compiler le modèle
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Enregistrer le modèle
model.save("data/gesture_model.h5")
