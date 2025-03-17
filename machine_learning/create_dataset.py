import os
import pickle
import mediapipe as mp
import cv2

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1)

# Répertoire contenant les images classées par gestes
DATA_DIR = './data'  # Remplace par le bon chemin si nécessaire

# Listes pour stocker les données et les labels
data = []
labels = []

# Vérifier que les dossiers existent
classes = ['0', '1', '2', '3']
for dir_ in classes:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.exists(dir_path):
        print(f"Le dossier {dir_path} est introuvable !")
        continue

    for img_name in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        # Charger l'image
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur de chargement : {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Détection des landmarks avec MediaPipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalisation des coordonnées
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(int(dir_))  # Convertir en entier

# Vérifier que les données ont été collectées
if len(data) == 0:
    print("Erreur : aucune donnée n'a été collectée !")
else:
    print(f"Dataset généré avec {len(data)} images.")

# Sauvegarde dans un fichier pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset sauvegardé dans 'data.pickle'.")