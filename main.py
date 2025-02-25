# Importation des bibliothèques nécessaires
from picamera2 import Picamera2  # Pour contrôler la caméra Raspberry Pi
import cv2  # OpenCV pour le traitement d'image
import mediapipe as mp  # Framework de détection de mains
import numpy as np  # Calculs mathématiques
import pygame  # Gestion audio
import time  # Gestion des délais


# =====================================================================
# Partie 1 : Génération des sons avec Pygame
# =====================================================================
# Initialisation du système audio Pygame
pygame.mixer.init()
# Chargement du fichier audio
movement_sound = pygame.mixer.Sound("media/metronome.wav")

# =====================================================================
# Partie 2 : Configuration de MediaPipe pour la détection des mains
# =====================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Détecte jusqu'à 2 mains
    min_detection_confidence=0.5,  # Seuil minimal pour considérer une détection valide
    min_tracking_confidence=0.5     # Seuil pour continuer le suivi
)

# =====================================================================
# Partie 3 : Configuration de la caméra Raspberry Pi
# =====================================================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"}  # Résolution réduite pour meilleures performances
)
picam2.configure(config)
picam2.start()  # Démarrage du flux vidéo

# =====================================================================
# Partie 4 : Variables globales
# =====================================================================
prev_y_left = 0    # Position Y précédente de la main gauche
prev_y_right = 0   # Position Y précédente de la main droite
prev_x_left = 0    # Position X précédente de la main gauche
prev_x_right = 0   # Position X précédente de la main droite
threshold = 30     # Seuil de mouvement en pixels (à ajuster empiriquement)
last_sound_time = 0  # Timestamp du dernier son joué
sound_delay = 0.3    # Délai minimal entre deux sons (évite les répétitions trop rapides)

# =====================================================================
# Partie 5 : Boucle principale
# =====================================================================
try:
    while True:
        # Capture d'une frame depuis la caméra
        frame = picam2.capture_array()  # Format BGR natif d'OpenCV
        
        # Conversion en RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection des mains
        results = hands.process(frame_rgb)

        # Debug : Affichage des détections dans la console
        if results.multi_hand_landmarks:
            print(f"Mains détectées: {len(results.multi_hand_landmarks)}")
        else:
            print("Aucune main détectée.")

        # Traitement de chaque main détectée
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Détermination de la main (gauche/droite)
                hand_type = handedness.classification[0].label  # "Left" ou "Right"
                
                # Récupération des coordonnées du poignet (landmark 0)
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                current_y = int(wrist.y * h)  # Conversion des coordonnées normalisées
                current_x = int(wrist.x * w)  # Conversion des coordonnées normalisées

                # Debug : Affichage des positions X et Y
                print(f"Main {hand_type} - X: {current_x}, Y: {current_y}")

                # Logique de déclenchement des sons
                if (time.time() - last_sound_time) > sound_delay:
                    if hand_type == "Left":
                        # Mouvement vers le haut
                        if prev_y_left - current_y > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (gauche, mouvement vers le haut)")
                        # Mouvement vers le bas
                        elif current_y - prev_y_left > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (gauche, mouvement vers le bas)")
                        # Mouvement vers la gauche
                        if prev_x_left - current_x > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (gauche, mouvement vers la gauche)")
                        # Mouvement vers la droite
                        elif current_x - prev_x_left > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (gauche, mouvement vers la droite)")
                        prev_y_left = current_y  # Mise à jour de la position Y
                        prev_x_left = current_x  # Mise à jour de la position X
                    else:  # Main droite
                        # Mouvement vers le haut
                        if prev_y_right - current_y > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (droit, mouvement vers le haut)")
                        # Mouvement vers le bas
                        elif current_y - prev_y_right > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (droit, mouvement vers le bas)")
                        # Mouvement vers la gauche
                        if prev_x_right - current_x > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (droit, mouvement vers la gauche)")
                        # Mouvement vers la droite
                        elif current_x - prev_x_right > threshold:
                            movement_sound.play()
                            last_sound_time = time.time()
                            print("Son joué (droit, mouvement vers la droite)")
                        prev_y_right = current_y  # Mise à jour de la position Y
                        prev_x_right = current_x  # Mise à jour de la position X

        # Affichage de l'image redimensionnée
        cv2.imshow("Conductor Tracker", cv2.resize(frame, (640, 480)))  # Agrandissement pour affichage

        # Condition de sortie avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"ERREUR: {e}")  # Gestion générique des erreurs
finally:
    # Nettoyage des ressources
    cv2.destroyAllWindows()  # Fermeture des fenêtres OpenCV
    picam2.stop()  # Arrêt de la caméra
    pygame.quit()  # Arrêt de Pygame
