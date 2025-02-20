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
def generate_high_beep():
    """Génère un son aigu (880Hz) avec Pygame."""
    sample_rate = 44100  # Fréquence d'échantillonnage standard
    frequency = 880      # Fréquences correspondant au La5 (aigu)
    duration = 0.1       # Durée courte pour un son "sec"
    
    # Création d'une onde sinusoïdale
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    
    # Conversion en format audio 16 bits et stéréo
    wave = np.int16(wave * 32767)
    stereo_wave = np.column_stack((wave, wave))
    
    return pygame.sndarray.make_sound(stereo_wave)

def generate_low_beep():
    """Génère un son grave (440Hz) avec Pygame (même logique que ci-dessus)."""
    # ... (code identique avec frequency=440)

# Initialisation du système audio Pygame
pygame.mixer.init()
high_beep = generate_high_beep()
low_beep = generate_low_beep()

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

                # Debug : Affichage de la position Y
                print(f"Main {hand_type} - Y: {current_y}")

                # Logique de déclenchement des sons
                if (time.time() - last_sound_time) > sound_delay:
                    if hand_type == "Left":
                        # Mouvement vers le haut
                        if prev_y_left - current_y > threshold:
                            high_beep.play()
                            last_sound_time = time.time()
                            print("Son HIGH joué (gauche)")
                        # Mouvement vers le bas
                        elif current_y - prev_y_left > threshold:
                            low_beep.play()
                            last_sound_time = time.time()
                            print("Son LOW joué (gauche)")
                        prev_y_left = current_y  # Mise à jour de la position
                    else:  # Main droite
                        # Même logique que pour la main gauche
                        if prev_y_right - current_y > threshold:
                            high_beep.play()
                            last_sound_time = time.time()
                            print("Son HIGH joué (droit)")
                        elif current_y - prev_y_right > threshold:
                            low_beep.play()
                            last_sound_time = time.time()
                            print("Son LOW joué (droit)")
                        prev_y_right = current_y

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
