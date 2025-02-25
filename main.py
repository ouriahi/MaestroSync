# Importation des bibliothèques nécessaires
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# =====================================================================
# Partie 1: Classe principale pour la gestion du projet
# =====================================================================
class ConductorTracker:
    def __init__(self):
        # Configuration initiale
        self.camera = None
        self.hands = None
        self.sound = None
        self.running = False
        
        # Paramètres de configuration
        self.config = {
            'camera_resolution': (320, 240),
            'hand_detection_confidence': 0.5,
            'movement_threshold': 30,  # en pixels
            'sound_delay': 0.3,        # en secondes
            'debug_mode': True
        }
        
        # Historique des positions
        self.positions = {
            'left': {'x': 0, 'y': 0},
            'right': {'x': 0, 'y': 0}
        }
        
        self.initialize_components()
    
    # =================================================================
    # Partie 2: Initialisation des composants
    # =================================================================
    def initialize_components(self):
        """Initialise tous les composants matériels et logiciels"""
        # Configuration de Pygame
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("media/metronome.wav")
        
        # Configuration MediaPipe
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=self.config['hand_detection_confidence'],
            min_tracking_confidence=self.config['hand_detection_confidence']
        )
        
        # Configuration de la caméra
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": self.config['camera_resolution'], "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
    
    # =================================================================
    # Partie 3: Logique de détection de mouvement
    # =================================================================
    def detect_movement(self, hand_type, current_x, current_y):
        """Détecte le mouvement et déclenche le son si nécessaire"""
        prev_x = self.positions[hand_type]['x']
        prev_y = self.positions[hand_type]['y']
        
        movement_detected = False
        directions = []
        
        # Vérification des mouvements verticaux
        if abs(current_y - prev_y) > self.config['movement_threshold']:
            directions.append('Haut' if current_y < prev_y else 'Bas')
            movement_detected = True
        
        # Vérification des mouvements horizontaux
        if abs(current_x - prev_x) > self.config['movement_threshold']:
            directions.append('Gauche' if current_x < prev_x else 'Droite')
            movement_detected = True
        
        if movement_detected and (time.time() - self.last_sound_time) > self.config['sound_delay']:
            self.sound.play()
            self.last_sound_time = time.time()
            if self.config['debug_mode']:
                print(f"Mouvement {hand_type} détecté: {', '.join(directions)}")
        
        # Mise à jour des positions
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y
    
    # =================================================================
    # Partie 4: Boucle principale de traitement
    # =================================================================
    def run(self):
        """Exécute la boucle principale de traitement"""
        self.running = True
        self.last_sound_time = 0
        
        try:
            while self.running:
                frame = self.camera.capture_array()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Détection des mains
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_type = handedness.classification[0].label.lower()
                        wrist = hand_landmarks.landmark[0]
                        
                        h, w = frame.shape[:2]
                        current_x = int(wrist.x * w)
                        current_y = int(wrist.y * h)
                        
                        self.detect_movement(hand_type, current_x, current_y)
                
                # Affichage de la vidéo
                resized_frame = cv2.resize(frame, (640, 480))
                cv2.imshow("Conductor Tracker", resized_frame)
                
                # Gestion des entrées clavier
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('d'):
                    self.config['debug_mode'] = not self.config['debug_mode']
        
        except Exception as e:
            print(f"Erreur: {str(e)}")
        finally:
            self.cleanup()
    
    # =================================================================
    # Partie 5: Nettoyage des ressources
    # =================================================================
    def cleanup(self):
        """Libère toutes les ressources"""
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        pygame.quit()
        print("Nettoyage terminé")

# =====================================================================
# Partie 6: Point d'entrée principal
# =====================================================================
if __name__ == "__main__":
    tracker = ConductorTracker()
    print("Démarrage du Conductor Tracker...")
    print("Appuyez sur:")
    print("- 'q' pour quitter")
    print("- 'd' pour activer/désactiver le mode debug")
    tracker.run()