# Importation des bibliothèques nécessaires
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import threading
import queue

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
    
    # Files d'attente pour les frames vidéo et les événements audio
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        
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
    # Partie 3: Détection de mouvement et déclenchement audio
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
    # Partie 4: Boucle de capture vidéo (Thread dédié)
    # =================================================================
    def capture_loop(self):
        """Capture en continu les frames depuis la caméra"""
        while self.running:
            frame = self.camera.capture_array()
            try:
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                # Si la file d'attente est pleine, on ignore la frame
                pass
            time.sleep(0.01)

    # =================================================================
    # Partie 5: Boucle de traitement d'image et affichage (Thread dédié)
    # =================================================================
    def processing_loop(self):
        """Traite les frames et affiche le flux vidéo"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Conversion et traitement avec MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label.lower()
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    self.detect_movement(hand_type, current_x, current_y)

            # Affichage du flux vidéo redimensionné
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Conductor Tracker", resized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.config['debug_mode'] = not self.config['debug_mode']

    # =================================================================
    # Partie 6: Boucle de gestion audio (Thread dédié)
    # =================================================================
    def audio_loop(self):
        """Joue les sons en fonction des événements reçus"""
        while self.running:
            try:
                event = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if event == "beep":
                self.sound.play()
    
    # =================================================================
    # Partie 7: Démarrage de l'application en multithreading
    # =================================================================
    def run(self):
        """Exécute la boucle principale avec plusieurs threads"""
        self.running = True
        self.last_sound_time = 0

        # Création des threads pour la capture, le traitement et l'audio
        threads = []
        capture_thread = threading.Thread(target=self.capture_loop)
        processing_thread = threading.Thread(target=self.processing_loop)
        audio_thread = threading.Thread(target=self.audio_loop)
        threads.extend([capture_thread, processing_thread, audio_thread])

        # Démarrage des threads
        for t in threads:
            t.start()

        # Le thread principal attend la fin des autres threads
        for t in threads:
            t.join()

        self.cleanup()
    
    # =================================================================
    # Partie 8: Nettoyage des ressources
    # =================================================================
    def cleanup(self):
        """Libère toutes les ressources et arrête les composants"""
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        pygame.quit()
        print("Nettoyage terminé")

# =====================================================================
# Partie 9: Point d'entrée principal
# =====================================================================
if __name__ == "__main__":
    tracker = ConductorTracker()
    print("Démarrage du Conductor Tracker...")
    print("Appuyez sur:")
    print("- 'q' pour quitter")
    print("- 'd' pour activer/désactiver le mode debug")
    tracker.run()