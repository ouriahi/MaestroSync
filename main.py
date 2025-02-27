# Importation des bibliothèques nécessaires
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk

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

        # Paramètres de configuration (valeurs par défaut)
        self.config = {
            'camera_resolution': (320, 240),
            'hand_detection_confidence': 0.5,
            'movement_threshold': 30,  # en pixels (sera ajusté par calibration)
            'sound_delay': 0.3,        # en secondes
            'debug_mode': True
        }

        # Historique des positions
        self.positions = {
            'left': {'x': 0, 'y': 0},
            'right': {'x': 0, 'y': 0}
        }

        # État de la LED
        self.led_on = False

        # Initialisation de beat_times
        self.beat_times = []

        self.initialize_components()

    # =================================================================
    # Partie 2: Initialisation des composants
    # =================================================================
    def initialize_components(self):
        """Initialise tous les composants matériels et logiciels"""
        # Configuration de Pygame pour la gestion audio
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
    # Partie 3: Calibration
    # =================================================================
    def calibrate(self, duration=5):
        """
        Lance une phase de calibrage pendant laquelle l'utilisateur effectue
        quelques mouvements standards. Le système calcule alors la moyenne
        des variations et ajuste le seuil de détection.
        """
        print(f"Calibration : Effectuez des mouvements standards pendant {duration} secondes.")
        start_time = time.time()
        deltas = []
        prev_positions = {}
        while time.time() - start_time < duration:
            frame = self.camera.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label.lower()
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    if hand_type in prev_positions:
                        dx = abs(current_x - prev_positions[hand_type]['x'])
                        dy = abs(current_y - prev_positions[hand_type]['y'])
                        delta = np.sqrt(dx*dx + dy*dy)
                        deltas.append(delta)
                    prev_positions[hand_type] = {'x': current_x, 'y': current_y}
            time.sleep(0.05)
        if deltas:
            average_delta = sum(deltas) / len(deltas)
            # On applique un facteur multiplicatif pour fixer le seuil
            new_threshold = max(30, int(average_delta * 1.2))
            self.config['movement_threshold'] = new_threshold
            print("Calibration terminée. Nouveau seuil de mouvement :", new_threshold)
        else:
            print("Calibration échouée : aucun mouvement détecté.")

    # =================================================================
    # Partie 4: Détection de mouvement et déclenchement audio
    # =================================================================
    def detect_movement(self, hand_type, current_x, current_y):
        """Détecte le mouvement et envoie un signal audio si nécessaire"""
        prev_x = self.positions[hand_type]['x']
        prev_y = self.positions[hand_type]['y']

        movement_detected = False
        self.led_on = False  # Éteindre la LED par défaut
        directions = []

        # Vérification des mouvements verticaux
        if abs(current_y - prev_y) > self.config['movement_threshold']:
            directions.append('Haut' if current_y < prev_y else 'Bas')
            movement_detected = True
            self.led_on = True

        # Vérification des mouvements horizontaux
        if abs(current_x - prev_x) > self.config['movement_threshold']:
            directions.append('Gauche' if current_x < prev_x else 'Droite')
            movement_detected = True
            self.led_on = True

        if movement_detected and (time.time() - self.last_sound_time) > self.config['sound_delay']:
            self.audio_queue.put("beep")
            self.last_sound_time = time.time()
            self.beat_times.append(self.last_sound_time)  # Ajouter le temps de battement
            if self.config['debug_mode']:
                print(f"Mouvement {hand_type} détecté: {', '.join(directions)}")

        # Mise à jour des positions
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y

    # =================================================================
    # Partie 5: Boucle de capture vidéo (Thread dédié)
    # =================================================================
    def capture_loop(self):
        """Capture en continu les frames depuis la caméra"""
        while self.running:
            frame = self.camera.capture_array()
            try:
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                pass
            time.sleep(0.01)

    # =================================================================
    # Partie 6: Boucle de traitement d'image et affichage (Thread dédié)
    # =================================================================
    def processing_loop(self):
        """Traite les frames et affiche le flux vidéo"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

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
            
            # Calcul du BPM à partir des beats enregistrés durant les 10 dernières secondes
            current_time = time.time()
            # Garder uniquement les beats des 10 dernières secondes
            self.beat_times = [t for t in self.beat_times if current_time - t <= 10]
            if len(self.beat_times) >= 2:
                total_time = self.beat_times[-1] - self.beat_times[0]
                avg_interval = total_time / (len(self.beat_times) - 1)
                bpm = 60 / avg_interval if avg_interval > 0 else 0
            else:
                bpm = 0

            # Dessiner la LED sur le frame
            led_color = (0, 255, 0) if self.led_on else (50, 50, 50)
            cv2.circle(frame, (50, 50), 20, led_color, -1)

            resized_frame = cv2.resize(frame, (640, 480))
            cv2.putText(resized_frame, f"BPM: {bpm:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Conductor Tracker", resized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.config['debug_mode'] = not self.config['debug_mode']

    # =================================================================
    # Partie 7: Boucle de gestion audio (Thread dédié)
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
    # Partie 8: Démarrage de l'application en multithreading (sans blocage)
    # =================================================================
    def run(self):
        """Démarre les threads de capture, traitement et audio"""
        self.running = True
        self.last_sound_time = 0

        self.threads = []
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.threads.extend([capture_thread, processing_thread, audio_thread])

        for t in self.threads:
            t.start()

    # =================================================================
    # Partie 9: Arrêt de l'application
    # =================================================================
    def stop(self):
        """Arrête les threads et nettoie les ressources"""
        self.running = False
        time.sleep(0.5)
        self.cleanup()

    # =================================================================
    # Partie 10: Nettoyage des ressources
    # =================================================================
    def cleanup(self):
        """Libère toutes les ressources et arrête les composants"""
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        pygame.quit()
        print("Nettoyage terminé")


# =====================================================================
# Partie 11: Interface de réglage avec Tkinter
# =====================================================================
class SettingsWindow:
    def __init__(self, tracker: ConductorTracker):
        self.tracker = tracker
        self.root = tk.Tk()
        self.root.title("Réglages Conductor Tracker")

        # Création des variables Tkinter associées aux paramètres
        self.threshold_var = tk.IntVar(value=self.tracker.config['movement_threshold'])
        self.delay_var = tk.DoubleVar(value=self.tracker.config['sound_delay'])
        self.debug_var = tk.BooleanVar(value=self.tracker.config['debug_mode'])

        # Construction de l'interface
        ttk.Label(self.root, text="Seuil de mouvement (pixels):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.threshold_scale = ttk.Scale(self.root, from_=10, to=150, orient="horizontal",
                                         variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.root, text="Délai sonore (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.delay_scale = ttk.Scale(self.root, from_=0.1, to=1.0, orient="horizontal",
                                     variable=self.delay_var, command=self.update_delay)
        self.delay_scale.grid(row=1, column=1, padx=5, pady=5)

        self.debug_check = ttk.Checkbutton(self.root, text="Mode Debug", variable=self.debug_var,
                                           command=self.update_debug)
        self.debug_check.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        self.calibrate_button = ttk.Button(self.root, text="Calibrer", command=self.calibrate)
        self.calibrate_button.grid(row=3, column=0, padx=5, pady=10)

        self.start_button = ttk.Button(self.root, text="Démarrer", command=self.start_tracker)
        self.start_button.grid(row=3, column=1, padx=5, pady=10)

        self.stop_button = ttk.Button(self.root, text="Arrêter", command=self.stop_tracker)
        self.stop_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

    def update_threshold(self, event):
        new_value = self.threshold_var.get()
        self.tracker.config['movement_threshold'] = int(new_value)
        print("Nouveau seuil de mouvement :", new_value)

    def update_delay(self, event):
        new_value = self.delay_var.get()
        self.tracker.config['sound_delay'] = float(new_value)
        print("Nouveau délai sonore :", new_value)

    def update_debug(self):
        new_value = self.debug_var.get()
        self.tracker.config['debug_mode'] = new_value
        print("Mode Debug :", new_value)

    def calibrate(self):
        # Désactivation temporaire des threads si besoin
        print("Lancement de la phase de calibrage...")
        self.tracker.calibrate()
        # Mise à jour de l'interface avec le nouveau seuil
        self.threshold_var.set(self.tracker.config['movement_threshold'])

    def start_tracker(self):
        print("Démarrage du Conductor Tracker...")
        self.tracker.run()

    def stop_tracker(self):
        print("Arrêt du Conductor Tracker...")
        self.tracker.stop()
        self.root.quit()

    def run(self):
        self.root.mainloop()


# =====================================================================
# Partie 12: Point d'entrée principal
# =====================================================================
if __name__ == "__main__":
    tracker = ConductorTracker()
    settings_window = SettingsWindow(tracker)
    settings_window.run()
