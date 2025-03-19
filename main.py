"""
MaestroSynch
-----------------
Application de suivi des mouvements du chef d'orchestre basée sur Picamera2, OpenCV, MediaPipe, Pygame et CustomTkinter.
Permet de détecter les mouvements des mains, de calibrer le seuil de mouvement, d'afficher le tempo (BPM) et de jouer un son de métronome.
"""

# =============================================================================
# Importation des bibliothèques nécessaires
# =============================================================================
from picamera2 import Picamera2    # Gestion de la caméra Picamera2
import cv2                          # Traitement d'images avec OpenCV
import mediapipe as mp              # Détection et suivi des mains via MediaPipe
import numpy as np                  # Calculs numériques
import pygame                       # Gestion de l'audio
import time                         # Gestion du temps et des délais
import threading                    # Multithreading pour l'exécution simultanée
import queue                        # File d'attente pour la communication inter-thread
import customtkinter as ctk         # Interface graphique CustomTkinter
from PIL import Image, ImageTk      # Conversion d'images pour Tkinter

# =============================================================================
# Classe principale: ConductorTracker
# =============================================================================
class ConductorTracker:
    """
    Classe principale gérant l'initialisation des composants, la calibration,
    la détection des mouvements et gestes, le traitement vidéo, l'audio et l'affichage.
    """
    def __init__(self):
        # -----------------------------
        # Attributs matériels et logiciels
        # -----------------------------
        self.camera = None        # Instance de la caméra
        self.hands = None         # Instance MediaPipe pour la détection des mains
        self.sound = None         # Son (métromone) chargé via Pygame
        self.running = False      # Indique si le système est en cours d'exécution

        # Files d'attente pour la communication entre threads
        self.frame_queue = queue.Queue(maxsize=10)   # Stockage des frames vidéo
        self.audio_queue = queue.Queue(maxsize=10)   # Stockage des événements audio

        # Paramètres de configuration par défaut
        self.config = {
            'camera_resolution': (320, 240),           # Résolution de la caméra
            'hand_detection_confidence': 0.5,          # Seuil de confiance pour MediaPipe
            'movement_threshold': 30,                  # Seuil de détection de mouvement (pixels)
            'sound_delay': 0.3,                        # Délai minimal entre deux sons (secondes)
            'debug_mode': True                         # Activation du mode debug
        }

        # Historique des positions des mains (pour calculer les mouvements)
        self.positions = {
            'left': {'x': 0, 'y': 0},   # Main gauche
            'right': {'x': 0, 'y': 0}   # Main droite
        }

        self.led_on = False          # État de la LED pour l'affichage visuel
        self.beat_times = []         # Historique des temps de battement (pour le BPM)
        self.video_label = None      # Label Tkinter pour afficher la vidéo

        # Attribution des rôles des mains :
        self.tempo_hand = 'left'     # Main pour le calcul du tempo (effet miroir)
        self.gesture_hand = 'right'  # Main pour la reconnaissance de gestes (effet miroir)

        # Initialisation des composants (caméra, audio, MediaPipe)
        self.initialize_components()

    # =========================================================================
    # Initialisation des composants matériels et logiciels
    # =========================================================================
    def initialize_components(self):
        """
        Initialise Pygame (audio), MediaPipe (détection des mains) et la caméra (Picamera2).
        """
        # Initialisation de Pygame pour l'audio
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("media/metronome.wav")  # Chargement du son du métromone

        # Configuration de MediaPipe pour la détection de mains
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=self.config['hand_detection_confidence'],
            min_tracking_confidence=self.config['hand_detection_confidence']
        )

        # Configuration de la caméra avec Picamera2
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": self.config['camera_resolution'], "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()

    # =========================================================================
    # Calibration du seuil de détection de mouvement
    # =========================================================================
    def calibrate(self, duration=5):
        """
        Lance une phase de calibration pendant laquelle l'utilisateur effectue des mouvements standards.
        Le système calcule la moyenne des variations et ajuste le seuil de détection.

        :param duration: Durée de la calibration en secondes (par défaut 5)
        """
        print(f"Calibration : Effectuez des mouvements standards pendant {duration} secondes.")
        start_time = time.time()
        deltas = []             # Stocke les variations de position
        prev_positions = {}     # Positions précédentes pour chaque main

        while time.time() - start_time < duration:
            frame = self.camera.capture_array()  # Capture d'une frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label.lower()
                    wrist = hand_landmarks.landmark[0]  # Utilisation du repère du poignet
                    h, w = frame.shape[:2]
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)

                    if hand_type in prev_positions:
                        dx = abs(current_x - prev_positions[hand_type]['x'])
                        dy = abs(current_y - prev_positions[hand_type]['y'])
                        delta = np.sqrt(dx*dx + dy*dy)
                        deltas.append(delta)

                    prev_positions[hand_type] = {'x': current_x, 'y': current_y}

            time.sleep(0.05)  # Délai pour limiter la fréquence des captures

        if deltas:
            average_delta = sum(deltas) / len(deltas)
            new_threshold = max(30, int(average_delta * 1.2))  # Application d'un facteur multiplicatif
            self.config['movement_threshold'] = new_threshold
            print("Calibration terminée. Nouveau seuil de mouvement :", new_threshold)
        else:
            print("Calibration échouée : aucun mouvement détecté.")

    # =========================================================================
    # Détection de mouvement et déclenchement audio pour la main de tempo
    # =========================================================================
    def detect_movement(self, hand_type, current_x, current_y):
        """
        Compare la position actuelle avec la précédente pour détecter un mouvement significatif.
        Si le seuil est dépassé et que le délai sonore est respecté, un signal audio est déclenché.

        :param hand_type: Type de main détectée ('left' ou 'right')
        :param current_x: Coordonnée x actuelle
        :param current_y: Coordonnée y actuelle
        """
        # Se limiter à la main utilisée pour le tempo (généralement la main gauche)
        if hand_type != 'left':
            return

        prev_x = self.positions[hand_type]['x']
        prev_y = self.positions[hand_type]['y']
        movement_detected = False
        self.led_on = False      # Réinitialisation de l'état de la LED
        directions = []          # Stockage des directions de mouvement

        # Vérification du mouvement vertical
        if abs(current_y - prev_y) > self.config['movement_threshold']:
            directions.append('Haut' if current_y < prev_y else 'Bas')
            movement_detected = True
            self.led_on = True

        # Vérification du mouvement horizontal
        if abs(current_x - prev_x) > self.config['movement_threshold']:
            directions.append('Gauche' if current_x < prev_x else 'Droite')
            movement_detected = True
            self.led_on = True

        # Déclenchement du son si un mouvement est détecté et que le délai est respecté
        if movement_detected and (time.time() - self.last_sound_time) > self.config['sound_delay']:
            self.audio_queue.put("beep")
            self.last_sound_time = time.time()
            self.beat_times.append(self.last_sound_time)
            if self.config['debug_mode']:
                print(f"Mouvement {hand_type} détecté: {', '.join(directions)}")

        # Mise à jour des positions
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y

    # =========================================================================
    # Reconnaissance des gestes pour la main dédiée aux gestes
    # =========================================================================
    def recognize_gesture(self, hand_type, current_x, current_y, landmarks):
        """
        Analyse l'état des doigts pour reconnaître certains gestes (ex. Crescendo, Decrescendo, Silence, Levee)
        uniquement pour la main droite.

        :param hand_type: Type de main détectée (seul 'right' est traité)
        :param current_x: Coordonnée x actuelle
        :param current_y: Coordonnée y actuelle
        :param landmarks: Liste des repères détectés par MediaPipe
        :return: Nom du geste détecté ou None
        """
        if hand_type != 'right':
            return None

        prev_y = self.positions[hand_type]['y']
        fingers = []    # Etat des doigts (1 pour levé, 0 pour baissé)
        tips = [4, 8, 12, 16, 20]  # Indices des extrémités des doigts

        for tip in tips:
            # Si la pointe du doigt est plus haute que la jointure située deux positions avant
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        gesture = None
        # Exemple de condition pour détecter un geste de crescendo/decrescendo
        if abs(current_x - prev_y) > self.config['movement_threshold'] and fingers == [1, 1, 1, 1, 1]:
            gesture = 'Crescendo' if current_y < prev_y else 'Decrescendo'
        # Conditions supplémentaires pour d'autres gestes
        if fingers == [0 or 1, 0, 0, 0, 0]:
            gesture = 'Silence'
        if fingers == [0 or 1, 1, 0, 0, 0]:
            gesture = 'Levee'

        # Mise à jour de la position pour la main droite
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y

        return gesture

    # =========================================================================
    # Boucle de capture vidéo (Thread dédié)
    # =========================================================================
    def capture_loop(self):
        """
        Capture continue des frames depuis la caméra et les place dans la file d'attente.
        """
        while self.running:
            frame = self.camera.capture_array()
            try:
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                # Si la file est pleine, la frame est ignorée
                pass
            time.sleep(0.01)

    # =========================================================================
    # Traitement des frames, calcul du BPM et affichage (Thread dédié)
    # =========================================================================
    def get_tempo_name(self, bpm):
        """
        Retourne le nom du tempo correspondant à la valeur du BPM.

        :param bpm: Battements par minute calculés
        :return: Nom du tempo
        """
        if bpm <= 25:
            return "Larghissimo"
        elif bpm <= 40:
            return "Grave"
        elif bpm <= 45:
            return "Grave / Largo"
        elif bpm <= 60:
            return "Largo / Lento"
        elif bpm <= 66:
            return "Larghetto"
        elif bpm <= 72:
            return "Adagio"
        elif bpm <= 76:
            return "Adagietto"
        elif bpm <= 80:
            return "Andante"
        elif bpm <= 92:
            return "Andantino"
        elif bpm <= 108:
            return "Andante Moderato"
        elif bpm <= 112:
            return "Moderato"
        elif bpm <= 116:
            return "Allegretto"
        elif bpm <= 120:
            return "Allegro Moderato"
        elif bpm <= 140:
            return "Allegro"
        elif bpm <= 168:
            return "Vivace"
        elif bpm <= 172:
            return "Presto"
        elif bpm <= 176:
            return "Vivacissimo / Allegrissimo"
        elif bpm <= 200:
            return "Presto"
        else:
            return "Prestissimo"

    def processing_loop(self):
        """
        Traite les frames issues de la file d'attente, détecte les mouvements et gestes,
        calcule le BPM et met à jour l'affichage vidéo.
        """
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Conversion de l'image en RGB pour MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Traitement des détections de mains
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label.lower()
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)

                    # Détection de mouvement pour la main de tempo (gauche)
                    self.detect_movement(hand_type, current_x, current_y)

                    # Détection de geste pour la main dédiée aux gestes (droite)
                    if hand_type == self.gesture_hand:
                        gesture = self.recognize_gesture(hand_type, current_x, current_y, hand_landmarks.landmark)
                        if gesture is not None:
                            cv2.putText(frame, f"Geste: {gesture}", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calcul du BPM à partir des temps de battement enregistrés durant les 10 dernières secondes
            current_time = time.time()
            self.beat_times = [t for t in self.beat_times if current_time - t <= 10]
            if len(self.beat_times) >= 2:
                total_time = self.beat_times[-1] - self.beat_times[0]
                avg_interval = total_time / (len(self.beat_times) - 1)
                bpm = 60 / avg_interval if avg_interval > 0 else 0
            else:
                bpm = 0

            tempo_name = self.get_tempo_name(bpm)

            # Affichage de la LED : verte si mouvement détecté, sinon gris
            led_color = (0, 255, 0) if self.led_on else (50, 50, 50)
            cv2.circle(frame, (50, 50), 20, led_color, -1)

            # Redimensionnement de la frame et ajout du texte du BPM
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.putText(resized_frame, f"BPM: {bpm:.1f} ({tempo_name})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Conversion de la frame pour affichage avec Tkinter
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            if self.video_label:
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Gestion des entrées clavier (q: quitter, d: toggle debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.config['debug_mode'] = not self.config['debug_mode']

    # =========================================================================
    # Boucle de gestion audio (Thread dédié)
    # =========================================================================
    def audio_loop(self):
        """
        Surveille la file d'attente audio et joue le son du métromone pour chaque événement "beep".
        """
        while self.running:
            try:
                event = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if event == "beep":
                self.sound.play()

    # =========================================================================
    # Démarrage de l'application en mode multithreading
    # =========================================================================
    def run(self):
        """
        Démarre les threads de capture vidéo, traitement d'image et gestion audio.
        """
        self.running = True
        self.last_sound_time = 0  # Initialisation du temps du dernier signal sonore

        # Création et démarrage des threads dédiés
        self.threads = []
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.threads.extend([capture_thread, processing_thread, audio_thread])

        for t in self.threads:
            t.start()

    # =========================================================================
    # Arrêt de l'application
    # =========================================================================
    def stop(self):
        """
        Arrête l'exécution, attend la terminaison des threads puis procède au nettoyage des ressources.
        """
        self.running = False
        time.sleep(0.5)  # Attente pour une terminaison correcte des threads
        self.cleanup()

    # =========================================================================
    # Nettoyage des ressources utilisées
    # =========================================================================
    def cleanup(self):
        """
        Libère les ressources : ferme les fenêtres OpenCV, arrête la caméra et quitte Pygame.
        """
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        pygame.quit()
        print("Nettoyage terminé")


# =============================================================================
# Classe pour l'interface graphique de réglage (CustomTkinter)
# =============================================================================
class SettingsWindow:
    """
    Fenêtre de réglage permettant de modifier les paramètres du Conductor Tracker,
    de lancer la calibration et de démarrer/arrêter l'application.
    """
    def __init__(self, tracker: ConductorTracker):
        self.tracker = tracker  # Référence à l'instance de ConductorTracker

        # Création de la fenêtre principale
        self.root = ctk.CTk()
        self.root.title("Réglages Conductor Tracker")

        # Variables liées aux paramètres
        self.threshold_var = ctk.IntVar(value=self.tracker.config['movement_threshold'])
        self.delay_var = ctk.DoubleVar(value=self.tracker.config['sound_delay'])
        self.debug_var = ctk.BooleanVar(value=self.tracker.config['debug_mode'])

        # Construction de l'interface graphique
        # --- Seuil de mouvement
        ctk.CTkLabel(self.root, text="Seuil de mouvement (pixels):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.threshold_scale = ctk.CTkSlider(self.root, from_=10, to=150, variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5)

        # --- Délai sonore
        ctk.CTkLabel(self.root, text="Délai sonore (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.delay_scale = ctk.CTkSlider(self.root, from_=0.1, to=1.0, variable=self.delay_var, command=self.update_delay)
        self.delay_scale.grid(row=1, column=1, padx=5, pady=5)

        # --- Mode Debug
        self.debug_check = ctk.CTkCheckBox(self.root, text="Mode Debug", variable=self.debug_var, command=self.update_debug)
        self.debug_check.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        # --- Boutons de calibration et de démarrage/arrêt
        self.calibrate_button = ctk.CTkButton(self.root, text="Calibrer", command=self.calibrate)
        self.calibrate_button.grid(row=3, column=0, padx=5, pady=10)
        self.start_button = ctk.CTkButton(self.root, text="Démarrer", command=self.start_tracker)
        self.start_button.grid(row=3, column=1, padx=5, pady=10)
        self.stop_button = ctk.CTkButton(self.root, text="Arrêter", command=self.stop_tracker)
        self.stop_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # --- Zone d'affichage de la vidéo
        self.tracker.video_label = ctk.CTkLabel(self.root)
        self.tracker.video_label.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

    def update_threshold(self, event):
        """Met à jour le seuil de mouvement dans la configuration du tracker."""
        new_value = self.threshold_var.get()
        self.tracker.config['movement_threshold'] = int(new_value)
        print("Nouveau seuil de mouvement :", new_value)

    def update_delay(self, event):
        """Met à jour le délai sonore dans la configuration du tracker."""
        new_value = self.delay_var.get()
        self.tracker.config['sound_delay'] = float(new_value)
        print("Nouveau délai sonore :", new_value)

    def update_debug(self):
        """Active/désactive le mode debug dans la configuration du tracker."""
        new_value = self.debug_var.get()
        self.tracker.config['debug_mode'] = new_value
        print("Mode Debug :", new_value)

    def calibrate(self):
        """Lance la phase de calibration et met à jour le seuil affiché dans l'interface."""
        print("Lancement de la phase de calibrage...")
        self.tracker.calibrate()
        self.threshold_var.set(self.tracker.config['movement_threshold'])

    def start_tracker(self):
        """Démarre le Conductor Tracker."""
        print("Démarrage du Conductor Tracker...")
        self.tracker.run()

    def stop_tracker(self):
        """Arrête le Conductor Tracker et ferme l'interface graphique."""
        print("Arrêt du Conductor Tracker...")
        self.tracker.stop()
        self.root.quit()

    def run(self):
        """Lance la boucle principale de l'interface graphique."""
        self.root.mainloop()


# =============================================================================
# Point d'entrée principal de l'application
# =============================================================================
if __name__ == "__main__":
    tracker = ConductorTracker()            # Création de l'instance principale
    settings_window = SettingsWindow(tracker) # Création de l'interface de réglage
    settings_window.run()                     # Lancement de l'interface graphique