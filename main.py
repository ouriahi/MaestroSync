# =============================================================================
# Importation des bibliothèques nécessaires
# =============================================================================
from picamera2 import Picamera2    # Gestion de la caméra (Picamera2)
import cv2                          # Traitement d'images avec OpenCV
import mediapipe as mp              # Détection et suivi des mains via MediaPipe
import numpy as np                  # Calculs numériques et opérations sur tableaux
import pygame                       # Gestion de l'audio (lecture du métronome)
import time                         # Gestion des temps et des délais
import threading                    # Pour l'exécution de plusieurs tâches en parallèle (multithreading)
import queue                        # Communication entre threads (files d'attente)
import customtkinter as ctk         # Interface graphique (CustomTkinter)
from PIL import Image, ImageTk      # Conversion d'images pour affichage avec Tkinter
import tensorflow as tf             # Utilisation de modèles de machine learning
import os                           # Pour la gestion des fichiers et des répertoires

# =============================================================================
# Partie 1: Classe principale pour la gestion du projet
# =============================================================================
class ConductorTracker:
    def __init__(self):
        # Initialisation des attributs de la classe
        
        self.camera = None        # Instance de la caméra
        self.hands = None         # Instance de MediaPipe Hands pour la détection des mains
        self.sound = None         # Son (métromone) chargé via Pygame
        self.running = False      # État d'exécution du système

        # Files d'attente pour stocker les images (frames) et les événements audio
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)

        # Paramètres de configuration par défaut
        self.config = {
            'camera_resolution': (320, 240),           # Taille de la capture vidéo
            'hand_detection_confidence': 0.5,            # Confiance minimale pour la détection des mains
            'movement_threshold': 30,                  # Seuil de détection de mouvement (pixels)
            'sound_delay': 0.3,                        # Délai minimal entre deux sons (secondes)
            'debug_mode': True                         # Mode debug (affichage de messages de débogage)
        }

        # Historique des positions pour les mains, utilisé pour détecter les mouvements
        self.positions = {
            'left': {'x': 0, 'y': 0},   # Position initiale de la main gauche
            'right': {'x': 0, 'y': 0}   # Position initiale de la main droite
        }

        # Indicateur de l'état de la LED (pour signal visuel dans l'interface)
        self.led_on = False

        # Liste pour enregistrer les temps des battements (pour le calcul du BPM)
        self.beat_times = []

        # Label qui sera utilisé pour afficher la vidéo dans l'interface graphique (sera défini plus tard)
        self.video_label = None

        # Main par défaut utilisée pour donner le tempo (ici la main droite)
        self.tempo_hand = 'right'

        # Initialisation des composants (caméra, audio, détection de mains)
        self.initialize_components()

        # Chargement du modèle de reconnaissance des gestes (machine learning)
        self.gesture_model = self.load_gesture_model()

        # Indicateur de collecte de données
        self.collecting_data = False
        # Répertoire pour stocker les données collectées
        self.data_dir = "data/dataset"
        os.makedirs(self.data_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas
        # Liste pour stocker les données de gestes
        self.gesture_data = []
        # Liste pour stocker les labels des gestes
        self.gesture_labels = []

    # =============================================================================
    # Partie 2: Initialisation des composants
    # =============================================================================
    def initialize_components(self):
        """
        Initialise tous les composants matériels et logiciels nécessaires :
         - Audio via Pygame
         - Détection de mains avec MediaPipe
         - Capture vidéo avec Picamera2
        """
        # Initialisation de Pygame pour l'audio
        pygame.mixer.init()
        # Chargement du son (métronome)
        self.sound = pygame.mixer.Sound("media/metronome.wav")

        # Configuration de MediaPipe pour la détection des mains
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,  # Autoriser la détection de deux mains
            min_detection_confidence=self.config['hand_detection_confidence'],
            min_tracking_confidence=self.config['hand_detection_confidence']
        )

        # Configuration de la caméra avec Picamera2
        self.camera = Picamera2()
        # Création d'une configuration de prévisualisation en spécifiant la résolution et le format RGB
        config = self.camera.create_preview_configuration(
            main={"size": self.config['camera_resolution'], "format": "RGB888"}
        )
        self.camera.configure(config)  # Application de la configuration à la caméra
        self.camera.start()            # Démarrage de la capture vidéo

    def load_gesture_model(self):
        """
        Charge le modèle TensorFlow pour la reconnaissance des gestes.
        Assurez-vous que le chemin vers le modèle est correct.
        """
        model = tf.keras.models.load_model("data/gesture_model.h5")
        return model

    # =============================================================================
    # Partie 3: Calibration
    # =============================================================================
    def calibrate(self, duration=5):
        """
        Lance une phase de calibrage durant laquelle l'utilisateur effectue des mouvements standards.
        Le système enregistre les variations de position pour ajuster le seuil de détection du mouvement.
        
        :param duration: Durée de la calibration en secondes.
        """
        print(f"Calibration : Effectuez des mouvements standards pendant {duration} secondes.")
        start_time = time.time()  # Début de la calibration
        deltas = []               # Liste pour stocker les variations (delta) de position
        prev_positions = {}       # Dictionnaire pour enregistrer les positions précédentes

        # Boucle de calibration pendant la durée spécifiée
        while time.time() - start_time < duration:
            # Capture d'une frame depuis la caméra
            frame = self.camera.capture_array()
            # Conversion de l'image en RGB (pour MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Détection des mains dans la frame
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                # Pour chaque main détectée
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Récupération du type de main (gauche ou droite)
                    hand_type = handedness.classification[0].label.lower()
                    # Sélection du landmark correspondant au poignet (landmark 0)
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    # Conversion des coordonnées normalisées en pixels
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    # Calcul de la variation de position si une position précédente existe
                    if hand_type in prev_positions:
                        dx = abs(current_x - prev_positions[hand_type]['x'])
                        dy = abs(current_y - prev_positions[hand_type]['y'])
                        # Calcul de la distance euclidienne
                        delta = np.sqrt(dx*dx + dy*dy)
                        deltas.append(delta)
                    # Mise à jour de la position précédente pour la main
                    prev_positions[hand_type] = {'x': current_x, 'y': current_y}
            time.sleep(0.05)  # Pause pour limiter la fréquence de capture

        # Calcul du nouveau seuil de mouvement basé sur les variations mesurées
        if deltas:
            average_delta = sum(deltas) / len(deltas)
            # Le seuil est la moyenne multipliée par un facteur, avec un minimum de 30 pixels
            new_threshold = max(30, int(average_delta * 1.2))
            self.config['movement_threshold'] = new_threshold
            print("Calibration terminée. Nouveau seuil de mouvement :", new_threshold)
        else:
            print("Calibration échouée : aucun mouvement détecté.")

    # =============================================================================
    # Partie 4: Détection de mouvement et déclenchement audio
    # =============================================================================
    def detect_movement(self, hand_type, current_x, current_y):
        """
        Compare la position actuelle d'une main avec sa position précédente pour détecter un mouvement.
        Si le mouvement dépasse le seuil défini, un signal audio est envoyé et la LED est activée.

        :param hand_type: Type de main ('left' ou 'right')
        :param current_x: Coordonnée x actuelle
        :param current_y: Coordonnée y actuelle
        """
        # Se limiter à la main définie pour donner le tempo
        if hand_type != self.tempo_hand:
            return

        # Récupérer la dernière position enregistrée pour cette main
        prev_x = self.positions[hand_type]['x']
        prev_y = self.positions[hand_type]['y']

        movement_detected = False  # Indique si un mouvement significatif est détecté
        self.led_on = False         # Par défaut, la LED est éteinte
        directions = []             # Liste pour indiquer la direction du mouvement

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

        # Si un mouvement significatif est détecté et que le délai sonore est respecté
        if movement_detected and (time.time() - self.last_sound_time) > self.config['sound_delay']:
            # Envoi d'un événement "beep" pour jouer le son
            self.audio_queue.put("beep")
            self.last_sound_time = time.time()
            # Enregistrement du battement pour le calcul du BPM
            self.beat_times.append(self.last_sound_time)
            if self.config['debug_mode']:
                print(f"Mouvement {hand_type} détecté: {', '.join(directions)}")

        # Mise à jour des positions pour cette main
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y

    # =============================================================================
    # Partie 5: Boucle de capture vidéo (Thread dédié)
    # =============================================================================
    def capture_loop(self):
        """
        Capture en continu les frames de la caméra et les insère dans la file d'attente.
        Cette boucle s'exécute dans un thread dédié.
        """
        while self.running:
            # Capture d'une image depuis la caméra
            frame = self.camera.capture_array()
            try:
                # Essai d'ajouter la frame dans la file d'attente
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                # Si la file est pleine, ignorer la frame
                pass
            time.sleep(0.01)  # Pause pour limiter l'utilisation CPU

    # =============================================================================
    # Partie 6: Boucle de traitement d'image et affichage (Thread dédié)
    # =============================================================================
    def get_tempo_name(self, bpm):
        """
        Retourne le nom du tempo en fonction du BPM calculé.
        Permet d'associer une étiquette musicale au rythme détecté.
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

    def recognize_gesture(self, landmarks):
        """
        Utilise le modèle de machine learning pour reconnaître le geste à partir
        des landmarks (points de repère) de la main.

        :param landmarks: Liste des landmarks détectés
        :return: Index du geste reconnu
        """
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
        input_data = np.expand_dims(input_data, axis=0)
        # Prédiction du geste via le modèle TensorFlow
        prediction = self.gesture_model.predict(input_data)
        gesture = np.argmax(prediction)
        return gesture

    def get_gesture_name(self, gesture):
        """
        Retourne le nom correspondant au geste reconnu à partir de son index.

        :param gesture: Index du geste
        :return: Nom du geste
        """
        gestures = ["Levée", "Battue", "Dynamique", "Arrêt"]
        return gestures[gesture]

    def processing_loop(self):
        """
        Boucle principale de traitement des frames :
         - Récupération et conversion des images
         - Détection des mains et des gestes
         - Calcul du BPM
         - Mise à jour de l'affichage dans l'interface graphique
         
         Cette boucle s'exécute dans un thread dédié.
        """
        while self.running:
            try:
                # Récupération d'une frame depuis la file d'attente
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue  # Si aucune frame n'est disponible, continuer

            # Conversion de la frame de BGR à RGB pour MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Détection des mains dans la frame
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Pour chaque main détectée, traiter les informations
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Récupération du type de main (left/right)
                    hand_type = handedness.classification[0].label.lower()
                    # Récupération du landmark du poignet pour déterminer la position
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    # Détection du mouvement pour la main actuelle
                    self.detect_movement(hand_type, current_x, current_y)

                    # Reconnaissance du geste effectué par la main
                    gesture = self.recognize_gesture(hand_landmarks.landmark)
                    gesture_name = self.get_gesture_name(gesture)
                    # Affichage du geste reconnu sur la frame
                    cv2.putText(frame, f"Geste: {gesture_name}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Collecte des données si activée
                    if self.collecting_data:
                        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                        self.gesture_data.append(landmarks)
                        self.gesture_labels.append(self.current_gesture_label)
            
            # Calcul du BPM à partir des battements enregistrés durant les 10 dernières secondes
            current_time = time.time()
            self.beat_times = [t for t in self.beat_times if current_time - t <= 10]
            if len(self.beat_times) >= 2:
                total_time = self.beat_times[-1] - self.beat_times[0]
                avg_interval = total_time / (len(self.beat_times) - 1)
                bpm = 60 / avg_interval if avg_interval > 0 else 0
            else:
                bpm = 0

            # Récupérer le nom du tempo en fonction du BPM calculé
            tempo_name = self.get_tempo_name(bpm)

            # Dessiner une LED sur l'image pour indiquer l'état du mouvement
            led_color = (0, 255, 0) if self.led_on else (50, 50, 50)
            cv2.circle(frame, (50, 50), 20, led_color, -1)

            # Redimensionnement de l'image pour l'affichage dans l'interface
            resized_frame = cv2.resize(frame, (640, 480))
            # Affichage du BPM et du tempo sur l'image
            cv2.putText(resized_frame, f"BPM: {bpm:.1f} ({tempo_name})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Conversion de l'image pour l'affichage via Tkinter
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mise à jour du label vidéo dans l'interface graphique, si défini
            if self.video_label:
                self.video_label.imgtk = imgtk  # Conserver une référence à l'image
                self.video_label.configure(image=imgtk)

            # Gestion simple des touches clavier pour quitter ou activer le mode debug
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.config['debug_mode'] = not self.config['debug_mode']

    # =============================================================================
    # Partie 7: Boucle de gestion audio (Thread dédié)
    # =============================================================================
    def audio_loop(self):
        """
        Écoute les événements dans la file audio et joue le son du métronome
        lorsque l'événement "beep" est reçu.
        Cette boucle s'exécute dans un thread dédié.
        """
        while self.running:
            try:
                event = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if event == "beep":
                self.sound.play()

    # =============================================================================
    # Partie 8: Démarrage de l'application en multithreading
    # =============================================================================
    def run(self):
        """
        Démarre les threads pour la capture vidéo, le traitement d'image et la gestion audio.
        Permet ainsi une exécution parallèle sans blocage de l'interface.
        """
        self.running = True
        self.last_sound_time = 0  # Initialisation du temps du dernier signal sonore

        # Création et démarrage des threads pour chaque tâche
        self.threads = []
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.threads.extend([capture_thread, processing_thread, audio_thread])

        for t in self.threads:
            t.start()

    # =============================================================================
    # Partie 9: Arrêt de l'application
    # =============================================================================
    def stop(self):
        """
        Arrête l'exécution en mettant fin aux threads et en nettoyant les ressources.
        """
        self.running = False
        time.sleep(0.5)  # Temps pour permettre l'arrêt en douceur des threads
        self.cleanup()

    # =============================================================================
    # Partie 10: Nettoyage des ressources
    # =============================================================================
    def cleanup(self):
        """
        Libère les ressources utilisées par le projet, notamment :
         - Fermeture des fenêtres OpenCV
         - Arrêt de la caméra
         - Quitter Pygame
        """
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        pygame.quit()
        print("Nettoyage terminé")

    # =============================================================================
    # Partie 11: Collecte des données de gestes
    # =============================================================================
    def start_data_collection(self, gesture_label):
        """Démarre la collecte des données pour un geste spécifique."""
        self.collecting_data = True
        self.current_gesture_label = gesture_label
        print(f"Collecte des données pour le geste: {gesture_label}")

    def stop_data_collection(self):
        """Arrête la collecte des données."""
        self.collecting_data = False
        print("Collecte des données arrêtée.")
        # Sauvegarder les données collectées
        np.save(os.path.join(self.data_dir, "X_data.npy"), np.array(self.gesture_data))
        np.save(os.path.join(self.data_dir, "Y_data.npy"), np.array(self.gesture_labels))
        print("Données sauvegardées.")

# =============================================================================
# Partie 12: Interface de réglage avec CustomTkinter
# =============================================================================
class SettingsWindow:
    def __init__(self, tracker: ConductorTracker):
        """
        Crée l'interface graphique pour régler les paramètres du Conductor Tracker.
        
        :param tracker: Instance de ConductorTracker à configurer.
        """
        self.tracker = tracker
        # Création de la fenêtre principale avec CustomTkinter
        self.root = ctk.CTk()
        self.root.title("Réglages Conductor Tracker")

        # Variables liées aux paramètres de configuration
        self.threshold_var = ctk.IntVar(value=self.tracker.config['movement_threshold'])
        self.delay_var = ctk.DoubleVar(value=self.tracker.config['sound_delay'])
        self.debug_var = ctk.BooleanVar(value=self.tracker.config['debug_mode'])

        # Construction de l'interface graphique
        # Curseur pour régler le seuil de mouvement
        ctk.CTkLabel(self.root, text="Seuil de mouvement (pixels):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.threshold_scale = ctk.CTkSlider(self.root, from_=10, to=150, variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5)

        # Curseur pour régler le délai sonore
        ctk.CTkLabel(self.root, text="Délai sonore (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.delay_scale = ctk.CTkSlider(self.root, from_=0.1, to=1.0, variable=self.delay_var, command=self.update_delay)
        self.delay_scale.grid(row=1, column=1, padx=5, pady=5)

        # Checkbox pour activer/désactiver le mode debug
        self.debug_check = ctk.CTkCheckBox(self.root, text="Mode Debug", variable=self.debug_var, command=self.update_debug)
        self.debug_check.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        # Option de sélection de la main utilisée pour le tempo
        self.tempo_hand_var = ctk.StringVar(value=self.tracker.tempo_hand)
        ctk.CTkLabel(self.root, text="Main pour le tempo:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.tempo_hand_option = ctk.CTkOptionMenu(self.root, variable=self.tempo_hand_var, values=["left", "right"], command=self.update_tempo_hand)
        self.tempo_hand_option.grid(row=5, column=1, padx=5, pady=5)

        # Boutons pour lancer la calibration, démarrer et arrêter le tracker
        self.calibrate_button = ctk.CTkButton(self.root, text="Calibrer", command=self.calibrate)
        self.calibrate_button.grid(row=3, column=0, padx=5, pady=10)
        self.start_button = ctk.CTkButton(self.root, text="Démarrer", command=self.start_tracker)
        self.start_button.grid(row=3, column=1, padx=5, pady=10)
        self.stop_button = ctk.CTkButton(self.root, text="Arrêter", command=self.stop_tracker)
        self.stop_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # Label d'affichage vidéo dans l'interface
        self.tracker.video_label = ctk.CTkLabel(self.root)
        self.tracker.video_label.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

        # Boutons pour démarrer et arrêter la collecte de données
        self.collect_data_button = ctk.CTkButton(self.root, text="Démarrer Collecte", command=self.start_data_collection)
        self.collect_data_button.grid(row=6, column=0, padx=5, pady=10)
        self.stop_collect_data_button = ctk.CTkButton(self.root, text="Arrêter Collecte", command=self.stop_data_collection)
        self.stop_collect_data_button.grid(row=6, column=1, padx=5, pady=10)

        # Option pour choisir le geste à collecter
        self.gesture_var = ctk.StringVar(value="Levée")
        ctk.CTkLabel(self.root, text="Geste à collecter:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.gesture_option = ctk.CTkOptionMenu(self.root, variable=self.gesture_var, values=["Levée", "Battue", "Dynamique", "Arrêt"])
        self.gesture_option.grid(row=7, column=1, padx=5, pady=5)

    def update_threshold(self, event):
        """Mise à jour du seuil de mouvement selon la valeur du curseur."""
        new_value = self.threshold_var.get()
        self.tracker.config['movement_threshold'] = int(new_value)
        print("Nouveau seuil de mouvement :", new_value)

    def update_delay(self, event):
        """Mise à jour du délai sonore selon la valeur du curseur."""
        new_value = self.delay_var.get()
        self.tracker.config['sound_delay'] = float(new_value)
        print("Nouveau délai sonore :", new_value)

    def update_debug(self):
        """Active ou désactive le mode debug selon la checkbox."""
        new_value = self.debug_var.get()
        self.tracker.config['debug_mode'] = new_value
        print("Mode Debug :", new_value)

    def update_tempo_hand(self, event):
        """Mise à jour de la main utilisée pour donner le tempo."""
        new_value = self.tempo_hand_var.get()
        self.tracker.tempo_hand = new_value
        print("Main pour le tempo mise à jour :", new_value)

    def calibrate(self):
        """Lance la phase de calibrage et met à jour le curseur avec le nouveau seuil calculé."""
        print("Lancement de la phase de calibrage...")
        self.tracker.calibrate()
        self.threshold_var.set(self.tracker.config['movement_threshold'])

    def start_tracker(self):
        """Démarre le Conductor Tracker en lançant les threads de capture, traitement et audio."""
        print("Démarrage du Conductor Tracker...")
        self.tracker.run()

    def stop_tracker(self):
        """Arrête le Conductor Tracker et ferme l'interface graphique."""
        print("Arrêt du Conductor Tracker...")
        self.tracker.stop()
        self.root.quit()

    def start_data_collection(self):
        """Démarre la collecte des données pour le geste sélectionné."""
        gesture_label = self.gesture_var.get()
        self.tracker.start_data_collection(gesture_label)

    def stop_data_collection(self):
        """Arrête la collecte des données."""
        self.tracker.stop_data_collection()

    def run(self):
        """Lance la boucle principale de l'interface graphique."""
        self.root.mainloop()

# =============================================================================
# Partie 13: Point d'entrée principal
# =============================================================================
if __name__ == "__main__":
    # Création de l'instance principale du tracker
    tracker = ConductorTracker()
    # Création de l'interface de réglage en lui passant le tracker
    settings_window = SettingsWindow(tracker)
    # Lancement de l'interface graphique
    settings_window.run()