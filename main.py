# Importation des bibliothèques nécessaires
from picamera2 import Picamera2    # Pour la gestion de la caméra Picamera2
import cv2                          # Pour le traitement d'images (OpenCV)
import mediapipe as mp              # Pour la détection des mains (MediaPipe)
import numpy as np                  # Pour les calculs numériques
import pygame                       # Pour la gestion de l'audio
import time                         # Pour la gestion du temps et des délais
import threading                    # Pour le multithreading
import queue                        # Pour les files d'attente entre threads
import customtkinter as ctk         # Pour l'interface graphique avec CustomTkinter
from PIL import Image, ImageTk      # Pour la conversion d'images pour Tkinter
import torch                        # Utilisation de PyTorch pour les modèles de machine learning
import torch.nn as nn               # Pour la définition des modèles de réseaux de neurones

# Définition du modèle de classification des gestes
class GestureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(x)
        return out

# =====================================================================
# Partie 1: Classe principale pour la gestion du projet
# =====================================================================
class ConductorTracker:
    def __init__(self):
        # Initialisation des attributs de la classe

        self.camera = None        # Instance de la caméra
        self.hands = None         # Instance de MediaPipe Hands
        self.sound = None         # Instance du son (métromone)
        self.running = False      # État du système (en cours ou arrêté)

        # Files d'attente pour stocker les frames vidéo et les événements audio
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)

        # Paramètres de configuration par défaut
        self.config = {
            'camera_resolution': (320, 240),           # Résolution de la caméra
            'hand_detection_confidence': 0.5,            # Confiance minimale pour la détection des mains
            'movement_threshold': 30,                  # Seuil de détection de mouvement en pixels (sera ajusté par calibration)
            'sound_delay': 0.3,                        # Délai minimal entre deux sons en secondes
            'debug_mode': True                         # Mode debug activé ou non (affiche des messages de débogage)
        }

        # Historique des positions pour chaque main (pour calculer les mouvements)
        self.positions = {
            'left': {'x': 0, 'y': 0},   # Position de la main gauche
            'right': {'x': 0, 'y': 0}   # Position de la main droite
        }

        # État de la LED (pour affichage visuel sur l'interface vidéo)
        self.led_on = False

        # Liste pour enregistrer les temps de battement (pour le calcul du tempo en BPM)
        self.beat_times = []

        # Label pour afficher la vidéo dans l'interface graphique (sera défini dans SettingsWindow)
        self.video_label = None

        # Main par défaut utilisée pour donner le tempo (toujours la main gauche à cause de l'effet miroir)
        self.tempo_hand = 'left'
        self.gesture_hand = 'right'  # Main opposée pour la reconnaissance des gestes

        # Chargement du modèle de reconnaissance des gestes (machine learning)
        self.gesture_model = self.load_gesture_model()

        # Appel à la méthode d'initialisation des composants (caméra, audio, MediaPipe)
        self.initialize_components()

    def load_gesture_model(self):
        """
        Charge le modèle PyTorch pour la reconnaissance des gestes.
        Si le modèle n'est pas disponible, retourne None.
        """
        try:
            input_dim = 42  # 21 landmarks * 2 coordonnées (x, y)
            num_classes = 4  # 4 classes de gestes
            model = GestureClassifier(input_dim, num_classes)
            model.load_state_dict(torch.load("model.pth"))
            model.eval()  # Mettre le modèle en mode évaluation
            print("Modèle de reconnaissance des gestes chargé avec succès.")
            return model
        except (IOError, ValueError) as e:
            print("Aucun modèle de reconnaissance des gestes trouvé. Mode collecte de données activé.")
            return None

    def recognize_gesture(self, landmarks):
        """
        Utilise le modèle de machine learning pour reconnaître le geste à partir
        des landmarks (points de repère) de la main.
        Si le modèle n'est pas disponible, retourne None.

        :param landmarks: Liste des landmarks détectés
        :return: Index du geste reconnu ou None si le modèle n'est pas disponible
        """
        if self.gesture_model is None:
            return None

        # Préparation des données d'entrée pour le modèle
        input_data = torch.tensor([[landmark.x, landmark.y] for landmark in landmarks]).flatten().unsqueeze(0)
        # Prédiction du geste via le modèle PyTorch
        with torch.no_grad():
            prediction = self.gesture_model(input_data)
        gesture = torch.argmax(prediction).item()
        return gesture

    def get_gesture_name(self, gesture):
        """Retourne le nom du geste en fonction de l'index de la classe."""
        gesture_names = ["Arrêt", "Levée", "Crescendo", "Decrescendo"]
        return gesture_names[gesture]

    # =================================================================
    # Partie 2: Initialisation des composants
    # =================================================================
    def initialize_components(self):
        """Initialise tous les composants matériels et logiciels nécessaires au projet."""
        # Initialisation de Pygame pour la gestion audio
        pygame.mixer.init()
        # Chargement du fichier audio (métromone)
        self.sound = pygame.mixer.Sound("media/metronome.wav")

        # Configuration de MediaPipe pour la détection des mains
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,  # Détection de deux mains maximum
            min_detection_confidence=self.config['hand_detection_confidence'],
            min_tracking_confidence=self.config['hand_detection_confidence']
        )

        # Configuration de la caméra avec Picamera2
        self.camera = Picamera2()
        # Création d'une configuration de prévisualisation avec la résolution et le format RGB
        config = self.camera.create_preview_configuration(
            main={"size": self.config['camera_resolution'], "format": "RGB888"}
        )
        self.camera.configure(config)  # Application de la configuration
        self.camera.start()            # Démarrage de la caméra

    # =================================================================
    # Partie 3: Calibration
    # =================================================================
    def calibrate(self, duration=5):
        """
        Lance une phase de calibrage pendant laquelle l'utilisateur effectue quelques mouvements standards.
        Le système calcule la moyenne des variations de position et ajuste le seuil de détection de mouvement.
        """
        print(f"Calibration : Effectuez des mouvements standards pendant {duration} secondes.")
        start_time = time.time()  # Temps de début de la calibration
        deltas = []               # Liste pour stocker les variations de positions
        prev_positions = {}       # Dictionnaire pour enregistrer les positions précédentes

        # Boucle de calibration pendant la durée spécifiée
        while time.time() - start_time < duration:
            # Capture d'une frame depuis la caméra
            frame = self.camera.capture_array()
            # Conversion de l'image BGR en RGB pour MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Traitement de l'image par MediaPipe pour détecter les mains
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                # Pour chaque main détectée
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Récupération du type de main (gauche ou droite)
                    hand_type = handedness.classification[0].label.lower()
                    # Sélection du point de repère du poignet (landmark 0)
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    # Conversion des coordonnées normalisées en pixels
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    # Si une position précédente existe pour cette main, calculer la variation (delta)
                    if hand_type in prev_positions:
                        dx = abs(current_x - prev_positions[hand_type]['x'])
                        dy = abs(current_y - prev_positions[hand_type]['y'])
                        # Calcul de la distance euclidienne entre les positions
                        delta = np.sqrt(dx*dx + dy*dy)
                        deltas.append(delta)
                    # Mise à jour de la position précédente pour cette main
                    prev_positions[hand_type] = {'x': current_x, 'y': current_y}
            time.sleep(0.05)  # Petite pause pour limiter la fréquence des captures

        # Si des variations ont été enregistrées, calculer le seuil moyen
        if deltas:
            average_delta = sum(deltas) / len(deltas)
            # Application d'un facteur multiplicatif pour fixer le seuil, minimum 30 pixels
            new_threshold = max(30, int(average_delta * 1.2))
            self.config['movement_threshold'] = new_threshold
            print("Calibration terminée. Nouveau seuil de mouvement :", new_threshold)
        else:
            print("Calibration échouée : aucun mouvement détecté.")

    # =================================================================
    # Partie 4: Détection de mouvement et déclenchement audio
    # =================================================================
    def detect_movement(self, hand_type, current_x, current_y):
        """
        Détecte le mouvement en comparant la position actuelle avec la position précédente.
        Si le mouvement dépasse le seuil défini, déclenche un signal audio et active la LED.
        """
        # Se limiter à la main gauche pour donner le tempo
        if hand_type != 'left':
            return

        # Récupération des positions précédentes pour la main gauche
        prev_x = self.positions[hand_type]['x']
        prev_y = self.positions[hand_type]['y']

        movement_detected = False  # Indique si un mouvement significatif est détecté
        self.led_on = False         # Par défaut, la LED est éteinte
        directions = []             # Liste pour stocker la direction du mouvement

        # Vérification du mouvement vertical
        if abs(current_y - prev_y) > self.config['movement_threshold']:
            # Détermination de la direction verticale
            directions.append('Haut' if current_y < prev_y else 'Bas')
            movement_detected = True
            self.led_on = True  # Activer la LED si le mouvement est détecté

        # Vérification du mouvement horizontal
        if abs(current_x - prev_x) > self.config['movement_threshold']:
            # Détermination de la direction horizontale
            directions.append('Gauche' if current_x < prev_x else 'Droite')
            movement_detected = True
            self.led_on = True  # Activer la LED en cas de mouvement

        # Si un mouvement est détecté et le délai sonore est respecté
        if movement_detected and (time.time() - self.last_sound_time) > self.config['sound_delay']:
            # Envoi d'un événement "beep" dans la file d'attente audio
            self.audio_queue.put("beep")
            # Mise à jour du temps du dernier signal sonore
            self.last_sound_time = time.time()
            # Ajout du temps du battement pour le calcul du BPM
            self.beat_times.append(self.last_sound_time)
            # Affichage d'un message de débogage si le mode debug est activé
            if self.config['debug_mode']:
                print(f"Mouvement {hand_type} détecté: {', '.join(directions)}")

        # Mise à jour des positions pour cette main
        self.positions[hand_type]['x'] = current_x
        self.positions[hand_type]['y'] = current_y

    # =================================================================
    # Partie 5: Boucle de capture vidéo (Thread dédié)
    # =================================================================
    def capture_loop(self):
        """Capture en continu les frames depuis la caméra et les place dans la file d'attente."""
        while self.running:
            # Capture d'une image via la caméra
            frame = self.camera.capture_array()
            try:
                # Ajout de la frame dans la file d'attente avec un délai maximum
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                # Si la file d'attente est pleine, on ignore la frame
                pass
            time.sleep(0.01)  # Pause courte pour limiter l'utilisation CPU

    # =================================================================
    # Partie 6: Boucle de traitement d'image et affichage (Thread dédié)
    # =================================================================
    def get_tempo_name(self, bpm):
        """Retourne le nom du tempo en fonction du BPM"""
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
        """Traite les frames vidéo, détecte les mouvements, calcule le BPM et met à jour l'affichage."""
        while self.running:
            try:
                # Récupère une frame dans la file d'attente
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue  # Si aucune frame n'est disponible, on continue

            # Conversion de la frame de BGR à RGB pour MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Traitement de la frame pour détecter les mains
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Pour chaque main détectée
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Récupération du type de main
                    hand_type = handedness.classification[0].label.lower()
                    # Sélection du landmark du poignet
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    # Conversion des coordonnées normalisées en pixels
                    current_x = int(wrist.x * w)
                    current_y = int(wrist.y * h)
                    # Détection de mouvement pour la main gauche uniquement
                    self.detect_movement(hand_type, current_x, current_y)

                    # Reconnaissance du geste effectué par la main droite
                    if hand_type == self.gesture_hand:
                        gesture = self.recognize_gesture(hand_landmarks.landmark)
                        if gesture is not None:
                            gesture_name = self.get_gesture_name(gesture)
                            # Affichage du geste reconnu sur la frame
                            cv2.putText(frame, f"Geste: {gesture_name}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calcul du BPM à partir des battements enregistrés pendant les 10 dernières secondes
            current_time = time.time()
            # Filtrer les battements pour ne conserver que ceux des 10 dernières secondes
            self.beat_times = [t for t in self.beat_times if current_time - t <= 10]
            if len(self.beat_times) >= 2:
                total_time = self.beat_times[-1] - self.beat_times[0]
                avg_interval = total_time / (len(self.beat_times) - 1)
                bpm = 60 / avg_interval if avg_interval > 0 else 0
            else:
                bpm = 0

            # Obtenir le nom du tempo correspondant au BPM calculé
            tempo_name = self.get_tempo_name(bpm)

            # Dessiner une LED sur l'image pour indiquer l'état du mouvement
            # Couleur verte si un mouvement est détecté (LED allumée), sinon gris
            led_color = (0, 255, 0) if self.led_on else (50, 50, 50)
            cv2.circle(frame, (50, 50), 20, led_color, -1)

            # Redimensionnement de la frame pour l'affichage
            resized_frame = cv2.resize(frame, (640, 480))
            # Affichage du BPM et du nom du tempo sur la frame
            cv2.putText(resized_frame, f"BPM: {bpm:.1f} ({tempo_name})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Conversion de l'image pour l'affichage dans l'interface Tkinter
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mise à jour du label vidéo dans l'interface graphique si défini
            if self.video_label:
                self.video_label.imgtk = imgtk  # Conservation d'une référence à l'image
                self.video_label.configure(image=imgtk)

            # Vérification des touches (bien que l'interface soit Tkinter, on conserve une gestion basique)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.config['debug_mode'] = not self.config['debug_mode']

    # =================================================================
    # Partie 7: Boucle de gestion audio (Thread dédié)
    # =================================================================
    def audio_loop(self):
        """Écoute les événements audio dans la file d'attente et joue le son correspondant."""
        while self.running:
            try:
                # Récupère un événement audio
                event = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Si l'événement est un "beep", joue le son de métronome
            if event == "beep":
                self.sound.play()

    # =================================================================
    # Partie 8: Démarrage de l'application en multithreading (sans blocage)
    # =================================================================
    def run(self):
        """Démarre les threads pour la capture vidéo, le traitement d'image et la gestion audio."""
        self.running = True
        self.last_sound_time = 0  # Initialisation du temps du dernier signal sonore

        # Création d'une liste de threads pour chaque tâche
        self.threads = []
        # Thread pour la capture vidéo
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        # Thread pour le traitement des frames et l'affichage
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        # Thread pour la gestion audio
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.threads.extend([capture_thread, processing_thread, audio_thread])

        # Démarrage de tous les threads
        for t in self.threads:
            t.start()

    # =================================================================
    # Partie 9: Arrêt de l'application
    # =================================================================
    def stop(self):
        """Arrête les threads en cours et lance le nettoyage des ressources."""
        self.running = False
        time.sleep(0.5)  # Attente pour permettre aux threads de se terminer correctement
        self.cleanup()

    # =================================================================
    # Partie 10: Nettoyage des ressources
    # =================================================================
    def cleanup(self):
        """Libère toutes les ressources et arrête les composants utilisés."""
        cv2.destroyAllWindows()  # Ferme toutes les fenêtres OpenCV
        if self.camera:
            self.camera.stop()   # Arrête la caméra
        pygame.quit()            # Quitte Pygame
        print("Nettoyage terminé")

# =====================================================================
# Partie 11: Interface de réglage avec CustomTkinter
# =====================================================================
class SettingsWindow:
    def __init__(self, tracker: ConductorTracker):
        # Référence à l'instance de ConductorTracker
        self.tracker = tracker
        # Création de la fenêtre principale avec CustomTkinter
        self.root = ctk.CTk()
        self.root.title("Réglages Conductor Tracker")

        # Création des variables liées aux paramètres de configuration
        self.threshold_var = ctk.IntVar(value=self.tracker.config['movement_threshold'])
        self.delay_var = ctk.DoubleVar(value=self.tracker.config['sound_delay'])
        self.debug_var = ctk.BooleanVar(value=self.tracker.config['debug_mode'])

        # Construction de l'interface graphique avec des labels, curseurs et boutons

        # Label et curseur pour régler le seuil de mouvement en pixels
        ctk.CTkLabel(self.root, text="Seuil de mouvement (pixels):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.threshold_scale = ctk.CTkSlider(self.root, from_=10, to=150, variable=self.threshold_var, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5)

        # Label et curseur pour régler le délai sonore en secondes
        ctk.CTkLabel(self.root, text="Délai sonore (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.delay_scale = ctk.CTkSlider(self.root, from_=0.1, to=1.0, variable=self.delay_var, command=self.update_delay)
        self.delay_scale.grid(row=1, column=1, padx=5, pady=5)

        # Checkbox pour activer/désactiver le mode debug
        self.debug_check = ctk.CTkCheckBox(self.root, text="Mode Debug", variable=self.debug_var, command=self.update_debug)
        self.debug_check.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        # Bouton pour lancer la phase de calibrage
        self.calibrate_button = ctk.CTkButton(self.root, text="Calibrer", command=self.calibrate)
        self.calibrate_button.grid(row=3, column=0, padx=5, pady=10)

        # Bouton pour démarrer le tracker
        self.start_button = ctk.CTkButton(self.root, text="Démarrer", command=self.start_tracker)
        self.start_button.grid(row=3, column=1, padx=5, pady=10)

        # Bouton pour arrêter le tracker
        self.stop_button = ctk.CTkButton(self.root, text="Arrêter", command=self.stop_tracker)
        self.stop_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # Label pour afficher la vidéo dans l'interface (sera mis à jour par ConductorTracker)
        self.tracker.video_label = ctk.CTkLabel(self.root)
        self.tracker.video_label.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

    def update_threshold(self, event):
        """Mise à jour du seuil de mouvement dans la configuration du tracker."""
        new_value = self.threshold_var.get()
        self.tracker.config['movement_threshold'] = int(new_value)
        print("Nouveau seuil de mouvement :", new_value)

    def update_delay(self, event):
        """Mise à jour du délai sonore dans la configuration du tracker."""
        new_value = self.delay_var.get()
        self.tracker.config['sound_delay'] = float(new_value)
        print("Nouveau délai sonore :", new_value)

    def update_debug(self):
        """Mise à jour du mode debug dans la configuration du tracker."""
        new_value = self.debug_var.get()
        self.tracker.config['debug_mode'] = new_value
        print("Mode Debug :", new_value)

    def calibrate(self):
        """Lance la phase de calibrage et met à jour le seuil dans l'interface."""
        print("Lancement de la phase de calibrage...")
        self.tracker.calibrate()
        # Mise à jour du curseur avec le nouveau seuil calculé
        self.threshold_var.set(self.tracker.config['movement_threshold'])

    def start_tracker(self):
        """Démarre le Conductor Tracker."""
        print("Démarrage du Conductor Tracker...")
        self.tracker.run()

    def stop_tracker(self):
        """Arrête le Conductor Tracker et ferme l'interface."""
        print("Arrêt du Conductor Tracker...")
        self.tracker.stop()
        self.root.quit()

    def run(self):
        """Lance la boucle principale de l'interface graphique."""
        self.root.mainloop()

# =====================================================================
# Partie 12: Point d'entrée principal
# =====================================================================
if __name__ == "__main__":
    # Création de l'instance du tracker
    tracker = ConductorTracker()
    # Création de l'interface de réglage en passant le tracker
    settings_window = SettingsWindow(tracker)
    # Lancement de l'interface graphique
    settings_window.run()