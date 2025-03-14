import cv2
import numpy as np
import time
import queue
import threading
import os
from picamera2 import Picamera2
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk

class DataCollector:
    def __init__(self):
        self.camera = None
        self.hands = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.collecting_data = False
        self.data_dir = "data/dataset"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.gesture_data = []
        self.gesture_labels = []
        self.current_gesture_label = None
        self.collecting_hand = 'left'

        self.initialize_components()

    def initialize_components(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (320, 240), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()

    def start_data_collection(self, gesture_label, hand):
        self.collecting_data = True
        self.current_gesture_label = gesture_label
        self.collecting_hand = hand
        print(f"Collecte des données pour le geste: {gesture_label} avec la main: {hand}")

    def stop_data_collection(self):
        self.collecting_data = False
        print("Collecte des données arrêtée.")
        generated_gesture_data = []
        generated_gesture_labels = []
        for landmarks, label in zip(self.gesture_data, self.gesture_labels):
            generated_gesture_data.append(landmarks)
            generated_gesture_labels.append(label)
            mirrored_landmarks = [[-x, y, z] for x, y, z in landmarks]
            generated_gesture_data.append(mirrored_landmarks)
            generated_gesture_labels.append(label)
        np.save(os.path.join(self.data_dir, "X_data.npy"), np.array(generated_gesture_data))
        np.save(os.path.join(self.data_dir, "Y_data.npy"), np.array(generated_gesture_labels))
        print("Données sauvegardées.")

    def capture_loop(self):
        while self.running:
            frame = self.camera.capture_array()
            try:
                self.frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                pass
            time.sleep(0.01)

    def processing_loop(self):
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
                    h, w = frame.shape[:2]
                    if self.collecting_data and hand_type == self.collecting_hand:
                        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                        self.gesture_data.append(landmarks)
                        self.gesture_labels.append(self.current_gesture_label)
                        for landmark in hand_landmarks.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            resized_frame = cv2.resize(frame, (640, 480))
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            if self.video_label:
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

    def run(self):
        self.running = True
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        capture_thread.start()
        processing_thread.start()

    def stop(self):
        self.running = False
        time.sleep(0.5)
        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        print("Nettoyage terminé")

class SettingsWindow:
    def __init__(self, collector: DataCollector):
        self.collector = collector
        self.root = ctk.CTk()
        self.root.title("Collecte de données")

        self.collecting_hand_var = ctk.StringVar(value=self.collector.collecting_hand)
        self.create_widgets()

    def create_widgets(self):
        data_frame = ctk.CTkFrame(self.root)
        data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.collect_data_button = ctk.CTkButton(data_frame, text="Démarrer Collecte", command=self.start_data_collection)
        self.collect_data_button.grid(row=0, column=0, padx=5, pady=10)
        self.stop_collect_data_button = ctk.CTkButton(data_frame, text="Arrêter Collecte", command=self.stop_data_collection)
        self.stop_collect_data_button.grid(row=0, column=1, padx=5, pady=10)

        ctk.CTkLabel(data_frame, text="Geste à collecter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.gesture_var = ctk.StringVar(value="Levée")
        self.gesture_option = ctk.CTkOptionMenu(data_frame, variable=self.gesture_var, values=["Levée", "Crescendo", "Diminuendo", "Arrêt"])
        self.gesture_option.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(data_frame, text="Main pour la collecte:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.collecting_hand_option = ctk.CTkOptionMenu(data_frame, variable=self.collecting_hand_var, values=["left", "right"])
        self.collecting_hand_option.grid(row=2, column=1, padx=5, pady=5)

        video_frame = ctk.CTkFrame(self.root)
        video_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.collector.video_label = ctk.CTkLabel(video_frame)
        self.collector.video_label.grid(row=0, column=0, padx=5, pady=10)

    def start_data_collection(self):
        gesture_label = self.gesture_var.get()
        collecting_hand = self.collecting_hand_var.get()
        self.collector.start_data_collection(gesture_label, collecting_hand)

    def stop_data_collection(self):
        self.collector.stop_data_collection()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    collector = DataCollector()
    settings_window = SettingsWindow(collector)
    collector.run()
    settings_window.run()
