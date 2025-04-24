import sys
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QFileDialog, QStyle
import joblib
import time
import os

model_path = os.path.join(os.getcwd(), 'random_forest_model.pkl')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

rf_classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def preprocess_hand_landmarks(landmarks):
    vector_points = []
    for landmark in landmarks.landmark:
        vector_points.append([landmark.x, landmark.y, landmark.z])
    vector_points = np.array(vector_points).flatten()
    return vector_points

def predict_hand_sign(vector_points):
    vector_points = np.array(vector_points).reshape(1, -1)
    vector_points = scaler.transform(vector_points)
    prediction = rf_classifier.predict(vector_points)
    return prediction[0]

class VideoCaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Converter - Video Feed")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setFixedHeight(100)
        pixmap = QPixmap("Images/logo.jpg")
        pixmap = pixmap.scaledToHeight(100, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.layout.addWidget(self.logo_label)

        self.asl_label = QLabel("ASL Letters:", self)
        self.layout.addWidget(self.asl_label)

        self.asl_text_box = QTextEdit(self)
        self.asl_text_box.setReadOnly(True)
        self.asl_text_box.setStyleSheet("background-color: #f0f0f0; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.asl_text_box.setFixedHeight(60)
        self.layout.addWidget(self.asl_text_box)

        self.video_layout = QVBoxLayout()
        self.video_layout.setAlignment(Qt.AlignCenter)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(680, 460)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.video_label)
        self.layout.addLayout(self.video_layout)

        self.chat_feed = QTextEdit(self)
        self.chat_feed.setReadOnly(True)
        self.chat_feed.setStyleSheet("background-color: #f0f0f0; font-size: 14px; padding: 5px; border-radius: 5px; color: black;")
        self.chat_feed.setFixedHeight(100)
        self.layout.addWidget(self.chat_feed)

        text_box_container = QHBoxLayout()
        text_box_container.addStretch(1)

        text_box_layout = QHBoxLayout()
        text_box_layout.setSpacing(10)

        self.add_image_button = QPushButton("+", self)
        self.add_image_button.setStyleSheet("""
            background-color: #d3d3d3;
            color: black;
            font-size: 24px;
            font-weight: bold;
            border-radius: 30px;
            min-width: 60px;
            min-height: 60px;
            max-width: 60px;
            max-height: 60px;
        """)
        self.add_image_button.clicked.connect(self.add_image)
        text_box_layout.addWidget(self.add_image_button)

        self.text_box = QTextEdit(self)
        self.text_box.setPlaceholderText("Type here...")
        self.text_box.setStyleSheet("border: 2px solid #0078D4; border-radius: 5px; padding: 5px;")
        self.text_box.setFixedHeight(60)
        text_box_layout.addWidget(self.text_box)

        self.send_button = QPushButton("", self)
        self.send_button.setStyleSheet("""
            background-color: #0078D4;
            border: none;
            border-radius: 30px;
            min-width: 60px;
            min-height: 60px;
            max-width: 60px;
            max-height: 60px;
        """)
        self.send_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_ArrowUp')))
        text_box_layout.addWidget(self.send_button)

        text_box_container.addLayout(text_box_layout, 8)
        text_box_container.addStretch(1)
        self.layout.addLayout(text_box_container)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)

        def create_button(symbol, label_text, color):
            container = QVBoxLayout()
            button = QPushButton(symbol, self)
            button.setStyleSheet(f"""
                background-color: {color};
                color: {'black' if color == 'white' else 'white'};
                font-size: 20px;
                font-weight: bold;
                border-radius: 35px;
                min-width: 70px;
                min-height: 70px;
                max-width: 70px;
                max-height: 70px;
            """
            )
            container.addWidget(button, alignment=Qt.AlignCenter)
            label = QLabel(label_text, self)
            label.setAlignment(Qt.AlignCenter)
            container.addWidget(label)
            return container, button

        call_layout, self.start_button = create_button("▶", "Call", "#28a745")
        mute_layout, self.mute_button = create_button("🔇", "Mute", "white")
        share_layout, _ = create_button("🖥️", "Share", "white")
        extra_layout, _ = create_button("⋯", "More", "white")
        end_layout, self.stop_button = create_button("✖", "End", "#dc3545")

        button_layout.addLayout(call_layout)
        button_layout.addLayout(mute_layout)
        button_layout.addLayout(share_layout)
        button_layout.addLayout(extra_layout)
        button_layout.addLayout(end_layout)

        self.layout.addLayout(button_layout)

        self.start_button.clicked.connect(self.start_video_stream)
        self.stop_button.clicked.connect(self.stop_video_stream)
        self.send_button.clicked.connect(self.send_message)
        self.mute_button.clicked.connect(self.toggle_mute)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)

        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'female' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.muted = False
        self.detected_text = ""
        self.last_spoken_letter = ""

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        self.last_detection_time = 0
        self.detection_interval = 3

        self.csun_logo = QLabel(self)
        self.csun_logo.setAlignment(Qt.AlignCenter)
        pixmap_cs = QPixmap("csunlogo.png")
        pixmap_cs = pixmap_cs.scaledToWidth(700, Qt.SmoothTransformation)
        screen_width = self.frameGeometry().width()
        scaled_width = min(pixmap_cs.width(), screen_width - 20)
        pixmap_cs = pixmap_cs.scaledToWidth(scaled_width, Qt.SmoothTransformation)
        self.csun_logo.setPixmap(pixmap_cs)
        self.layout.addWidget(self.csun_logo)

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.mute_button.setText("🔈")
            self.mute_button.setToolTip("Unmute")
        else:
            self.mute_button.setText("🔇")
            self.mute_button.setToolTip("Mute")

    def add_image(self):
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if image_path:
            image = QPixmap(image_path)
            scaled_image = image.scaled(100, 100, Qt.KeepAspectRatio)
            self.asl_text_box.append(f'<img src="{image_path}" width="100" height="100">')

    def start_video_stream(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not access webcam.")
            return
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_video_stream(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.clear()

    def send_message(self):
        message = self.text_box.toPlainText()
        if message.strip():
            current_chat = self.chat_feed.toPlainText()
            new_chat = f"{current_chat}\nYou: {message}"
            self.chat_feed.setPlainText(new_chat)
            self.text_box.clear()

    def update_video_feed(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                        vector_points = preprocess_hand_landmarks(landmarks)
                        current_time = time.time()
                        if current_time - self.last_detection_time >= self.detection_interval:
                            prediction = predict_hand_sign(vector_points)
                            self.last_detection_time = current_time
                            if prediction and prediction != self.last_spoken_letter:
                                self.detected_text += prediction + " "
                                if not self.muted:
                                    self.engine.say(prediction)
                                    self.engine.runAndWait()
                                self.last_spoken_letter = prediction
                            self.asl_text_box.setPlainText(self.detected_text.strip())

                h, w, c = frame.shape
                bytes_per_line = c * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoCaptureWindow()
    window.show()
    sys.exit(app.exec_())
