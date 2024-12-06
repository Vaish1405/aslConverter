import sys
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QFileDialog
import joblib
import time  

# Load the trained model and scaler
rf_classifier = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_hand_landmarks(landmarks):
    vector_points = []
    for landmark in landmarks.landmark:
        vector_points.append([landmark.x, landmark.y, landmark.z])
    vector_points = np.array(vector_points).flatten()  # Flatten into a 1D array
    return vector_points

def predict_hand_sign(vector_points):
    vector_points = np.array(vector_points).reshape(1, -1)
    vector_points = scaler.transform(vector_points)  # Normalize
    prediction = rf_classifier.predict(vector_points)
    return prediction[0]

class VideoCaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("ASL Converter - Video Feed")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Logo section
        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setFixedHeight(100)  
        pixmap = QPixmap("Images/logo.jpg")  
        pixmap = pixmap.scaledToHeight(100, Qt.SmoothTransformation)  
        self.logo_label.setPixmap(pixmap)
        self.layout.addWidget(self.logo_label)

        # ASL detection result section
        self.asl_label = QLabel("ASL Letters:", self)
        self.layout.addWidget(self.asl_label)

        # ASL detected text box
        self.asl_text_box = QTextEdit(self)
        self.asl_text_box.setReadOnly(True)
        self.asl_text_box.setStyleSheet("background-color: #f0f0f0; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.asl_text_box.setFixedHeight(60)  
        self.layout.addWidget(self.asl_text_box)

        # Video feed section
        self.video_layout = QVBoxLayout()
        self.video_layout.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(680, 460)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.video_label)

        self.layout.addLayout(self.video_layout)

        # Text section (user message input)
        self.chat_feed = QTextEdit(self)
        self.chat_feed.setReadOnly(True)
        self.chat_feed.setStyleSheet("background-color: #f0f0f0; font-size: 14px; padding: 5px; border-radius: 5px; color: black;")
        self.chat_feed.setFixedHeight(100)  
        self.layout.addWidget(self.chat_feed)

        # Text box and send button for user message input
        text_box_layout = QHBoxLayout()
        
        # Circular "+" Button for adding images
        self.add_image_button = QPushButton("+", self)
        self.add_image_button.setStyleSheet("""
            background-color: #0078D4;
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 70px;
            width: 15px;
            height: 30px;
        """)
        self.add_image_button.clicked.connect(self.add_image)
        text_box_layout.addWidget(self.add_image_button, stretch=1)

        self.text_box = QTextEdit(self)
        self.text_box.setPlaceholderText("Type here...")
        self.text_box.setStyleSheet("border: 2px solid #0078D4; border-radius: 5px; padding: 5px;")
        self.text_box.setFixedHeight(60)  
        text_box_layout.addWidget(self.text_box, stretch=5)

        self.send_button = QPushButton("Send", self)
        self.send_button.setStyleSheet("""
            background-color: #0078D4;
            color: white;
            font-size: 14px;
            padding: 5px;
            border-radius: 5px;
        """)
        self.send_button.setFixedWidth(80)  # Fixed width for the send button
        text_box_layout.addWidget(self.send_button, stretch=1)
        self.layout.addLayout(text_box_layout)

        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Video Stream", self)
        self.stop_button = QPushButton("Stop Video Stream", self)
        
        self.start_button.setStyleSheet("""
            background-color: #28a745;
            color: white;
            font-size: 14px;
            padding: 5px;
            border-radius: 5px;
        """)
        self.stop_button.setStyleSheet("""
            background-color: #dc3545;
            color: white;
            font-size: 14px;
            padding: 5px;
            border-radius: 5px;
        """)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.layout.addLayout(button_layout)

        # Connect buttons to actions
        self.start_button.clicked.connect(self.start_video_stream)
        self.stop_button.clicked.connect(self.stop_video_stream)
        self.send_button.clicked.connect(self.send_message)

        # Timer for video feed
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)

        # TTS engine
        self.engine = pyttsx3.init()
        self.detected_text = ""

        # Mediapipe hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Add timestamp for rate-limiting detection
        self.last_detection_time = 0
        self.detection_interval = 3  # Interval in seconds between detections

       # Centered image at the bottom
        self.csun_logo = QLabel(self)
        self.csun_logo.setAlignment(Qt.AlignCenter)

        # Centered image at the bottom
        self.csun_logo = QLabel(self)
        self.csun_logo.setAlignment(Qt.AlignCenter)

        # Load the CSUN logo image
        pixmap_cs = QPixmap("csunlogo.png")  # Path to your CSUN logo
        pixmap_cs = pixmap_cs.scaledToWidth(700, Qt.SmoothTransformation)  # Scale logo width to fit

        # Ensure the logo fits within the screen
        screen_width = self.frameGeometry().width()
        scaled_width = min(pixmap_cs.width(), screen_width - 20)  # Adjust for screen width with padding
        pixmap_cs = pixmap_cs.scaledToWidth(scaled_width, Qt.SmoothTransformation)

        self.csun_logo.setPixmap(pixmap_cs)

        # Add the logo label to the bottom of the layout
        self.layout.addWidget(self.csun_logo)

    def add_image(self):
        # Open file dialog to select an image
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
        if set:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Preprocess the hand landmarks
                    vector_points = preprocess_hand_landmarks(landmarks)

                    # Get the current time to rate-limit predictions
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.detection_interval:
                        prediction = predict_hand_sign(vector_points)
                        self.last_detection_time = current_time  

                        # Append the predicted letter to the detected text
                        if prediction:  # Ensure prediction is not empty
                            self.detected_text += prediction + " "

                        # Update the ASL text box with the accumulated text
                        self.asl_text_box.setPlainText(self.detected_text.strip())

            # Convert frame to QImage and update the video label
            h, w, c = frame.shape
            bytes_per_line = c * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

# Main 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoCaptureWindow()
    window.show()
    sys.exit(app.exec_()) 