import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
# Load the trained model and scaler
rf_classifier = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess hand landmarks for model input
def preprocess_hand_landmarks(landmarks):
    vector_points = []
    for landmark in landmarks.landmark:
        vector_points.append([landmark.x, landmark.y, landmark.z])
    vector_points = np.array(vector_points).flatten()  # Flatten into a 1D array
    return vector_points

# Function to predict the hand sign
def predict_hand_sign(vector_points):
    vector_points = np.array(vector_points).reshape(1, -1)
    vector_points = scaler.transform(vector_points)  # Normalize
    prediction = rf_classifier.predict(vector_points)
    return prediction[0]

# Initialize OpenCV to capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If hand landmarks are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the vector points (landmarks)
            vector_points = preprocess_hand_landmarks(hand_landmarks)

            # Predict the hand sign
            hand_sign = predict_hand_sign(vector_points)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the predicted hand sign on the frame (optional)
            cv2.putText(frame, f'{hand_sign}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Sign Prediction', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Print output when 'p' is pressed
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                vector_points = preprocess_hand_landmarks(hand_landmarks)
                hand_sign = predict_hand_sign(vector_points)
                print(f'Predicted Hand Sign: {hand_sign}')
    elif key == 27:  # Exit with the 'Esc' key
        break

# Release the camera and close OpenCV window
cap.release()
cv2.destroyAllWindows()