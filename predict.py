## MAIN INTERFACE FOR CNN currently 
# Uses the cv2 to open the camera and uses the pretrained model
# the issue seems to be coming from poor quality of images 
# need to fix the functionality -- professor said might be because of overfitting 
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

model = load_model('trained_model_full_dataset.h5')  # Replace with your .h5 file path
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

img_size = 64

label_mapping = {i: chr(65 + i) for i in range(26)}  # Mapping 0-25 to 'A'-'Z'
label_mapping.update({26: 'del', 27: 'space', 28: 'nothing'})

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    results = hands.process(frame)
    if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get the bounding box of the hand based on landmarks
                x_min = min([lm.x for lm in landmarks.landmark]) * frame.shape[1]
                x_max = max([lm.x for lm in landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in landmarks.landmark]) * frame.shape[0]
                y_max = max([lm.y for lm in landmarks.landmark]) * frame.shape[0]

                padding = 40  
                x_min = max(0, int(x_min - padding))
                x_max = min(frame.shape[1], int(x_max + padding))
                y_min = max(0, int(y_min - padding))
                y_max = min(frame.shape[0], int(y_max + padding))
                
                # Crop the frame to focus on the hand region
                hand_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                # Ensure the image is valid before saving
                

                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Sign Prediction', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Print output when 'p' is pressed
        # Resize the cropped hand frame to match the model input size (64x64)
        hand_frame_resized = cv2.resize(hand_frame, (img_size, img_size))

        # hand_frame_resized = cv2.resize(hand_frame, (64,64))

        cv2.imwrite("hand.jpg", hand_frame_resized)  

        # Convert the image to RGB for the model
        hand_frame_rgb = cv2.cvtColor(hand_frame_resized, cv2.COLOR_BGR2RGB)          

        # Expand dimensions to match the model input
        img_array = np.expand_dims(hand_frame_rgb, axis=0)

        # Normalize the image (same as in training)
        img_array = img_array / 255.0

             

        predictions = model.predict(img_array)

        # Get the class index with the highest probability
        predicted_class = np.argmax(predictions)

        # Get the predicted label
        predicted_label = label_mapping.get(predicted_class, "Unknown")

        print(predicted_label)

        # Display the label on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('Camera Prediction', frame)


# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Back up option to change the image location
# Manually change where the imread is coming from -- Accuracy is almost 100% 
# uncomment to test this functionality 
# # Load the image
# img = cv2.imread('./asl_alphabet_test 2/V_test.jpg')
# if img is None:
#     raise ValueError(f"Image could not be loaded.")

# # Resize the image to the model's expected input size
# img_resized = cv2.resize(img, (img_size, img_size))

# # Convert to RGB (if using OpenCV, which loads images as BGR by default)
# img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# # Normalize the image
# img_normalized = img_rgb / 255.0

# # Add batch dimension
# img_array = np.expand_dims(img_normalized, axis=0)

# # Make the prediction
# predictions = model.predict(img_array)

# # Get the predicted class index
# predicted_class = np.argmax(predictions)

# # Map the predicted class to its label
# predicted_label = label_mapping.get(predicted_class)

# print(predicted_label)

## RFC STARTS HERE 
# working interface for the first 5 letters of RFC 
# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib
# from tensorflow.keras.models import load_model
# Load the trained model and scaler
# rf_classifier = joblib.load('random_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Initialize MediaPipe Hand module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_drawing = mp.solutions.drawing_utils

# # Function to preprocess hand landmarks for model input
# def preprocess_hand_landmarks(landmarks):
#     vector_points = []
#     for landmark in landmarks.landmark:
#         vector_points.append([landmark.x, landmark.y, landmark.z])
#     vector_points = np.array(vector_points).flatten()  # Flatten into a 1D array
#     return vector_points

# # Function to predict the hand sign
# def predict_hand_sign(vector_points):
#     vector_points = np.array(vector_points).reshape(1, -1)
#     vector_points = scaler.transform(vector_points)  # Normalize
#     prediction = rf_classifier.predict(vector_points)
#     return prediction[0]

# # Initialize OpenCV to capture video
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     # Convert the frame to RGB for MediaPipe processing
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     # If hand landmarks are found
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Extract the vector points (landmarks)
#             vector_points = preprocess_hand_landmarks(hand_landmarks)

#             # Predict the hand sign
#             hand_sign = predict_hand_sign(vector_points)

#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Display the predicted hand sign on the frame (optional)
#             cv2.putText(frame, f'{hand_sign}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow('Hand Sign Prediction', frame)

#     # Check for key presses
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('p'):  # Print output when 'p' is pressed
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 vector_points = preprocess_hand_landmarks(hand_landmarks)
#                 hand_sign = predict_hand_sign(vector_points)
#                 print(f'Predicted Hand Sign: {hand_sign}')
#     elif key == 27:  # Exit with the 'Esc' key
#         break

# # Release the camera and close OpenCV window
# cap.release()
# cv2.destroyAllWindows()