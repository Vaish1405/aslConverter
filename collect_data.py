import cv2
import mediapipe as mp
import csv

cap = cv2.VideoCapture(0)  # 0 -> default, any other number -> the camera number

# concentrate only on the hands (for now)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils # Draw the vector points so visible on screen

# Save hand vector points of the hands into a file
def save_landmarks(hand_landmarks, file):
    """
    Save hand vector points in the csv file

    Args:
        hand_landmarks: Vector points for each hand sign.
        file: name of csv file to store information in.
    """
    with open(file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Vector points -> array
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        writer.writerow(landmarks)

file_path = "hand_landmarks.csv"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find the vector points (using mediapipe)
    results = hands.process(image)

    # Show the points on hands on live scren
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Display in BGR for OpenCV
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Show the hand with the lines for vectors
    cv2.imshow('ASL Converter', cv2.flip(frame, 1))

    # Press 's' to save or 'q' to quit
    key = cv2.waitKey(1) & 0xFF  # wait till some key is pressed, 0 -> take data infinitely

    if key == ord('s'):
        # Save landmarks when 's' is pressed
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                save_landmarks(hand_landmarks, file_path)
        else:
            print("No hand landmarks detected. Try again.")
    elif key == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all windows
