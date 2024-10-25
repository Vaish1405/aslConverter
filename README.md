# ASL Converter

This project is an ASL (American Sign Language) hand sign recognition system that uses a trained Random Forest model and the MediaPipe library for real-time prediction of hand signs from webcam input.

## Project Structure

* **train_model.py**: Script to train the Random Forest classifier and save the model and scaler.
* **predict.py**: Script to capture video from a webcam, process hand landmarks, and predict hand signs.
* **random_forest_model.pkl**: Saved Random Forest model for hand sign prediction.
* **scaler.pkl**: Scaler for normalizing input data.
* **hand_landmarks.csv**: CSV file with vector points (features) and labels for training the model.
* **__pycache__**: Auto-generated cache files.

## Requirements

The project requires the following Python libraries:

* `opencv-python`
* `numpy`
* `mediapipe`
* `scikit-learn`
* `joblib`
* `pandas`

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd aslConverter
   ```

2. Install all required libraries in one command:
   ```bash
   pip install opencv-python numpy mediapipe scikit-learn joblib pandas
   ```

## Usage

1. Train the Model

    If **random_forest_model.pkl** and **scaler.pkl** are not present, train the model by running **train_model.py**:

    ```bash
    python train_model.py
    ```

    This will:
    * Load the **hand_landmarks.csv** file.
    * Train a Random Forest model.
    * Save the model and scaler as **random_forest_model.pkl** and **scaler.pkl**.

2. Run the Prediction Script

   Run **predict.py** to start real-time hand sign prediction using your webcam:

    ```bash
    python predict.py
    ```

    This script:
    * Opens a webcam video feed.
    * Detects hand landmarks and predicts hand signs.
    * Displays the predicted hand sign on the video frame.
    
    Controls
    * p: Print the predicted hand sign to the console.
    * Esc: Exit the program and close the webcam.

## Troubleshooting

If you encounter any missing module errors, ensure all libraries are installed as described in the Installation section.

## License

Need to add when this project is made public.
   