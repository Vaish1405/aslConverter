
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load the pretrained model
model = load_model('/content/drive/MyDrive/1 COMP 542/Project/aslConverter/trained_model.h5')

# Define the test dataset directory
test_dir = '/content/drive/MyDrive/1 COMP 542/Project/aslConverter/alphabet_dataset/asl_alphabet_test/asl_alphabet_test'

# Image dimensions (make sure this matches what your model was trained on)
img_size = 64

# Load test images and labels
test_images = []
test_labels = []

# Initialize label_mapping
label_mapping = {}  # Dictionary to map labels to numeric values

# Loop through all images in the test directory
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    
    # Skip .DS_Store or other non-image files
    if img_file.endswith('.DS_Store'):
        continue
    
    # Read the image and preprocess it
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        continue  # Skip if the image cannot be read
    
    img = cv2.resize(img, (img_size, img_size))  # Resize to model's expected input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize the image (assuming BGR to RGB)
    
    test_images.append(img)
    
    # Extract label from the first character of the file name (adjust if your naming is different)
    label = img_file[0]
    
    # Add the label to label_mapping if it isn't already
    if label not in label_mapping:
        label_mapping[label] = len(label_mapping)
    
    test_labels.append(label)

# Convert test_images to a numpy array
test_images = np.array(test_images)

# Map test labels to numeric values based on the label_mapping
test_labels = np.array([label_mapping[label] for label in test_labels])

# One-hot encode the labels
test_labels = to_categorical(test_labels, num_classes=len(label_mapping))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

