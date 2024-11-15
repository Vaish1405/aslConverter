
from tensorflow.keras.models import load_model
import cv2
import os
import re
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load the pretrained model
model = load_model('/content/drive/MyDrive/1 COMP 542/Project/aslConverter/trained_model_full_dataset.h5')

# Define the test dataset directory
test_dir = '/content/drive/MyDrive/1 COMP 542/Project/full dataset/asl_alphabet_test'

# Image dimensions (make sure this matches what your model was trained on)
img_size = 64

label_mapping = {chr(65 + i): i for i in range(26)}  # Mapping labels A-Z to 0-25
<<<<<<< HEAD
label_mapping.update({'del': 26, 'space': 27, 'nothing': 28})
print(label_mapping)
num_classes = len(label_mapping)

=======
num_classes = len(label_mapping)

print(label_mapping)

>>>>>>> f4e0f2b938fda7abb1ef25d223912831030fd147
# Load test images and labels
test_images = []
test_labels = []

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

    label = re.split(r'_test', img_file)[0]
    if label in label_mapping:
      test_labels.append(label_mapping[label])

# Convert test_images to a numpy array
test_images = np.array(test_images)

# Map test labels to numeric values based on the label_mapping
test_labels = np.array(test_labels)

test_labels = to_categorical(test_labels, num_classes=num_classes)
<<<<<<< HEAD
=======

print(test_labels)
>>>>>>> f4e0f2b938fda7abb1ef25d223912831030fd147

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

