
import os
import numpy as np
import pandas as pd
import cv2

# Paths to the training and testing datasets
train_dir = '/content/drive/MyDrive/1 COMP 542/Project/full dataset/asl_alphabet_train'

# Image dimensions
img_size = 64

# Initialize lists to store images and labels
images = []
labels = []

# Load training images
for label in os.listdir(train_dir):
    label_path = os.path.join(train_dir, label)
    if os.path.isdir(label_path):
      counter = 0
      for img_file in os.listdir(label_path):
          img_path = os.path.join(label_path, img_file)
          # Read, resize, and normalize imageS
          img = cv2.imread(img_path)
          if img is None:
              continue
          img = cv2.resize(img, (img_size, img_size))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
          images.append(img)
          labels.append(label)
      print('finished', label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
np.save('/content/drive/MyDrive/1 COMP 542/Project/aslConverter/images.npy', images)
np.save('/content/drive/MyDrive/1 COMP 542/Project/aslConverter/labels.npy', labels)
