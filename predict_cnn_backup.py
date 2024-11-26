import cv2
from tensorflow.keras.models import load_model
import numpy as np

img_size = 224

model = load_model('trained_model_full_dataset.h5')  # Replace with your .h5 file path
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

label_mapping = {i: chr(65 + i) for i in range(26)}  # Mapping 0-25 to 'A'-'Z'
label_mapping.update({26: 'del', 27: 'space', 28: 'nothing'})

# Load the image
img = cv2.imread('./asl_alphabet_test 2/L_test.jpg')
if img is None:
    raise ValueError(f"Image could not be loaded.")

# Resize the image to the model's expected input size
img_resized = cv2.resize(img, (img_size, img_size))

# Convert to RGB (if using OpenCV, which loads images as BGR by default)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Normalize the image
img_normalized = img_rgb / 255.0

# Add batch dimension
img_array = np.expand_dims(img_normalized, axis=0)

# Make the prediction
predictions = model.predict(img_array)

# Get the predicted class index
predicted_class = np.argmax(predictions)

# Map the predicted class to its label
predicted_label = label_mapping.get(predicted_class)

print(predicted_label)
