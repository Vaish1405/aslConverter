from cnn_model import *

# Load test images
test_images = []
test_labels = []

for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    # Read and preprocess the test image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    test_images.append(img)
    
    # Extract label from the filename (e.g., 'A.jpg' -> 'A')
    label = img_file[0]  
    test_labels.append(label_mapping[label])

# Convert to numpy arrays and one-hot encode labels
test_images = np.array(test_images)
test_labels = to_categorical(np.array(test_labels), num_classes=len(label_mapping))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()