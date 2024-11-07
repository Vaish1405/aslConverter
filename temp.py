import os

train_dir = './alphabet_dataset/asl_alphabet_train/asl_alphabet_train'
test_dir = './alphabet_dataset/asl_alphabet_test/asl_alphabet_test'

for label in os.listdir(train_dir):
    
    if label == 'R':
        label_path = os.path.join(train_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                print(img_file)
                # Read, resize, and normalize image
            

'./alphabet_dataset/asl_alphabet_train/asl_alphabet_train/R/.DS_Store'