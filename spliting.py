import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit

# Path to the data directory
data_dir = 'augmented_dataset_new'

# Class names
classes = ['healthy', 'mild', 'moderate', 'severe']

# Split ratio for train and test
train_ratio = 0.7
test_ratio = 0.3

# Create directories for train and test if they don't exist
train_dir = os.path.join(data_dir, 'train_new')
test_dir = os.path.join(data_dir, 'test_new')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Check if the directories already exist
if not os.listdir(train_dir) or not os.listdir(test_dir):
    # Split the data for each class using StratifiedShuffleSplit
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = os.listdir(class_path)
        
        # Use StratifiedShuffleSplit to ensure balanced splitting
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        
        # Labels for the images (all images in a class share the same label)
        labels = [class_name] * len(images)
        
        for train_index, test_index in sss.split(images, labels):
            train_images = [images[i] for i in train_index]
            test_images = [images[i] for i in test_index]
            
            # Create class directories in train and test folders
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            
            if not os.path.exists(test_class_dir):
                os.makedirs(test_class_dir)
            
            # Move the images to the respective directories
            for image in train_images:
                src = os.path.join(class_path, image)
                dst = os.path.join(train_class_dir, image)
                shutil.copy(src, dst)
            
            for image in test_images:
                src = os.path.join(class_path, image)
                dst = os.path.join(test_class_dir, image)
                shutil.copy(src, dst)
    
    print("Data split completed with stratified sampling, saved in 'train_new' and 'test_new'!")
else:
    print("Data already split, skipping the split process.")

# Print the number of images per class in train
for class_name in classes:
    class_train_dir = os.path.join(train_dir, class_name)
    num_images = len(os.listdir(class_train_dir))
    print(f"Number of images in {class_name} class (train): {num_images}")

# Print the number of images per class in test
for class_name in classes:
    class_test_dir = os.path.join(test_dir, class_name)
    num_images = len(os.listdir(class_test_dir))
    print(f"Number of images in {class_name} class (test): {num_images}")