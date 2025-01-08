import pandas as pd
import os 
import shutil

# Read CSV file
data = pd.read_csv('IR Labels.csv')

# Add class column
def assign_class(value):
    """
    Assign a class label to a given IR Index value based on the following rules:

    0: Healthy
    1-25: Mild
    26-50: Moderate
    51-100: Severe
    else: Unknown

    Args:
        value (int): IR Index value

    Returns:
        str: Class label
    """
    if value == 0:
        # If the value is 0, the plant is healthy
        return "Healthy"
    elif 1 <= value <= 25:
        # If the value is between 1 and 25, the plant has a mild infection
        return "Mild"
    elif 26 <= value <= 50:
        # If the value is between 26 and 50, the plant has a moderate infection
        return "Moderate"
    elif 51 <= value <= 100:
        # If the value is between 51 and 100, the plant has a severe infection
        return "Severe"
    else:
        # If the value is outside the above ranges, it's unknown
        return "Unknown"

data['Class'] = data['IRIndex'].apply(assign_class)
data.to_csv('classified_data.csv', index=False)
print("Data classification completed and saved.")

# Prepare data
data = pd.read_csv('classified_data.csv')
# Add the directory containing the images
image_directory = "RSM-PC_dataset"  # Replace with the actual path to your image dataset
data['Image Path'] = data['ImageNo.'].apply(lambda x: os.path.join(image_directory, f"{x}.jpg"))

# Save the modified data back to a new CSV file
data.to_csv('classified_data_with_paths.csv', index=False)
print("done")
# Create organized_images folder and subfolders for each class
organized_images_dir = "organized_images"
os.makedirs(organized_images_dir, exist_ok=True)

for class_name in data['Class'].unique():
    class_folder = os.path.join(organized_images_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

# Move images to their respective class folders
for _, row in data.iterrows():
    image_path = row['Image Path']
    class_name = row['Class']
    class_folder = os.path.join(organized_images_dir, class_name)
    
    if os.path.exists(image_path):
        shutil.copy(image_path, class_folder)
    else:
        print(f"Image not found: {image_path}")

print("Images organized by class.")

