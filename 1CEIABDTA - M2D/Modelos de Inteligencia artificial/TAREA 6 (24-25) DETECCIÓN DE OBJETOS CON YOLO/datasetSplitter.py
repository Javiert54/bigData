import os
from sklearn.model_selection import train_test_split
import shutil
import os
import zipfile

dataset_path = "Dice.v2-medium-color.yolov8"

images_path = "Dice.v2-medium-color.yolov8/export/images"
labels_path = "Dice.v2-medium-color.yolov8/export/labels"


# Get the list of all image files
image_files = [images_path+"/"+f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split the dataset into train, val, and test sets
train_files, val_files = train_test_split(image_files, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Function to copy files to the respective directories
def copy_files(file_list, destination_folder, label_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in file_list:
        file_name = file.split("/")[-1].rsplit(".", 1)[0]
        
        shutil.copy(file, destination_folder)
        shutil.copy("Dice.v2-medium-color.yolov8/export/labels/"+file_name + ".txt", label_folder)
        

# Copy the files to the respective directories
copy_files(train_files, "dataset/images/train", "dataset/labels/train")
copy_files(val_files, "dataset/images/val", "dataset/labels/val")

print("Dataset split into train, val, and test sets.")