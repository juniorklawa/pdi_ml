import os
from sklearn.model_selection import train_test_split
import shutil

def create_train_test_dirs(base_path, train_path, test_path):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    flower_folders = os.listdir(base_path)

    for folder in flower_folders:
        train_flower_path = os.path.join(train_path, folder)
        test_flower_path = os.path.join(test_path, folder)
        
        os.makedirs(train_flower_path, exist_ok=True)
        os.makedirs(test_flower_path, exist_ok=True)

        flower_images = os.listdir(os.path.join(base_path, folder))
        train_images, test_images = train_test_split(flower_images, test_size=0.3, random_state=42)

        for train_image in train_images:
            src = os.path.join(base_path, folder, train_image)
            dst = os.path.join(train_flower_path, train_image)
            shutil.copy(src, dst)

        for test_image in test_images:
            src = os.path.join(base_path, folder, test_image)
            dst = os.path.join(test_flower_path, test_image)
            shutil.copy(src, dst)

base_path = './flower_images'
train_path = './train_images'
test_path = './test_images'

create_train_test_dirs(base_path, train_path, test_path)