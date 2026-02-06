print("prepare_data.py started")

from sklearn.ensemble import RandomForestClassifier

import os
import numpy as np
from PIL import Image

# Path to dataset
DATASET_PATH = "../dataset"
IMAGE_SIZE = (128, 128)

x = []      #container for image
y = []      #container to store labels

# Get class names (subdirectories in dataset)
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
print("Classes found:", classes)

for label, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Ensure 3 channels
                img = img.resize(IMAGE_SIZE)
                x.append(np.array(img))
                y.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)

print("Data preparation complete.")
print(f"Total images: {len(x)}, Labels: {len(y)}")