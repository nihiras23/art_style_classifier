from PIL import Image 
import matplotlib.pyplot as plt 
import os

# Path to dataset (relative to this file)
dataset_path = "../dataset"

# List all art style folders
classes = os.listdir(dataset_path)
print("Classes found:", classes)

# Pick the first class
class_name = classes[0]
class_path = os.path.join(dataset_path, class_name)

# Pick the first image from that class
image_name = os.listdir(class_path)[0]
image_path = os.path.join(class_path, image_name)

# Load the image
img = Image.open(image_path)

# Display the image
plt.figure(figsize=(6, 6)) 
plt.imshow(img)
plt.title(f"Art Style: {class_name}")
plt.axis("off")
plt.show(block=True)


