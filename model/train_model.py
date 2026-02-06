import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
from PIL import Image


# -----------------------------
# Load data again (same logic)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
IMAGE_SIZE = (128, 128)

X = []
y = []

classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
print("Classes:", classes)

for label, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE)
                X.append(np.array(img))
                y.append(label)
        except:
            pass

X = np.array(X)
y = np.array(y)

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# Flatten images
# -----------------------------

X = X.reshape(X.shape[0], -1)
print("Flattened X shape:", X.shape)

# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# Train model (SVM)
# -----------------------------

model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
