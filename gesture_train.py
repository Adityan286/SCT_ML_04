import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

DATASET_PATH = "leapGestRecog"
IMG_SIZE = 64

images = []
labels = []
label_map = {}
label_counter = 0

# SUBJECT folders: 00–09
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    # Ignore anything that is not a subject folder
    if not subject.isdigit():
        continue

    # GESTURE folders inside subject
    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        # 🔥 VERY IMPORTANT FILTER
        # Gestures ALWAYS contain '_'
        if "_" not in gesture:
            continue

        if gesture not in label_map:
            label_map[gesture] = label_counter
            label_counter += 1

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_map[gesture])

# Convert to numpy arrays
X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(labels, num_classes=len(label_map))

print("Total images:", len(X))
print("Total gesture classes:", len(label_map))
print("Gesture labels:", label_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CNN model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("gesture_cnn_model.h5")

# Save labels
with open("gesture_labels.txt", "w") as f:
    for name, idx in label_map.items():
        f.write(f"{idx}:{name}\n")
