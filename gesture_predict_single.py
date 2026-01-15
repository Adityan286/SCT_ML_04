import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# ------------------- CONFIG -------------------
MODEL_PATH = "gesture_cnn_model.h5"
LABELS_PATH = "gesture_labels.txt"

# ------------------- LOAD MODEL -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Expected input shape: {model.input_shape}")

# Get expected input dimensions
IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]
IMG_CHANNELS = model.input_shape[3]  # 1 for grayscale, 3 for RGB

# ------------------- LOAD LABELS -------------------
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

labels = {}
with open(LABELS_PATH, "r") as f:
    for line in f:
        k, v = line.strip().split(":")
        labels[int(k)] = v

# ------------------- SELECT IMAGE -------------------
root = Tk()
root.withdraw()  # hide the root window

IMAGE_PATH = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not IMAGE_PATH:
    raise ValueError("No image selected!")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Failed to read image: {IMAGE_PATH}")

# ------------------- PROCESS IMAGE -------------------
# Convert to grayscale if model expects 1 channel
if IMG_CHANNELS == 1:
    img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to model input
img_resized = cv2.resize(img_processed, (IMG_WIDTH, IMG_HEIGHT))
img_normalized = img_resized / 255.0

# Add channel dimension if grayscale
if IMG_CHANNELS == 1:
    img_normalized = np.expand_dims(img_normalized, axis=-1)

# Add batch dimension
img_input = np.expand_dims(img_normalized, axis=0)

# ------------------- PREDICT -------------------
prediction = model.predict(img_input)
predicted_index = np.argmax(prediction)
gesture = labels.get(predicted_index, "Unknown")
confidence = float(np.max(prediction))

print(f"Predicted Gesture: {gesture}")
print(f"Confidence: {confidence:.2f}")

# ------------------- PLOT -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original Image
if IMG_CHANNELS == 1:
    axes[0].imshow(img_resized, cmap='gray')
else:
    axes[0].imshow(img_resized)
axes[0].set_title("Input Image")
axes[0].axis("off")

# Prediction Bar Plot
axes[1].bar(range(len(prediction[0])), prediction[0], color="skyblue")
axes[1].set_xticks(range(len(prediction[0])))
axes[1].set_xticklabels([labels[i] for i in range(len(prediction[0]))], rotation=45)
axes[1].set_ylim([0, 1])
axes[1].set_title(f"Prediction (Gesture: {gesture})")

plt.tight_layout()
plt.show()
