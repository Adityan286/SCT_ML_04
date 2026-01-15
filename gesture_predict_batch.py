import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------- CONFIG (CHANGE THESE) -------------------
IMG_DIM = 64  # change to your model's input size
MODEL_FILE = r"C:\Users\adhi_\OneDrive\Desktop\Codes\SCT_ML_04\gesture_cnn_model.h5"
LABEL_FILE = r"C:\Users\adhi_\OneDrive\Desktop\Codes\SCT_ML_04\gesture_labels.txt"
DATASET_PATH = r"C:\Users\adhi_\OneDrive\Desktop\Codes\SCT_ML_04\leapGestRecog"

NUM_SAMPLES = 4  # number of images to randomly sample for display

# ------------------- LOAD MODEL -------------------
model = load_model(MODEL_FILE)

# Load gesture labels
with open(LABEL_FILE, "r") as f:
    gestures = [line.strip() for line in f.readlines()]

# ------------------- GATHER ALL IMAGE PATHS -------------------
all_imgs = []
for sub in os.listdir(DATASET_PATH):
    sub_path = os.path.join(DATASET_PATH, sub)
    if not os.path.isdir(sub_path):
        continue
    for gest in os.listdir(sub_path):
        gest_path = os.path.join(sub_path, gest)
        if not os.path.isdir(gest_path):
            continue
        for img_file in os.listdir(gest_path):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_imgs.append(os.path.join(gest_path, img_file))

if len(all_imgs) == 0:
    raise ValueError("No images found in dataset path!")

# ------------------- RANDOM SAMPLE -------------------
samples = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))

# ------------------- PLOT SETUP -------------------
fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4*len(samples)))

# Ensure axes is 2D array even if len(samples)=1
if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

# ------------------- PREDICT AND PLOT -------------------
for i, img_path in enumerate(samples):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_DIM, IMG_DIM))
    img_input = img_resized.reshape(1, IMG_DIM, IMG_DIM, 1)/255.0

    # Predict
    pred = model.predict(img_input, verbose=0)
    index = np.argmax(pred)
    confidence = pred[0][index]*100
    label = gestures[index]

    # Display image
    axes[i][0].imshow(img, cmap='gray')
    axes[i][0].set_title(f"Predicted: {label}")
    axes[i][0].axis('off')

    # Display confidence bar
    axes[i][1].barh([label], [confidence], color='skyblue')
    axes[i][1].set_xlim(0,100)
    axes[i][1].set_title("Confidence (%)")

plt.tight_layout()
plt.show()
