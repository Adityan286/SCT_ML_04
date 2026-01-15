✋🖐 Hand Gesture Recognition using CNN
This project is part of Task 04 of my Machine Learning Internship at SkillCraft Technology. The goal of this task is to build a hand gesture recognition system using Convolutional Neural Networks (CNN) to classify images of hand gestures from the LeAP GestRecog dataset.

📌 Project Overview
Gesture recognition is a key problem in computer vision, useful for HCI (Human-Computer Interaction), sign language recognition, and robotics. This project uses a CNN trained on grayscale images of hand gestures to predict gestures accurately. The workflow is divided into two main stages:

Training the model
Predicting new images (single or batch predictions)

📂 Project Structure
SCT_ML_04/
│
├── gesture_train.py            # Training script
├── gesture_predict_single.py   # Single image prediction script
├── gesture_predict_batch.py    # Batch prediction script
├── gesture_cnn_model.h5        # Trained CNN model
├── gesture_labels.txt          # Gesture label mapping
│
├── leapGestRecog/              # Dataset
│   ├── 00/
│   │   ├── gesture_01/
│   │   │   ├── img1.png
│   │   │   └── ...
│   │   └── gesture_02/
│   └── 01/
│       └── ...
│
└── test_images/                # Optional images for testing
    └── test1.png

🧠 Approach
1️⃣ Image Preprocessing
Images are loaded in grayscale.
Resized to 64 × 64 pixels (matching CNN input).
Normalized to [0,1] for better model convergence.
2️⃣ Model Architecture
Input: 64×64×1 (grayscale)
Conv2D → MaxPooling layers (2 blocks)
Flatten → Dense → Dropout → Output Dense layer
Output: Softmax probabilities over all gesture classes
3️⃣ Model Training
Loss: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Train-test split: 80% training, 20% testing
Number of epochs: 10
Batch size: 32
4️⃣ Model Saving
CNN model saved as gesture_cnn_model.h5
Gesture labels saved as gesture_labels.txt
5️⃣ Prediction
Load the trained model
Preprocess a new image
Predict gesture class
For batch predictions, visualize multiple images with confidence bar charts

🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
scikit-learn
Matplotlib

📊 Output
Classification result (gesture name)
Confidence percentage for each class
Visualization of predictions for single or batch images

🎯 Learning Outcomes
Understanding CNN architectures for image classification
Preprocessing grayscale images for deep learning
Training and evaluating CNN models
Saving and loading trained models
Building single and batch prediction pipelines
Visualizing model predictions using Matplotlib

📌 Internship Task
This project was completed as part of the SkillCraft Technology Machine Learning Internship, focusing on applying deep learning techniques to real-world image classification and gesture recognition problems.
