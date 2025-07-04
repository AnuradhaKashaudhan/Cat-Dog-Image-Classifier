# Cat-Dog-Image-Classifier
CNN-based binary image classifier to distinguish cats from dogs using TensorFlow and Keras.


# 🐱🐶 Cat vs Dog Image Classifier

This project is a **Convolutional Neural Network (CNN)** that classifies images as either **cat** or **dog**. Built using **TensorFlow/Keras** in Google Colab, it demonstrates basic image preprocessing, data augmentation, model training, and prediction.

---

## 🧠 Project Summary

- 📁 Uses a dataset of labeled cat and dog images
- 🧼 Includes resizing, rescaling, and data augmentation
- ⚙️ CNN architecture with Conv2D, MaxPooling, Flatten, Dense layers
- 📉 Trained with `binary_crossentropy` loss
- 📊 Outputs training/validation accuracy and loss graphs

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib
- Google Colab
- ImageDataGenerator for augmentation

---

## 🔧 Model Architecture

python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])


predict_image("test/cat1.jpg")  # → "Predicted: Cat"
predict_image("test/dog5.jpg")  # → "Predicted: Dog"



