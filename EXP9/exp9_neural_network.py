import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ------------------ LOAD AND PREPARE DATA ------------------
# Using the MNIST dataset (images of handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# ------------------ BUILD THE MODEL ------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),   # Flatten 28x28 images to 784 input neurons
    Dense(128, activation='relu'),   # Hidden layer 1
    Dense(64, activation='relu'),    # Hidden layer 2
    Dense(10, activation='softmax')  # Output layer (10 classes)
])

# ------------------ COMPILE THE MODEL ------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------ TRAIN THE MODEL ------------------
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ------------------ EVALUATE THE MODEL ------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ------------------ VISUALIZE TRAINING PERFORMANCE ------------------
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------ CONCLUSION ------------------
print("Conclusion: Thus, we have successfully implemented an Artificial Neural Network using Keras/TensorFlow.")