import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# --------------------------------------------------
# 🔹 EXPECTED INPUT / OUTPUT
# --------------------------------------------------
"""
INPUT:
Enter test size (e.g., 0.2): 0.2
Enter number of epochs: 5

EXPECTED OUTPUT (approx):
Epoch 1/5 → accuracy ~0.80
Epoch 5/5 → accuracy ~0.95

Test Accuracy: ~0.90 to 0.98

Sample Predictions:
Predicted: 3 Actual: 3
Predicted: 8 Actual: 8
Predicted: 1 Actual: 1
...
"""

# --------------------------------------------------
# 🔹 STEP 1: USER INPUT
# --------------------------------------------------

test_size = float(input("Enter test size (e.g., 0.2): "))
epochs = int(input("Enter number of epochs: "))

# --------------------------------------------------
# 🔹 STEP 2: LOAD TOY DATASET
# --------------------------------------------------

digits = load_digits()

X = digits.images   # shape (1797, 8, 8)
y = digits.target   # labels 0–9

# Normalize data (0–16 → 0–1)
X = X / 16.0

# Reshape for CNN (add channel dimension)
X = X.reshape(-1, 8, 8, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# --------------------------------------------------
# 🔹 STEP 3: BUILD CNN MODEL
# --------------------------------------------------

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(8,8,1)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# --------------------------------------------------
# 🔹 STEP 4: COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# 🔹 STEP 5: TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train, epochs=epochs)

# --------------------------------------------------
# 🔹 STEP 6: EVALUATE MODEL
# --------------------------------------------------

loss, acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", acc)

# --------------------------------------------------
# 🔹 STEP 7: SAMPLE PREDICTIONS
# --------------------------------------------------

print("\nSample Predictions:")
pred = model.predict(X_test[:5])

for i in range(5):
    print("Predicted:", np.argmax(pred[i]), "Actual:", y_test[i])