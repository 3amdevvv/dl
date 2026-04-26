import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --------------------------------------------------
# 🔹 EXPECTED INPUT / OUTPUT
# --------------------------------------------------
"""
INPUT:
Enter number of samples: 100
Enter sequence length: 5
Enter number of epochs: 10

EXPECTED OUTPUT:
- Predictions close to actual values
- Graph showing decreasing loss over epochs
"""

# --------------------------------------------------
# 🔹 STEP 1: USER INPUT
# --------------------------------------------------

n = int(input("Enter number of samples: "))
seq_len = int(input("Enter sequence length: "))
epochs = int(input("Enter number of epochs: "))

# --------------------------------------------------
# 🔹 STEP 2: CREATE DATA
# --------------------------------------------------

X = np.array([[i + j for j in range(seq_len)] for i in range(n)])
y = np.array([i + seq_len for i in range(n)])

X = X.reshape((n, seq_len, 1))

# --------------------------------------------------
# 🔹 STEP 3: BUILD MODEL
# --------------------------------------------------

model = tf.keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(seq_len, 1)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# --------------------------------------------------
# 🔹 STEP 4: TRAIN MODEL (STORE HISTORY)
# --------------------------------------------------

history = model.fit(X, y, epochs=epochs)

# --------------------------------------------------
# 🔹 STEP 5: PREDICTIONS
# --------------------------------------------------

print("\nSample Predictions:")
pred = model.predict(X[:5])

for i in range(5):
    print("Input:", X[i].flatten(), "-> Predicted:", pred[i][0], "Actual:", y[i])

# --------------------------------------------------
# 🔹 STEP 6: PLOT GRAPH
# --------------------------------------------------

plt.plot(history.history['loss'])
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()