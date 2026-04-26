import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --------------------------------------------------
# 🔹 EXPECTED INPUT / OUTPUT
# --------------------------------------------------
"""
INPUT:
Enter number of epochs: 5

EXPECTED OUTPUT:
- Loss decreases over epochs
- Reconstructed images look similar to original digits
- Graph of loss vs epochs

Example:
Epoch 1 → loss ~0.08
Epoch 5 → loss ~0.03
"""

# --------------------------------------------------
# 🔹 STEP 1: USER INPUT
# --------------------------------------------------

epochs = int(input("Enter number of epochs: "))

# --------------------------------------------------
# 🔹 STEP 2: LOAD DATASET (MNIST)
# --------------------------------------------------

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten (28x28 → 784)
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# --------------------------------------------------
# 🔹 STEP 3: BUILD AUTOENCODER
# --------------------------------------------------

# Encoder (compress data)
input_layer = tf.keras.Input(shape=(784,))
encoded = layers.Dense(64, activation='relu')(input_layer)

# Decoder (reconstruct data)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Model
autoencoder = tf.keras.Model(input_layer, decoded)

# --------------------------------------------------
# 🔹 STEP 4: COMPILE MODEL
# --------------------------------------------------

autoencoder.compile(optimizer='adam', loss='mse')

# --------------------------------------------------
# 🔹 STEP 5: TRAIN MODEL
# --------------------------------------------------

history = autoencoder.fit(X_train, X_train, epochs=epochs)

# --------------------------------------------------
# 🔹 STEP 6: RECONSTRUCTION
# --------------------------------------------------

decoded_imgs = autoencoder.predict(X_test)

# --------------------------------------------------
# 🔹 STEP 7: DISPLAY IMAGES
# --------------------------------------------------

n = 5  # number of images to display
plt.figure(figsize=(10, 4))

for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()

# --------------------------------------------------
# 🔹 STEP 8: PLOT LOSS GRAPH
# --------------------------------------------------

plt.plot(history.history['loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()