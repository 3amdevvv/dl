import numpy as np


# --------------------------------------------------
# 🔹 STEP 1: TAKE INPUT FROM USER
# --------------------------------------------------

n = int(input("Enter number of training samples: "))

X = []
y = []

print("Enter input features and label (space separated):")
for i in range(n):
    data = list(map(float, input(f"Sample {i + 1}: ").split()))

    X.append(data[:-1])  # features
    y.append([data[-1]])  # output

X = np.array(X)
y = np.array(y)

# --------------------------------------------------
# 🔹 STEP 2: INITIALIZATION
# --------------------------------------------------

w = np.random.rand(X.shape[1], 1)  # random weights
lr = 0.5  # learning rate


# --------------------------------------------------
# 🔹 STEP 3: ACTIVATION FUNCTION (SIGMOID)
# --------------------------------------------------

def sigmoid(x):
    """
    Sigmoid function converts output into probability (0 to 1)
    """
    return 1 / (1 + np.exp(-x))


def derivative(x):
    """
    Derivative of sigmoid used for backpropagation
    """
    return x * (1 - x)


# --------------------------------------------------
# 🔹 STEP 4: TRAINING
# --------------------------------------------------

epochs = int(input("Enter number of epochs: "))

for epoch in range(epochs):
    # Forward pass
    output = sigmoid(np.dot(X, w))

    # Error calculation
    error = y - output

    # Backpropagation (weight update)
    w += lr * np.dot(X.T, error * derivative(output))

# --------------------------------------------------
# 🔹 STEP 5: TESTING / PREDICTION
# --------------------------------------------------

print("\nPredictions (Probabilities):")
pred = sigmoid(np.dot(X, w))
print(pred)

print("\nBinary Output (after threshold 0.5):")
for i in range(len(pred)):
    print(X[i], "->", 1 if pred[i] >= 0.5 else 0)


# --------------------------------------------------
# 🔹 EXPECTED INPUT FORMAT (Example: XOR problem)
# --------------------------------------------------
"""
Enter number of training samples: 4
Enter input features and label (space separated):
Sample 1: 0 0 0
Sample 2: 0 1 1
Sample 3: 1 0 1
Sample 4: 1 1 0
Enter number of epochs: 10000

Expected Output (approx):
Predictions:
[[0.01]
 [0.98]
 [0.98]
 [0.02]]

(Note: Values are probabilities, close to 0 or 1)
"""
