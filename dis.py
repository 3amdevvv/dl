import numpy as np

# STEP 1: TAKE INPUT FROM USER

n = int(input("Enter number of training samples: "))

X = []
y = []

print("Enter input features and label (space separated):")
for i in range(n):
    data = list(map(int, input(f"Sample {i + 1}: ").split()))

    # Last value is output, rest are inputs
    X.append(data[:-1])
    y.append(data[-1])

X = np.array(X)
y = np.array(y)

# STEP 2: INITIALIZATION

w = np.zeros(X.shape[1])  # Initialize weights as 0
b = 0  # Initialize bias
lr = 0.1  # Learning rate

# STEP 3: ACTIVATION FUNCTION

def step(x):
    """
    Step activation function:
    If input >= 0 → output = 1
    Else → output = 0
    """
    return 1 if x >= 0 else 0

# STEP 4: TRAINING THE MODEL

epochs = int(input("Enter number of epochs: "))

for epoch in range(epochs):
    for i in range(len(X)):
        # Linear combination: w.x + b
        linear = np.dot(X[i], w) + b

        # Apply activation function
        y_pred = step(linear)

        # Calculate error
        error = y[i] - y_pred

        # Update weights
        w += lr * error * X[i]

        # Update bias
        b += lr * error

# STEP 5: OUTPUT TRAINED MODEL


print("\nFinal Weights:", w)
print("Final Bias:", b)

# STEP 6: TESTING

print("\nPredictions:")
for i in range(len(X)):
    prediction = step(np.dot(X[i], w) + b)
    print(X[i], "->", prediction)

    # Enter
    # number
    # of
    # training
    # samples: 4
    # Enter
    # input
    # features and label(space
    # separated):
    # Sample
    # 1: 0
    # 0
    # 0
    # Sample
    # 2: 0
    # 1
    # 0
    # Sample
    # 3: 1
    # 0
    # 0
    # Sample
    # 4: 1
    # 1
    # 1
    # Enter
    # number
    # of
    # epochs: 10
    #
    # Final
    # Weights: [0.2 0.1]
    # Final
    # Bias: -0.20000000000000004
    #
    # Predictions:
    # [0 0] -> 0
    # [0 1] -> 0
    # [1 0] -> 0
    # [1 1] -> 1
