import numpy as np

n = int(input("Enter number of samples: "))

X = []
y = []

print("Enter data (x y):")
for i in range(n):
    a, b_val = map(float, input(f"Sample {i + 1}: ").split())
    X.append([a])
    y.append(b_val)

X = np.array(X)
y = np.array(y)

epochs = int(input("Enter epochs: "))
optimizer = input("Choose optimizer (momentum/rmsprop/adam): ").lower()
mode = input("Choose mode (batch/mini/sgd): ").lower()
batch_size = int(input("Enter batch size (for mini, else 1): "))

# --------------------------------------------------
# 🔹 STEP 2: INITIALIZATION
# --------------------------------------------------

w = np.random.randn()
b = 0

lr = 0.01

# optimizer variables
v_w, v_b = 0, 0
s_w, s_b = 0, 0

beta = 0.9
beta2 = 0.999
epsilon = 1e-8


# --------------------------------------------------
# 🔹 HELPER: BATCH CREATION
# --------------------------------------------------

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


# --------------------------------------------------
# 🔹 TRAINING
# --------------------------------------------------

for epoch in range(epochs):

    # Select batching strategy
    if mode == "batch":
        batches = [(X, y)]
    elif mode == "sgd":
        batches = [(X[i:i + 1], y[i:i + 1]) for i in range(len(X))]
    else:
        batches = get_batches(X, y, batch_size)

    for xb, yb in batches:

        # Forward pass (linear regression)
        pred = w * xb + b

        # Compute gradients
        error = pred.flatten() - yb
        dw = np.mean(error * xb.flatten())
        db = np.mean(error)

        # -------------------------
        # OPTIMIZERS
        # -------------------------

        if optimizer == "momentum":
            # Momentum update
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db

            w -= lr * v_w
            b -= lr * v_b

        elif optimizer == "rmsprop":
            # RMSProp update
            s_w = beta * s_w + (1 - beta) * (dw ** 2)
            s_b = beta * s_b + (1 - beta) * (db ** 2)

            w -= lr * dw / (np.sqrt(s_w) + epsilon)
            b -= lr * db / (np.sqrt(s_b) + epsilon)

        elif optimizer == "adam":
            # Adam update
            v_w = beta * v_w + (1 - beta) * dw
            v_b = beta * v_b + (1 - beta) * db

            s_w = beta2 * s_w + (1 - beta2) * (dw ** 2)
            s_b = beta2 * s_b + (1 - beta2) * (db ** 2)

            # bias correction
            v_w_hat = v_w / (1 - beta)
            v_b_hat = v_b / (1 - beta)

            s_w_hat = s_w / (1 - beta2)
            s_b_hat = s_b / (1 - beta2)

            w -= lr * v_w_hat / (np.sqrt(s_w_hat) + epsilon)
            b -= lr * v_b_hat / (np.sqrt(s_b_hat) + epsilon)

# --------------------------------------------------
# 🔹 OUTPUT
# --------------------------------------------------

print("\nFinal Parameters:")
print("w =", w)
print("b =", b)

print("\nPredictions:")
for i in range(len(X)):
    print(X[i][0], "->", w * X[i][0] + b)

"""
Enter number of samples: 5
Enter data (x y):
Sample 1: 1 5
Sample 2: 2 8
Sample 3: 3 11
Sample 4: 4 14
Sample 5: 5 17

Enter epochs: 100
Choose optimizer (momentum/rmsprop/adam): adam
Choose mode (batch/mini/sgd): mini
Enter batch size (for mini, else put 1): 2

Expected Output (approx):
Final w ≈ 3
Final b ≈ 2

Predictions:
1 → ~5
2 → ~8
3 → ~11
...
"""
