def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(target, Y_hat):
    # cross-entropy https://en.wikipedia.org/wiki/Cross_entropy
    L_sum = np.sum(np.multiply(target, np.log(Y_hat)))
    return -L_sum / target.shape[1]


X = X_train
Y = Y_train

# n_epochs 20..1000 should be ok
n_epochs = 50
for i in range(n_epochs):
    # Forward pass
    z1 = np.matmul(W1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(W2, a1) + b2
    # softmax https://en.wikipedia.org/wiki/Softmax_function
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)

    m = 60000

    # Backward pass
    dz2 = a2 - Y
    dW2 = np.matmul(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    da1 = np.matmul(W2.T, dz2)
    dz1 = da1 * sigmoid(z1) * (1 - sigmoid(z1))
    dW1 = np.matmul(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # Update network parameters
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    # do not overshoot with many epochs
    alpha = alpha * (1 - 0.1 / n_epochs)

    print("Epoch", i, "loss: ", loss(Y, a2))

