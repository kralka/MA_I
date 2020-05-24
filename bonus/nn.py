# download the files from: http://yann.lecun.com/exdb/mnist/
# train-images-idx3-ubyte.gz
# train-labels-idx1-ubyte.gz
# t10k-images-idx3-ubyte.gz
# t10k-labels-idx1-ubyte.gz

# This code was inspired by the blogpost: https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/

import gzip
import numpy as np
import struct
from math import floor

# https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def pretty_print(a):
    """Print ASCII-art representation of the number"""
    if len(a.shape) == 1:
        # reshape to original dimension
        return pretty_print(a.reshape(28, 28))
    for l in a:
        print("".join(" .:-=+*#%@"[min(floor(10*x), 9)] for x in l))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(target, Y_hat):
    # cross-entropy https://en.wikipedia.org/wiki/Cross_entropy
    L_sum = np.sum(np.multiply(target, np.log(Y_hat)))
    return -L_sum / target.shape[1]


# train data
train_imgs = read_idx('train-images-idx3-ubyte.gz') / 255
train_labels = read_idx('train-labels-idx1-ubyte.gz')

# test data
test_imgs = read_idx('t10k-images-idx3-ubyte.gz') / 255  # normalize to [0, 1], max = 255
test_labels = read_idx('t10k-labels-idx1-ubyte.gz')

# flatten
train_imgs = train_imgs.reshape(train_imgs.shape[0], 28 * 28)
test_imgs = test_imgs.reshape(test_imgs.shape[0], 28 * 28)

# turn train labels into one-hot encoding
digits = 10
train_labels = np.eye(digits)[train_labels]


# Transposing helps with batches
X_train = train_imgs.T
Y_train = train_labels.T
X_test = test_imgs.T
Y_test = test_labels.T

n_hidden_activations = 100
alpha = 1

# Initialize weights at random
W1 = np.random.randn(n_hidden_activations, X_train.shape[0]) / np.sqrt(n_hidden_activations)
b1 = np.random.randn(n_hidden_activations, 1) / np.sqrt(n_hidden_activations)
W2 = np.random.randn(digits, n_hidden_activations) / np.sqrt(digits)
b2 = np.random.randn(digits, 1) / np.sqrt(digits)

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


# Make predictions (one forward pass with test images)
z1 = np.matmul(W1, X_test) + b1
a1 = sigmoid(z1)
z2 = np.matmul(W2, a1) + b2
a2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)

predictions = np.argmax(a2, axis=0)
result = predictions == test_labels
print("Correct in:", np.sum(result) * 100.0 / result.shape[0], "% cases")

# Pretty print first few mislabeled
found_mislabeled = 0
for i in range(result.shape[0]):
    if found_mislabeled >= 10:
        break
    if not result[i]:
        found_mislabeled = found_mislabeled + 1
        pretty_print(X_test[:, i])
        print("Predicted:", predictions[i], "but should be:", test_labels[i])

