from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist

# Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images: each image becomes a 784-length vector (28*28)
x_train_flat = x_train.reshape(x_train.shape[0], -1)# check numpy notes for reshaping
x_train_flat = x_train_flat / 255.0 # normalize the data to be between 0 and 1

y_train = y_train.reshape(-1, 1) # convert y_train into a column vector

x_test_flat = x_test.reshape(x_test.shape[0], -1)
x_test_flat = x_test_flat / 255.0
y_test = y_test.reshape(-1, 1)
test_data = np.hstack((y_test, x_test_flat))
np.random.shuffle(test_data)


try:
    data = np.hstack((y_train, x_train_flat))  # hstack, will be ValueError if the shapes are not compatible
except ValueError:
    print("Shapes are not compatible")

# Get the number of rows and columns
m, n = data.shape
print("Rows:", m, "Columns:", n)

# Shuffle the data randomly
np.random.shuffle(data)

# after transposing, each training example is a column
data_dev = data[0: 1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1: n]

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1: n] # (784, 59000)


# init_params randomly for forward propagation
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

# element wise operation so it goes through each element in Z
def ReLu(x):
    return np.maximum(x, 0)

def deriv_Relu(Z):
    return Z > 0 # works due to boolean to int conversion


def softmax(z):
    z_stable = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    Y = Y.astype(int)
    # Create a 2D array of zeros with:
    # - number of rows equal to the number of elements in Y (i.e., one row per label)
    # - number of columns equal to the maximum value in Y plus 1
    #   (assuming classes are labeled from 0 to Y.max())
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    
    # np.arange(Y.size) creates an array of indices [0, 1, 2, ..., Y.size-1]
    # The expression one_hot_Y[np.arange(Y.size), Y] selects the position in each row
    # corresponding to the label in Y, and sets that position to 1.
    one_hot_Y[np.arange(Y.size), Y] = 1

    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_propagation(Z1, A1, A2, W2, X, Y):
    m = Y.size 
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y 
    dW2 = 1 / m * dZ2.dot(A1.T) 
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_Relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.mean(predictions == Y)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    print("These are the final values: ", W1, b1, W2, b2)
    return W1, b1, W2, b2




def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def test_accuracy(index):
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 100)
    # select all rows at the column index
    # we add None to ensure that the shape of X is still correct 
    
    predictions = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    current_image = X_train[:, index, None]
    label = Y_train[index]
    print("Prediction: ", predictions)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_accuracy(40)