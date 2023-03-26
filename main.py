import cupy as np
import os
import matplotlib.pyplot as plt
import copy
import time

os.chdir('C:/Users/Will Schulte/Desktop/train_data')

# load training data from csv file
X_train_csv = np.array(np.genfromtxt('training_data.csv', delimiter=',')).T  # X
Y_train_csv = np.array(np.genfromtxt('training_data_key.csv', delimiter=','))  # Y

os.chdir('C:/Users/Will Schulte/Desktop/test_data')
X_test_csv = np.array(np.genfromtxt('test_data.csv', delimiter=',')).T  # X
Y_test_csv = np.array(np.genfromtxt('test_data_key.csv', delimiter=','))  # Y

os.chdir('C:/Users/Will Schulte/Desktop/train_data')

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros(dim)
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]
    # forward propagation
    small = 1e-5
    a = sigmoid(np.dot(w, X.T) + b)
    cost = (-1 / m) * (np.sum(Y * np.log(a + small) + (1 - Y) * np.log(1 - a + small)))
    # backward propagation
    dw = np.dot(X.T, (a - Y))
    db = (a - Y)
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_Iterations, learning_rate, print_cost):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_Iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = np.zeros((1, m))
    # removed bias due to shape issues
    A = sigmoid(np.dot(w, X.T))
    A = A.reshape(A.shape[0], 1)

    for i in range(A.shape[0]):
        if A[i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 7000, learning_rate = 0.00001, print_cost=True):
    w, b = initialize_with_zeros(X_train[0].shape)

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "test_accuracy": 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}

    return d


m = model(X_train_csv, Y_train_csv, X_test_csv, Y_test_csv)

# x = 50
# y = 0.01
# best = 0
# best_m = dict
# for num_I in range(10):
#    for LR in range(100):
#        dict = model(X_train_csv, Y_train_csv, X_test_csv, Y_test_csv, x, y)
#        if dict["test_accuracy"] > best:
#            best = dict["test_accuracy"]
#            best_m = dict
#        y += 0.01
#    x += 25

# print(best_m)

#plt.plot(numpy.squeeze(m["costs"]), label=str(m["learning_rate"]))
#plt.ylabel('cost')
#plt.xlabel('iterations (hundreds)')
#legend = plt.legend(loc='upper center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()
