import cupy as cp
import os
import matplotlib.pyplot as plt
import copy

DATA_DIR = 'C:/Users/Will Schulte/Desktop'
TRAIN_DATA = os.path.join(DATA_DIR, 'train_data')
TEST_DATA = os.path.join(DATA_DIR, 'test_data')


def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = cp.zeros((1, m))
    # removed bias due to shape issues
    A = sigmoid(cp.dot(w, X.T))
    A = A.reshape(A.shape[0], 1)

    for i in range(A.shape[0]):
        if A[i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def sigmoid(z):
    return 1 / (1 + cp.exp(-z))


def propagate(w, b, X, Y):
    m = X.shape[0]
    # forward propagation
    small = 1e-5
    a = sigmoid(cp.dot(w, X.T) + b)
    cost = (-1 / m) * (cp.sum(Y * cp.log(a + small) + (1 - Y) * cp.log(1 - a + small)))
    # backward propagation
    dw = cp.dot(X.T, (a - Y))
    db = cp.sum(a - Y)
    return {"dw": dw, "db": db}, cp.squeeze(cp.array(cost))


def optimize(w, b, X, Y, num_Iterations, learning_rate, print_info):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []
    accuracy = []

    for i in range(num_Iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            accuracy.append((100 - cp.mean(cp.abs(predict(w, b, X) - Y)) * 100)/100)
            if print_info:
                print("Cost after iteration %i: %f" % (i, cost))
                print("Accuracy after iteration %i: %f" % (i, accuracy[int(i/100)]/100))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs, accuracy


def initialize_with_zeros(dim):
    w = cp.zeros(dim)
    b = cp.random.randn()
    return w, b


def model(X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.00001, print_info=True):
    w, b = initialize_with_zeros(X_train[0].shape)

    params, grads, costs, accuracy = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_info)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_info:
        print("train accuracy: {} %".format(100 - cp.mean(cp.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - cp.mean(cp.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "accuracy": accuracy,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "test_accuracy": 100 - cp.mean(cp.abs(Y_prediction_test - Y_test)) * 100}

    return d


def load_data():
    os.chdir(TRAIN_DATA)
    X_train = cp.genfromtxt('training_data.csv', delimiter=',').T
    Y_train = cp.genfromtxt('training_data_key.csv', delimiter=',')
    os.chdir(TEST_DATA)
    X_test = cp.genfromtxt('test_data.csv', delimiter=',').T
    Y_test = cp.genfromtxt('test_data_key.csv', delimiter=',')
    os.chdir(TRAIN_DATA)
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data()

m = model(X_train, Y_train, X_test, Y_test)

costs = [cp.asnumpy(cost) for cost in m["costs"]]
accuracy = [cp.asnumpy(accuracy) for accuracy in m["accuracy"]]
iterations = range(0, len(costs) * 100, 100)

plt.plot(iterations, costs, label='Cost')
plt.plot(iterations, accuracy, label='Training Accuracy')

test_accuracy = m["test_accuracy"]
accuracy_text = f"Test Accuracy: {test_accuracy:.2f}%"
plt.text(0.6 * max(iterations), 0.5 * max(costs), accuracy_text, fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

plt.xlabel('Iterations (hundreds)')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.grid(True)
plt.show()
