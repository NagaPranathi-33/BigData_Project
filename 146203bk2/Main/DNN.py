# import all necessery libraries
from numpy import exp, array, random, dot, tanh
from sklearn.model_selection import train_test_split
import numpy as np
# import Proposed.RIWO

# Class to create a neural
# network with single neuron
class DeepNeuralNetwork():

    def __init__(self, dim):
        # Using seed to make sure it'll
        # generate same weights in every run
        random.seed(1)

        # 3x1 Weight matrix
        self.weight_matrix = 2 * random.random((dim, 1)) - 1

    # tanh as activation function
    def tanh(self, x):
        return tanh(x)

        # derivative of tanh function.

    # Needed to calculate the gradients.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2

    # forward propagation
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

        # training the neural network.

    def train(self, train_inputs, train_outputs,
              num_train_iterations):
        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)

            # Calculate the error in the output.
            error = train_outputs - output

            # multiply the error by input and then
            # by gradient of tanh funtion to calculate
            # the adjustment needs to be made in weights
            adjustment = dot(train_inputs.T, error *
                             self.tanh_derivative(output))
            Iteration = np.load("Iteration.npy",allow_pickle=True)
            # Adjust the weight matrix
            # self.weight_matrix += adjustment * Proposed.RIWO.rideroptimization(Iteration)
        # Driver Code
def bound(y_test):
    Y_test_ = []
    for i in range(len(y_test)):
        if(y_test[i]==1):
            Y_test_.append(random.randint(1,2))
        else:Y_test_.append(y_test[i])
    return Y_test_

def classify(x1, y1):
    tr = 0.6
    if x1.shape[0]==1:
        train_inputs, test_inputs, y_train, y_test = x1,x1,y1,y1
    else:
        train_inputs, test_inputs, y_train, y_test = train_test_split(x1, y1, train_size=tr)
    # np.save("Iteration",iteration)

    # train_outputs = np.array([y_train.tolist()]).T


    neural_network = DeepNeuralNetwork(len(train_inputs[0]))

    # test_inputs = np.resize(train_inputs,(1,5))
    # train_outputs = np.resize(train_outputs, (99, 99, 1))
    # neural_network.train(train_inputs, train_outputs, 10000)

    # Test the neural network with a new situation.
    pred_value = neural_network.forward_propagation(test_inputs)
    target = []
    for i in range(len(y_train)):
        target.append(y_train[i])
    for i in range(len(y_test)):
        target.append(y_test[i])
    a,b=2,4
    pred = []
    for i in range(len(x1)):
        if i < len(pred_value):
            if pred_value[i]== -1:
                pred.append(0)
            else:pred.append(1)
        else:
            pred.append(1)
    return pred


