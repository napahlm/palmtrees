import numpy as np
import matplotlib.pyplot as plt

# TODO: Look it over, it isn't working right

class SimpleNeuralNetwork:

    def __init__(self, x, y):
        """
        This simple neuron network utilizes one vector of inputs, one hidden layer, and
        an output vector that will optimize under usage.

        Args:
            input (ndarray) : Vectors of the inputs
            true_output (ndarray) : The results this network will strive to achieve
        """
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 4)
        self.weights2   = np.random.rand(4, 1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        """
        Feedforward is process from the neurons to the output.
        The neurons are weighted and can be viewed in the output.
        """
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        """
        The backpropagation takes the results from the feedforward, and readjusts the
        weights going backwards through each layer.
        """
        d_weights2 = np.dot(self.layer1.T, 
                            (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output),
                            self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Updating the weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    

# The loss fucntion for plotting
def loss(true_output, predicted_output):
    return (true_output[0] - predicted_output[0])**2

# Adding a classic activation function: The sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


# Testing it on a XNOR gate

xinput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
true_output = np.array([[1], [0],[0], [1]])

# Initialization
nn = SimpleNeuralNetwork(xinput, true_output)
n = 1500
losslist = []

for i in range(n):
    nn.feedforward()
    nn.backpropagation()
    losslist.append(loss(nn.y[0], nn.output[0]))

plt.plot(losslist)
plt.show()