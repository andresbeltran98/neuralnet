import numpy as np

# Learning rate
LR = 0.1
MAX_EPOCHS = 10000


class Unit:
    """
    Represents a single neuron unit
    """
    def __init__(self, bias):
        self.bias = bias
        self.output = 0
        self.error = 0


class NeuralNetwork:
    """
    Represents a NN with one hidden layer
    """
    def __init__(self, lr, hidden_weights, output_weights, hidden_units, output_units):
        self.lr = lr
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_units = [Unit(b) for b in hidden_units]
        self.output_units = [Unit(b) for b in output_units]

    def train(self, dataset):
        """
        Trains this neural network with a given dataset
        :param dataset: input dataset
        """
        epochs = 0
        while epochs < MAX_EPOCHS:
            for ex in dataset:
                self.feed_forward(ex)
                self.back_prop(ex)
            epochs += 1

    def predict(self, example):
        """
        Predicts the class of a novel example
        """
        output = []

        output_hidden = []
        for i in range(len(self.hidden_weights[0])):
            h_unit = self.hidden_units[i]
            weights = self.hidden_weights[:, i]
            output_hidden.append(sigmoid(np.dot(weights, example) + h_unit.bias))

        for i in range(len(self.output_weights[0])):
            o_unit = self.output_units[i]
            weights = self.output_weights[:, i]  # weights of output unit i
            weights = [i for i in weights if i]
            output.append(sigmoid(np.dot(weights, output_hidden) + o_unit.bias))

        output = [(1 if x >= 0.5 else 0) for x in output]
        return output

    def feed_forward(self, example):
        """
        Feed a given example to the network
        :param example: training example
        """
        # hidden layer
        for i in range(len(self.hidden_weights[0])):
            h_unit = self.hidden_units[i]
            weights = self.hidden_weights[:, i]
            h_unit.output = sigmoid(np.dot(weights, example[:-1]) + h_unit.bias)

        # output layer
        for i in range(len(self.output_weights[0])):
            o_unit = self.output_units[i]
            weights = self.output_weights[:, i]  # weights of output unit i
            out_hidden = self.out_hidden(weights)  # output of hidden units
            weights = [i for i in weights if i]
            o_unit.output = sigmoid(np.dot(weights, out_hidden) + o_unit.bias)

    def back_prop(self, example):
        """
        Backpropagation algorithm
        :param example: training example
        """
        lbl_idx = 0
        cl_lbls = example[-1]

        # Calculate errors in neurons
        for unit in self.output_units:
            out = unit.output
            unit.error = out * (1 - out) * (cl_lbls[lbl_idx]-out)
            lbl_idx += 1

        for i in range(len(self.hidden_units)):
            unit_hidden = self.hidden_units[i]
            out = unit.output

            weights = self.output_weights[i]  # weights from hidden unit i
            errors = self.error_output(weights)  # errors from output units connected to hidden unit i
            weights = [i for i in weights if i]
            unit_hidden.error = out * (1 - out) * np.dot(errors, weights)

        # Update weights and biases
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[0])):
                self.output_weights[i][j] += self.lr * self.output_units[j].error * self.hidden_units[i].output

        for i in range(len(self.hidden_weights)):
            for j in range(len(self.hidden_weights[0])):
                self.hidden_weights[i][j] += self.lr * self.hidden_units[j].error * example[i]

        for unit in self.output_units:
            unit.bias += self.lr * unit.error

        for unit in self.hidden_units:
            unit.bias += self.lr * unit.error

    def out_hidden(self, weights):
        unit_idx = [i for i in range(len(weights)) if weights[i] is not None]
        out_hidden = []
        for idx in unit_idx:
            out_hidden.append(self.hidden_units[idx].output)
        return out_hidden

    def error_output(self, weights):
        unit_idx = [i for i in range(len(weights)) if weights[i] is not None]
        err_out = []
        for idx in unit_idx:
            err_out.append(self.output_units[idx].error)
        return err_out


def sigmoid(x):
    """
    Outputs the result of the sigmoid function
    """
    return 1/(1+np.exp(-x))


# Rows are input units, cols are hidden units
hidden_weights = np.array(([0.1, 0, 0.3],
                           [-0.2, 0.2, -0.4]))

# Rows are hidden units, cols are output units
output_weights = np.array(([-0.4, 0.2],
                           [0.1, -0.1],
                           [0.6, -0.2]))

# Hidden-layer biases
hidden_biases = [0.1, 0.2, 0.5]

# Output-layer biases
output_biases = [-0.1, 0.6]

# Dataset
examples = [[0.6, 0.1, [1, 0]],
            [0.2, 0.3, [0, 1]]]

# Train Neural Network
n = NeuralNetwork(LR, hidden_weights, output_weights, hidden_biases, output_biases)
n.train(examples)

print('Resulting NN after', MAX_EPOCHS, 'epochs...')
print('Weights to Output units\n', n.output_weights)
print('Weights to Hidden units\n', n.hidden_weights)
print('Biases of Output units: ', [o.bias for o in n.output_units])
print('Biases of Hidden units: ', [o.bias for o in n.hidden_units])

# Predictions
ex_1 = [0.6, 0.1]
ex_2 = [0.2, 0.3]
print('\nPredict', ex_1, ':', n.predict(ex_1))
print('Predict', ex_2, ':', n.predict(ex_2))

