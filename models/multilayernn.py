import numpy as np
import matplotlib.pyplot as plt


class MultiLayerNN:
    def __init__(self, data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs):
        self.data = data
        self.labels = labels
        self.classes = classes
        self.hidden_layers = hidden_layers
        self.num_nodes = num_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = []
        self.biases = []

        # Forward propagation parameters
        self.weighted_sums = [0 for _ in range(hidden_layers + 1)]
        self.activations = [self.data] + [0 for _ in range(hidden_layers + 1)]

        # Backward propagation parameters
        self.activation_derivatives = [0 for _ in range(hidden_layers + 1)]
        self.weighted_sum_derivatives = [0 for _ in range(hidden_layers + 1)]
        self.weight_derivatives = [0 for _ in range(hidden_layers + 1)]
        self.bias_derivatives = [0 for _ in range(hidden_layers + 1)]

    def train(self):
        errors = []

        num_examples = self.data.shape[1]

        # Initialize weights and biases
        for i in range(self.hidden_layers + 1):
            self.weights.append(np.random.uniform(low=-0.01, high=0.01, size=(self.num_nodes[i + 1], self.num_nodes[i])))
            self.biases.append(np.random.uniform(low=-0.01, high=0.01, size=self.num_nodes[i + 1]).reshape(self.num_nodes[i + 1], 1))

        # Main loop
        for epoch in range(self.epochs):
            # Forward propagation
            for i in range(self.hidden_layers + 1):
                self.weighted_sums[i] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
                self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))

            # Backward propagation
            for i in range(self.hidden_layers, -1, -1):
                if i == self.hidden_layers:
                    self.activation_derivatives[i] = -(self.labels / self.activations[-1]) + \
                                                     (1 - self.labels) / (1 - self.activations[-1])
                else:
                    self.activation_derivatives[i] = np.dot(self.weights[i + 1].T, self.weighted_sum_derivatives[i + 1])
                if i == self.hidden_layers:
                    self.weighted_sum_derivatives[i] = self.activations[-1] - self.labels
                else:
                    self.weighted_sum_derivatives[i] = self.activation_derivatives[i] * \
                                                       (1 / (1 + np.exp(-self.weighted_sums[i]))) * \
                                                       (1 - (1 / (1 + np.exp(-self.weighted_sums[i]))))
                self.weight_derivatives[i] = (1/num_examples) * np.dot(self.weighted_sum_derivatives[i], self.activations[i].T)
                self.bias_derivatives[i] = (1/num_examples) * np.sum(self.weighted_sum_derivatives[i], axis=1, keepdims=True)

                self.weights[i] = self.weights[i] - self.learning_rate * self.weight_derivatives[i]
                self.biases[i] = self.biases[i] - self.learning_rate * self.bias_derivatives[i]

                errors.append(self.get_error())
        self.plot_error(errors)
        print(self.activations[-1])

    def get_error(self):
        classifier = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        output = classifier(self.activations[-1])
        results = output == self.labels
        correct = np.count_nonzero(results == True)
        incorrect = np.count_nonzero(results == False)
        classified = correct + incorrect
        accuracy = correct / classified
        error = 1 - accuracy
        return error

    def plot_error(self, errors):
        plt.plot(errors)
        plt.ylabel('Error')
        plt.savefig("error.png")
