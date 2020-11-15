import numpy as np
import matplotlib.pyplot as plt


class MultiLayerNN:
    def __init__(self, data, raw_data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs):
        self.data = data
        self.raw_data = raw_data
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

        # Multiclass variables
        self.multicls_labels = dict()

        # Training statistics
        self.training_stats = {
            'correct': 0,
            'incorrect': 0,
            'total': 0,
            'accuracy': 0.0,
            'error': 0.0,
            'mse': 0.0,
        }
        self.errors = []
        self.classifications = None

        # Testing statistics
        self.testing_stats = {
            'correct': 0,
            'incorrect': 0,
            'total': 0,
            'accuracy': 0.0,
            'error': 0.0,
            'mse': 0.0,
        }
        self.testing_errors = []

    def train(self):
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

                self.update_error()
        self.plot_error()

    def initialize_multiclass_labels(self):
        """
        Separates out class labels in case of multiclass problems
        :return: None
        """
        for i, cls in enumerate(self.classes):
            self.multicls_labels[i] = np.where(self.raw_data['class'] == cls, 1, 0)

    def update_error(self):
        classifier = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.classifications = classifier(self.activations[-1])
        results = self.classifications == self.labels
        self.training_stats['correct'] = np.count_nonzero(results == True)
        self.training_stats['incorrect'] = np.count_nonzero(results == False)
        self.training_stats['total'] = self.training_stats['correct'] + self.training_stats['incorrect']
        accuracy = self.training_stats['correct'] / self.training_stats['total']
        self.training_stats['error'] = 1 - accuracy
        self.errors.append(1 - accuracy)

    def update_multi_error(self):
        results = self.classifications == self.labels
        self.training_stats['correct'] = np.count_nonzero(results == True)
        self.training_stats['incorrect'] = np.count_nonzero(results == False)
        self.training_stats['total'] = self.training_stats['correct'] + self.training_stats['incorrect']
        accuracy = self.training_stats['correct'] / self.training_stats['total']
        self.training_stats['error'] = 1 - accuracy
        self.errors.append(1 - accuracy)

    def update_regression_error(self):
        squared_diffs = np.power(np.abs(self.activations[-1] - self.labels), 2)
        results = np.abs(self.activations[-1] - self.labels) <= 10.0
        self.training_stats['correct'] = np.count_nonzero(results == True)
        self.training_stats['incorrect'] = np.count_nonzero(results == False)
        self.training_stats['total'] = self.training_stats['correct'] + self.training_stats['incorrect']
        accuracy = self.training_stats['correct'] / self.training_stats['total']
        self.training_stats['error'] = 1 - accuracy
        self.training_stats['mse'] = np.divide(np.sum(squared_diffs), self.labels.shape[1])
        self.errors.append(self.training_stats['mse'])

    def multi_classify(self, data, output):
        fake_labels = [_ for _ in range(len(self.classes))]
        actual_labels = {k:v for (k, v) in zip(fake_labels, self.classes)}
        o = output.T
        classifications = []
        for example in o:
            label, max_value = None, -np.inf
            for i, value in enumerate(example):
                if value > max_value:
                    max_value = value
                    label = actual_labels[i]
            classifications.append(label)
        self.classifications = classifications

    def plot_error(self):
        plt.plot(self.errors)
        plt.ylabel('Error')
        plt.savefig("error.png")

    def multi_train(self):
        self.initialize_multiclass_labels()
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
                if i == self.hidden_layers:
                    self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))
                else:
                    self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))

            # Backward propagation
            for i in range(self.hidden_layers, -1, -1):
                if i == self.hidden_layers:
                    self.weighted_sum_derivatives[i] = [0 for _ in range(len(self.classes))]
                    for j in range(len(self.classes)):
                        self.weighted_sum_derivatives[i][j] = self.activations[-1][j] - self.multicls_labels[j]
                    self.weight_derivatives[i] = (1/num_examples) * np.dot(self.weighted_sum_derivatives[i], self.activations[i].T)
                    self.bias_derivatives[i] = (1/num_examples) * np.sum(self.weighted_sum_derivatives[i], axis=1, keepdims=True)
                    self.weights[i] = self.weights[i] - self.learning_rate * self.weight_derivatives[i]
                    self.biases[i] = self.biases[i] - self.learning_rate * self.bias_derivatives[i]
                else:
                    self.activation_derivatives[i] = np.dot(self.weights[i + 1].T, self.weighted_sum_derivatives[i + 1])
                    self.weighted_sum_derivatives[i] = self.activation_derivatives[i] * \
                                                       (1 / (1 + np.exp(-self.weighted_sums[i]))) * \
                                                       (1 - (1 / (1 + np.exp(-self.weighted_sums[i]))))
                    self.weight_derivatives[i] = (1/num_examples) * np.dot(self.weighted_sum_derivatives[i], self.activations[i].T)
                    self.bias_derivatives[i] = (1/num_examples) * np.sum(self.weighted_sum_derivatives[i], axis=1, keepdims=True)
                    self.weights[i] = self.weights[i] - self.learning_rate * self.weight_derivatives[i]
                    self.biases[i] = self.biases[i] - self.learning_rate * self.bias_derivatives[i]

                self.multi_classify(self.data, self.activations[-1])
                self.update_multi_error()
        self.multi_classify(self.data, self.activations[-1])
        self.plot_error()

    def regression(self):
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
                if i == self.hidden_layers:
                    self.activations[i + 1] = self.weighted_sums[i]
                else:
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

                self.update_regression_error()
        self.plot_error()
        self.update_regression_error()

    def test(self, data, labels):
        self.activations[0] = data
        for i in range(self.hidden_layers + 1):
            self.weighted_sums[i] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))

        classifier = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.classifications = classifier(self.activations[-1])
        results = self.classifications == labels
        self.testing_stats['correct'] = np.count_nonzero(results == True)
        self.testing_stats['incorrect'] = np.count_nonzero(results == False)
        self.testing_stats['total'] = self.testing_stats['correct'] + self.testing_stats['incorrect']
        accuracy = self.testing_stats['correct'] / self.testing_stats['total']
        self.testing_stats['error'] = 1 - accuracy
        self.testing_errors.append(1 - accuracy)

    def multi_test(self, data, labels):
        self.activations[0] = data
        for i in range(self.hidden_layers + 1):
            self.weighted_sums[i] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            if i == self.hidden_layers:
                self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))
            else:
                self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))
        fake_labels = [_ for _ in range(len(self.classes))]
        actual_labels = {k:v for (k, v) in zip(fake_labels, self.classes)}
        o = self.activations[-1].T
        classifications = []
        for example in o:
            label, max_value = None, -np.inf
            for i, value in enumerate(example):
                if value > max_value:
                    max_value = value
                    label = actual_labels[i]
            classifications.append(label)
        self.classifications = classifications
        results = self.classifications == labels
        self.testing_stats['correct'] = np.count_nonzero(results == True)
        self.testing_stats['incorrect'] = np.count_nonzero(results == False)
        self.testing_stats['total'] = self.testing_stats['correct'] + self.testing_stats['incorrect']
        accuracy = self.testing_stats['correct'] / self.testing_stats['total']
        self.testing_stats['error'] = 1 - accuracy
        self.testing_errors.append(1 - accuracy)

    def regression_test(self, data, labels):
        self.activations[0] = data
        for i in range(self.hidden_layers + 1):
            self.weighted_sums[i] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            if i == self.hidden_layers:
                self.activations[i + 1] = self.weighted_sums[i]
            else:
                self.activations[i + 1] = 1 / (1 + np.exp(-self.weighted_sums[i]))
        squared_diffs = np.power(np.abs(self.activations[-1] - labels), 2)
        results = np.abs(self.activations[-1] - labels) <= 10.0
        self.testing_stats['correct'] = np.count_nonzero(results == True)
        self.testing_stats['incorrect'] = np.count_nonzero(results == False)
        self.testing_stats['total'] = self.testing_stats['correct'] + self.testing_stats['incorrect']
        accuracy = self.testing_stats['correct'] / self.testing_stats['total']
        self.testing_stats['error'] = 1 - accuracy
        self.testing_stats['mse'] = np.divide(np.sum(squared_diffs), self.testing_stats['total'])
        self.testing_errors.append(self.testing_stats['mse'])

    def report_classifications(self):
        print(self.classifications)

    def get_training_error(self):
        return self.training_stats['error']

    def get_training_mse(self):
        return self.training_stats['mse']

    def get_testing_error(self):
        return self.testing_stats['error']

    def get_testing_mse(self):
        return self.testing_stats['mse']
