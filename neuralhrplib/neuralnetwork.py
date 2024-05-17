import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def step_function(x):
    return np.where(x <= 0, 0, 1)


def step_derivative(x):
    return 0  # Aproximación para la derivada de la función escalón


def identity_function(x):
    return x


def identity_derivative(x):
    return 1


class NeuralNetwork:
    def __init__(self, layers, activations='sigmoid', activations_output='step'):
        if activations == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activations == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivative


        if activations_output == 'step':
            self.activation_output = step_function
            self.activation_output_prime = step_derivative
        elif activations_output == 'identity':
            self.activation_output = identity_function
            self.activation_output_prime = identity_derivative


        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.biases = [np.random.randn(1, layers[i+1]) for i in range(len(layers) - 1)]
        self.losses = []


    def get_weighted_sum(self, weights, feature, bias):
        return np.dot(feature, weights) + bias


    def cross_entropy(self, target, prediction):
        epsilon = 1e-10  # Valor pequeño para evitar log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)  # Limitar predicción para evitar log(0)
        cost = -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))
        return cost


    def update_weights(self, learning_rate, target, prediction, feature):
        new_weights = []
        for x, w in zip(feature, self.weights):
            new_w = w + learning_rate * (target - prediction) * x
            new_weights.append(new_w)
        return new_weights


    def update_bias(self, bias, learning_rate, target, prediction):
        return bias + learning_rate * (target - prediction)


    def training(self, X, targets, epochs=200, learning_rate=0.01, log_epoch_rate=300):
        for epoch in range(epochs):
            epoch_deltas = []  # List to store the magnitudes of deltas for each iteration
            for i in range(X.shape[0]):
                # Forward propagation
                activations = [X[i]] # Take the value for one of the rows in the input data and put it in an array (e.g., [1.0, 0.5, 1.0])
                for l in range(len(self.weights)):
                    weighted_sum = self.get_weighted_sum(self.weights[l], activations[l], self.biases[l]) # Compute weighted sum
                    activation = self.activation(weighted_sum) if l < len(self.weights) - 1 else self.activation_output(weighted_sum) # Apply activation function
                    activations.append(activation) # Append activation to the list

                # Calculate the error (difference between target and output)
                error = targets[i] - activations[-1] # Calculate error for the output layer

                # Backward propagation
                deltas = [error * (self.activation_prime(activations[-1]))] # Compute delta for the output layer

                for l in range(len(self.weights) - 2, -1, -1): # Iterate backward through the layers
                    error = deltas[-1].dot(self.weights[l + 1].T) # Calculate error for the current layer
                    delta = error * self.activation_prime(activations[l + 1]) # Compute delta for the current layer
                    deltas.append(delta) # Append delta to the list

                deltas.reverse() # Reverse deltas to match layer order

                # Calculate the magnitude of the deltas
                epoch_deltas.append(np.mean([np.linalg.norm(delta) for delta in deltas])) # Store mean magnitude of deltas


                # Update weights and biases
                for l in range(len(self.weights)):
                    self.weights[l] += learning_rate * np.outer(activations[l], deltas[l]) # Update weights using gradient descent
                    self.biases[l] += learning_rate * deltas[l]  # Update biases


            # Store the average magnitude of deltas for the epoch
            self.losses.append(np.mean(epoch_deltas)) # Store mean delta magnitude for the epoch


            # Check for unexpected increase in deltas
            if epoch > 0 and np.mean(epoch_deltas) > 2 * self.losses[-1]:  # Arbitrary threshold to detect abrupt increases
                print(f"Warning: Abrupt increase in deltas at epoch {epoch}, avg_delta: {np.mean(epoch_deltas)}")


            # Print the epoch number and average delta magnitude every log_epoch_rate epochs
            if epoch % log_epoch_rate == 0:
                print(f'epochs: {epoch}, avg_cost_entropy: {np.mean(epoch_deltas)}')


    def predict(self, x):
        a = x  # Initialize 'a' with the input data 'x'
        for l in range(len(self.weights)): # Iterate through each layer
            # For all layers except the last, apply the hidden layer activation function
            # For the last layer, apply the output layer activation function
            a = self.activation(np.dot(a, self.weights[l]) + self.biases[l]) if l < len(self.weights) - 1 else self.activation_output(np.dot(a, self.weights[l]) + self.biases[l])
        return a # Return the final output after passing through all layers


    def get_losses(self):
        return self.losses

