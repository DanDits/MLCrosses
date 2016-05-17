import numpy as np
import random

# Cost: How we measure the error the network did
class CrossEntropyCost: # works best with sigmoid neuron
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) # non negative value indicating the error

    @staticmethod
    def delta(z, y, Neuron): # derivative of fn(a(z),y) wrt. z; vector result for starting backpropagation
        if Neuron is SigmoidNeuron:
            return Neuron.activation(z) - y # simplified, as in sigmoid case: activation prime(z) = a*(1-a)
        else:
            a = Neuron.activation(z)
            return np.nan_to_num((a - y) * Neuron.activation_prime(z) / (a * (1 - a)))


class QuadraticCost: # works best with linear neuron
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2
    @staticmethod
    def delta(z, y, Neuron):
        return (Neuron.activation(z) - y) * Neuron.activation_prime(z)

# How a neuron in the network is activated, that is a continuous mapping from real values to the range [0,1]
class SigmoidNeuron():
    @staticmethod
    def activation(z):
        # When the neuron is activated and fired, a vectorized value between 0 and 1
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def activation_prime(z):
        # The derivative of the activation function
        return SigmoidNeuron.activation(z) * (1. - SigmoidNeuron.activation(z))


class LinearNeuron():
    @staticmethod
    def activation(z):
        return np.clip(z, -0.5, 0.5) + 0.5
    @staticmethod
    def activation_prime(z):
        x = np.ones(z.shape)
        x[np.where(x < -0.5)] = 0.
        x[np.where(x > 0.5)] = 0.
        return x


class Network:

    def __init__(self, in_size, out_size, cost, neuron):
        self.cost = cost
        self.neuron = neuron
        self.sizes = [in_size, out_size]

        # so later: weight matrix * input + bias = estimate
        self.bias = np.random.randn(out_size, 1)
        self.weight = np.random.randn(out_size, in_size) / np.sqrt(in_size)

    def feedforward(self, x):
        # Feed data vector in network, giving out an estimate
        return self.neuron.activation(np.dot(self.weight, x) + self.bias)

    def accuracy(self, data, print_errors=True):
        # given the vectorized data, check how many estimates match the actual targets
        errors = [name for (x, y, name) in data if int((np.argmax(self.feedforward(x)) != np.argmax(y)))]
        if print_errors:
            for name in errors:
                print(name)
        return len(data) - len(errors)

    def total_cost(self, data):
        # Given vectorized data, calculate the cost to the estimate
        cost = 0.
        for x, y, _ in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y) / len(data)
        return cost

    def backprop(self, x, y):
        # Return the gradient of the cost function for the weights and bias

        # feed forward x, do not yet activate it
        z = np.dot(self.weight, x) + self.bias

        # backward pass
        delta = self.cost.delta(z, y, self.neuron)
        nabla_b = delta
        nabla_w = np.dot(delta, x.transpose())
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        # update network weight and bias using gradient descent
        nabla_b = np.zeros(self.bias.shape)
        nabla_w = np.zeros(self.weight.shape)
        for x, y, _ in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b += delta_nabla_b
            nabla_w += delta_nabla_w
        self.weight -= (eta / len(mini_batch)) * nabla_w
        self.bias -= (eta / len(mini_batch)) * nabla_b

    def SGD(self, training_data, epochs, mini_batch_size, eta, evaluation_data=None, print_errors=False):
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch %s training complete" % j)
            if evaluation_data:
                cost = self.total_cost(evaluation_data)
                print("Cost on evaluation data: {}".format(cost))
                accuracy = self.accuracy(evaluation_data, print_errors)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, len(evaluation_data)))
