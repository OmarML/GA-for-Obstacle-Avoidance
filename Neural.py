import random
import math as m
import numpy as np


class NeuralNetwork:
		def __init__(self, inputs, hidden_layers, hidden_neurons, outputs):
				self.inputs = inputs
				self.hidden_layers = hidden_layers
				self.hidden_neurons = hidden_neurons
				self.outputs = outputs
				self.weights = self.create_weights()

		def activation(self, z, logistic=False, htan=False):
				if logistic:
						return 1 / 1 + np.exp(-z)
				elif htan:
						return np.tanh(z)

		def create_weights(self):
				w1 = np.random.randn(self.inputs, self.hidden_neurons)
				wl = np.random.randn(self.hidden_neurons, self.outputs)
				self.weights = []
				self.weights.append(w1)
				for i in range(self.hidden_layers - 1):
						self.weights.append(np.random.rand(self.hidden_neurons, self.hidden_neurons))
				self.weights.append(wl)
				return self.weights

		def forward(self, X):
				new_X = self.activation(np.dot(X, self.weights[0]), htan=True)
				for i in self.weights[1:]:
						new_X = np.dot(new_X, i)
						new_X = self.activation(new_X, htan=True)
				return new_X

# testing github

# nn = NeuralNetwork(inputs=3, hidden_layers=1, hidden_neurons=3, outputs=2)
# a = nn.weights
# print(a)
# print([np.ravel(i) for i in a])

# print(nn.forward([55,0.03,0.3,9]))