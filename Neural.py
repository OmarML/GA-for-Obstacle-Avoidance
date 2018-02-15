import random
import math as m
import numpy as np
import functools


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


def convert_to_genome(weights_array):
		return np.concatenate([np.ravel(i) for i in a])


def convert_to_weights(genome, weights_array):
		shapes = [np.shape(i) for i in weights_array]
		# print(shapes)
		products = ([(i[0] * i[1]) for i in shapes])
		# print (products)
		out = []
		start = 0
		for i in range(len(products)):
				# print(sum(products[:i+1]))
				# print(start, sum(products[:i+1]))
				out.append(np.reshape(genome[start:sum(products[:i+1])], shapes[i]))
				start += products[i]
		return out






if __name__ == '__main__':
		nn = NeuralNetwork(inputs=4, hidden_layers=1, hidden_neurons=3, outputs=2)
		a = nn.weights
		print(a)
		genome = convert_to_genome(a)
		# print(genome)
		weights = convert_to_weights(genome, a)
		print(weights)
		# print(np.array_equal(a, weights))
		print(np.array_equal(weights[0], a[0]))
		print(np.array_equal(weights[1], a[1]))
		# print(np.array_equal(weights[2], a[2]))
		print(a is type(weights))
# 		print(len(a))
# 		shapes = ([np.shape(i) for i in a])
# 		# print(shapes)
# 		# functools.reduce(lambda (x: x*y, [np.shape(i) for i in a])
# 		print([(i[0]*i[1]) for i in shapes])
# 		print(a)
#
# 		print([np.ravel(i) for i in a])
# 		b = (np.concatenate([np.ravel(i) for i in a]))
# 		print(b)
# 		# print(np.shape(b))
# 		# c = np.reshape(b[0:(9*16)], (9,16))
# 		# d = np.reshape(b[(9*16):(16*16)+(9*16)], (16,16))
# 		# e = np.reshape(b[(9*16)+(16*16):], (16, 2))
# 		# print(a[0])
# 		# print(np.array_equal(c, a[0]))
# 		# print(np.array_equal(d, a[1]))
# 		# print(np.array_equal(e, a[2]))
# # print(nn.forward([55,0.03,0.3,9]))