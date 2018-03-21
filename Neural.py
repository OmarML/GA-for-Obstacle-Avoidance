import random
import math as m
import numpy as np
import functools


class NeuralNetwork:
		def __init__(self, inputs, hidden_layers, hidden_neurons, outputs, given_weights):
				self.inputs = inputs
				self.hidden_layers = hidden_layers
				self.hidden_neurons = hidden_neurons
				self.outputs = outputs
				self.given_weights = given_weights
				self.weights = given_weights

		def activation(self, z, logistic=False, htan=False):
				if logistic:
						return 1 / 1 + np.exp(-z)
				elif htan:
						return np.tanh(z)

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



g = [np.array([[-0.93505587,  0.9224942 ,  1.9651044 ],
       [ 0.31516236, -2.59470964,  1.19106615],
       [-1.43315243, -0.1763587 , -1.52025321],
       [ 0.07718855, -0.34538581,  0.56921631]]), np.array([[-1.52457222,  0.47035277],
       [-1.19483094,  0.97760238],
       [-1.03914763,  0.03431391]])]




if __name__ == '__main__':
		nn = NeuralNetwork(inputs=4, hidden_layers=1, hidden_neurons=3, outputs=2, given_weights=g)
		a = nn.weights
		# print(a)
		print(nn.weights)
		# nn.weights = [np.array([[-0.18569203, -1.15955254,  0.98317866],
     #   [-0.6967923 ,  0.68024923, -0.15005902],
     #   [ 0.6443928 , -2.061897  ,  1.10600649],
     #   [ 1.59231743, -2.21725663, -1.20812529]]), np.array([[-0.21087811, -0.59785633],
     #   [-0.07869691,  0.04648555],
     #   [-1.10778754,  1.7161734 ]])]
		print(nn.weights)
		# print(a)
		# print(nn.forward([21, 34, 55, 12]))
		# print(nn.forward([21, 34, 55, 12]))
		genome = convert_to_genome(a)
		print(len(genome))
		print(genome[random.randint(0, len(genome))])
		print(genome)
		weights = convert_to_weights(genome, a)
		print(weights)
		print(np.array_equal(a, weights))
		# print(np.array_equal(weights[0], a[0]))
		# print(np.array_equal(weights[1], a[1]))
		# print(np.array_equal(weights[2], a[2]))
		# print(a is type(weights))
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

# array([-0.84544668,  0.79015025])
