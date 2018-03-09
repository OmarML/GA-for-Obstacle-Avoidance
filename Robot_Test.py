from Sensor import Robot
from Pedestrian import all, Darwin
import numpy as np



a = Robot(200, 300, 8, 360, 9, all)
b = Robot(200, 300, 8, 360, 9, all)
print(a, b)
print(a.fitness)
robot_array = [a, b]
print([i.fitness for i in robot_array])
darwin = Darwin(robot_array, 2, 1)
genome = darwin.convert_to_genome(a.DNA)
weights = darwin.convert_to_weight(genome, a.DNA)
print(np.shape(weights), np.shape(a.DNA))
c = darwin.create_child(a, b)
# print(np.array_equal(a.DNA, a.brain.weights))
# print(a.DNA)
# print(a.brain.weights)
a.DNA = c


for i in range(3):
	print(np.array_equal(a.DNA[i], a.brain.weights[i]))