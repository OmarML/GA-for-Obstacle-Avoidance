from Sensor import Robot
from Pedestrian import all, Darwin
import numpy as np



a = Robot(200, 300, 8, 360, 9, all)
b = Robot(200, 300, 8, 360, 9, all)
print(a.fitness)
robot_array = [a, b]
print([i.fitness for i in robot_array])
darwin = Darwin(robot_array, 2, 1)
genome = darwin.convert_to_genome(a.DNA)
weights = darwin.convert_to_weight(genome, a.DNA)
print(np.shape(weights), np.shape(a.DNA))
c = darwin.create_child(a, b)
for i in range(3):
	print(np.array_equal(b.DNA[i], c[i]))