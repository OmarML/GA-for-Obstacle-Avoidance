from Pygame import screen, width, height
import pygame
import math as m
import numpy as np
import random
import Neural
import time
from Vec2d import Vec2d

class Robot:
		def __init__(self, x, y, size, field_of_view, num_sensors, manager, set_weights, own_weights=False):
				self.x = x
				self.y = y
				self.start_x = x
				self.start_y = y
				self.size = size
				self.colour = (255, 255, 255)
				self.max_range = 75
				self.sensors = [(Sensor(self.x, self.y, self.size, 0, self.max_range, 0))]
				self.robot_angle = 0 # robot's orientation make sensor class take this into account
				self.manager = manager
				self.speed = random.random()
				self.angle = random.uniform(-0.5, 0.5)
				self.alive = True
				self.set_weights = set_weights
				if own_weights:
						self.brain = Neural.NeuralNetwork(inputs=num_sensors, hidden_layers=2, hidden_neurons=16, outputs=2, given_weights=set_weights)
				else:
						self.brain = Neural.NeuralNetwork(inputs=num_sensors, hidden_layers=2, hidden_neurons=16, outputs=2, given_weights=self.create_weights(num_sensors, 16, 2, 2)) # this will be the neural network which makes the decision based on sensor inputs
				self.DNA = self.brain.weights # this will be an array which contains the all weights of the NN
				self.fitness = 0
				self.time_alive = time.time()

				angle = field_of_view / num_sensors
				angle_const = field_of_view / num_sensors
				temp = int(num_sensors / 2)
				id = 1
				for i in range(1, temp+1):
						self.sensors.append(Sensor(self.x, self.y, self.size, angle, self.max_range, id))
						id += 1
						self.sensors.append(Sensor(self.x, self.y, self.size, -angle, self.max_range, id))
						angle += angle_const
						id += 1

		def create_weights(self, inputs, hidden_neurons, hidden_layers, outputs):
				w1 = np.random.randn(inputs, hidden_neurons)
				wl = np.random.randn(hidden_neurons, outputs)
				self.weights = []
				self.weights.append(w1)
				for i in range(hidden_layers - 1):
						self.weights.append(np.random.randn(hidden_neurons, hidden_neurons))
				self.weights.append(wl)
				return self.weights

		def move(self):
				if self.alive:
					self.x += 4*self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[0] * m.cos(m.radians(self.robot_angle))
					self.y += 4*self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[0] * m.sin(m.radians(self.robot_angle))
					self.robot_angle += self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[1] # this changes between -1 and 1 degrees per decision
				# 	print(self.brain.forward([(sensor.reading) for sensor in self.sensors]))
				# 	self.x += self.speed * m.cos(m.radians(self.robot_angle))
				# 	self.y += self.speed * m.sin(m.radians(self.robot_angle))
				# 	self.robot_angle += self.angle

		def update(self):
				if self.alive:
						pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, 0)
						for sensor in self.sensors:
								sensor.draw_sensor(self)
								# sensor.detect(self.manager)
								sensor.collide(self, self.manager)
				if time.time() - self.time_alive > 35: # add condition to check if fitness isnt changing much not just time for killing
						self.alive = False
								# sensor.evaluate_fitness(self)
						# print([sensor.reading for sensor in self.sensors])
						# print("Sensor reading {} is:{}".format(sensor.id, sensor.reading)) # need to think about this line

		def evaluate_fitness(self):
				if self.alive:
						target_pos = Vec2d(800, 200)
						start = Vec2d(self.start_x, self.start_y)
						robot_pos = Vec2d(self.x, self.y)
						angle = robot_pos.get_angle_between(target_pos)
						total = start.get_distance(target_pos)
						distance = robot_pos.get_distance(target_pos)
						# print( (total - distance) / (total) )
						pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (800, 200))
						try:
								# self.fitness = 1/(distance*m.cos(m.radians(angle)))
							self.fitness = (total - distance) / total
						except ZeroDivisionError:
								self.fitness = 1



class Sensor:
		def __init__(self, robot_x, robot_y, robot_r, angle, max_range, id):
				self.robot_x = robot_x
				self.robot_y = robot_y
				self.max_range = max_range
				self.angle = angle
				self.radius = robot_r
				self.id = id
				self.reading = self.max_range
				self.x0 = 0
				self.y0 = 0
				self.x1 = 0
				self.x1 = 0
				self.y1 = 0

		def update_sensor_pos(self, robot):
				self.x0 = robot.x + self.radius * m.cos(m.radians(self.angle+robot.robot_angle)) # do robot_angle + self.angle here
				self.y0 = robot.y + self.radius * m.sin(m.radians(self.angle+robot.robot_angle))
				self.x1 = robot.x + (self.radius + self.max_range) * m.cos(m.radians(self.angle+robot.robot_angle))
				self.y1 = robot.y + (self.radius + self.max_range) * m.sin(m.radians(self.angle+robot.robot_angle))

		def draw_sensor(self, robot):
				self.update_sensor_pos(robot)
				if self.id == 0:
						pygame.draw.line(screen, (0, 0, 0), (robot.x, robot.y), (self.x0, self.y0))
				# pygame.draw.line(screen, (0, 0, 255), (self.x0, self.y0), (self.x1, self.y1))
				# pygame.draw.circle(screen, (255, 0, 0), (int(self.x0), int(self.y0)), 1, 0)
				# pygame.draw.circle(screen, (255, 0, 0), (int(self.x1), int(self.y1)), 1, 0)

		def detect(self, manager):
				# M = (self.y1 - self.y0) / (self.x1 - self.x0)
				# c = self.y0 - (self.x0 * M)
				for pedestrian in manager.start_pedestrians:
						# https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle
						h = (pedestrian.x + width/2)
						k = (pedestrian.y + height/2)
						q = Vec2d(h, k)
						r = pedestrian.size
						v = Vec2d(self.x1, self.y1) - Vec2d(self.x0, self.y0)
						p = Vec2d(self.x0, self.y0)
						A = v.dot(v)
						B = 2*(v.dot((p-q)))
						C = p.dot(p) + q.dot(q) - ((2*p).dot(q)) - r**2
						if (B**2 - 4*A*C) >= 0:             # Sensors delayed updating could be something to do with this line
								x_solution = np.roots([A, B, C])
								if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:
										y_solution = [p + x_solution[0]*v, p + x_solution[1]*v]
										# pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
										pygame.draw.circle(screen, (0, 255, 0), (int(y_solution[1][0]), int(y_solution[1][1])),1, 0) # here
										pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (y_solution[1][0], y_solution[1][1]))
										# print("Sensor {}s reading is {}".format(self.id, p.get_distance(y_solution[1])))
										self.reading = p.get_distance(y_solution[1])
								else:
										# print("Sensor {}s reading is {}".format(self.id, self.max_range))
										self.reading = self.max_range
						# print("Sensor {}s reading is {}".format(self.id, self.reading))
						# else:
						# 		self.reading = self.max_range # need to think about this line



						# print ("Sensor {}'s reading is: {}".format(self.id, self.reading))

		def collide(self, robot, manager):
				robot_pos = Vec2d(robot.x, robot.y)
				# for pedestrian in manager.start_pedestrians:
				# 		pedestrian_pos = Vec2d(pedestrian.x + width/2, pedestrian.y + height/2)
				# 		distance = robot_pos.get_distance(pedestrian_pos)
				# 		if distance <= robot.size + pedestrian.size:
				# 				robot.alive = False
				if robot.x <= 10 or robot.y <= 10 or robot.y >= height-20 or robot.x >= width -20 :
						robot.alive = False
				target_distance = robot_pos.get_distance(Vec2d(800, 200))
				if target_distance <= robot.size + 10:
						robot.alive = False

		# def evaluate_fitness(self, robot):
		# 		robot_pos = Vec2d(robot.x, robot.y)
		# 		target_pos = Vec2d(800, 300)
		# 		pygame.draw.line(screen, (255,0,0), (robot.x, robot.y), (800, 300))
		# 		# print(robot_pos.get_distance(target_pos))
		# 		try:
		# 				return 1 / robot_pos.get_distance(target_pos)
		# 		except ZeroDivisionError:
		# 				return 1






						# if (B**2 - 4*A*C) >= 0:
						# 		x_solution = np.roots([A, B, C])
						# 		y_solution = [(M*x_solution[0])+c, (M*x_solution[1])+c]
						# 		# if x_solution[0] > self.x0 and x_solution[1] <= self.x1: # not fully correct need to also consider y co ordinates
						# 		if m.hypot((self.x0 - x_solution[1]), (self.y0 - y_solution[1])) <= self.max_range:
						# 			pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
						# 			pygame.draw.circle(screen, (0, 255, 0), (int(x_solution[1]), int(y_solution[1])),1, 0)



