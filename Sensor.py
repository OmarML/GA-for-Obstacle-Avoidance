from Pygame import screen, width, height
import pygame
import math as m
import numpy as np
import random
import Neural
import time
from Vec2d import Vec2d
from LineIntersections import calculateIntersectPoint


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
				self.alive = True
				self.set_weights = set_weights
				if own_weights:
						self.brain = Neural.NeuralNetwork(inputs=num_sensors, hidden_layers=1, hidden_neurons=32, outputs=2, given_weights=set_weights)
				else:
						self.brain = Neural.NeuralNetwork(inputs=num_sensors, hidden_layers=1, hidden_neurons=32, outputs=2, given_weights=self.create_weights(num_sensors, 32, 1, 2)) # this will be the neural network which makes the decision based on sensor inputs
				self.DNA = self.brain.weights # this will be an array which contains the all weights of the NN
				self.fitness = 0
				self.time_alive = time.time()
				self.best = 1e6
				self.considered = False
				self.hitTarget = False

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
						robot_pos = Vec2d(self.x, self.y)
						target_pos = Vec2d(800, 300)
						angle_between = target_pos.get_angle_between(robot_pos)
						angle_out = self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[1]
						# self.robot_angle = np.interp(angle_out, [-1, 1], [-60, 60])
						triggered = False
						for sensor in self.sensors:
								if sensor.reading < self.max_range:
										triggered = True
						if triggered:
								self.robot_angle = np.interp(angle_out, [-1, 1], [-60, 60])
						else:
								self.robot_angle = np.interp(angle_out, [-1, 1], [-60, 60])
								delta_y = 300 - self.y
								delta_x = 800 - self.x
								# self.robot_angle = m.degrees(m.atan(delta_y/delta_x))
						# self.robot_angle += self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[1] # this changes between -1 and 1 degrees per decision
						self.x += 7*self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[0] * m.cos(m.radians(self.robot_angle))
						self.y += 7*self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[0] * m.sin(m.radians(self.robot_angle))
				# 	self.robot_angle += self.brain.forward([(sensor.reading/self.max_range) for sensor in self.sensors])[1] # this changes between -1 and 1 degrees per decision
				# 		print(self.brain.forward([(sensor.reading) for sensor in self.sensors]))
				# 	self.x += self.speed * m.cos(m.radians(self.robot_angle))
				# 	self.y += self.speed * m.sin(m.radians(self.robot_angle))
				# 	self.robot_angle += self.angle

		def update(self):
				if self.alive:
						pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, 0)
						for sensor in self.sensors:
								sensor.draw_sensor(self)
								for obstacle in obstacleArray:
										sensor.detectObstacle(obstacle)
								# sensor.detectBoundary()
								# sensor.detect(self.manager)
								# sensor.collide(self, self.manager)
								# sensor.detect_static()
								# sensor.detect_static2()
				if time.time() - self.time_alive > 10: # add condition to check if fitness isnt changing much not just time for killing
						self.alive = False
				if not self.alive and not self.considered:
						self.time_alive = time.time() - self.time_alive
						self.considered = True
						# print(self.time_alive)
								# sensor.evaluate_fitness(self)
						# print([sensor.reading for sensor in self.sensors])
						# print("Sensor reading {} is:{}".format(sensor.id, sensor.reading)) # need to think about this line

		def collide(self):
				robot_pos = Vec2d(self.x, self.y)
				# for pedestrian in self.manager.start_pedestrians:
				# 		pedestrian_pos = Vec2d(pedestrian.x + width/2, pedestrian.y + height/2)
				# 		distance = robot_pos.get_distance(pedestrian_pos)
				# 		if distance <= self.size + pedestrian.size:
				# 				self.alive = False
				if self.x <= 10 or self.y <= 10 or self.y >= height-20 or self.x >= width -20 :
						self.alive = False
				target_distance = robot_pos.get_distance(Vec2d(800, 300))
				obstacle_distance = robot_pos.get_distance(Vec2d(500, 300))
				obstacle_distance2 = robot_pos.get_distance(Vec2d(200, 300))
				if target_distance <= self.size + 10:
						self.alive = False
						self.hitTarget = True
						# self.completion_time -= time.time()*(-1)
						# print(self.completion_time)
				for obstacle in obstacleArray:
						if obstacle.shape == 'Circle':
								obstacleDistance = robot_pos.get_distance(obstacle.position)
								if obstacleDistance <= self.size + obstacle.radius:
										self.alive = False
				# 		elif obstacle.shape == 'Line':
				# 				q = robot_pos
				# 				r = self.size
				# 				v = Vec2d(obstacle.position[1]) - Vec2d(obstacle.position[0])
				# 				p = Vec2d(obstacle.position[0])
				# 				A = v.dot(v)
				# 				B = 2 * (v.dot((p - q)))
				# 				C = p.dot(p) + q.dot(q) - ((2 * p).dot(q)) - r ** 2
				# 				if (B ** 2 - 4 * A * C) >= 0:  # Sensors delayed updating could be something to do with this line
				# 						x_solution = np.roots([A, B, C])
				# 						if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:
				# 								self.alive = False


				# if obstacle_distance <= self.size + 100:
				# 		self.alive = False
				# if obstacle_distance2 <= self.size + 75:
				# 		self.alive = False

		def evaluate_fitness(self):
				if self.alive:
						target_pos = Vec2d(800, 300)
						start = Vec2d(self.start_x, self.start_y)
						robot_pos = Vec2d(self.x, self.y)
						angle = robot_pos.get_angle_between(target_pos)
						total = start.get_distance(target_pos)
						distance = robot_pos.get_distance(target_pos)
						closest = total
						# completion_time = self.sensors[0].completion_time
						if distance < self.best:
								self.best = distance
						if self.hitTarget:
								targetFactor = 1
						else:
								targetFactor = 0
						try:
								# self.fitness = (1 / self.best)
								# self.fitness = 1/(distance*m.cos(m.radians(angle)))
								self.fitness = (1 / distance) + (0.5*(1 / self.best)) + (1*(1 / self.time_alive)) + (0.5*targetFactor)
								# self.fitness = distance
								# self.fitness = (total - distance) / total
						except ZeroDivisionError:
								print("tried dividing by zero")
								self.fitness = 100



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
				self.activated = False
				self.timer = 8
				# self.completion_time = time.time()

		def update_sensor_pos(self, robot):
				# print(self.x0, self.y0)
				# print(robot.robot_angle)
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
						if (B**2 - 4*A*C) >= 0:            # Does the line intersect, extended OR not extended
								x_solution = np.roots([A, B, C])
								if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:  # does the non extended line intersect
										y_solution = [p + x_solution[0]*v, p + x_solution[1]*v]
										# pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
										# pygame.draw.circle(screen, (0, 255, 0), (int(y_solution[1][0]), int(y_solution[1][1])),1, 0) # here
										pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (y_solution[1][0], y_solution[1][1]))
										# print("Sensor {}s reading is {}".format(self.id, p.get_distance(y_solution[1])))
										self.reading = p.get_distance(y_solution[1])
										self.activated = True
								else:
										# self.reading = self.max_range
										# pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
										# print("Sensor {}s reading is {}".format(self.id, self.max_range))
										self.activated = False
						# print("Sensor {}s reading is {}".format(self.id, self.reading))
						else:
								if not self.activated:
										self.reading = self.max_range
						# 		pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
						# 		self.reading = self.max_range
						# print(self.reading)
						# 		self.reading = self.reset_sensor()
				print("Sensor {}s reading is {}".format(self.id, self.reading))

		def detectObstacle(self, obstacle):
				if obstacle.shape == 'Circle':
						q = Vec2d(obstacle.position)
						r = obstacle.radius
						v = Vec2d(self.x1, self.y1) - Vec2d(self.x0, self.y0)
						p = Vec2d(self.x0, self.y0)
						A = v.dot(v)
						B = 2 * (v.dot((p - q)))
						C = p.dot(p) + q.dot(q) - ((2 * p).dot(q)) - r ** 2
						# print(Vec2d(self.x1, self.y1), Vec2d(self.x0, self.y0))
						# print(A, B, C)
						# print(p, Vec2d(self.x1, self.y1))
						if (B ** 2 - 4 * A * C) >= 0: # Sensors delayed updating could be something to do with this line
								x_solution = np.roots([A, B, C])
								if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:
										y_solution = [p + x_solution[0] * v, p + x_solution[1] * v]
										pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (y_solution[1][0], y_solution[1][1]))
										self.reading = p.get_distance(y_solution[1])
										# self.activated = True
										# print("Sensor {} is {}".format(self.id, self.reading))
								else:
										# self.activated = False
										self.reading = self.max_range
						else:
								self.reading = self.max_range
								# if not self.activated:
								# 		self.reading = self.max_range # PROBLEM WHEN THERE IS MORE THAN ONE OBSTACLE SENSOR READINGS GET RESET FOR THE OBSTACLE NOT YET INTERACTED WITH
								# print('here')
						# 		# self.reading = self.detect_static()
						# 		self.reading = self.max_range
				# print("Sensor {} is {}".format(self.id, self.reading))

				# elif obstacle.shape == 'Line':
				# 		start = (self.x0, self.y0)
				# 		end = (self.x1, self.y1)
				# 		solution = calculateIntersectPoint(start, end, obstacle.position[0], obstacle.position[1])
				# 		if solution is not None:
				# 				pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (solution[0], solution[1]))
				# 				self.reading = Vec2d(start).get_distance(Vec2d(solution))
				# 		else:
				# 				self.reading = self.detect_static()
				#
				#
				# elif obstacle.shape == 'Rectangle':
				# 		w = obstacle.position[2]
				# 		h = obstacle.position[3]
				# 		X = (self.x0, self.y0)
				# 		Y = (self.x1, self.y1)
				# 		R1 = (obstacle.position[0], obstacle.position[1] + h)
				# 		R2 = (obstacle.position[0] + w, obstacle.position[1])
				# 		R3 = (obstacle.position[0] + w, obstacle.position[1] + h)
				# 		R4 = (obstacle.position[0] + w, obstacle.position[1] + h)
				# 		F1 = (obstacle.position[0], obstacle.position[1])
				# 		F2 = (obstacle.position[0], obstacle.position[1])
				# 		F3 = (obstacle.position[0], obstacle.position[1] + h)
				# 		F4 = (obstacle.position[0] + w, obstacle.position[1])
				# 		FArray = [F1, F2, F3 ,F4]
				# 		RArray = [R1, R2, R3, R4]
				# 		solution1 = calculateIntersectPoint(X, Y, F1, R1)
				# 		solution2 = calculateIntersectPoint(X, Y, F2, R2)
				# 		solution3 = calculateIntersectPoint(X, Y, F3, R3)
				# 		solution4 = calculateIntersectPoint(X, Y, F4, R4)
				# 		solutionArray = [solution1, solution2, solution3, solution4]
				# 		distanceArray = [Vec2d(X).get_distance(Vec2d(i)) for i in solutionArray if i != None]
				# 		# print(solutionArray)
				# 		# print(distanceArray)
				# 		for i in solutionArray:
				# 				if i != None:
				# 						pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (i[0], i[1]))
				# 						self.reading = Vec2d(X).get_distance(Vec2d(i))
				# 						self.timer = 8
				# 				else:
				# 						self.timer -= 1
				# 						if self.timer == 0:
				# 								self.reading = self.max_range
				# 						# print(self.reading)
				#
				# 		if distanceArray:
				# 				trueSolution = solutionArray[distanceArray.index(min(distanceArray))]
								# print(trueSolution)
								# pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (trueSolution[0], trueSolution[1]))
						# if solution1 != None:
						# 		[i = None for i in solutionArray]
						# for i in range(len(FArray)):
						# 		solution = calculateIntersectPoint(X, Y, FArray[i], RArray[i])
						# 		if solution != None:
						# 				print(solution)
						# 				self.reading = Vec2d(X).get_distance(Vec2d(solution))
						# 				pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (solution[0], solution[1]))
						# 		# else:
						# 		# 		self.reading = 75



						# R1 = (obstacle.position[0], obstacle.position[1] + h)
						# R2 = (obstacle.position[0] + w, obstacle.position[1])
						# R3 = (obstacle.position[0] + w, obstacle.position[1] + h)
						# R4 = (obstacle.position[0] + w, obstacle.position[1] + h)
						# P = Vec2d(-E[1], E[0])
						# F1 = (obstacle.position[0], obstacle.position[1]) - Vec2d(obstacle.position[0], obstacle.position[1] + h)
						# F2 = (obstacle.position[0], obstacle.position[1]) - Vec2d(obstacle.position[0] + w, obstacle.position[1])
						# F3 = (obstacle.position[0], obstacle.position[1] + h) - Vec2d(obstacle.position[0] + w, obstacle.position[1] + h)
						# F4 = (obstacle.position[0] + w, obstacle.position[1]) - Vec2d(obstacle.position[0] + w, obstacle.position[1] + h)
						# FArray = [F1, F2, F3 ,F4]
						# RArray = [R1, R2, R3, R4]
						# for i in range(len(FArray)):
						# 		h = ((X - RArray[i]).dot(P)) / (FArray[i].dot(P))
						# 		# print(h)
						# 		# print(FArray[i])
						# 		if 0 <= h <= 1:
						# 				solution = RArray[i] + FArray[i]*h
						# 				pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (solution[0], solution[1]))
						# 				self.reading = X.get_distance(solution)
						# 				print(self.reading)
						# 		# else:
						# 		# 		self.reading = self.max_range
						#

		def detectBoundary(self):
				boundary_coords = [((10, 10), (10, 590)), ((10, 10), (990, 10)), ((10, 590), (990, 590)), ((990, 10), (990, 590))]
				for i in range(len(boundary_coords)):
						hit_boundary = calculateIntersectPoint((self.x0, self.y0), (self.x1, self.y1), boundary_coords[i][0], boundary_coords[i][1])
						if hit_boundary is not None:
								pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (hit_boundary[0], hit_boundary[1]))
								distance = Vec2d(self.x0, self.y0).get_distance(Vec2d(hit_boundary))
								self.reading = distance
								self.timer = 5
						else:
								# self.reading = self.max_range
								self.timer -= 1
								if self.timer == 0:
										# pygame.draw.line(screen, (0, 0, 255), (self.x0, self.y0), (self.x1, self.y1))
										self.reading = self.max_range
										self.timer = 8

						# 		self.reading = self.max_range
				print(self.reading)


		def detect_static(self):
				q = Vec2d(500, 300)
				r = 100
				v = Vec2d(self.x1, self.y1) - Vec2d(self.x0, self.y0)
				p = Vec2d(self.x0, self.y0)
				A = v.dot(v)
				B = 2*(v.dot((p-q)))
				C = p.dot(p) + q.dot(q) - ((2 * p).dot(q)) - r ** 2
				if (B ** 2 - 4 * A * C) >= 0:  # Sensors delayed updating could be something to do with this line
						x_solution = np.roots([A, B, C])
						if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:
								y_solution = [p + x_solution[0] * v, p + x_solution[1] * v]
								# pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
								# pygame.draw.circle(screen, (0, 255, 0), (int(y_solution[1][0]), int(y_solution[1][1])), 1, 0)  # here
								pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (y_solution[1][0], y_solution[1][1]))
								# print("Sensor {}s reading is {}".format(self.id, p.get_distance(y_solution[1])))
								self.reading = p.get_distance(y_solution[1])
						else:
								# print("Sensor {}s reading is {}".format(self.id, self.max_range))
								self.reading = self.max_range
				else:
						self.reading = self.max_range
				return self.reading

		def detect_static2(self): #first obstacle
				q = Vec2d(200, 300)
				r = 75
				v = Vec2d(self.x1, self.y1) - Vec2d(self.x0, self.y0)
				p = Vec2d(self.x0, self.y0)
				A = v.dot(v)
				B = 2*(v.dot((p-q)))
				C = p.dot(p) + q.dot(q) - ((2 * p).dot(q)) - r ** 2
				if (B ** 2 - 4 * A * C) >= 0:  # Sensors delayed updating could be something to do with this line
						x_solution = np.roots([A, B, C])
						if 0 <= x_solution[0] <= 1 or 0 <= x_solution[1] <= 1:
								y_solution = [p + x_solution[0] * v, p + x_solution[1] * v]
								# pygame.draw.line(screen, (255, 0, 0), (self.x0, self.y0), (self.x1, self.y1))
								# pygame.draw.circle(screen, (0, 255, 0), (int(y_solution[1][0]), int(y_solution[1][1])), 1, 0)  # here
								pygame.draw.line(screen, (0, 255, 0), (self.x0, self.y0), (y_solution[1][0], y_solution[1][1]))
								# print("Sensor {}s reading is {}".format(self.id, p.get_distance(y_solution[1])))
								self.reading = p.get_distance(y_solution[1])
						else:
								# print("Sensor {}s reading is {}".format(self.id, self.max_range))
								self.reading = self.max_range
				else:
						self.reading = self.detect_static()


class Obstacle():
		def __init__(self, shape, position, colour, thickness, radius=None):
				self.shape = shape
				self.position = position
				self.colour = colour
				self.thickness = thickness
				self.radius = radius
				self.has_reached_top = False
				self.has_reached_bottom = False
				self.direction = 1

		def drawShape(self):
				if self.shape == 'Circle':
						pygame.draw.circle(screen, self.colour, (self.position[0], self.position[1]), self.radius, self.thickness)
				elif self.shape == 'Rectangle':
						pygame.draw.rect(screen, self.colour, self.position, self.thickness)
				elif self.shape == 'Line':
						pygame.draw.line(screen, self.colour, self.position[0], self.position[1], self.thickness)

		def move_y(self):
				upper_limit = 450
				lower_limit = 150
				speed = 3


				if self.position[1] > 450 and not self.has_reached_bottom:
						self.direction *= -1
						self.has_reached_top = False
						self.has_reached_bottom = True

				if self.position[1] < 150 and not self.has_reached_top:
						self.direction *= -1
						self.has_reached_top = True
						self.has_reached_bottom = False
				# if self.position[1] > 150:
				# 		direction *= -1
				self.position[1] += speed*self.direction

				# print(self.direction, self.position[1])






obstacleArray = [Obstacle('Circle', [300, 300], (0, 0, 255), 0, 75)]
                 # Obstacle('Circle', [500, 300], (0, 0, 255), 0, 100)]
                 # Obstacle('Circle', [200, 300], (0, 255, 0), 0, 75)]
                 #Obstacle('Line', ((200, 10), (200, 148)), (255, 255, 255), 10),
                 #Obstacle('Line', ((200, 450), (200, 588)), (255, 255, 255), 10),
                 #]

# obstacleArray = [Obstacle('Circle', (200, 300), (0, 255, 0), 0, 75), Obstacle('Circle', (500, 300), (0, 0, 255), 0, 100)]


								# def collide(self, robot, manager):
		# 		robot_pos = Vec2d(robot.x, robot.y)
		# 		# for pedestrian in manager.start_pedestrians:
		# 		# 		pedestrian_pos = Vec2d(pedestrian.x + width/2, pedestrian.y + height/2)
		# 		# 		distance = robot_pos.get_distance(pedestrian_pos)
		# 		# 		if distance <= robot.size + pedestrian.size:
		# 		# 				robot.alive = False
		# 		if robot.x <= 10 or robot.y <= 10 or robot.y >= height-20 or robot.x >= width -20 :
		# 				robot.alive = False
		# 		target_distance = robot_pos.get_distance(Vec2d(800, 300))
		# 		obstacle_distance = robot_pos.get_distance(Vec2d(500, 300))
		# 		obstacle_distance2 = robot_pos.get_distance(Vec2d(200, 300))
		# 		if target_distance <= robot.size + 10:
		# 				robot.alive = False
		# 				# self.completion_time -= time.time()*(-1)
		# 				# print(self.completion_time)
		# 		if obstacle_distance <= robot.size + 100:
		# 				robot.alive = False
		# 		if obstacle_distance2 <= robot.size + 75:
		# 				robot.alive = False

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


