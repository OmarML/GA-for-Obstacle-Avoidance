import random
import pygame
from Pygame import width, height
from ProcessData import My_Trajectory_Dict, Pedestrian_IDs, new_list
from Sensor import Robot
import numpy as np

pygame.init()
background_colour = (0, 0, 0)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Omar's Simulation")
screen.fill(background_colour)
target_location = (700, 300)


class Pedestrian:
		def __init__(self, x, y, size, id, trajectory):
				self.x = x
				self.y = y
				self.size = size
				self.id = id
				self.trajectory = trajectory
				self.colour = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
				self.game_width = width
				self.game_height = height
				self.present = True
				self.considered = False
				self.destination = 0
				self.paths = []

		def update(self):
				if self.present:
						pygame.draw.circle(screen, self.colour, ((int(self.x) + int(width / 2)), (int(self.y) + int(height / 2))),self.size, 0)
					
		def move(self):
				if self.destination < len(self.trajectory[0]):
						self.x = self.trajectory[0][self.destination]
						self.y = -self.trajectory[1][self.destination]
						self.destination += 1
						self.paths.append([self.x + (width/2), self.y + (height/2)])
				else:
						self.present = False



class Manager(Pedestrian):
		def __init__(self, all_pedestrians, limit):
				self.limit = limit
				self.all_pedestrians = all_pedestrians
				self.start_pedestrians = all_pedestrians[:self.limit]

		def introduce(self):
				for pedestrian in self.all_pedestrians:
						if not pedestrian.present:
								self.all_pedestrians.remove(pedestrian)
								self.start_pedestrians.remove(pedestrian)
								for i in self.all_pedestrians:
										if i not in self.start_pedestrians and len(self.start_pedestrians) <= self.limit:
												self.start_pedestrians.append(i)
								# print ([i.id for i in self.all_pedestrians[:20]])
								# print ([i.id for i in self.start_pedestrians])

						


all_pedestrians = []
for pedestrian in Pedestrian_IDs:
		starting_x = My_Trajectory_Dict[pedestrian][0][0]
		starting_y = My_Trajectory_Dict[pedestrian][0][1]
		all_pedestrians.append(Pedestrian(starting_x, starting_y, 8, pedestrian, My_Trajectory_Dict[pedestrian]))

all = Manager(all_pedestrians, 15)


population_size = 10
elitism = 4

robots = []
for i in range(population_size):
	robots.append(Robot(50, 300, 8, 360, 9, all, set_weights=None))


# some class here maybe to manage all robots
class Darwin:
		def __init__(self, robot_array, elitism, mutation_rate):
				self.robot_array = robot_array
				self.generation = 0
				self.population_size = population_size
				self.elitism = elitism
				self.mutation_rate = mutation_rate
				self.best_fitness = 0
				self.number_of_parents = 4
				self.dead_count = 0

		def check_if_all_dead(self):
				self.dead_count = 0
				for robot in self.robot_array:
						if not robot.alive:
								self.dead_count += 1
				if self.dead_count == len(self.robot_array):
						return True
				# print(self.dead_count)

		def choose_parents(self):
				self.robot_array.sort(key=lambda x: x.fitness)
				max_fitness = max([robot.fitness for robot in self.robot_array])
				if max_fitness > self.best_fitness:
						self.best_fitness = max_fitness
				print("Highest fitness is: {}".format(self.best_fitness))
				return self.robot_array[(self.population_size - self.elitism):]
				# upper_limit = sum([robot.fitness for robot in self.robot_array])
				# pick = random.uniform(0, upper_limit)
				# current = 0
				# for robot in self.robot_array:
				# 		current += robot.fitness
				# 		if current > pick:
				# 				print('here')
				# 				return robot
				# 		else:
				# 				return self.choose_parents()

		def convert_to_genome(self, weights_array):
				return np.concatenate([np.ravel(i) for i in weights_array])

		def convert_to_weight(self, genome, weights_array):
				shapes = [np.shape(i) for i in weights_array]
				products = ([(i[0] * i[1]) for i in shapes])
				out = []
				start = 0
				for i in range(len(products)):
						out.append(np.reshape(genome[start:sum(products[:i + 1])], shapes[i]))
						start += products[i]
				return out

		def create_child(self, parent1, parent2):  # Uniform crossover
				parent1_genome = self.convert_to_genome(parent1.brain.weights)  # but is DNA actually being updated over time (over the generations)
				parent2_genome = self.convert_to_genome(parent2.brain.weights)
				child_genome = []
				for i in range(len(parent1_genome)):
						if random.random() > 0.5:
								child_genome.append(parent1_genome[i])
						else:
								child_genome.append(parent2_genome[i])
				child_weights = self.convert_to_weight(child_genome, parent1.brain.weights) # the return value is a weights array
				# return child_weights
				return Robot(50, 300, 8, 360, 9, all, child_weights, own_weights=True)

		def mutate(self, individual):
				genome = self.convert_to_genome(individual.brain.weights)
				weight_to_mutate = random.randint(0, len(genome))
				genome[weight_to_mutate] = genome[weight_to_mutate] * random.uniform(0.9, 1.1)
				new_weights = self.convert_to_weight(genome, individual.brain.weights)
				return Robot(50, 300, 8, 360, 9, all, new_weights, own_weights=True)


				# code for mutation goes here


		def make_next_generation(self):
				breeders = self.choose_parents()
				# breeders = []
				# for i in range(self.number_of_parents):
				# 		breeders.append(self.choose_parents())
				# print(breeders)
				# offspring = [self.convert_to_genome(self.robot_array[i].DNA) for i in range(len(breeders))]
				offspring = [] # need to include breeders in offspring list i.e parents from nth gen should be in the n+1th generation
				# offspring += breeders # Include parents in the n+1th generation
				# number_of_children = (self.population_size - len(breeders)) / (len(breeders) / 2)
				for i in range(int(len(breeders)/2)):
						for j in range(int(5)):
								offspring.append(self.create_child(breeders[i], breeders[len(breeders) - 1 - i])) # make best parents breed with eachother
				# print([np.array_equal(offspring[i].brain.weights, offspring[i+1].brain.weights) for i in range(len(offspring)-1)])
				for i in range(len(offspring)):
						if random.uniform(0, 1)  <= self.mutation_rate:
								offspring[i] = self.mutate(offspring[i])
								print("mutated")

				print("Breeders:", len(breeders))
				print("Offspring:", len(offspring))
				# print("Robot array")
				# print(self.robot_array)
				# print("offspring")
				# print(offspring)
				self.robot_array = offspring
				# print(len(offspring))
				# weights = [self.convert_to_weight(i, self.robot_array[0].DNA) for i in offspring]
				# This is stupid Implementation change it, Should be able to choose DNA upon instantiation
				# self.robot_array.clear()
				# print(self.robot_array)
				# for i in range(len(offspring)):
				# 		self.robot_array.extend(offspring[i])
				# print(self.robot_array)
				# self.robot_array = offspring
				# [print(offspring[i].alive) for i in range(len(offspring))]
				# self.robot_array.extend([Robot(50, 300, 8, 360, 9, all) for i in range(self.population_size)])
				# for i in range(self.population_size):
				# 		if self.robot_array[i].alive:
				# 				self.robot_array[i].brain.weights = offspring[i]  # hmmmm?
				# print(self.robot_array)
				self.generation += 1
				# self.dead_count = 0
				print("Generation ", self.generation)





darwin = Darwin(robot_array=robots, elitism=4, mutation_rate=0.2)

c = 0

if __name__ == '__main__':
		# begin = input("Press any letter to begin")

		running = True
		while running:
				for event in pygame.event.get():
						if event.type == pygame.QUIT:
								running = False
				screen.fill(background_colour)
				pygame.draw.rect(screen, (255, 255, 255), (10, 10, width-20, height-20), 1)
				pygame.draw.circle(screen, (255, 10, 0), target_location, 10, 0)
				pygame.draw.circle(screen, (0, 0, 255), (500, 300), 100, 0)
				pygame.draw.circle(screen, (0, 255, 20), (200, 300), 75, 0)
				# pygame.draw.polygon(screen, (255, 255, 255), new_list, 1)
				# for pedestrian in all.start_pedestrians:
				# 		pedestrian.move()
				# 		pedestrian.update()
				# 		all.introduce()
				for robot in darwin.robot_array:
						robot.move()
						robot.update()
						robot.evaluate_fitness()
				if darwin.check_if_all_dead():
						darwin.make_next_generation()


				pygame.display.update()
				# pygame.time.Clock().tick(10000)