import random
import pygame
from Pygame import width, height
from ProcessData import My_Trajectory_Dict, Pedestrian_IDs, new_list
from Sensor import Robot

pygame.init()
background_colour = (0, 0, 0)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Omar's Simulation")
screen.fill(background_colour)
target_location = (800, 300)


elitism = 0.1


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

robots = []
for i in range(1):
	robots.append(Robot(200, 300, 8, 360, 9, all))

# some class here maybe to manage all robots


def check_if_all_dead():
		dead_count = 0
		for robot in robots:
				if not robot.alive:
						dead_count += 1
		if dead_count == len(robots):
				return True


def choose_fittest(elitism):
		if check_if_all_dead():
				robots.sort(key=lambda x: x.fitness)
				print("1: ", [robot.fitness for robot in robots])
				return robots[-elitism:]


def make_next_generation():
		pass








running = True
while running:
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
						running = False
		screen.fill(background_colour)
		pygame.draw.rect(screen, (255, 255, 255), (10, 10, width-20, height-20), 1)
		pygame.draw.circle(screen, (255, 10, 0), target_location, 10, 0)
		# pygame.draw.polygon(screen, (255, 255, 255), new_list, 1)
		for pedestrian in all.start_pedestrians:
				pedestrian.move()
				pedestrian.update()
				all.introduce()
		for robot in robots:
				robot.move()
				robot.update()
				robot.evaluate_fitness()
		choose_fittest(3)
		make_next_generation()

		pygame.display.update()
		# pygame.time.Clock().tick(10000)