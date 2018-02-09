import pygame
from pygame.locals import *
from math import *
# from Sensor import Sensor
from Vec2d import Vec2d
pygame.init()

screen = pygame.display.set_mode((600, 500))
screen.fill((255, 255, 255))

colour = (0, 255, 0)
running = True
while running:
		for event in pygame.event.get():
				if event.type in (QUIT, KEYDOWN):
						running = False

		center = (300, 250)
		radius = 25
		sensor_range = 50
		angled_sensor1_start = (center[0]+radius*cos(pi/6), center[1]-radius*sin(pi/6))
		angled_sensor1_end = (center[0]+(radius+sensor_range)*cos(pi/6), center[1]-(radius+sensor_range)*sin(pi/6))
		angled_sensor2_start = (center[0]+radius*cos(pi/6), center[1]+radius*sin(pi/6))
		angled_sensor2_end = (center[0]+2*radius*cos(pi/6), center[1]+2*radius*sin(pi/6))

		pygame.draw.circle(screen, colour, center, radius, 0)
		pygame.draw.line(screen, (255, 0, 0), (center[0]+radius, center[1]),(center[0]+(radius+sensor_range), center[1]), 1)
		pygame.draw.line(screen, (255, 0, 0), angled_sensor1_start, angled_sensor1_end, 1)
		pygame.draw.line(screen, (255, 0, 0), angled_sensor2_start, angled_sensor2_end, 1)
		# pygame.transfrom.rotate(pygame.draw.circle(screen, colour, center, radius, 0), 1)
		pygame.display.update()

class RobotOld:
		def __init__(self, field_of_view=90, num_of_sensors=5, start_pos=Vec2d(0,0)):
				self.pos = start_pos
				self.sensors = []
				for i in range(num_of_sensors):
						sensor = Sensor()
						self.sensors.append(sensor)


		# def update(self, particles):
		# 		# Update robot position
		#
		# 		# Using new posititon update sensor position
		# 		for sensor in self.sensors:
		# 				sensor.update(self.pos, particles)
		# 				# After updating sensor
		#
		# 				# Get reading of sensor
		#
		#
		#
		# 		#  Draw new robot
		#
		#
		# 		# Loop throw sensor draw sensor lines
		#
		# 		pass