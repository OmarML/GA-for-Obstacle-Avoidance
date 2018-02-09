import pygame
import random
from ProcessData import My_Trajectory_Dict, Pedestrian_IDs

pygame.init()

background_colour = (255, 255, 255)
(width, height) = (1000, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Omar's Simulation")
screen.fill(background_colour)

counter = 0
pedestrian_count = 0

# Can give target destination for robot to reach and end the game if target is reached

# Need to give each particle and ID which is its pedestrian ID that way I can access it more easily
class Particle:
    def __init__(self, x, y, size, ID):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (random.randint(20,255), random.randint(20,255),  random.randint(20,255))
        self.thickness = 0
        self.present = True
        self.paths = []
        self.ID = ID
        self.considered = False

    def display(self):
        if self.present:
            pygame.draw.circle(screen, self.colour, ((int(self.x) + int(width/2)), (int(self.y) + int(height/2))), self.size, self.thickness)
            if len(self.paths) > 2:
                for i in range(len(self.paths)):
                    pygame.draw.circle(screen, self.colour, (int(self.paths[i][0]), int(self.paths[i][1])), 0, self.thickness)


    def move(self, index, x_pos, y_pos):
        try:
            self.x = x_pos[index]
            self.y = y_pos[index] * -1
            self.paths.append([self.x + (width/2), self.y + (height/2)])
        except IndexError:
            self.present = False
            # pass

    def state(self):
        return self.present


my_particles = []
temp = []

bigArr2 = []
for pedestrian in Pedestrian_IDs:
    size = 8
    x1 = My_Trajectory_Dict[pedestrian][0][0]
    y1 = My_Trajectory_Dict[pedestrian][1][0]
    x = My_Trajectory_Dict[pedestrian][0]
    y = My_Trajectory_Dict[pedestrian][1]
    bigArr2.append([x, y])
    my_particles.append(
        Particle(x1, y1, size, pedestrian))

running = False
max = 10
current = max
existing_particles = my_particles[:10]
to_remove_existing = []
to_remove_all = []
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(background_colour)
    indexing = 0  # make sure right trajectory is being accessed for each instance as they all use common of bigArr2
    for particle in existing_particles: # Need to rethink here..., try using a queue, no just limit it to 10 pedestrians at a time
        if current <= max:
            particle.move(counter, bigArr2[indexing][0], bigArr2[indexing][1])
            particle.display()
            indexing += 1
            counter += 1    
            if not particle.state():
                if not particle.considered:
                    pedestrian_count += 1
                    # to_remove_existing.append(particle)
                    # my_particles.remove(particle)
                    # existing_particles.append(my_particles[0])
                    particle.considered = True
            current = max - pedestrian_count # number of pedestrians that have dissappeard im on to something... embed in a while loop to limit number on screen
    print("Existing",[particle.ID for particle in existing_particles])
    # print("All",[particle.ID for particle in my_particles[:10]])
    pygame.display.update()
    pygame.time.Clock().tick(20)