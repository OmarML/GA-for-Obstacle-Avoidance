import csv
import pickle
import matplotlib.pyplot as plt
import pygame

myfile = '/Users/omardiab/Documents/University/atc-20121114.csv'


def Get_Trajectory(file, Person_ID):
	data = open(file)
	reader = csv.reader(data)
	xpos = []
	ypos = []
	for row in reader:
		if Person_ID in row:
			xpos.append(float(row[2]) / 1000.0)  # convert from mm to m
			ypos.append(float(row[3]) / 1000.0)
	data.close()
	return [xpos, ypos]


def Get_Pedestrian_IDs(file):
	data = open(file)
	reader = csv.reader(data)
	Pedestrian_IDs = []
	for row in reader:
		if row[1] not in Pedestrian_IDs:
			Pedestrian_IDs.append(row[1])
	data.close()
	print("This Dataset contains {} unique pedestrians".format(len(Pedestrian_IDs)))
	return Pedestrian_IDs


def check_rows(file):
		data = open(file)
		reader = csv.reader(data)
		count = 0
		for row in reader:
				if count == 1:
						print(row[0])
				if count == 1048575:
						print(row[0])
				count += 1
		return count



# Pedestrian_IDs = Get_Pedestrian_IDs(myfile)
# pickle_out = open("pedestrianIDs.pickle", "wb")
# pickle.dump(Pedestrian_IDs, pickle_out)
# pickle_out.close()
IDs_in = open("pedestrianIDs.pickle", "rb")
Pedestrian_IDs = pickle.load(IDs_in)


# Generate Dictionary Entries for first 15 Pedestrian IDs
# Use Pickle to save the really large Dictionary and load back into memory
def Generate_Tracjectories():
	Trajectory_Dict = {}
	c = 0
	for i in Pedestrian_IDs:
		Trajectory_Dict[i] = Get_Trajectory(myfile, i)
		c += 1
		print(c)
	return Trajectory_Dict


# My_Trajectory_Dict2 = Generate_Tracjectories()
# pickle_out = open("mydict2.pickle", "wb")
# pickle.dump(My_Trajectory_Dict2, pickle_out)
# pickle_out.close()
pickle_in2 = open("mydict2.pickle", "rb")
My_Trajectory_Dict2 = pickle.load(pickle_in2)

# My_Trajectory_Dict = Generate_Tracjectories()
# pickle_out = open("mydict.pickle", "wb")
# pickle.dump(My_Trajectory_Dict, pickle_out)
# pickle_out.close()
pickle_in = open("mydict.pickle", "rb")
My_Trajectory_Dict = pickle.load(pickle_in)
# print (My_Trajectory_Dict['9315400'])

def Plot_Trajectories(legend=False):
		for pedestrian in Pedestrian_IDs[0:10]:
				plt.plot(My_Trajectory_Dict2[pedestrian][0], My_Trajectory_Dict2[pedestrian][1], label=pedestrian)
				plt.xlabel("X (m)")
				plt.ylabel("Y (m)")
				plt.title("Pedestrian Trajectories in ATC Shopping Centre: Osaka, Japan")
				if legend:
						plt.legend()
		plt.show()

# Plot_Trajectories(legend=True)




pygame.init()
background_colour = (255, 255, 255)
(width, height) = (1000, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Omar's Simulation")
screen.fill(background_colour)

# def polygon_list():
# 		points = []
# 		for pedestrian in Pedestrian_IDs:
# 				points.append((width/2 + My_Trajectory_Dict[pedestrian][0][-1], height/2 + My_Trajectory_Dict[pedestrian][1][-1]))
# 		return points

def polygon_list():
		data_x = []
		data_y = []
		for pedestrian in Pedestrian_IDs:
				data_x.append(My_Trajectory_Dict[pedestrian][0][-1])
				data_y.append(My_Trajectory_Dict[pedestrian][1][-1])
		return data_x, data_y

x, y = polygon_list()
# plt.plot(x, y)
# plt.show()

x0 = min(x)
y0 = y[x.index(x0)]

x1 = max(x)
y1 = y[x.index(x1)]

y2 = min(y)
x2 = x[y.index(y2)]

y3 = max(y)
x3 = x[y.index(y3)]

points_list = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

# print(points_list)

new_list = []
for i in points_list:
		new_list.append((i[0] + width/2, -i[1] + height /2))

# print (new_list)

# plt.scatter(x, y)
# plt.show()





running = False
while running:
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
						running = False
		screen.fill(background_colour)
		# pygame.draw.ellipse(screen, (0, 0, 255), (100, 100, 200, 300), 2)
		# pygame.draw.circle(screen, (0, 0, 255), (100, 100), 5, 0)
		pygame.draw.polygon(screen, (0, 0, 255), new_list, 1)
		pygame.display.update()
