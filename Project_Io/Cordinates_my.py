import numpy as np
import csv
import math

with open('csv') as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

x = []
y = []
z = []


theta = hot_spots_data[:, 0]
phi = hot_spots_data[:, 1]
r = np.ones(343)

for i in range(343):
    x.append(float((r[i]*math.cos(theta[i])*math.sin(phi[i]))))
    y.append(float((r[i]*math.sin(theta[i])*math.sin(phi[i]))))
    z.append(float((r[i]*math.cos(phi[i]))))



