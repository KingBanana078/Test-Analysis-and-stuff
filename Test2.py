import csv
import numpy as np
import matplotlib.pyplot  as plt
import math

from scipy.interpolate import Rbf
from scipy.spatial import Delaunay, SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d


with open('Positiondata.csv') as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

longitude = hot_spots_data[:, 1]
latitude = hot_spots_data[:, 0]

"""triangulations = Delaunay(hot_spots_data)

plt.triplot(hot_spots_data[:, 0], hot_spots_data[:, 1], triangulations.simplices)
plt.plot(hot_spots_data[:, 0], hot_spots_data[:, 1], 'o')
plt.show()"""

x = []
y = []
z = []

theta = hot_spots_data[:, 1]
phi = hot_spots_data[:, 0]
r = np.ones(343)

#for i in range(len(theta)):
    #if theta[i]> 180:
        #value = 180- theta[i]
        #theta[i]=value
for i in range(len(theta)):
    value = 180-theta[i]
    theta[i]=value


for i in range(343):
    x.append(float((r[i]*math.cos(theta[i])*math.sin(phi[i]))))
    y.append(float((r[i]*math.sin(theta[i])*math.sin(phi[i]))))
    z.append(float((r[i]*math.cos(phi[i]))))

point = []
for k in range(343):
    point.append([x[k], y[k], z[k]])

long, lat = np.meshgrid(longitude, latitude)
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')

ax.scatter(theta/180*math.pi, phi/180*math.pi)
plt.show()
