import csv
import numpy as np
import matplotlib.pyplot  as plt
import math

from scipy.interpolate import Rbf
from scipy.spatial import Delaunay, SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d


with open('csv') as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

longitude = hot_spots_data[:, 0]
latitude = hot_spots_data[:, 1]

"""triangulations = Delaunay(hot_spots_data)

plt.triplot(hot_spots_data[:, 0], hot_spots_data[:, 1], triangulations.simplices)
plt.plot(hot_spots_data[:, 0], hot_spots_data[:, 1], 'o')
plt.show()"""

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

point = []
for k in range(343):
    point.append([x[k], y[k], z[k]])

print(point)


points = np.array(point)
radius = 1
center = np.array([0, 0, 0])
sv = SphericalVoronoi(points, radius, center)

# sort vertices (optional, helpful for plotting)
sv.sort_vertices_of_regions()
t_vals = np.linspace(0, 1, 2000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the unit sphere for reference (optional)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='y', alpha=0.1)
# plot generator points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
# plot Voronoi vertices
ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
                   c='g')
# indicate Voronoi regions (as Euclidean polygons)
for region in sv.regions:
   n = len(region)
   for i in range(n):
       start = sv.vertices[region][i]
       end = sv.vertices[region][(i + 1) % n]
       result = geometric_slerp(start, end, t_vals)
       ax.plot(result[..., 0],
               result[..., 1],
               result[..., 2],
               c='k')
ax.azim = 10
ax.elev = 40
_ = ax.set_xticks([])
_ = ax.set_yticks([])
_ = ax.set_zticks([])
fig.set_size_inches(10, 10)
plt.show()


areas = sv.calculate_areas()
