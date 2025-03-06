import numpy as np
import csv
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

latitude_longitude = []
with open('io_volcanoes.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        latitude_longitude.append([float(x) for x in line])

array = np.array(latitude_longitude)
tri = Delaunay(array)

import matplotlib.pyplot as plt

plt.triplot(array[:, 1], array[:, 0], tri.simplices)
plt.plot(array[:, 1], array[:, 0], 'o')


# Step 3: Compute areas of the triangles
def triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given its three points using the determinant method."""
    return 0.5 * np.abs(
        p1[0] * (p2[1] - p3[1]) +
        p2[0] * (p3[1] - p1[1]) +
        p3[0] * (p1[1] - p2[1])
    )


# Step 4: Compute the inverse sum of surrounding triangle areas
point_density = np.zeros(len(array))

for i, point in enumerate(array):
    surrounding_triangles = np.where(tri.simplices == i)[0]  # Get indices of triangles that contain this point
    total_area = sum(triangle_area(array[t[0]], array[t[1]], array[t[2]]) for t in tri.simplices[surrounding_triangles])

    # Avoid division by zero
    point_density[i] = 1 / total_area if total_area > 0 else 0

# Step 5: Normalize densities for better visualization
normalized_density = (point_density - np.min(point_density)) / (np.max(point_density) - np.min(point_density))
# Step 6: Create a continuous heatmap (interpolation)
# Define grid
grid_x, grid_y = np.mgrid[
                 np.min(array[:, 1]):np.max(array[:, 1]):100j,  # Longitude range
                 np.min(array[:, 0]):np.max(array[:, 0]):100j  # Latitude range
                 ]

# Interpolate density values onto the grid
grid_z = griddata((array[:, 1], array[:, 0]), normalized_density, (grid_x, grid_y), method='cubic')

# Step 7: Plot the density field
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='plasma')  # Continuous heatmap
plt.colorbar(label='Density (1/Total Triangle Area)')

# Overlay original points for reference
plt.scatter(array[:, 1], array[:, 0], c='black', s=10, label='Volcano Locations')

# Labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Continuous 2D Density Field of Io Volcanoes")
plt.legend()
plt.show()