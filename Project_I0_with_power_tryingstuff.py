import csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, geometric_slerp
from scipy.interpolate import Rbf

R_Io = 1821600
# Read the position data (latitude, longitude) from Positiondata.csv
with open('Positiondata.csv') as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

# Extract latitude and longitude from hot_spots_data
longitude = hot_spots_data[:, 1]
latitude = hot_spots_data[:, 0]

# Convert spherical coordinates (latitude, longitude) to 3D Cartesian coordinates
x = []
y = []
z = []
theta = hot_spots_data[:, 0]  # Latitude
phi = hot_spots_data[:, 1]  # Longitude
r = np.ones(343)  # Assume the radius is 1 (unit sphere)

# Convert spherical to Cartesian coordinates
for i in range(343):
    x.append(float(r[i] * math.cos(theta[i]) * math.sin(phi[i])))
    y.append(float(r[i] * math.sin(theta[i]) * math.sin(phi[i])))
    z.append(float(r[i] * math.cos(phi[i])))

point = []
for k in range(343):
    point.append([x[k], y[k], z[k]])

# Convert points to a numpy array for processing
points = np.array(point)
radius = 1
center = np.array([0, 0, 0])

# Generate the Spherical Voronoi diagram
sv = SphericalVoronoi(points, radius, center)

# Sort vertices (optional, helpful for plotting)
sv.sort_vertices_of_regions()

# Calculate the areas of the Voronoi regions
areas = sv.calculate_areas()


'''
# Prepare for plotting the Voronoi diagram
t_vals = np.linspace(0, 1, 2000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the unit sphere for reference (optional)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='y', alpha=0.1)

# Plot generator points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')

# Plot Voronoi vertices
ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')

# Indicate Voronoi regions (as Euclidean polygons)
for region in sv.regions:
    n = len(region)
    for i in range(n):
        start = sv.vertices[region][i]
        end = sv.vertices[region][(i + 1) % n]
        result = geometric_slerp(start, end, t_vals)
        ax.plot(result[..., 0], result[..., 1], result[..., 2], c='k')
'''

# Read the power data (in Watts) from Powerdata.csv
power_data, area_data = [], []
with open('power.csv') as power:
    reader = csv.reader(power)
    powerandarea = list(reader)

for i in range(len(powerandarea)): 
    for j in range(len(powerandarea[i])):
        powerandarea[i][j]=float(powerandarea[i][j])

power_data = [row[0] for row in powerandarea]
area_data = [row[1] for row in powerandarea]

#total_surface_area = 4*np.pi
#areas_in_m2 = area_data * total_surface_area
#scaled_areas = areas_in_m2 * (R_Io**2)

# Convert the power data into a numpy array
power_data = np.array(power_data)  # Already in Watts (W)
area_data = np.array(area_data)
# Compute centroids of Voronoi regions
centroids = np.array([np.mean(sv.vertices[region], axis=0) for region in sv.regions])

# Calculate intensity (W/m²) for each Voronoi region
# Intensity = Power (W) / Area (m²)
mask = area_data != 0

centroids1, area_data_1, power_data_1 = centroids[mask], area_data[mask], power_data[mask]
intensity = power_data_1 / area_data_1
intensity1 = np.sort(intensity)
centroids1_1 = np.sort(centroids1)
centroids1_1 = centroids1_1[:-6]
intensity1_1 = intensity1[:-6]
centroids2, power_data_2 = centroids[:-10], power_data[:-10]


# Create interpolation function (RBF) using centroids and intensity values
rbf1 = Rbf(centroids1[:, 0], centroids1[:, 1], centroids1[:, 2], intensity, function='linear')
rbf1_1 = Rbf(centroids1_1[:, 0], centroids1_1[:, 1], centroids1_1[:, 2], intensity1_1, function='linear')
rbf2 = Rbf(centroids[:, 0], centroids[:, 1], centroids[:, 2], power_data, function='linear')
rbf2_2 = Rbf(centroids2[:, 0], centroids2[:, 1], centroids2[:, 2], power_data_2, function='linear')
rbf3 = Rbf(centroids1[:, 0], centroids1[:, 1], centroids1[:, 2], area_data_1, function='linear')

# Generate grid for visualization
num_grid = 360
grid_phi = np.linspace(-np.pi, np.pi, num_grid)
grid_theta = np.linspace(-np.pi/2, np.pi/2, num_grid)
phi_grid, theta_grid = np.meshgrid(grid_phi, grid_theta)
x_grid = np.sin(theta_grid) * np.cos(phi_grid)
y_grid = np.sin(theta_grid) * np.sin(phi_grid)
z_grid = np.cos(theta_grid)

# Interpolate intensity on the grid
intensity_grid = rbf1(x_grid, y_grid, z_grid)
intensity_grid_2 = rbf1_1(x_grid, y_grid, z_grid)
power_grid = rbf2(x_grid, y_grid, z_grid)
power_grid_2 = rbf2_2(x_grid, y_grid, z_grid)
area_grid = rbf3(x_grid, y_grid, z_grid)

# Plot the intensity map (W/m²)
plt.figure(figsize=(10, 5))
plt.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='auto', cmap='viridis')
plt.colorbar(label='Intensity (W/m²)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Intensity Field (W/m²) from Spherical Voronoi')

#other plots
plt.figure(figsize=(10, 5))
plt.pcolormesh(phi_grid, theta_grid, intensity_grid_2, shading='auto', cmap='viridis')
plt.colorbar(label='Intensity - excluding 6 highest (W/m²)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Intensity Field (W/m²) - excluding 6 highest')

plt.figure(figsize=(10, 5))
plt.pcolormesh(phi_grid, theta_grid, power_grid, shading='auto', cmap='viridis')
plt.colorbar(label='Power (GW)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Power Field (GW)')

plt.figure(figsize=(10, 5))
plt.pcolormesh(phi_grid, theta_grid, power_grid_2, shading='auto', cmap='viridis')
plt.colorbar(label='Power (GW) - excluding 10 highest')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Power Field (GW) - excluding 10 highest')

plt.figure(figsize=(10, 5))
plt.pcolormesh(phi_grid, theta_grid, area_grid, shading='auto', cmap='viridis')
plt.colorbar(label='Area (km²)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Area Field (km²)')

# Add a scatter plot for the power values at the generator points
#scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=power_data, cmap='coolwarm', marker='o', label="Power (W)")
#fig.colorbar(scatter, ax=ax, label='Power (W)', shrink=0.5, aspect=5)

plt.show()

