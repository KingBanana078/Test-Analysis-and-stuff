import numpy as np
import matplotlib.pyplot as plt
import csv
import pyshtools

# load data
with open(r"Positiondata.csv") as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)
longitude_deg = hot_spots_data[:, 1]   # Longitude in degrees
latitude_deg = hot_spots_data[:, 0]    # Latitude in degrees

# convert to radians
lon_rad = np.radians(longitude_deg)
lat_rad = np.radians(latitude_deg)

# set up spherical grid
n_lat, n_lon = 180, 360
lat_grid = np.linspace(np.pi / 2, -np.pi / 2, n_lat)
lon_grid = np.linspace(2*np.pi, 0, n_lon)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# initialize density map
density = np.zeros_like(lat_mesh)

# gaussian kernel width (in radians)
sigma = 0.1

# compute haversine distances and sum Gaussians
for lon0, lat0 in zip(lon_rad, lat_rad):
    a = np.sin((lat_mesh-lat0)/2) ** 2 + np.cos(lat0) * np.cos(lat_mesh) * np.sin((lon_mesh-lon0)/2) ** 2
    d = 2.0 * np.arcsin(np.sqrt(a))
    density += np.exp(-(d ** 2) / (2.0 * sigma ** 2))
#np.arccos(np.sin(lat_mesh) * np.sin(lat0) +np.cos(lat_mesh) * np.cos(lat0) * np.cos(lon_mesh - lon0))

# flip for pyshtools (lat: +90 to -90)
density_flipped = np.flipud(density)

# spherical harmonic expansion
grid = pyshtools.SHGrid.from_array(density_flipped)
clm = grid.expand()
clm_trunc = clm.pad(lmax=40)

# reconstruct field
reconstructed = clm_trunc.expand(grid='DH').to_array()

# create lat/lon radian grids for plotting
lat_rad_plot = np.linspace(-np.pi / 2, np.pi / 2, reconstructed.shape[0])
lon_rad_plot = np.linspace(-np.pi, np.pi, reconstructed.shape[1])

# plot
plt.figure(figsize=(10, 5))
plt.contourf(lon_rad_plot, lat_rad_plot, reconstructed, levels=40, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('Longitude (radians)')
plt.ylabel('Latitude (radians)')
plt.title('Reconstructed Density from Spherical Harmonics')
plt.show()
