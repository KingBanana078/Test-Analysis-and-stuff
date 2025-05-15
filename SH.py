import numpy as np
import matplotlib.pyplot as plt
import csv
import pyshtools

with open(r"Positiondata.csv") as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)
longitude_deg = hot_spots_data[:, 1]   # Longitude in degrees
latitude_deg = hot_spots_data[:, 0]  # Latitude in degrees

# Convert to radians and spherical coordinates
theta = np.radians(longitude_deg)   # longitude
phi = np.radians(90 - latitude_deg)      # co-latitude

print(theta, phi)

# Define a grid
n_theta, n_phi = 100, 200
theta_grid = np.linspace(0, np.pi, n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi)
phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)

# Initialize scalar field
density = np.zeros_like(theta_mesh)

# Choose kernel width (radians)
sigma = 0.1


# Sum Gaussian contributions from each point
for t, p in zip(theta, phi):
    a = np.sqrt((np.sin((theta_mesh-t) / 2) ** 2 + np.sin((phi_mesh-p) / 2) ** 2 * np.cos(t) * np.cos(theta_mesh)))
    d = 2 * np.arcsin(a)
    density += np.exp(-d**2 / (2 * sigma**2))

# Plot Result
plt.figure(figsize=(12, 6))
plt.contourf(np.degrees(phi_mesh), 90 - np.degrees(theta_mesh), density, levels=40, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('KDE-based Density on Sphere (Spherical Gaussian)')
plt.show()


# Spherical Harmonics Part
# Make sure your density is flipped in latitude (pyshtools uses +90 to -90 lat ordering)
density_flipped = np.flipud(density)

# Create a SHGrid object
grid = pyshtools.SHGrid.from_array(density_flipped)

# Expand into spherical harmonics
clm = grid.expand()  # Returns SHCoeffs object

# Optional: truncate to a maximum degree (e.g., lmax = 20)
clm_trunc = clm.pad(lmax=20)

# Reconstruct the smoothed field (can change grid resolution if needed)
reconstructed = clm_trunc.expand(grid='DH').to_array()

# Plot Result
plt.figure(figsize=(12, 6))
plt.contourf(np.linspace(0, 360, reconstructed.shape[1]),
             np.linspace(-90, 90, reconstructed.shape[0]),
             reconstructed, levels=40, cmap='viridis')
plt.title('Reconstructed Density from Spherical Harmonics')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.colorbar(label='Density')
plt.show()
