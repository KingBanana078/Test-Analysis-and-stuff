from scipy.special import sph_harm
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from matplotlib import cm, colors

with open(r"Positiondata.csv")as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

longitude_deg = hot_spots_data[:,1]  # Longitude in degrees
latitude_deg = hot_spots_data[:,0] # Latitude in degrees


# Convert to radians
theta = np.radians(longitude_deg) # Azimuthal angle θ in [0, 2π]
latitude_rad = np.radians(latitude_deg)
phi = math.pi/2 - latitude_rad # Convert latitude to co-latitude φ in [0, π]

# Create Meshgrid (θ first, then φ)
phi, theta = np.meshgrid(theta, phi)

m, l = 2 , 3

print(theta)
print(phi)

print(sph_harm(m, l, theta, phi))

# phi = np.linspace(0, np.pi, 100)
# theta = np.linspace(0, 2*np.pi, 100)


# The Cartesian Coordinates of a Unit Sphere
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)


# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fcolors = sph_harm(m, l, theta, phi).real
f_max, f_min = fcolors.max(), fcolors.min()
fcolors = (fcolors - f_min)/(f_max - f_min)

# Set the aspect ratio to 1 so our sphere looks spherical
fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
# Turn off the axis planes
ax.set_axis_off()
plt.show()



