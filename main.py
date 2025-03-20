import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, geometric_slerp
from scipy.interpolate import Rbf
import math


from scipy.interpolate import Rbf, RBFInterpolator
from scipy.spatial import Delaunay, SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d

def read_csv(filename):
    """Reads a CSV file and converts it to a NumPy array of floats."""
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        return np.array([list(map(float, row)) for row in reader])

def transform_coordinates(hot_spots_data):
    """Converts latitude and longitude to spherical coordinates."""
    theta = 180 - hot_spots_data[:, 1]  # Adjust longitude
    phi = 90 - hot_spots_data[:, 0]     # Adjust latitude

    return theta, phi

def Mollweide(hot_spots_data):

    theta = 180 - hot_spots_data[:, 1]  # Adjust longitude
    phi = hot_spots_data[:, 0]     # Adjust latitude


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.scatter(theta/180*math.pi, phi/180*math.pi)
    plt.show()


def plot_scatter(theta, phi):
    """Plots a scatter plot of transformed coordinates."""
    plt.scatter(theta, phi)
    plt.xlabel("Theta (Longitude Adjusted)")
    plt.ylabel("Phi (Latitude Adjusted)")
    plt.title("Transformed Hot Spot Coordinates")
    plt.show()

def spherical_to_cartesian(theta, phi, r=1):
    """Converts spherical coordinates (theta, phi) to Cartesian (x, y, z)."""
    x = r * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    y = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = r * np.cos(np.radians(phi))
    return np.column_stack((x, y, z))

def compute_voronoi(points):
    """Computes the Spherical Voronoi diagram."""
    radius = 1
    center = np.array([0, 0, 0])
    
    sv = SphericalVoronoi(points, radius, center)
    sv.sort_vertices_of_regions()  # Sort for better visualization

    return sv

def plot_voronoi(sv, points):
    """Plots the Spherical Voronoi diagram."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='y', alpha=0.1)

    # Plot generator points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label="Hot Spots")

    # Plot Voronoi vertices
    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g', label="Voronoi Vertices")

    # Plot Voronoi edges
    t_vals = np.linspace(0, 1, 2000)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            ax.plot(result[..., 0], result[..., 1], result[..., 2], c='k')

    ax.legend()
    plt.title("Spherical Voronoi Diagram of Hot Spots")
    plt.show()

def compute_density(sv):
    """Computes density based on Voronoi cell areas."""
    areas = (sv.calculate_areas())*1.5  # Compute Voronoi region areas
    
    # Compute centroids of Voronoi regions
    centroids = np.array([np.mean(sv.vertices[region], axis=0) for region in sv.regions])

    # Normalize centroids to lie on the unit sphere
    centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]

    # Define density as inverse of area (higher area = lower density)
    densities = 1 /  areas

    # Create interpolation function (RBF) using centroids
    rbf = Rbf(centroids[:, 0], centroids[:, 1], centroids[:, 2], densities, function='linear')

    return rbf

def plot_density(rbf):
    """Plots a 2D density map from the RBF interpolation."""
    num_grid = 360
    grid_theta = np.linspace(-np.pi, np.pi, num_grid)
    grid_phi = np.linspace(-np.pi/2, np.pi/2, num_grid)
    phi_grid, theta_grid = np.meshgrid(grid_phi, grid_theta)
    
    # Convert grid points to Cartesian coordinates
    x_grid = np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = np.cos(theta_grid)

    # Interpolate density on grid
    density_grid = rbf(x_grid, y_grid, z_grid)

    # Plot the density map
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(np.linspace(-180, 180, num_grid), np.linspace(-90, 90, num_grid), density_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Interpolated Density Field from Spherical Voronoi')
    plt.show()

def main():
    """Main function to execute the workflow."""
    filename = 'Positiondata.csv'  # Update with the correct file path if needed
    hot_spots_data = read_csv(filename)

    theta, phi = transform_coordinates(hot_spots_data)
    plot_scatter(theta, phi)

    Mollweide(hot_spots_data)
    
    points = spherical_to_cartesian(theta, phi)
    sv = compute_voronoi(points)
    plot_voronoi(sv, points)

    # Compute and plot density
    rbf = compute_density(sv)
    plot_density(rbf)

if __name__ == "__main__":
    main()