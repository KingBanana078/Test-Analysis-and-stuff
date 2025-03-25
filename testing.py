import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
import math
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

def read_csv(filename):
    """Reads a CSV file and converts it to a NumPy array of floats."""
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        return np.array([list(map(float, row)) for row in reader])

#lat&lon to cartesian
def transform_coordinates(hot_spots_data):
    lon_rad = np.radians(hot_spots_data[:, 1])
    lat_rad = np.radians(hot_spots_data[:, 0])

    r = 1 

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    #print("Cartesian Coordinates (x,y,z):", x,y,z)

    return np.column_stack([x, y, z])

#cartesian to spherical
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  
    phi = np.arctan2(y, x)  

    #print("Spherical Coordinates (r, θ, φ):", r,theta,phi)
    
    return np.column_stack([r, theta, phi])

def Mollweide_plot(hot_spots_data):

    theta = 180 - hot_spots_data[:, 1]  # Adjust longitude
    phi = hot_spots_data[:, 0]     # Adjust latitude

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.scatter(theta/180*math.pi, phi/180*math.pi)
    plt.show()

def compute_voronoi(points):
    points = np.asarray(points)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    sv = SphericalVoronoi(points, radius=1, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    return sv

def compute_area(sv):
    areas = sv.calculate_areas()
    #density = 1/areas
    
    #print(areas)
    #print(len(areas))
    #print(np.sum(areas))
    return areas

def plot_voronoi_cells(sv, areas):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], color='r', s=50, label='Sites')

    # Color cells based on density (1/area)
    densities = 1 / areas

    # Handle NaN or infinite densities
    densities = np.nan_to_num(densities, nan=0)  # Replace NaN values with 0

    max_density = max(densities)

    for i, region in enumerate(sv.regions):
        if len(region) > 0:  # Skip empty regions
            polygon = sv.vertices[region]
            if len(polygon) >= 3:
                # Use Poly3DCollection for 3D polygons
                color = mcolors.to_rgba(cm.plasma(densities[i] / max_density))  # Convert density to color
                poly3d = Poly3DCollection([polygon], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
                ax.add_collection3d(poly3d)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spherical Voronoi Diagram with Density')

    plt.show()

def compute_centroids(vertices, regions):
    """Computes the centroids of the Voronoi regions."""
    centroids = []
    for region in regions:
        if len(region) < 3:  # Skip degenerate regions with fewer than 3 vertices
            centroids.append([0, 0, 0])
            continue
        polygon = vertices[region]
        centroid = np.mean(polygon, axis=0)
        centroids.append(centroid)

        #print(len(centroids))
    return np.array(centroids)


#needs working on
def mollweide_plot(centroids, densities, interpolator = None):
    theta = np.pi - centroids[:, 2]  # Convert longitude (phi) to theta for Mollweide
    phi = centroids[:, 1]  # Latitude (theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "mollweide")

    # Normalize densities for color mapping
    max_density = max(densities)
    norm = plt.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma

    if interpolator:
        # Interpolation of densities to a grid of points
        theta_grid, phi_grid = np.meshgrid(np.linspace(-180, 180, 300), np.linspace(-90, 90, 150))
        theta_grid = np.radians(theta_grid)  # Convert to radians
        phi_grid = np.radians(phi_grid)  # Convert to radians
        
        # Flatten the grid for interpolation
        grid_points = np.column_stack([np.cos(phi_grid.flatten()) * np.cos(theta_grid.flatten()), 
                                       np.cos(phi_grid.flatten()) * np.sin(theta_grid.flatten()), 
                                       np.sin(phi_grid.flatten())])
        
        # Interpolate densities on the grid
        grid_densities = interpolator(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2])

        # Plot interpolated grid
        ax.pcolormesh(theta_grid, phi_grid, grid_densities.reshape(theta_grid.shape), shading='auto', cmap='plasma')

    # Plot the centroids with color based on density
    for i in range(len(centroids)):
        color = cmap(norm(densities[i]))  # Map density to color
        ax.scatter(theta[i] / 180 * math.pi, phi[i] / 180 * math.pi, c=[color], marker='o', s=50)

    ax.set_title('Voronoi Density on Mollweide Projection')
    plt.show()

def main():
    """Main function to execute the workflow."""
    filename = 'Positiondata.csv'
    hot_spots_data = read_csv(filename)

    # Convert to Cartesian coordinates
    points = transform_coordinates(hot_spots_data)

    # Compute Voronoi diagram
    sv = compute_voronoi(points)

    # Calculate areas of Voronoi cells
    areas = compute_area(sv)
    densities = 1/areas

    # Plot Voronoi diagram with densities
    plot_voronoi_cells(sv, areas)

    centroids = compute_centroids(sv.vertices, sv.regions)

    #print(len(densities))
    interpolator = NearestNDInterpolator(centroids, densities)

    # Plot the results on a Mollweide projection
    mollweide_plot(centroids, densities, interpolator)

if __name__ == "__main__":
    main()