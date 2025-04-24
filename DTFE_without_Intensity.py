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
    longitude = np.radians(180- hot_spots_data[:, 1]) #in radians, converts already to spherical reference frame
    latitude = np.radians(90- hot_spots_data[:, 0]) #in randians, converts already to spherical reference frame

    r = 1 
    #now spherical to cartesian
    x = r * np.sin(latitude) * np.cos(longitude)
    y = r * np.sin(latitude) * np.sin(longitude)
    z = r * np.cos(latitude)

    #print("Cartesian Coordinates (x,y,z):", x,y,z)

    return np.column_stack([x, y, z])
#cartesian to spherical
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  
    phi = np.arctan2(y, x)  

    #print("Spherical Coordinates (r, θ, φ):", r,theta,phi)
    
    return np.column_stack([r, theta, phi])

#makes points good for a mollweide plot
def Mollweide_plot_points(hot_spots_data):

    theta = 180 - hot_spots_data[:, 1]  # Adjust longitude
    phi = hot_spots_data[:, 0]     # Adjust latitude

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.scatter(np.radians(theta), np.radians(phi))
    plt.show()

    #points = np.column_stack([ theta, phi])
    #print(points)
    return 

#creates the vonoroi regions, which is Delaunay triangulation
def compute_voronoi(points):
    points = np.asarray(points)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    sv = SphericalVoronoi(points, radius=1, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    return sv

def compute_area(sv): #areas of the vonoroi regions
    areas = sv.calculate_areas()
    #density = 1/areas
    #print(areas)
    #print(len(areas))
    #print(np.sum(areas))
    return areas

def plot_voronoi_cells(sv, areas):  #makes the vonoroi cells, which also create the polygons

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], color='c', s=50, label='Sites')

    densities = 1 / areas
    max_density = max(densities)

    for i, region in enumerate(sv.regions):
        if len(region) > 0: 
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

    plt.show() #plot the 3d map of the vonoroi diagram

def compute_centroids(vertices, regions):
    centroids = []
    for region in regions:
        if len(region) < 3:  # Skip degenerate regions with fewer than 3 vertices
            centroids.append([0, 0, 0])
            continue
        polygon = vertices[region]
        centroid = np.mean(polygon, axis=0)
        centroids.append(centroid)
        #print(polygon)
        #print(len(centroids))
    return np.array(centroids)

def interpolator_rbf(centroids, areas):
    #theta = np.radians(centroids[:, 2])  # Convert longitude to radians
    #phi = np.radians(centroids[:, 1])  # Convert latitude to radians

    # Convert to Cartesian coordinates for RBF interpolation
    #x = np.cos(phi) * np.cos(theta)
    #y = np.cos(phi) * np.sin(theta)
    #z = np.sin(phi)

    areas = np.array(areas)
    densities = 1 / areas

    rbf = Rbf(centroids[:, 0], centroids[:, 1], centroids[:, 2], densities, function='linear')
    #rbf = Rbf(x, y, z, densities, function='linear')  # 'linear', 'cubic', 'multiquadric', etc.

    return rbf

def mollweide_plot(centroids, data, interpolator=None):
    theta = np.radians(centroids[:, 2])  
    phi = np.radians(centroids[:, 1]) 

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    #ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], color='c', s=50, label='Sites')

    max_density = max(data)
    norm = plt.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar


    if interpolator:
        # Create grid for interpolation
        theta_grid, phi_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 360), np.linspace(-np.pi/2, np.pi/2, 360))

        # Convert grid points to Cartesian coordinates
        x_grid = np.cos(phi_grid) * np.cos(theta_grid)
        y_grid = np.cos(phi_grid) * np.sin(theta_grid)
        z_grid = np.sin(phi_grid)

        # Interpolate densities using RBF
        grid_densities = interpolator(x_grid, y_grid, z_grid).reshape(theta_grid.shape)

        # Plot interpolated grid with proper normalization
        ax.pcolormesh(theta_grid, phi_grid, grid_densities, shading='auto', cmap='plasma', norm=norm)

    # Plot the centroids with color based on density
    #for i in range(len(centroids)):
        #color = cmap(norm(data[i]))  # Map density to color
        #ax.scatter(theta[i], phi[i], c=[color], marker='o', s=50)

    # Add color bar (legend)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.1)
    cbar.set_label('Density')

    ax.set_title('Voronoi Density on Mollweide Projection')
    plt.show()

def rectangularPlot(centroids, data, interpolator=None):
    theta = np.radians(centroids[:, 2])  
    phi = np.radians(centroids[:, 1]) 

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    #ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], color='c', s=50, label='Sites')

    max_density = max(data)
    norm = plt.Normalize(vmin=0, vmax=max_density)
    cmap = plt.cm.plasma
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar


    if interpolator:
        # Create grid for interpolation
        theta_grid, phi_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 360), np.linspace(-np.pi/2, np.pi/2, 360))

        # Convert grid points to Cartesian coordinates
        x_grid = np.cos(phi_grid) * np.cos(theta_grid)
        y_grid = np.cos(phi_grid) * np.sin(theta_grid)
        z_grid = np.sin(phi_grid)

        # Interpolate densities using RBF
        grid_densities = interpolator(x_grid, y_grid, z_grid).reshape(theta_grid.shape)

        # Plot interpolated grid with proper normalization
        ax.pcolormesh(theta_grid, phi_grid, grid_densities, shading='auto', cmap='plasma', norm=norm)

    # Plot the centroids with color based on density
    #for i in range(len(centroids)):
        #color = cmap(norm(data[i]))  # Map density to color
        #ax.scatter(theta[i], phi[i], c=[color], marker='o', s=50)

    # Add color bar (legend)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.1)
    cbar.set_label('Density')

    ax.set_title('Voronoi Density on Mollweide Projection')
    plt.show()


def main():
    filename = 'Positiondata.csv'
    hot_spots_data = read_csv(filename)

    points = transform_coordinates(hot_spots_data)#calls the datapoints
    #Mollweide_plot_points(hot_spots_data)

    sv = compute_voronoi(points)#computes the vonoroi function
    #print((sv.vertices))
    areas = compute_area(sv)#calculates the areas for the vonoroi cells
    #print(np.sum(areas)) #checks whether the total area is 4pi
    densities = 1/areas

    plot_voronoi_cells(sv, areas) #plots the 3d map
    centroids = compute_centroids(sv.vertices, sv.regions) #computes centroids of the vonoroi regions

    interpolator = NearestNDInterpolator(centroids, densities) #interpolates the areas
    #interpolator = interpolator_rbf(centroids, areas) #interpolates using a radial basis

    mollweide_plot(centroids, densities, interpolator) #plots mollweide of the density
    rectangularPlot(centroids, densities, interpolator) #plots regular map of the density

if __name__ == "__main__":
    main()