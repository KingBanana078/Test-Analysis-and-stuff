import csv
import numpy as np
import math
from scipy.spatial import SphericalVoronoi
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        return np.array([list(map(float, row)) for row in reader])

def read_power_area_csv():
    with open('powerANDarea.csv') as power:
        reader = csv.reader(power)
        powerandarea = list(reader)
        for i in range(len(powerandarea)):
            for j in range(len(powerandarea[i])):
                powerandarea[i][j] = float(powerandarea[i][j])
        power_data = np.array([row[0] for row in powerandarea])
        area_data = np.array([row[1] for row in powerandarea])
        return power_data, area_data

def transform_coordinates(hot_spots_data):
    lon_rad = np.radians(180 - hot_spots_data[:, 1])
    lat_rad = np.radians(hot_spots_data[:, 0])

    r = 1
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.column_stack([x, y, z])

def compute_voronoi(points):
    points = np.asarray(points)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    sv = SphericalVoronoi(points, radius=1, center=[0, 0, 0])
    sv.sort_vertices_of_regions()
    return sv

def compute_area(sv):
    areas = sv.calculate_areas()
    return areas

def compute_centroids(vertices, regions):
    centroids = []
    for region in regions:
        if len(region) < 3:  # Skip degenerate regions with fewer than 3 vertices
            centroids.append([0, 0, 0])
            continue
        polygon = vertices[region]
        centroid = np.mean(polygon, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def interpolator_rbf(centroids, values):
    return Rbf(centroids[:, 0], centroids[:, 1], centroids[:, 2], values, function='multiquadric')

def cartesian_to_lat_lon(x, y, z):
    # Convert cartesian coords back to lat, lon
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    
    # Convert to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    # Adjust longitude to be in [0, 360] range
    lon_deg = (lon_deg + 360) % 360
    
    return lat_deg, lon_deg

def generate_global_grid(interpolator, resolution=(36, 72)):
    # Create 36×72 grid (5° resolution)
    nlat, nlon = resolution
    
    # Create arrays of latitudes (-90 to 90) and longitudes (0 to 360)
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    
    # Create grid matrices
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Convert to Cartesian coordinates for interpolation
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    
    x_grid = np.cos(lat_rad) * np.cos(lon_rad)
    y_grid = np.cos(lat_rad) * np.sin(lon_rad)
    z_grid = np.sin(lat_rad)
    
    # Calculate intensity values using our interpolator
    intensity_values = interpolator(x_grid, y_grid, z_grid)
    
    # Reshape to match our grid dimensions
    intensity_grid = intensity_values.reshape(nlat, nlon)
    
    # Create output data structure
    grid_data = []
    for i in range(nlat):
        for j in range(nlon):
            grid_data.append([lats[i], lons[j], intensity_grid[i, j]])
    
    return np.array(grid_data)

def save_grid_to_csv(grid_data, filename='global_intensity_grid.csv'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Latitude', 'Longitude', 'Intensity'])
        for row in grid_data:
            writer.writerow(row)
    print(f"Grid data saved to {filename}")

def main():
    # Read input data
    filename = 'Positiondata.csv'
    hot_spots_data = read_csv(filename)
    powers, areas = read_power_area_csv()
    
    # Transform coordinates and compute Voronoi
    points = transform_coordinates(hot_spots_data)
    sv = compute_voronoi(points)
    
    # Calculate areas and intensity values
    r_io = 1821  # km (radius of Io)
    areas_vor = compute_area(sv) * r_io**2
    intensity = (powers / areas_vor) * 1000  # W/m²
    
    # Calculate centroids of Voronoi regions
    centroids = compute_centroids(sv.vertices, sv.regions)
    
    # Create interpolator using RBF
    interpolator = interpolator_rbf(centroids, intensity)
    
    # Generate global 36×72 grid of intensity values
    grid_data = generate_global_grid(interpolator)
    
    # Save the grid data to CSV
    save_grid_to_csv(grid_data)
    
    # Also return grid data in the desired format
    nlat, nlon = 36, 72
    formatted_grid = np.zeros((nlat, nlon))
    
    for point in grid_data:
        lat, lon, intensity = point
        i = int((lat + 90) / 5)  # Convert latitude to index
        j = int(lon / 5)         # Convert longitude to index
        
        # Ensure we don't go out of bounds
        i = min(nlat-1, max(0, i))
        j = min(nlon-1, max(0, j))
        
        formatted_grid[i, j] = intensity
    
    # Plot the intensity grid
    plt.figure(figsize=(10, 6))
    plt.imshow(formatted_grid, cmap='inferno', origin='lower', aspect='auto', extent=[0, 360, -90, 90])
    plt.colorbar(label='Intensity (W/m²)')
    plt.title('Global Intensity Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    return formatted_grid

if __name__ == "__main__":
    global_intensity = main()
    
    # Print the shape of the resulting grid
    print(f"Grid shape: {global_intensity.shape}")
    
    # Save as a numpy array if needed
    np.save('global_intensity_36x72.npy', global_intensity)
    print("Grid saved as global_intensity_36x72.npy")
