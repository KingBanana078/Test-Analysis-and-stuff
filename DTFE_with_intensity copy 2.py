import csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

def read_temp_csv():
    with open('Temperature.csv') as temp:
        reader = csv.reader(temp)
        return [list(map(float, row)) for row in reader]


def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        return np.array([list(map(float, row)) for row in reader])

#lat&lon to cartesian
def transform_coordinates(hot_spots_data):
    lon_rad = np.radians(180 - hot_spots_data[:, 1])
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

#OK
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

def plot_voronoi_cells(sv, areas, region_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    densities = 1 / areas
    max_density = max(densities)

    for i, idx in enumerate(region_indices):
        region = sv.regions[idx]
        polygon = sv.vertices[region]
        if len(polygon) >= 3:
            color = mcolors.to_rgba(cm.cividis(densities[i] / max_density))
            poly3d = Poly3DCollection([polygon], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
            ax.add_collection3d(poly3d)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voronoi Diagram with Density')
    plt.show()



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

    rbf = Rbf(centroids[:, 0], centroids[:, 1], centroids[:, 2], densities, function='multiquadric')
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


    # Add color bar (legend)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.1)
    cbar.set_label('Instensity(W/m$^2$)')

    ax.set_title('Voronoi Intensity on Mollweide Projection')
    plt.show()


#----- INTENSITY ADJUSTMENTS START HERE -----#

def read_power_area_csv():
    with open('powerANDarea.csv') as power:
        reader = csv.reader(power)
        powerandarea = list(reader)
        for i in range(len(powerandarea)): 
            for j in range(len(powerandarea[i])):
                powerandarea[i][j]=float(powerandarea[i][j])
        power_data = np.array([row[0] for row in powerandarea])
        area_data = np.array([row[1] for row in powerandarea])
        return power_data, area_data


def read_temp_csv():
    with open('Temperature.csv') as temp:
        reader = csv.reader(temp)
        return [list(map(float, row)) for row in reader]

def main():
    filename = 'Positiondata.csv'
    hot_spots_data = read_csv(filename)
    powers, areas = read_power_area_csv()
    #powers = powers[:-2]
    #areas = areas[:-2]
    temps = read_temp_csv()
    points = transform_coordinates(hot_spots_data)
    print(len(powers))
    mask = areas != 0
    r_io = 1821 #km
    #points1, area_data_1, power_data_1 = points[mask], areas[mask], powers[mask]

    #-----IF WE USE THE VORONOI AREAS---------------------
    sv = compute_voronoi(points)
    #print((sv.vertices)
    areas_vor = compute_area(sv)*r_io**2
    areas_vor = areas_vor
    densities = 1/areas_vor
    intensity1 = (np.sort(powers / areas_vor)*1000) #w/m^2
    intensity1log = np.log((np.sort(powers / areas_vor)*1000)) #w/m^2

    region_indices = np.arange(len(sv.regions))

    plot_voronoi_cells(sv, areas_vor, region_indices)

    centroids = compute_centroids(sv.vertices, sv.regions)

    #interpolator = NearestNDInterpolator(centroids, intensity1)
    interpolator = interpolator_rbf(centroids, 1/intensity1)

    #interpolatorLog = NearestNDInterpolator(centroids, intensity1log)
    interpolatorLog = interpolator_rbf(centroids, 1/intensity1log)

    mollweide_plot(centroids, intensity1, interpolator)
    mollweide_plot(centroids, intensity1log, interpolatorLog)
    #print(intensity1)



    '''
    #-----IF WE USE THE DATASET AREAS---------------------
    sv = compute_voronoi(points[:-2])
    powers = powers[mask]
    areas_data = areas[mask]

    region_indices = np.where(mask)[0] #chatgpt
    intensity1 = np.sort(powers / areas_data )
    centroids = compute_centroids(sv.vertices, sv.regions)
    centroids = centroids[region_indices]  

    plot_voronoi_cells(sv, areas_data, region_indices) #chatgpt
    
    interpolator = NearestNDInterpolator(centroids, intensity1)

    mollweide_plot(centroids, intensity1, interpolator)
    '''


#----- BELOW ARE CALCULATIONS ABOUT THE LATITUDINAL DISTRIBUTION OF VOLCANOES AND INTENSITIES -----#

    '''
    #intensity_unsorted = powers / areas_vor
    #intensity_unsorted = powers / areas_data
    plusminus45, poles, INTplusminus45, INTpoles = [], [], [], [] 
    N = min(len(points), len(intensity_unsorted))
    
    for i in range(N):
        if centroids[i,2] > math.sqrt(2)/2:
            poles.append(centroids[i,2])
            INTpoles.append(intensity_unsorted[i])
        elif centroids[i,2] < - math.sqrt(2)/2:
            poles.append(centroids[i,2])
            INTpoles.append(intensity_unsorted[i])
        else:
            plusminus45.append(centroids[i,2])
            INTplusminus45.append(intensity_unsorted[i])
 
    totalintensity = sum(intensity_unsorted)
    #avgintensity = totalintensity/sum(areas_vor)
    #avgintensity = totalintensity/sum(areas)

    print(totalintensity, avgintensity, sum(INTplusminus45), sum(INTpoles), sum(INTplusminus45)/totalintensity, sum(INTpoles)/totalintensity)
    print(len(plusminus45)/(len(plusminus45)+len(poles)), len(poles)/(len(plusminus45)+len(poles)))
    '''


if __name__ == "__main__":
    main()