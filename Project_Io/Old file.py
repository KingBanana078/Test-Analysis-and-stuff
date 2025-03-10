import csv
import numpy as np
import matplotlib.pyplot  as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import proj3d


with open('csv') as csvfile:
    reader = csv.reader(csvfile)
    hot_spots_data = list(reader)

hot_spots_data = np.array(hot_spots_data, dtype=float)

longitude = hot_spots_data[:, 0]
latitude = hot_spots_data[:, 1]

triangulations = Delaunay(hot_spots_data)

plt.triplot(hot_spots_data[:, 0], hot_spots_data[:, 1], triangulations.simplices)
plt.plot(hot_spots_data[:, 0], hot_spots_data[:, 1], 'o')
plt.show()