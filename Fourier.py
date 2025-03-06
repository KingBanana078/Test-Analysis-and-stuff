import csv
import numpy as np
import matplotlib.pyplot  as plt
from scipy.spatial import Delaunay


with open('io_volcanoes.csv') as f:
    reader = csv.reader(f)
    hot_spots_data = list(reader)

hot_spots_data.pop(0)

hot_spots_data = np.array(hot_spots_data, dtype=float)

x = hot_spots_data[:, 0]
y = hot_spots_data[:, 1]


