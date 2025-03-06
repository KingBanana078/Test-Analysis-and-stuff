import csv
import numpy as np

with open('IO-data.csv') as f:
    reader = csv.reader(f)
    hot_spots_data = list(reader)


print(hot_spots_data)

hot_spots_data = np.array(hot_spots_data, dtype=float)

longitude = hot_spots_data[:, 5]
latitude = hot_spots_data[:, 6]

print(longitude, latitude)
