import csv
import numpy as np
import matplotlib.pyplot  as plt
from scipy.spatial import Delaunay
import pandas as pd

pd.read_csv('NewTestData(2).csv', comment = ";" )
 
with open('NewTestData(2).csv') as f:
    reader = csv.reader(f)
    hot_spots_data = list(reader)

print(hot_spots_data)