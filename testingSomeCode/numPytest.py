import numpy as np
import pandas as pd

player_1 = pd.read_csv("data/tracker_1_clean.csv")

arr = np.array(player_1)
print(arr.flatten())


list = np.array([[1,2,3],[6,7,8]])
list.flatten()
print(list.flatten())