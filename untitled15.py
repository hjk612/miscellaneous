import os
import pandas as pd

os.chdir('/Users/Hatim/Desktop/ivani/')

data = pd.read_csv('sample_data.txt', sep = ', ')
data['Timestamp'] = data['Timestamp'].map(lambda x: x.lstrip('[').rstrip(']'))
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
mapping = {True: 1, False: 0}
data = data.applymap(lambda s: mapping.get(s) if s in mapping else s)

