import pandas as pd
import numpy as np

W, H = 1200, 1920
dw = 1. / W
dh = 1. / H

data = pd.read_csv('/Users/meshams/Downloads/object-detection-crowdai/labels.csv')

labels = data.Label.unique()

data['x1'] = np.min(data[['xmin', 'xmax']], axis=1)
data['x2'] = np.max(data[['xmin', 'xmax']], axis=1)
data['y1'] = np.min(data[['ymin', 'ymax']], axis=1)
data['y2'] = np.max(data[['ymin', 'ymax']], axis=1)

data['w'] = dw * (data['x2'] - data['x1'])
data['h'] = dh * (data['y2'] - data['y1'])
data['x'] = dw * ((data['x1'] + data['x2']) / 2 - 1)
data['y'] = dh * ((data['y1'] + data['y2']) / 2 - 1)

data.to_csv('/Users/meshams/Downloads/object-detection-crowdai/labels2.csv', index=False)