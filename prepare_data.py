import pandas as pd
import os
import numpy as np

data = pd.read_csv(os.path.expanduser('~/Downloads/object-detection-crowdai/labels_crowdai.csv'))

W, H = 1920, 1200
xs, ys = 416./W, 416./H

W, H = 416, 416
dw = 1. / W
dh = 1. / H

labels = data.Label.unique()

# data['x1'] = np.min(data[['xmin', 'xmax']], axis=1) * xs
# data['x2'] = np.max(data[['xmin', 'xmax']], axis=1) * xs
# data['y1'] = np.min(data[['ymin', 'ymax']], axis=1) * ys
# data['y2'] = np.max(data[['ymin', 'ymax']], axis=1) * ys

# data['x1'] = data['xmin'] * xs
# data['x2'] = data['xmax'] * xs
# data['y1'] = data['ymin'] * ys
# data['y2'] = data['ymax'] * ys

data['w'] = dw * (data['x2'] - data['x1'])
data['h'] = dh * (data['y2'] - data['y1'])
data['x'] = dw * ((data['x1'] + data['x2']) / 2)
data['y'] = dh * ((data['y1'] + data['y2']) / 2)

data.to_csv(os.path.expanduser('~/Downloads/object-detection-crowdai/labels2.csv'), index=False)

