import pandas as pd
import os
import numpy as np
import cv2

data = pd.read_csv('./data/object-detection-crowdai/labels_crowdai.csv')

labels = data.Label.unique()

frames = data.Frame.unique()

images = []
boxes = []

for frame in frames:
    img = cv2.imread(os.path.join('./data/object-detection-crowdai', frame))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

    frame_boxes = data[data.Frame == frame]
    boxes.append(frame_boxes[['xmin','ymin','xmax','ymax']].as_matrix())

np.save('data.npy', {'images':np.array(images), 'boxes':boxes})