from PIL import Image, ImageDraw
from matplotlib import image as mpimg

import os
import pandas as pd

path = os.path.expanduser('~/Downloads/object-detection-crowdai/')
data = pd.read_csv(os.path.expanduser('~/Downloads/object-detection-crowdai/labels2.csv'))

sample = data.sample(frac=0.01, random_state=100)

rec = sample[0:1]
filename = rec.Frame.values[0]
img = mpimg.imread(path + filename)

H, W = img.shape[0:2]

records = data[data.Frame == filename]
image = Image.fromarray(img)
draw = ImageDraw.Draw(image)

for index, rec in records.iterrows():
    w, h, x, y = rec.w, rec.h, rec.x, rec.y
    cx, cy = x * W, y * H
    cw, ch = w * W, h * H

    x1 = int(cx - cw / 2)
    x2 = int(cx + cw / 2)
    y1 = int(cy - ch / 2)
    y2 = int(cy + ch / 2)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))

image.save(os.path.splitext(filename)[0] + '_out.jpg')
