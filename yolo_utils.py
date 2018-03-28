import numpy as np
import os
import tensorflow as tf
import copy
import cv2
from sklearn.utils import shuffle

NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13, 13
BATCH_SIZE = 8
BOX = 5
THRESHOLD = 0.2
ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
CLASS = 3


class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w * self.h + box.w * box.h - intersection
        return intersection / union

    def intersect(self, box):
        width = self.__overlap([self.x - self.w / 2, self.x + self.w / 2], [box.x - box.w / 2, box.x + box.w / 2])
        height = self.__overlap([self.y - self.h / 2, self.y + self.h / 2], [box.y - box.h / 2, box.y + box.h / 2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3


# def interpret_netout(image, netout):
#     boxes = []
#
#     # interpret the output by the network
#     for row in range(GRID_H):
#         for col in range(GRID_W):
#             for b in range(BOX):
#                 box = BoundBox(CLASS)
#
#                 # first 5 weights for x, y, w, h and confidence
#                 box.x, box.y, box.w, box.h, box.c = netout[row, col, b, :5]
#
#                 box.x = (col + sigmoid(box.x)) / GRID_W
#                 box.y = (row + sigmoid(box.y)) / GRID_H
#                 box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
#                 box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
#                 box.c = sigmoid(box.c)
#
#                 # rest of weights for class likelihoods
#                 classes = netout[row, col, b, 5:]
#                 box.probs = softmax(classes) * box.c
#                 box.probs *= box.probs > THRESHOLD
#
#                 boxes.append(box)
#
#     # suppress non-maximal boxes
#     for c in range(CLASS):
#         sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))
#
#         for i in range(len(sorted_indices)):
#             index_i = sorted_indices[i]
#
#             if boxes[index_i].probs[c] == 0:
#                 continue
#             else:
#                 for j in range(i + 1, len(sorted_indices)):
#                     index_j = sorted_indices[j]
#
#                     if boxes[index_i].iou(boxes[index_j]) >= 0.4:
#                         boxes[index_j].probs[c] = 0
#
#     print("Number of initial boxes: {}".format(len(boxes)))
#
#     # draw the boxes using a threshold
#     for box in boxes:
#         max_indx = np.argmax(box.probs)
#         max_prob = box.probs[max_indx]
#         print("Highest box probability for box: {}".format(max_prob))
#
#         if max_prob > THRESHOLD:
#             xmin = int((box.x - box.w / 2) * image.shape[1])
#             xmax = int((box.x + box.w / 2) * image.shape[1])
#             ymin = int((box.y - box.h / 2) * image.shape[0])
#             ymax = int((box.y + box.h / 2) * image.shape[0])
#
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
#             cv2.putText(image, labels[max_indx], (xmin, ymin - 12), 0, 1e-3 * image.shape[0], (0, 255, 0), 2)
#
#     return image


def read_imagenet_labels(label_file):
    labels = {}
    with open(label_file) as f:
        for line in f:
            wnid, _, label = line.split()
            labels[wnid] = label
    return labels


class DataGenerator(object):
    def __init__(self, data_path, anchors, dim_x=416, dim_y=416, batch_size=32, shuffle=True):
        'Initialization'
        self._data_path = data_path
        self.anchors = anchors
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, data):
        'Generates batches of samples'
        # Infinite loop
        files = data.Frame.unique()
        while 1:
            # Generate batches
            imax = int(len(files) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                batch_files = list(files[i * self.batch_size:(i + 1) * self.batch_size])
                batch = data[data.Frame.isin(batch_files)]

                # Generate data
                yield self.__data_generation(batch)

    def __read_image(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (NORM_W, NORM_H))
        return img.astype(np.float32) / 255.

    def __data_generation(self, batch):
        files = shuffle(batch.Frame.unique())
        image_files = [os.path.join(self._data_path, f) for f in files]
        images = [self.__read_image(f) for f in image_files]

        x_batch = np.zeros((self.batch_size, NORM_W, NORM_H, 3))
        y_batch = np.zeros((self.batch_size, GRID_W, GRID_H, BOX, 5 + CLASS))

        boxes = []
        max_boxz = 0
        for idx, p in enumerate(zip(files, images)):
            f, img = p
            rows = batch[batch.Frame == f]

            for _, obj in rows.iterrows():
                center_x = .5 * (obj['xmin'] + obj['xmax'])  # xmin, xmax
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = .5 * (obj['ymin'] + obj['ymax'])  # ymin, ymax
                center_y = center_y / (float(NORM_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_idx = obj['c']
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

                    y_batch[idx, grid_y, grid_x, :, 0:4] = BOX * [box]
                    y_batch[idx, grid_y, grid_x, :, 4] = BOX * [1.]
                    y_batch[idx, grid_y, grid_x, :, 5:] = BOX * [[0.] * CLASS]
                    y_batch[idx, grid_y, grid_x, :, 5 + obj_idx] = 1.0

            x_batch[idx] = img

        return x_batch, y_batch


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
