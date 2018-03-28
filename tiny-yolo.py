from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import random
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from yolo_utils import *


DATA_PATH = os.path.join(os.getcwd(), 'data/object-detection-crowdai')


def create_model():
    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=(416, 416, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2 - 5
    for i in range(0, 4):
        model.add(Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Layer 7 - 8
    for _ in range(0, 2):
        model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    # Layer 9
    model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))

    return model


def custom_loss(y_true, y_pred):
    ### Adjust prediction
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)

    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    print("Y_pred shape: {}".format(y_pred.shape))

    ### Adjust ground truth
    # adjust x and y
    center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W) / GRID_W), (float(NORM_H) / GRID_H)], [1, 1, 1, 1, 2])
    true_box_xy = center_xy - tf.floor(center_xy)

    # adjust w and h
    true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

    # adjust confidence
    true_box_prob = y_true[:, :, :, :, 5:]

    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    print("Y_true shape: {}".format(y_true.shape))
    # y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor

    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf

    weight_prob = tf.concat(CLASS * [true_box_conf], 4)
    weight_prob = SCALE_PROB * weight_prob

    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    print("Weight shape: {}".format(weight.shape))

    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W * GRID_H * BOX * (4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss


def _main(args):
    # model_path = '/Users/meshams/Documents/Learn/temp/YAD2K/model_data/tiny-yolo.h5'
    anchor_path = 'tiny-yolo_anchors.txt'

    data = pd.read_csv(os.path.join(DATA_PATH, 'labels2.csv'))
    class_names = data.Label.unique()

    lenc = LabelEncoder()
    data['c'] = lenc.fit_transform(data.Label)

    if args.subsample:
        data = data.sample(frac=0.002, random_state=1000)

    mask = np.random.rand(len(data)) < 0.9

    train_data = data[mask]
    valid_data = data[~mask]

    model = create_model()

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
    logging = TensorBoard()

    # sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    batch_size = args.batch_size
    print(f"Running on train={len(train_data)}, validation={len(valid_data)}, Batch size = {batch_size}")

    train_generator = DataGenerator(DATA_PATH, ANCHORS, batch_size=batch_size).generate(train_data)
    valid_generator = DataGenerator(DATA_PATH, ANCHORS, batch_size=batch_size).generate(valid_data)

    model.compile(loss=custom_loss, optimizer=optimizer)
    model.fit_generator(generator=train_generator, steps_per_epoch=len(train_data) // batch_size,
                        validation_data=valid_generator, validation_steps=len(valid_data) // batch_size,
                        callbacks=[logging, early_stop, checkpoint], epochs=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO')
    parser.add_argument("--subsample", action="store_true", default=False)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    _main(args)
