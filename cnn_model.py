import glob
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn import utils as sklearn_utils
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from utils import add_heat, color_heat_map, draw_boxes
import argparse
import cv2
import os
import shutil


class NeuralNetModel(object):
    epochs = 50
    batch_size = 32
    diag_kernel = [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]

    def __init__(self, weight_file_name=None):
        self._weight_file_name = weight_file_name
        if weight_file_name is not None:
            # Limit the image in height to the ROI
            input_shape = (660 - 400, 1280, 3)

            self._model = NeuralNetModel.create_model(input_shape=input_shape)
            self._model.load_weights(weight_file_name)

    @staticmethod
    def create_model(input_shape=(64, 64, 3)):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255., input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.5))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(8, 8)))
        model.add(Dropout(0.5))

        model.add(Conv2D(1, (8, 8), activation='sigmoid'))
        return model

    def train(self):
        batch_size = NeuralNetModel.batch_size
        epochs = NeuralNetModel.epochs

        datagen = ImageDataGenerator()

        input_shape = (64, 64, 3)

        model = NeuralNetModel.create_model(input_shape=input_shape)
        model.add(Flatten())

        print(model.summary())

        gen_train = datagen.flow_from_directory(
            './data/split/train',
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False)

        nb_train = len(glob.glob('data/split/train/*/*'))
        nb_valid = len(glob.glob('data/split/valid/*/*'))

        print(f'nb_train = {nb_train}, nb_valid = {nb_valid}')

        gen_valid = datagen.flow_from_directory(
            './data/split/valid',
            target_size=(64, 64),
            batch_size=NeuralNetModel.batch_size,
            class_mode='binary',
            shuffle=False)

        checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.5f}.h5', monitor='val_acc', verbose=1,
                                     save_best_only=True)

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit_generator(gen_train, steps_per_epoch=nb_train // batch_size, epochs=epochs,
                                      validation_data=gen_valid, validation_steps=nb_valid // batch_size,
                                      callbacks=[checkpoint])

        model.save('model.h5')

        joblib.dump(history, 'history.pkl')

    def copy(self, data, cls, train=True):
        for idx, fpath in enumerate(data):
            fname = os.path.split(fpath)[-1]
            ext = os.path.splitext(fname)[-1]
            new_path = os.path.join('data/split/{}/{}'.format('train' if train else 'valid', cls), str(idx + 1) + ext)
            shutil.copy(os.path.join(os.getcwd(), fpath), os.path.join(os.getcwd(), new_path))

    def split(self):
        vehicles = glob.glob('data/vehicles/*/*.png')
        non_vehicles = glob.glob('data/non-vehicles/*/*.png')

        vehicles = sklearn_utils.shuffle(vehicles)
        non_vehicles = sklearn_utils.shuffle(non_vehicles)

        vehicle_split = int(len(vehicles) * .9)
        non_vehicle_split = int(len(non_vehicles) * .9)

        train_vehicles, valid_vehicles = vehicles[:vehicle_split], vehicles[vehicle_split:]
        train_non_vehicles, valid_non_vehicles = non_vehicles[:non_vehicle_split], non_vehicles[non_vehicle_split:]

        if not os.path.exists('data/split'):
            os.mkdir('data/split')
            os.mkdir('data/split/train')
            os.mkdir('data/split/valid')

        classes = ['vehicle', 'non-vehicle']
        for c in classes:
            for mode in ['train', 'valid']:
                if not os.path.exists('data/split/{}/{}'.format(mode, c)):
                    os.mkdir('data/split/{}/{}'.format(mode, c))

        self.copy(train_vehicles, 'vehicle')
        self.copy(train_non_vehicles, 'non-vehicle')

        self.copy(valid_vehicles, 'vehicle', train=False)
        self.copy(valid_non_vehicles, 'non-vehicle', train=False)

    def predict(self, img):
        model = self._model

        crop = (400, 660)
        roi = img[400:660, :, :]

        inp = np.expand_dims(roi, axis=0)

        map = model.predict(inp)
        map = map.reshape(map.shape[1], map.shape[2])
        map = map >= 0.99

        roi_h, roi_w = roi.shape[:2]
        prediction_map_h, prediction_map_w = map.shape

        labels = label(map, structure=NeuralNetModel.diag_kernel)
        bboxes = []

        ratio_h, ratio_w = roi_h / prediction_map_h, roi_w / prediction_map_w
        detection_size = 64

        for car_number in range(labels[1]):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number + 1).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y

            xmin = np.min(nonzerox) - 32
            xmax = np.max(nonzerox) + 32

            ymin = np.min(nonzeroy)
            ymax = np.max(nonzeroy) + 64

            spanX = xmax - xmin
            spanY = ymax - ymin

            for x, y in zip(nonzerox, nonzeroy):
                offset_x = (x - xmin) / spanX * detection_size
                offset_y = (y - ymin) / spanY * detection_size

                # Getting boundaries in ROI coordinates scale (multiplying by ratio_w, ratio_h)
                top_left_x = int(round(x * ratio_w - offset_x, 0))
                top_left_y = int(round(y * ratio_h - offset_y, 0))
                bottom_left_x = top_left_x + detection_size
                bottom_left_y = top_left_y + detection_size

                top_left = (top_left_x, crop[0] + top_left_y)
                bottom_right = (bottom_left_x, crop[0] + bottom_left_y)

                bboxes.append((top_left, bottom_right))

        mask = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat_map = add_heat(mask, bboxes)

        heat_map[heat_map <= 10] = 0

        labels = label(heat_map, structure=NeuralNetModel.diag_kernel)

        heat_map_color = color_heat_map(heat_map, cmap=cv2.COLORMAP_JET)

        # global boxes
        box_list = []
        for i in range(labels[1]):
            nz = (labels[0] == i + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])

            tlX = np.min(nzX)
            tlY = np.min(nzY)
            brX = np.max(nzX)
            brY = np.max(nzY)

            box_list.append([tlX, tlY, brX, brY])

        boxes, _ = cv2.groupRectangles(rectList=np.array(box_list).tolist(),
                                       groupThreshold=10, eps=10)
        if len(boxes) == 0:
            boxes = box_list

        map2 = cv2.addWeighted(img, 1, heat_map_color, 0.7, gamma=0)
        draw_boxes(map2, boxes)
        return map2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--split", action="store_true", help="split data")

    args = parser.parse_args()

    nn = NeuralNetModel()
    if args.train:
        nn.train()
    elif args.split:
        nn.split()
