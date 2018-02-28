import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn import utils as sklearn_utils
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import joblib
import cv2
import glob
import shutil
import os
from moviepy.editor import VideoFileClip
from matplotlib import image as mpimg

epochs = 50
batch_size = 32
top_model_weights_path = 'bottleneck_fc_model.h5'

diagKernel = [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]

def generator(X, y, batch_size=32, shuffle=True):
    """
    This is the generator that loads, extends and returns the images and their corresponding steering wheel
    angle for a given batch
    """
    num_samples = len(X)
    while 1:
        if shuffle:
            X, y = sklearn_utils.shuffle(X, y)

        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_X, batch_y = X[offset:end], y[offset:end]

            images = []
            for row in range(len(batch_X)):
                image = X[row]
                # image = cv2.resize(image, (224, 224))

                images.append(image)

            images = np.array(images)

            print(images.shape)
            print(batch_y.shape)

            yield sklearn_utils.shuffle(images, batch_y)


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

def main():
    datagen = ImageDataGenerator(rescale=1./255)

    # model = VGG16(weights='imagenet', include_top=False)

    input_shape = (64, 64, 3)

    model = create_model()
    model.add(Flatten())

    # model = Sequential()
    # model.add(Lambda(lambda x: x/255., input_shape=input_shape))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(8, 8)))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv2D(1, (8, 8), activation='sigmoid'))
    # model.add(Flatten())

    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy')

    print(model.summary())

    # train = joblib.load('train.pkl')
    # test = joblib.load('test.pkl')

    # X, y = train['X'], train['y']
    # X_test, y_test = test['X'], test['y']
    #
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state = 42)

    gen_train = datagen.flow_from_directory(
        './data/split/train',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    train_classes = gen_train.classes

    # nb_train = len(X_train)
    # nb_valid = len(X_valid)

    nb_train = len(glob.glob('data/split/train/*/*'))
    nb_valid = len(glob.glob('data/split/valid/*/*'))

    print(f'nb_train = {nb_train}, nb_valid = {nb_valid}')

    # gen = generator(X_train, y_train, batch_size=batch_size, shuffle=False)
    # bottleneck_features_train = model.predict_generator(gen, nb_train // batch_size, verbose=True)

    # np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    np.save(open('train_classes.npy','wb'), train_classes)

    print('Saved train features and classes')

    # gen = generator(X_valid, y_valid, batch_size=batch_size, shuffle=False)
    gen_valid = datagen.flow_from_directory(
        './data/split/valid',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    valid_classes = gen_valid.classes

    # bottleneck_features_valid = model.predict_generator(gen, nb_valid // batch_size)

    np.save(open('valid_classes.npy', 'wb'), valid_classes)
    # np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_valid)

    print('Saved valid features and classes')

    checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.5f}.h5', monitor='val_acc', verbose=1,
                                 save_best_only=True)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(gen_train, steps_per_epoch=nb_train//batch_size, epochs=epochs,
                        validation_data=gen_valid, validation_steps=nb_valid//batch_size,
                        callbacks=[checkpoint])

    model.save('model.h5')

    joblib.dump(history, 'history.pkl')


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.load(open('train_classes.npy', 'rb'))

    validation_data = np.load(open('bottleneck_features_test.npy', 'rb'))
    validation_labels = np.load(open('valid_classes.npy', 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    model.save("model.h5")


def addHeat(mask, bBoxes):
    """
    Creates the actual heat map. Overlaps build-up the 'heat'
    :param mask: the image where the 'heat' being projected
    :param bBoxes: bounding boxes formatted as list of tuples of tuples: [((x, y), (x, y)),...]
    :return: 1-channel Heat map image
    """
    for box in bBoxes:
        # box as ((x, y), (x, y))
        topY = box[0][1]
        bottomY = box[1][1]
        leftX = box[0][0]
        rightX = box[1][0]

        mask[topY:bottomY, leftX:rightX] += 1

        mask = np.clip(mask, 0, 255)

    return mask

vehicleBoxesHistory = []

def predict(model, img):
    crop = (400, 660)
    roi = img[400:660, :, :]

    # print(model.summary())

    inp = np.expand_dims(roi, axis=0)

    map = model.predict(x=inp)

    # print(map.shape)

    map = map.reshape(map.shape[1], map.shape[2])
    map = map >= 0.99

    roiH, roiW = roi.shape[:2]
    predictionMapH, predictionMapW = map.shape

    # map = np.expand_dims(map, axis=2).astype(np.uint8) * 255
    #
    # map2 = cv2.resize(map, (1280-660, 720-400))
    # map2 = np.dstack([map2, np.zeros_like(map2), np.zeros_like(map2)])
    #
    # print(map2.shape)
    #
    # map2 = cv2.addWeighted(roi, 1, map2, 0.7, gamma=0)
    #
    # plt.imshow(map2)
    # plt.waitforbuttonpress()

    labels = label(map, structure=diagKernel)
    hotPoints = []

    ratioH, ratioW = roiH / predictionMapH, roiW / predictionMapW
    detectionPointSize = 64

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
            # Adjustment offsets for a box starting point.
            # Ranges from 0 for the left(upper)-most to detectionPointSize for right(bottom)-most
            offsetX = (x - xmin) / spanX * detectionPointSize
            offsetY = (y - ymin) / spanY * detectionPointSize

            # Getting boundaries in ROI coordinates scale (multiplying by ratioW, ratioH)
            topLeftX = int(round(x * ratioW - offsetX, 0))
            topLeftY = int(round(y * ratioH - offsetY, 0))
            bottomLeftX = topLeftX + detectionPointSize
            bottomLeftY = topLeftY + detectionPointSize

            topLeft = (topLeftX, crop[0] + topLeftY)
            bottomRight = (bottomLeftX, crop[0] + bottomLeftY)

            hotPoints.append((topLeft, bottomRight))

        # bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # # Draw the box on the image
        # cv2.rectangle(roi, bbox[0], bbox[1], (0, 0, 255), 6)

    sampleMask = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatMap = addHeat(sampleMask, hotPoints)

    heatMap[heatMap <= 10] = 0

    currentLabels = label(heatMap, structure=diagKernel)

    heatColor = colorHeatMap(heatMapMono=heatMap, cmap=cv2.COLORMAP_JET)

    # self.updateHistory(currentLabels=currentLabels)

    # global vehicleBoxesHistory
    vehicleBoxesHistory = []
    for i in range(currentLabels[1]):
        nz = (currentLabels[0] == i + 1).nonzero()
        nzY = np.array(nz[0])
        nzX = np.array(nz[1])

        tlX = np.min(nzX)
        tlY = np.min(nzY)
        brX = np.max(nzX)
        brY = np.max(nzY)

        vehicleBoxesHistory.append([tlX, tlY, brX, brY])

        # vehicleBoxesHistory = vehicleBoxesHistory[-50:]

    boxes, _ = cv2.groupRectangles(rectList=np.array(vehicleBoxesHistory).tolist(),
                                   groupThreshold=10, eps=10)
    if len(boxes) == 0:
        boxes = vehicleBoxesHistory
    # print(boxes)

    map2 = cv2.addWeighted(img, 1, heatColor, 0.7, gamma=0)
    drawBoxes(map2, boxes)
    # plt.imshow(map2)
    # plt.waitforbuttonpress()
    return map2


def drawBoxes(img, bBoxes, color=(0, 255, 0), thickness=4):
    """
    Universal bounding box painter, regardless of bBoxes format
    :param img: image of interest
    :param bBoxes: list of bounding boxes.
    :param color:
    :param thickness:
    :return:
    """
    for bBox in bBoxes:

        bBox = np.array(bBox)
        bBox = bBox.reshape(bBox.size)

        cv2.rectangle(img=img, pt1=(bBox[0], bBox[1]), pt2=(bBox[2], bBox[3]),
                      color=color, thickness=thickness)

def colorHeatMap(heatMapMono, cmap=cv2.COLORMAP_HOT):
    """
    Makes an RGB version of the 1-channel heatMap
    :param heatMapMono:
    :param cmap: The color map of choice
    :return: RGB heatMap
    """
    heatMapInt = cv2.equalizeHist(heatMapMono.astype(np.uint8))
    heatColor = cv2.applyColorMap(heatMapInt, cmap)
    heatColor = cv2.cvtColor(heatColor, code=cv2.COLOR_BGR2RGB)

    return heatColor

def copy(data, cls, train=True):
    for idx, fpath in enumerate(data):
        fname = os.path.split(fpath)[-1]
        ext = os.path.splitext(fname)[-1]
        new_path = os.path.join('data/split/{}/{}'.format('train' if train else 'valid', cls), str(idx + 1) + ext)
        shutil.copy(os.path.join(os.getcwd(), fpath), os.path.join(os.getcwd(), new_path))


def split():
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

    copy(train_vehicles, 'vehicle')
    copy(train_non_vehicles, 'non-vehicle')

    copy(valid_vehicles, 'vehicle', train=False)
    copy(valid_non_vehicles, 'non-vehicle', train=False)


if __name__ == '__main__':
    # train_top_model()
    input_shape = (660-400, 1280, 3)

    model = create_model(input_shape=input_shape)
    model.load_weights('model2.h5')
    model.load_weights('ppico.h5')
    # model.load_weights('weights.08-0.98762.h5')


    # video_file = 'test_video.mp4'
    # video_file = 'project_video.mp4'
    video_file = 'clip2.mp4'
    clip = VideoFileClip(video_file)
    clip = clip.fl_image(lambda x: predict(model, x))
    clip.write_videofile(os.path.splitext(video_file)[0] + '_out.mp4', audio=False)

    # for f in glob.glob("test_images/*.jpg"):
    #     print(f)
    #     img = cv2.imread(f)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     out = predict(model, img)
    #     cv2.imwrite(os.path.splitext(os.path.split(f)[-1])[0] + "_out.jpg", out)

    # img = mpimg.imread('frames/frame1044.jpg')
    # print(img.shape)
    # print(img.max())

    # img = cv2.imread('frames/frame1044.jpg')
    img = cv2.imread('temp.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # model = create_model()
    # model.load_weights('model2.h5')
    # x =np.expand_dims(img[550:550+64,700:700+64,:], axis=0)
    # print(x.shape)
    # print(model.predict(x))

    out = predict(model, img)
    cv2.imwrite("frame_1044_out.jpg", out)