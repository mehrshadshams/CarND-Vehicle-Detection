import numpy as np
import glob

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, LeakyReLU, Flatten, Dense, Activation, Reshape
from yolo_utils import load_weights


def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(448, 448, 3), border_mode='same', subsample=(1, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Conv2D(64, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Conv2D(128, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Conv2D(256, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Conv2D(512, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Conv2D(1024, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Conv2D(1024, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Conv2D(1024, (3, 3), border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Dense(1470))
    print(model.summary())

    return model


def main():
    model = create_model()

    load_weights(model, './yolo-tiny.weights')

    


if __name__ == '__main__':
    main()
