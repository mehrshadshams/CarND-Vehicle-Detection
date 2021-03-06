import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import glob
import pickle
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib

from utils import *
import logging


def main():
    COLOR_SPACES = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

    vehicles = list(filter(lambda x: not x.startswith('.'), glob.glob('data/vehicles/**/*')))
    non_vehicles = list(filter(lambda x: not x.startswith('.'), glob.glob('data/non-vehicles/**/*')))

    for cspace in COLOR_SPACES:
        st = time.time()

        logger.info(f'Trainig on {cspace}')

        logger.info('Loading features ...')
        X = extract_features(vehicles + non_vehicles, cspace=cspace, single_hog=True)
        logger.info('Features loaded in {}'.format(time.time() - st))

        y = np.concatenate([np.ones(len(vehicles)), np.zeros(len(non_vehicles))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        logger.info('Scaling features ...')

        scaler = StandardScaler()
        scaler.fit(X_train)

        joblib.dump(scaler, f'scaler{cspace}.pkl')

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        joblib.dump({'X': X_train_scaled, 'y': y_train}, f'train{cspace}.pkl')
        joblib.dump({'X': X_test_scaled, 'y': y_test}, f'test{cspace}.pkl')

        train_start = time.time()
        logger.info('Training ...')

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svr = SVC()
        clf = GridSearchCV(svr, parameters)
        clf.fit(X_train_scaled, y_train)

        logger.info('Training finished in {}ms'.format(time.time() - train_start))
        logger.info(clf.best_params_)

        score = clf.score(X_test_scaled, y_test)

        logger.info('Score on test {}, Time={}ms'.format(score, time.time() - st))

        joblib.dump(clf, f'svm{cspace}.pkl')

        logger.info('\n==========================================\n')


if __name__ == '__main__':
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('train.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(ch)

    main()
