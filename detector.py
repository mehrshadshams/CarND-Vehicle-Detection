import logging
from utils import *
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from cnn_model import NeuralNetModel


class Detector(object):
    def __init__(self):
        pass

    def detect_cars(self, img):
        pass

    @staticmethod
    def create(mode, args):
        if mode == 'svm':
            return SvmDetector(args)
        elif mode == 'nn':
            return DeepDetector(args)

        raise RuntimeError("Unknown vehicle detector {}".format(mode))


class DeepDetector(Detector):
    def __init__(self, args):
        super().__init__()

        self._nn = NeuralNetModel(args.weight_file)

    def detect_cars(self, img):
        return self._nn.predict(img)


class SvmDetector(Detector):
    # Regions of interest: (window_size, ROI bounds, scale)
    regions = [(32, (400, 464, 500, 1280), 1.0),
               (64, (416, 480, 500, 1280), 1.0),
               (64, (400, 496, 500, 1280), 1.5),
               (64, (432, 528, 500, 1280), 1.5),
               (64, (400, 528, 500, 1280), 2.0),
               (64, (432, 560, 500, 1280), 2.0),
               (64, (400, 596, 500, 1280), 3.0),
               (64, (464, 660, 500, 1280), 3.0),
               (64, (400, 596, 500, 1280), 3.5),
               (64, (464, 660, 500, 1280), 3.5)]

    def __init__(self, args):
        super().__init__()

        clf_name = args.svm_file

        logging.info(f"Loading classifier '{clf_name}'")

        clf = joblib.load(clf_name)
        self._clf = clf.best_estimator_

        logging.info("Loading scaler information")
        self._scaler = joblib.load('scaler.pkl')
        self._verbose = args.verbose
        self._filename_no_ext = os.path.splitext(os.path.split(args.filename)[-1])[0]

    def detect_cars(self, img):
        logging.info(f"Running detection on {self._filename_no_ext}")
        verbose = self._verbose

        time_start = time.time()

        regions = SvmDetector.regions
        windows = []
        for r in regions:
            feature_extraction_start = time.time()

            bounds = r[1]
            scale = r[2]

            windows += self.find_car_single_hog(img, self._clf, self._scaler,
                                                bounds[0], bounds[1], bounds[2], bounds[3], scale)

            if verbose:
                t = (time.time() - feature_extraction_start)
                logging.info(f'Region: {bounds}, Scale: {scale}, Time: {t}s')

        if self._verbose:
            draw_img = draw_boxes(img, windows)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        heat = add_heat(heat, windows)

        if self._verbose:
            save_image(f'{self._filename_no_ext}_windows.jpg', draw_img)
            save_image(f'{self._filename_no_ext}_heat.jpg', heat, cmap='hot')

        heat = apply_threshold(heat, 2)

        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)

        if self._verbose:
            save_image(f'{self._filename_no_ext}_labels.jpg', cmap='gray', img=labels[0])

        draw_img, _ = draw_labeled_bboxes(np.copy(img), labels)

        time_end = time.time()

        if self._verbose:
            logging.info('Total Time: {}s'.format((time_end - time_start)))

        return draw_img

    @staticmethod
    def find_car_single_hog(img, clf, scaler, ystart, yend, xstart, xend, scale=1.5, hog_single_channel=True):
        img_to_search = img[ystart:yend, xstart:xend, :]

        if scale != 1:
            imshape = img_to_search.shape
            img_to_search = cv2.resize(img_to_search, (np.int(imshape[1] / scale), (np.int(imshape[0] / scale))))

        pix_per_cell = 8
        cell_per_block = 2
        orient = 9
        spatial_size = (32, 32)
        hist_bins = 32

        ch1 = img_to_search[:, :, 0]
        ch2 = img_to_search[:, :, 1]
        ch3 = img_to_search[:, :, 2]

        # Define blocks and steps as above
        # nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        # nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nxblocks = (ch1.shape[1] // pix_per_cell) + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) + 1
        nfeat_per_block = orient * cell_per_block ** 2

        all_windows = []

        if hog_single_channel:
            image_gray = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2GRAY)
            hog = get_hog_features(image_gray, orient, pix_per_cell,
                                   cell_per_block,
                                   vis=False, feature_vec=False)
        else:
            hog1 = get_hog_features(ch1, orient, pix_per_cell,
                                    cell_per_block,
                                    vis=False, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

        imshape = img.shape
        h, w, _ = imshape

        window = 64
        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        # nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # cells_per_step = 2  # Instead of overlap, define how many cells to step
        # nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        # nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch

                if hog_single_channel:
                    hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                else:
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_to_search[ytop:min(h, ytop + window), xleft:min(w, xleft + window)],
                                    (window, window))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins, feature_vec=True)

                features = np.hstack([spatial_features, hist_features, hog_features])
                features = scaler.transform(features.reshape(1, -1))
                prediction = clf.predict(features)

                if prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    # win_draw = np.int(window)
                    win_draw = np.int(window * scale)
                    all_windows.append(
                        ((xbox_left + xstart, ytop_draw + ystart),
                         (xbox_left + xstart + win_draw, ytop_draw + win_draw + ystart)))
                    # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    #               (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                    #               (0, 0, 255), 6)

        return all_windows

    """
    This method find features by running sliding window through all regions and extracting feature everytime (SLOW)
    """

    @staticmethod
    def find_car_patch_features(img, clf, scaler):

        all_windows = []

        for ws, r, s in SvmDetector.regions:
            windows = SvmDetector.slide_window(img, x_start_stop=(r[2], r[3]), y_start_stop=(r[0], r[1]),
                                               xy_window=(ws, ws),
                                               xy_overlap=(0.5, 0.5))
            windows = SvmDetector.search_windows(img, windows, clf, scaler)
            [all_windows.append(w) for w in windows]

        return all_windows

    """
    Define a function that takes an image,
    start and stop positions in both x and y,
    window size (x and y dimensions),
    and overlap fraction (for both x and y)
    """

    @staticmethod
    def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        # Compute the span of the region to be searched
        # Compute the number of pixels per step in x/y
        # Compute the number of windows in x/y
        # Initialize a list to append window positions to
        window_list = []
        h, w = img.shape[:2]
        x1, x2 = 0, w
        y1, y2 = 0, h
        if x_start_stop[0] is not None:
            x1 = x_start_stop[0]
        if x_start_stop[1] is not None:
            x2 = x_start_stop[1]
        if y_start_stop[0] is not None:
            y1 = y_start_stop[0]
        if y_start_stop[1] is not None:
            y2 = y_start_stop[1]

        step = int(xy_window[0] * xy_overlap[0])
        for x in range(x1, x2, step):
            for y in range(y1, y2, step):
                window_list.append(((x, y), (x + xy_window[0], y + xy_window[1])))
                # Loop through finding x and y window positions
                #     Note: you could vectorize this step, but in practice
                #     you'll be considering windows one by one with your
                #     classifier, so looping makes sense
                # Calculate each window position
                # Append window position to list
        # Return the list of windows
        return window_list

    @staticmethod
    def search_windows(img, windows, clf, scaler):
        hit_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = single_image_extract_features(test_img, clf, scaler)
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = clf.predict(features_scaled)
            if prediction == 1:
                hit_windows.append(window)
        return hit_windows
