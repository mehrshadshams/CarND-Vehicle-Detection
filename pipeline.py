import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from utils import *
import pickle
import traceback
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import os


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
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


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler):
    hit_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_image_extract_features(test_img)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = clf.predict(features_scaled)
        if prediction == 1:
            hit_windows.append(window)
    return hit_windows


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


clf = joblib.load('svm.pkl')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


regions = [(32, (405, 450)), (64, (400, 550)), (96, (400, 600)), (128, (400, 650))]
ystart, yend = 400, 650


def find_car(img, export=True):
    time_start = time.time()

    img_to_search = img[ystart:yend, :, :]

    scale = 1.5

    if scale != 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(img_to_search, (np.int(imshape[1]/scale), (np.int(imshape[0]/scale))))

    ch1 = img_to_search[:, :, 0]
    ch2 = img_to_search[:, :, 1]
    ch3 = img_to_search[:, :, 2]

    pix_per_cell = 8
    cell_per_block = 2
    orient = 9
    spatial_size = (32, 32)
    hist_bins = 32

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    draw_img = np.copy(img)

    all_windows = []

    hog1 = get_hog_features(cv2.cvtColor(img_to_search, cv2.COLOR_RGB2GRAY), orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    # hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    # hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    default_window = 64
    h, w, _ = imshape

    for scale in [0.5, 1, 1.5, 2]:
        window = int(default_window * scale)
        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        nblocks_per_window = (default_window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch

                hog_img = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window]

                hog_feat1 = hog_img.ravel()
                # hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                # hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                # hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                hog_features = hog_feat1

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_to_search[ytop:min(h, ytop + window), xleft:min(w, xleft + window)], (window, window))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins, feature_vec=True)

                features = np.hstack([spatial_features, hist_features, hog_features])
                features = scaler.transform(features.reshape(1, -1))
                prediction = clf.predict(features)

                if prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window)
                    all_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                              (0, 0, 255), 6)

    # # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    # window = 64
    # nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # cells_per_step = 2  # Instead of overlap, define how many cells to step
    # nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    # nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    #
    # all_windows = []
    # for ws, r in regions:
    #     windows = slide_window(img, y_start_stop=r, xy_window=(ws, ws), xy_overlap=(0.3, 0.3))
    #     windows = search_windows(img, windows, clf, scaler)
    #     [all_windows.append(w) for w in windows]
    #
    # out_img = draw_boxes(img, all_windows)
    #
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    heat = add_heat(heat, all_windows)

    if export:
        mpimg.imsave('windows.png', draw_img)
        mpimg.imsave('heat.png', heat)

    heat = apply_threshold(heat, 2)

    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    # plt.imshow(out_img)

    if export:
        mpimg.imsave('final.png', draw_img)

    time_end = time.time()

    if export:
        print('Time: {}s'.format((time_end - time_start)))

    return draw_img

# plt.imshow(draw_img)
# plt.waitforbuttonpress()

def process_frame(frame):
    try:
        return find_car(frame, export=False)
    except Exception as e:
        traceback.print_exc()
        return frame


# video_file = 'test_video.mp4'
# video_file = 'project_video.mp4'
# clip = VideoFileClip(video_file)
# clip = clip.fl_image(process_frame)
# clip.write_videofile(os.path.splitext(video_file)[0] + '_out.mp4', audio=False)

sample_img = load_image('./test_images/test6.jpg')
find_car(sample_img, export=True)