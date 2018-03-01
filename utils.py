import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
import time
import os


OUT_DIR = "output_images"
IMAGES_DIR = "images"


def load_image(path):
    img = mpimg.imread(path)
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    return img


def color_hist(img, nbins=32, bins_range=(0, 256), feature_vec=False):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    hist_features = np.concatenate([rhist[0], ghist[0], bhist[0]])

    if feature_vec:
        return hist_features

    return rhist, ghist, bhist, bin_centers, hist_features


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    if feature_image is None:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    out = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
              visualise=vis, feature_vector=feature_vec)
    return out


# def single_image_extract_features(img, cspace='RGB', spatial_size=(32, 32),
#                                   hist_bins=32, hist_range=(0, 256), orient=9,
#                                   pix_per_cel=8, cell_per_block=2):
#     if cspace != 'RGB':
#         img = cv2.cvtColor(img, getattr(cv2, 'COLOR_RGB2' + cspace))
#     spatial_features = bin_spatial(img, spatial_size)
#     hist_features = color_hist(img, hist_bins, hist_range, feature_vec=True)
#     hog_features = get_hog_features(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), orient, pix_per_cel,
#                                     cell_per_block)
#     return np.concatenate([spatial_features, hist_features, hog_features.ravel()]).astype(np.float64)

def single_image_extract_features(img, cspace='RGB', spatial_size=(32, 32),
                                  hist_bins=32, hist_range=(0, 256), orient=9,
                                  pix_per_cel=8, cell_per_block=2, single_hog=True):
    if cspace != 'RGB':
        img = cv2.cvtColor(img, getattr(cv2, 'COLOR_RGB2' + cspace))
    spatial_features = bin_spatial(img, spatial_size)
    hist_features = color_hist(img, hist_bins, hist_range, feature_vec=True)
    if single_hog:
        hog_features = get_hog_features(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), orient, pix_per_cel,
                                    cell_per_block)
    else:
        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]
        hog_features = []
        for ch in [ch1, ch2, ch3]:
            hf = get_hog_features(ch, orient, pix_per_cel, cell_per_block).ravel()
            hog_features.append(hf)
        hog_features = np.concatenate(hog_features)
    return np.concatenate([spatial_features, hist_features, hog_features.ravel()]).astype(np.float64)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cel=8, cell_per_block=2, single_hog=True):
    # Create a list to append feature vectors to
    features = []
    for img_name in imgs:
        img = load_image(img_name)
        features.append(single_image_extract_features(img, cspace, spatial_size,
                                                      hist_bins, hist_range, orient,
                                                      pix_per_cel, cell_per_block, single_hog))
    return np.array(features).astype(np.float64)


def draw_color_histogram(rh, gh, bh, bincen):
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()


def plot3d(pixels, colors_rgb,
           axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


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


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap = np.clip(heatmap, 0, 255)
    return heatmap


def color_heat_map(heat_map, cmap=cv2.COLORMAP_HOT):
    """
    Makes an RGB version of the 1-channel heatMap
    :param heatMapMono:
    :param cmap: The color map of choice
    :return: RGB heatMap
    """
    heat_map = cv2.equalizeHist(heat_map.astype(np.uint8))
    heat_map_color = cv2.applyColorMap(heat_map, cmap)
    heat_map_color = cv2.cvtColor(heat_map_color, code=cv2.COLOR_BGR2RGB)

    return heat_map_color


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    boxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img, boxes


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=4):
    for bbox in boxes:

        bbox = np.array(bbox)
        bbox = bbox.reshape(bbox.size)

        cv2.rectangle(img=img, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]),
                      color=color, thickness=thickness)

    return img


def save_image(fname, img, cmap=None, out_dir=IMAGES_DIR, **kwargs):
    mpimg.imsave(os.path.join(out_dir, fname), img, cmap=cmap, **kwargs)