
## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/hog_per_channel.png
[test_1_windows]: ./images/test1_windows.jpg
[test_1_heat]: ./images/test1_heat.jpg
[test_1_output]: ./output_images/test1.jpg
[test_2_windows]: ./images/test2_windows.jpg
[test_2_heat]: ./images/test2_heat.jpg
[test_2_output]: ./output_images/test2.jpg
[test_3_windows]: ./images/test3_windows.jpg
[test_3_heat]: ./images/test3_heat.jpg
[test_3_output]: ./output_images/test3.jpg
[test_4_windows]: ./images/test4_windows.jpg
[test_4_heat]: ./images/test4_heat.jpg
[test_4_output]: ./output_images/test4.jpg
[test_5_windows]: ./images/test5_windows.jpg
[test_5_heat]: ./images/test5_heat.jpg
[test_5_output]: ./output_images/test5.jpg
[test_6_windows]: ./images/test6_windows.jpg
[test_6_heat]: ./images/test6_heat.jpg
[test_6_output]: ./output_images/test6.jpg

[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally decided to with the parameters shown above.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `GridSearchCV` to find the best parameters. The code for the training is in file `train.py`. I ran this code with different color schemes and HOG features on gray-scale image or all channel. Then I chose the classifier that obtained the best test score. The result of the training can be found in `train.log` file. Here's an excerpt:

```
Trainig on RGB
Loading features ...
Features loaded in 36.67877268791199
Scaling features ...
Training ...
Training finished in 1667.149317741394ms
{'C': 10, 'kernel': 'rbf'}
Score on test 0.9952226582494454, Time=1772.7060058116913ms

==========================================

Trainig on HSV
Loading features ...
Features loaded in 37.050665855407715
Scaling features ...
Training ...
Training finished in 1914.3402307033539ms
{'C': 10, 'kernel': 'rbf'}
Score on test 0.9921515099812319, Time=2030.0881683826447ms

==========================================

Trainig on LUV
Loading features ...
Features loaded in 38.144434452056885
Scaling features ...
Training ...
Training finished in 1644.4318776130676ms
{'C': 10, 'kernel': 'rbf'}
Score on test 0.994369561508275, Time=1753.6057770252228ms
```

Then I stored the model in `svm.pkl` to be used later in detection. I also applied standardization step using `sklearn.StandardScaler` on the features and stored the scaler in a pickle file. I use the same scaler during prediction. (`train.py` line 39-40 and 44-45)

### Sliding Window Search

I decided to use different regions of interest with different window scale. The regions are (from `detector.py` line 37)

```
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
```

The first value in each triple is the window size, the second value is the bounds of the region of interest and the third value is the scale to which the HOG features should be scaled to.

|                             |                          |
| --------------------------- | ------------------------ |
| ![alt text][test_1_windows] | ![alt text][test_1_heat] |
| ![alt text][test_2_windows] | ![alt text][test_2_heat] |
| ![alt text][test_3_windows] | ![alt text][test_3_heat] |
| ![alt text][test_4_windows] | ![alt text][test_4_heat] |
| ![alt text][test_5_windows] | ![alt text][test_5_heat] |
| ![alt text][test_6_windows] | ![alt text][test_6_heat] |

### Video Implementation

Here's a [link to my video result](./output_images/project_video.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The images above show the heat map and the following image shows the result of `scipy.ndimage.measurements.label()`

### Here is the output of `scipy.ndimage.measurements.label()` on an image:
![alt text][image6]

### Here the resulting bounding boxes on test images:

|                             |                          |
| --------------------------- | ------------------------ |
| ![alt text][test_1_output] | ![alt text][test_2_output]|
| ![alt text][test_3_output] | ![alt text][test_4_output]|
| ![alt text][test_5_output] | ![alt text][test_6_output]|

---

# Deep learning approach

Thanks to great article by Ivan Kazakov (https://towardsdatascience.com/vehicle-detection-and-tracking-44b851d70508) I decided to try training a CNN model and do the prediction based on that.

So I trained the following model implemented in Keras. Basically this model starts with the images and runs 3 convolution layers to learn different features and finally we apply a MaxPooling layer to get patches of size 8x8. Then we feed these patches in a final 1x1 convolution layer which acts as our prediction layer (using a sigmoid activation)

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 16)        448
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 64, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 32)        4640
_________________________________________________________________
dropout_2 (Dropout)          (None, 64, 64, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 64)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 1)           4097
=================================================================
Total params: 27,681
Trainable params: 27,681
Non-trainable params: 0
________________________________________________________________
```

Then at prediction we set the input_shape to a patch of size (260, 1280, 3) The model will generate feature map of size (25x153) which we need to scale up. Once we get this prediction map we can filter out low confidence predictions and only keep predictions that are above a certain threshold. Then as before we apply heat map and labeling technique to filter out false positives and find different cars.

The result for this approach is [here](./output_images/project_video_cnn.mp4)

The results needs more work but unlike the traditional approach of the previous section, deep learning approach can run at 1.5 FPS

### Discussion

One of the best techniques for object detection is YOLO which I would like to train on Udacity images and apply to this exercise.

