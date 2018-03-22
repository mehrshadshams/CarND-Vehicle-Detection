import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import colorsys
import random

from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

from PIL import Image, ImageDraw, ImageFont


YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

DATA_PATH = os.path.expanduser('~/Downloads/object-detection-crowdai')


def yolo_head(feats, anchors, num_classes):
    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_loss(args, anchors, num_classes, rescore_confidence=False, print_loss=False):
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args

    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    if print_loss:
        total_loss = tf.Print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss


def create_model(class_names, prediction=False):
    n_classes = len(class_names)
    n_anchors = 5

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Conv1
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(image_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D()(x)

    # Conv2
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D()(x)

    # Conv3
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D()(x)

    # Conv4
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D()(x)

    # Conv5
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D()(x)

    # Conv6
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(padding='same', pool_size=2, strides=(1, 1))(x)

    # Conv7
    x = Conv2D(1024, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Conv8
    x = Conv2D(1024, (3, 3), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Conv9
    x = Conv2D(n_anchors * (n_classes + 5), (1, 1), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)(x)

    model_body = Model(image_input, x)

    if not prediction:
        model_loss = Lambda(
                    yolo_loss,
                    output_shape=(1, ),
                    name='yolo_loss',
                    arguments={'anchors': YOLO_ANCHORS,
                               'num_classes': len(class_names)})([
                                   model_body.output, boxes_input,
                                   detectors_mask_input, matching_boxes_input
                               ])

        model = Model(
                [model_body.input, boxes_input, detectors_mask_input,
                 matching_boxes_input], model_loss)
    else:
        model = model_body

    print(model.summary())

    return model_body, model


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box[None,...][0], anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)


def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes


class DataGenerator(object):
    def __init__(self, anchors, dim_x=416, dim_y=416, batch_size=32, shuffle=True):
        'Initialization'
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
                images, boxes, detectors_mask, matching_true_boxes = self.__data_generation(batch)

                yield [images, boxes, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)

    def __read_image(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416))
        return img.astype(np.float32) / 255.

    def __data_generation(self, batch):
        files = batch.Frame.unique()
        image_files = [os.path.join(DATA_PATH, f) for f in files]
        images = [self.__read_image(f) for f in image_files]

        boxes = []
        max_boxz = 0
        for f in files:
            rows = batch[batch.Frame == f]
            frame_boxes = []
            for _, row in rows.iterrows():
                frame_boxes.append(row[['x', 'y', 'w', 'h', 'c']].as_matrix())
            max_boxz = max(max_boxz, len(frame_boxes))
            boxes.append(np.array(frame_boxes))

        for i, box in enumerate(boxes):
            if len(box) < max_boxz:
                zero_padding = np.zeros((max_boxz - len(box), 5))
                boxes[i] = np.vstack((box, zero_padding))

        boxes = np.array(boxes)
        detectors_mask, matching_true_boxes = get_detector_mask(boxes, self.anchors)

        # print((boxes.shape, detectors_mask.shape, matching_true_boxes.shape))

        return np.array(images), boxes, detectors_mask, matching_true_boxes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes


def _eval():
    model_path = 'trained_stage_3_best.h5'
    anchor_path = 'tiny-yolo_anchors.txt'
    # Load anchors
    with open(anchor_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    data = pd.read_csv(os.path.join(DATA_PATH, 'labels2.csv'))
    class_names = data.Label.unique()

    lenc = LabelEncoder()
    data['c'] = lenc.fit_transform(data.Label)

    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    print('Creating model...')

    model_body, yolo_model = create_model(lenc.classes_, prediction=True)
    yolo_model.load_weights(model_path)

    image_file = '/Users/meshams/Documents/Learn/temp/YAD2K/images/test1.jpg'

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = image.astype(np.float32) / 255.
    image = image[np.newaxis,:]

    sess = K.get_session()

    model_image_size = yolo_model.layers[0].input_shape[1:3]

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    input_image_shape = K.placeholder(shape=(2,))

    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=0.3,
        iou_threshold=0.5)

    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image,
                input_image_shape: [image.shape[1], image.shape[0]],
                K.learning_phase(): 0
            })
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.shape[1] + 0.5).astype('int32'))
    thickness = (image.shape[0] + image.shape[1]) // 300

    image_orig = Image.open(image_file)
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image_orig)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_orig.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_orig.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        image_orig.save(os.path.split(image_file)[-1], quality=90)

    sess.close()



def _main():
    # model_path = '/Users/meshams/Documents/Learn/temp/YAD2K/model_data/tiny-yolo.h5'
    anchor_path = 'tiny-yolo_anchors.txt'

    data = pd.read_csv(os.path.join(DATA_PATH, 'labels2.csv'))
    class_names = data.Label.unique()

    lenc = LabelEncoder()
    data['c'] = lenc.fit_transform(data.Label)

    data = data.sample(frac=0.01, random_state=100)

    train_data = data[:int(len(data) * 0.8)]
    valid_data = data[int(len(data) * 0.8):]

    # Load anchors
    with open(anchor_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    print('Creating model...')

    model_body, model = create_model(lenc.classes_)

    model.compile(optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
    })

    batch_size = 32

    train_generator = DataGenerator(anchors, batch_size=batch_size).generate(train_data)
    valid_generator = DataGenerator(anchors, batch_size=batch_size).generate(valid_data)

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("train.{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss',
                                 save_weights_only=False, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit_generator(generator=train_generator, steps_per_epoch=len(train_data) // batch_size,
                        validation_data=valid_generator, validation_steps=len(valid_data) // batch_size,
                        callbacks=[logging], epochs=10)

    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(lenc.classes_)

    model.load_weights('trained_stage_1.h5')

    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred
    })

    model.fit_generator(generator=train_generator, steps_per_epoch=len(train_data) // batch_size,
                        validation_data=valid_generator, validation_steps=len(valid_data) // batch_size,
                        callbacks=[logging, checkpoint, early_stopping], epochs=30)


if __name__ == "__main__":
    _main()
    # _eval()