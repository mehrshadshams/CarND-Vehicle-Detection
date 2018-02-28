from moviepy.editor import VideoFileClip
import os
import argparse
import cv2
import logging
from detector import *
from utils import IMAGES_DIR, save_image


class Pipeline(object):
    def __init__(self, args):
        self._filename = args.filename
        self._verbose = args.verbose
        self._filename_no_ext = os.path.splitext(os.path.split(args.filename)[-1])[0]
        self._frame = 0

        if self._verbose and not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)

        self._detector: Detector = Detector.create(args.detector, args)

    def process_image(self, image):
        self._frame += 1

        out_img = self._detector.detect_cars(image)

        if self._verbose:
            save_image(f'{self._filename_no_ext}.jpg', out_img, out_dir=OUT_DIR)

        return out_img

    @staticmethod
    def factory(args):
        if args.mode == 'image':
            return ImagePipeline(args)
        elif args.mode == 'video':
            return VideoPipeline(args)

        raise RuntimeError("Unknown mode {}".format(args.mode))


class ImagePipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        logging.info("Loading '{}'".format(args.filename))
        image = cv2.imread(args.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._image = image

    def run(self):
        self.process_image(self._image)


class VideoPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self._clip = VideoFileClip(self._filename)

    def process_image(self, image):
        return super().process_image(image)

    def run(self):
        output = os.path.join(OUT_DIR, self._filename)
        out_clip = self._clip.fl_image(self.process_image)
        out_clip.write_videofile(output, audio=False)


def main(args):
    if not os.path.exists(args.filename):
        print('File {} not found.'.format(args.filename))
        return

    pipeline = Pipeline.factory(args)
    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the file for processing")
    parser.add_argument("--mode", choices=['image', 'video'], help="Mode to operate in (image or video)", default='image')
    parser.add_argument("--detector", choices=['svm', 'nn'], help="Detector mode", default='svm')
    parser.add_argument("--svm_file", help="Path to the svm file", default='svm.pkl')
    parser.add_argument("--weight_file", help="Path to the model file", default='model.h5')
    parser.add_argument("--verbose", action="store_true", help="Enable verbosity")

    args = parser.parse_args()

    if args.mode != 'video':
        logging.basicConfig(level=logging.INFO)

    main(args)
