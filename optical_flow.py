import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip


class OpticalFlow(object):
    def __init__(self, p0 = None):
        self._old_frame = None
        self._old_gray = None
        self._p0 = p0
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))

    def process_frame(self, frame):
        if self._old_frame is None:
            self._old_frame = frame
            self._old_gray = cv2.cvtColor(self._old_frame, cv2.COLOR_BGR2GRAY)
            self._p0 = cv2.goodFeaturesToTrack(self._old_gray, mask=None, **self.feature_params)
            print(self._p0.shape)
            return frame
        else:
            mask = np.zeros_like(self._old_frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self._old_gray, frame_gray, self._p0, None, **self.lk_params)
            if p1 is not None:
                # Select good points
                good_new = p1[st == 1]
                good_old = self._p0[st == 1]
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
                img = cv2.add(frame, mask)
                # Now update the previous frame and previous points
                self._old_gray = frame_gray.copy()
                self._p0 = good_new.reshape(-1, 1, 2)
                return img
            else:
                return frame

    def process(self):
        # video_file = 'test_video.mp4'
        video_file = 'project_video.mp4'
        clip = VideoFileClip(video_file)
        clip = clip.fl_image(self.process_frame)
        clip.write_videofile(os.path.splitext(video_file)[0] + '_out.mp4', audio=False)


OpticalFlow().process()

