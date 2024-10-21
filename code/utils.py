import platform

import cv2
import numpy as np


def get_video_capture():
    if platform.system() == "Darwin":  # macOS
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = None # idk, havent tested elsewhere
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise Exception('failed to open camera')
    return cap


def rgb2grey(im: np.ndarray):
    if (im is None) or (len(im.shape) == 2):
        return im
    return im[..., :3].dot([0.2989, 0.5870, 0.1140])


def normalize(im):
    """ Make im zero mean and unit variance, then scale all pixels are [0, 1]
        valued.
    """
    im = im.astype('float')
    n = (im - np.mean(im)) / np.std(im) # zero mean, unit variance
    n = n - np.min(n) # zero min
    n = n / np.max(n) # 1 max
    return n
