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
    """ scale im pixels to [0, 1] """
    im = im.astype('float')
    if np.min(im) == np.max(im):
        return (im * 0) + 0.5 # set all pixels to 0.5 if im is a constant
    im = im - np.min(im) # zero min
    im = im / np.max(im) # unit max
    return im
