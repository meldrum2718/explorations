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


def get_appropriate_dims_for_ax_grid(n) -> tuple:
    """
    Minimize np.abs(nrows - ncols),
    subject to the constraints that:
        nrows, ncols are both ints
        nrows * ncols >= n

    return (nrows, ncols)
    """
    best_dims = None
    best_delta = np.inf
    for ncols in range(1, n):
    #  for ncols in range(np.ceil(n**.5), 0, -1): # need to test it before leaving this line uncommented
        nrows = np.ceil(n / ncols).astype(int)
        delta = np.abs(nrows - ncols)
        if delta < best_delta:
            best_delta = delta
            best_dims = (nrows, ncols)
    return best_dims


def normalize(im):
    """ scale im pixels to [0, 1] """
    im = im.astype('float')
    if np.min(im) == np.max(im):
        return (im * 0) + 0.5 # set all pixels to 0.5 if im is a constant
    im = im - np.min(im) # zero min
    im = im / np.max(im) # unit max
    return im


def inspect(label, im):
    """ Print some basic image stats."""
    print()
    print(label + ':')
    print('shape:', im.shape)
    print('dtype:', im.dtype)
    print('max:', np.max(im))
    print('min:', np.min(im))
    print('mean:', np.mean(im))
    print('std:', np.std(im))
    print()
