import cv2
import random
import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def read_left(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = (img - np.mean(img)) / np.std(img)
    dx = sig.convolve2d(img, kx, 'same')
    dy = sig.convolve2d(img, ky, 'same')
    img = np.expand_dims(img.astype('float32'), -1)
    dx = np.expand_dims(dx.astype('float32'), -1)
    dy = np.expand_dims(dy.astype('float32'), -1)
    return img, dx, dy


def read_right(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = (img - np.mean(img)) / np.std(img)
    return np.expand_dims(img.astype('float32'), -1)
