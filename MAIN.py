import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



IMGL_PATH = './images/a.jpg'
IMGR_PATH = './images/b.jpg'

class main():
    def __init__(self):
        self.img_l = cv.imread(IMGL_PATH)
        self.img_r = cv.imread(IMGR_PATH)


