import os
import cv2
import math
import numpy as np

class MobileFaceEnhance():
    """MobileFace enhance for dark or bright face.
    """ 
    def __init__(self, **kwargs):
        super(MobileFaceEnhance, self).__init__(**kwargs)

    def hist_statistic(self, img, dark_th=80, bright_th=200, dark_shift=0.4, bright_shift=2.5):
        """Face gamma correction.
        Parameters
        ----------
        img: mat 
            The Mat data format of reading from the original image using opencv.
        dark_th: int, default is 80.
            Black pixel threshold whith typical values from 50 to 100.
        bright_th: int, default is 200.
            White pixel threshold whith typical values from 180 to 220.
        dark_shift: float, default is 0.4.
            Gamma shift value for gamma correction to brighten the face. 
            The typical values are from 0.3 to 0.5.
        bright_shift: float, default is 2.5.
            Gamma shift value for gamma correction to darken the face. 
            The typical values are from 2.0 to 3.0.            
        Returns
        -------
        gamma: float
            The gamma value for gamma correction.
        hist: list
            The gray histogram for face.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # method 1
        # hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        
        # method 2
        hist = cv2.calcHist(img_gray, [0], None, [256], [0, 256])

        dark_rate = np.sum(hist[:dark_th]) / np.sum(hist)
        normal_rate = np.sum(hist[dark_th:bright_th]) / np.sum(hist)
        bright_rate = np.sum(hist[bright_th:]) / np.sum(hist)
        rate = [dark_rate, normal_rate, bright_rate]
        if np.max(rate) == dark_rate:
            gamma = np.minimum(np.maximum(1.0 - math.pow(dark_rate, 2) + dark_shift, dark_shift), 1.0)
        elif np.max(rate) == bright_rate:
            gamma = math.pow(bright_rate, 3) + bright_shift
        else:
            gamma = 1.0
        return gamma, hist

    def gamma_trans(self, img, gamma):
        """Face gamma correction.
        Parameters
        ----------
        img: mat 
            The Mat data format of reading from the original image using opencv. 
        gamma: float
            The gamma value for gamma correction.          
        Returns
        -------
        type: mat
            Face BGR image after gamma correction.
        """
        gamma_table = [np.power(x / 255.0, gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)