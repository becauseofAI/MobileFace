import cv2
import numpy as np

class MobileFaceMakeup():
    """MobileFace makeup.
    """ 
    def __init__(self, **kwargs):
        super(MobileFaceMakeup, self).__init__(**kwargs)

    def face_whiten(self, im_bgr, whiten_rate=0.15):
        """Face whitening.
        Parameters
        ----------
        im_bgr: mat 
            The Mat data format of reading from the original image using opencv.
        whiten_rate: float, default is 0.15
            The face whitening rate.
        Returns
        -------
        type: mat
            The result of face whitening.
        """  
        im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        im_hsv[:,:,-1] = np.minimum(im_hsv[:,:,-1] * (1 + whiten_rate), 255).astype('uint8')
        im_whiten = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im_whiten

    def face_smooth(self, im_bgr, smooth_rate=0.7, bi_ksize=15, sigma=100, ga_ksize=3):
        """Face smoothing.
        Parameters
        ----------
        im_bgr: mat 
            The Mat data format of reading from the original image using opencv.
        smooth_rate: float, default is 0.7.
            The face smoothing rate.
        bi_ksize: int, default is 15.
            The kernel size of bilateral filter.
        sigma: int, default is 100.
            The value of sigmaColor and sigmaSpace for bilateral filter.
        ga_ksize: int, default is 3.
            The kernel size of gaussian blur filter.
        Returns
        -------
        type: mat
            The result of face smoothing.
        """
        im_bi = cv2.bilateralFilter(im_bgr, bi_ksize, sigma, sigma)
        im_ga = cv2.GaussianBlur(im_bi, (ga_ksize, ga_ksize), 0, 0)
        im_smooth = np.minimum(smooth_rate * im_ga + (1 - smooth_rate) * im_bgr, 255).astype('uint8')
        return im_smooth


