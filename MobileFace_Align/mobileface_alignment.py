"""MobileFaceAttribute Demo script based on SE-ResNetV2-18."""
from __future__ import absolute_import
from __future__ import division
import time
import cv2
import numpy as np


class MobileFaceAlign():
    """MobileFace Align Tool.
    Parameters
    ----------
    model: str, default is '../MobileFace_Align/mobileface_align_v1.npy'.
         Align model params path.
    mode: int, default is 3.
        Points for aligning, 3 points mode is affine transformation.   
    """
    def __init__(self, model, mode = 3, **kwargs):
        super(MobileFaceAlign, self).__init__(**kwargs)
        self._mode = mode
        self.scale = np.load(model)

    def get_align(self, image, keypoints, align_size = (96, 96)): 
        """Face attribute predictor.
        Parameters
        ----------
        image: Mat.
            The Mat data format of reading from the original image using opencv.
        keypoints: list.
            The coordinates of keypoints of multiple faces in an image:
            [[(left_eye_x,left_eye_y),(right_eye_x,right_eye_y),(nose_x,nose_y),
            (left_mouth_x,left_mouth_y),(right_mouth_x,right_mouth_y)],[(...),...]].
        align_size: tuple, default is (96, 96).
            The face image size after aligned.
        Returns
        -------
        type: list
            Results of aligned faces:
            [[aligned_1], [aligned_2], [...]].
        """  
        target_size = np.array([(align_size[0], align_size[1])], dtype = 'float32') 
        target_points = target_size * self.scale
        aligned_list = []
        keypoints_ = [[0.0, 0.0] for i in range(self._mode)]
        for i in range(len(keypoints)):
            keypoints_[0] = keypoints[i][0]
            keypoints_[1] = keypoints[i][1]
            keypoints_[2][0] = (keypoints[i][3][0] + keypoints[i][4][0]) * 0.5
            keypoints_[2][1] = (keypoints[i][3][1] + keypoints[i][4][1]) * 0.5
            source_points = np.array(keypoints_, dtype = 'float32')
            trans_matrix = cv2.getAffineTransform(source_points, target_points)
            image_aligned = cv2.warpAffine(image, trans_matrix, align_size, flags = cv2.INTER_CUBIC, 
                                           borderMode = cv2.BORDER_CONSTANT, borderValue = 0.0)
            aligned_list.append(image_aligned)            
        return aligned_list

