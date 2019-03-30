"""MobileFacePose: A face predictor for fast face pose prediction."""
import numpy as np
import pickle

class MobileFacePose():
    """MobileFacePose V1 for fast face pose prediction.
    Parameters
    ----------
    model: str, default is '../MobileFace_Pose/mobileface_pose_emnme_v1.dat'.
        Pretrained model path.
    landmark_num : int, default is 5
        Landmarks numbers.  
    """
    def __init__(self, model, landmark_num = 5, **kwargs):
        super(MobileFacePose, self).__init__(**kwargs)
        self._landmark_num = landmark_num 

        model_file = open(model,'rb')
        self._model = pickle.load(model_file)
    
    def get_pose(self, points): 
        """Face pose predictor.
        Parameters
        ----------
        points: list
            The landmarks coordinate:
            [[left_eye_x,left_eye_y],[right_eye_x,right_eye_y],[nose_x,nose_y],
            [left_mouth_x,left_mouth_y],[right_mouth_x,right_mouth_y]].
        Returns
        -------
        type: matrix
            Results of Face Pose:
            [float(pitch), float(roll), float(yaw)].
        """   
        x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        miny = 0.0
        maxy = 0.0
        sumx = 0.0
        sumy = 0.0
        _, miny = np.min(points, axis = 0)
        _, maxy = np.max(points, axis = 0)
        sumx, sumy = np.sum(points, axis = 0)
        dist = maxy - miny
        if dist <= 0:
            print('Value Err: The value of dist needs to be greater than 0.')
            return            
        sumx = sumx / self._landmark_num
        sumy = sumy / self._landmark_num
        for i in range(self._landmark_num):
            x[i*2] = int(((points[i][0] - sumx) / dist) * 100)
            x[i*2+1] = int(((points[i][1] - sumy) / dist) * 100)

        x = np.matrix(x)
        result = x * (np.matrix(self._model).T) * 100
        return result