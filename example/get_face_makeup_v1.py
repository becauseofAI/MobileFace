import os, sys
import argparse
import cv2
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Makeup/')
from mobileface_makeup import MobileFaceMakeup

def parse_args():
    parser = argparse.ArgumentParser(description='Test MobileFace Makeup.')
    parser.add_argument('--image', type=str, default='./makeup_result/girl.png',
                        help='Test images of face makeup.')
    parser.add_argument('--whiten-rate', type=float, default=0.15,
                        help='The face whitening rate.')
    parser.add_argument('--smooth-rate', type=float, default=0.7,
                        help='The face smoothing rate.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 
    makeup_tool = MobileFaceMakeup()
    im_bgr = cv2.imread(args.image)
    face_whitened = makeup_tool.face_whiten(im_bgr, whiten_rate=args.whiten_rate)
    face_whitened_smoothed = makeup_tool.face_smooth(face_whitened, smooth_rate=args.smooth_rate)
    cv2.imshow('face_original', im_bgr)
    cv2.imshow('face_whitened', face_whitened)
    cv2.imshow('face_whitened_smoothed', face_whitened_smoothed)
    cv2.imwrite('./makeup_result/girl_makeup.png', face_whitened_smoothed)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
