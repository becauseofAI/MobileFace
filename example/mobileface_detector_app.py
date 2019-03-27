import os, sys
import argparse
import cv2
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/')
from mobileface_detector import MobileFaceDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--model', type=str, 
                        default='../MobileFace_Detection/model/mobilefacedet_v1_gluoncv.params',
                        help="Pretrained model path.")
    parser.add_argument('--images', type=str, default='./friends.jpg',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Default is cpu , you can specify 1,3 for example with GPUs.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

def run_app():
    args = parse_args()
    mfdet = MobileFaceDetection(args.model, args.gpus)
    image_list = [x.strip() for x in args.images.split(',') if x.strip()]
    for img_dir in image_list:
        img_mat = cv2.imread(img_dir)
        results = mfdet.mobileface_detector(img_dir, img_mat)
        if results == None or len(results) < 1:
            continue
        for i, result in enumerate(results):
            xmin, ymin, xmax, ymax, score, classname = result
            cv2.rectangle(img_mat, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
            cv2.putText(img_mat, str('%s%0.2f' % (classname, score)), 
                       (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX , 0.8, (0,0,255), 2)
        cv2.imwrite('friends_result.jpg', img_mat)
        cv2.imshow('result', img_mat)
        cv2.waitKey(2000)

if __name__ == "__main__":
    run_app()
