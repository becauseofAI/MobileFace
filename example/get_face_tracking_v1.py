"""MobileFaceDetection Demo script based on YOLOV3."""
# from __future__ import absolute_import
# from __future__ import division
import os, sys
import argparse
import cv2
import time
import numpy as np
import mxnet as mx
from mxnet import nd
import gluoncv as gcv
from mxnet.gluon.nn import BatchNorm
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/utils/')
from data_presets import data_trans
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/')
from mobilefacedetnet import mobilefacedetnet_v2
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Tracking/')
from mobileface_sort_v1 import Sort


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--model', type=str, 
                        default='../MobileFace_Detection/model/mobilefacedet_v2_gluoncv.params',
                        help='Pretrained model path.')
    parser.add_argument('--video', type=str, default='friends1.mp4',
                        help='Test video path.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Default is cpu , you can specify 1,3 for example with GPUs.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--sort_max_age', type=int, default=10,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--sort_min_hits', type=int, default=3,
                        help='Threshold of object score when visualize the bboxes.')
    parser.add_argument('--output', type=str, 
                        default='./tracking_result/result_friends1_tracking.avi',
                        help='Output video path and name.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    net = mobilefacedetnet_v2(args.model)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    mot_tracker = Sort(args.sort_max_age, args.sort_min_hits) 

    img_short = 256   
    colors = np.random.rand(32, 3) * 255

    winName = 'MobileFace for face detection and tracking'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(args.video)
    output_video = args.output
    # video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # while(cap.isOpened()):
    while cv2.waitKey(1) < 0:
        ret, frame = cap.read()
        if not ret:
            print("Done processing !!!")
            print("Output file is stored as ", output_video)
            cv2.waitKey(3000)
            break
        
        dets = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_nd = nd.array(frame_rgb)
        x, img = data_trans(frame_nd, short=img_short)
        x = x.as_in_context(ctx[0])
        # ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        tic = time.time()
        result = net(x)
        toc = time.time() - tic
        print('Detection inference time:%fms' % (toc*1000))
        ids, scores, bboxes = [xx[0].asnumpy() for xx in result]

        h, w, c = frame.shape
        scale = float(img_short) / float(min(h, w))
        for i, bbox in enumerate(bboxes):
            if scores[i]< args.thresh:
                continue
            xmin, ymin, xmax, ymax = [int(x/scale) for x in bbox]
            # result = [xmin, ymin, xmax, ymax, ids[i], scores[i]]
            result = [xmin, ymin, xmax, ymax, ids[i]]
            dets.append(result)

        dets = np.array(dets)    
        tic = time.time()
        trackers = mot_tracker.update(dets)
        toc = time.time() - tic
        print('Tracking time:%fms' % (toc*1000))

        for d in trackers:
            color = (int(colors[int(d[4]) % 32, 0]), int(colors[int(d[4]) % 32,1]), int(colors[int(d[4]) % 32, 2]))
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color, 3)
            # cv2.putText(frame, str('%s%0.2f' % (net.classes[int(d[4])], d[5])), 
            #            (d[0], d[1] - 5), cv2.FONT_HERSHEY_COMPLEX , 0.8, color, 2)
            cv2.putText(frame, str('%s%d' % ('face', d[4])), 
                       (int(d[0]), int(d[1]) - 5), cv2.FONT_HERSHEY_COMPLEX , 0.8, color, 2)

        video_writer.write(frame.astype(np.uint8))  
        cv2.imshow(winName, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
