"""MobileFaceDetection Demo script based on YOLOV3."""
# from __future__ import absolute_import
# from __future__ import division
import os, sys
import argparse
import cv2
import time
import mxnet as mx
import gluoncv as gcv
from mxnet.gluon.nn import BatchNorm
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Detection/')
from mobilefacedetnet import mobilefacedetnet_v2


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--model', type=str, 
                        default='../MobileFace_Detection/model/mobilefacedet_v2_gluoncv.params',
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

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    net = mobilefacedetnet_v2(args.model)

    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    img_short = 256     
    for image in image_list:
        ax = None
        x, img = presets.yolo.load_test(image, short=img_short)
        x = x.as_in_context(ctx[0])
        # ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        tic = time.time()
        result = net(x)
        toc = time.time() - tic
        print('Inference time:%fms' % (toc*1000))
        ids, scores, bboxes = [xx[0].asnumpy() for xx in result]

        # # pyplot show
        # ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
        #                             class_names=net.classes, colors={0: (0, 1, 0)}, ax=ax)
        # # plt.savefig("friends_result.jpg")
        # plt.show()

        # opencv show
        im = cv2.imread(image)
        h, w, c = im.shape
        scale = float(img_short) / float(min(h, w))
        for i, bbox in enumerate(bboxes):
            if scores[i]< args.thresh:
                continue
            xmin, ymin, xmax, ymax = [int(x/scale) for x in bbox]
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
            cv2.putText(im, str('%s%0.2f' % (net.classes[int(ids[i])], scores[i])), 
                       (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX , 0.8, (0,0,255), 2)
        cv2.imwrite('result_detect_v2.jpg', im)
        cv2.imshow('result_detect', im)
        cv2.waitKey(2000)

    # Uncomment for loop test method 1.
    '''
    loop = 100
    count = loop
    loop_toc = 0
    while count > 0:
        image_num = 0
        for image in image_list:
            image_num += 1
            x, img = presets.yolo.load_test(image, short=img_short)
            x = x.as_in_context(ctx[0])        
            tic = time.time()
            result = net(x)
            toc = time.time() - tic            
            print('Inference time:%fms' % (toc*1000))
            loop_toc += toc
        count -= 1
    print('Average inference time:%fms' % (loop_toc*1000/(loop*image_num)))
    '''

    # Uncomment for loop test method 2.
    '''
    for image in image_list:
        loop = 100
        count = loop
        loop_toc = 0
        x, img = presets.yolo.load_test(image, short=img_short)
        x = x.as_in_context(ctx[0])
        while count > 0:
            tic = time.time()
            result = net(x)
            toc = time.time() - tic            
            print('Inference time:%fms' % (toc*1000))
            loop_toc += toc
            count -= 1
        print('Average inference time:%fms' % (loop_toc*1000/loop))
    '''







