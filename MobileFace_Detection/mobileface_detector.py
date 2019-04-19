"""MobileFaceDetection Demo script based on YOLOV3."""
from __future__ import absolute_import
from __future__ import division
import time
import mxnet as mx
from gluoncv.data.transforms import presets
from mobilefacedetnet import mobilefacedetnet_v1


class MobileFaceDetection():
    """MobileFaceDetNet V1 detection network.
    Parameters
    ----------
    model: str, default is '../MobileFace_Detection/model/mobilefacedet_v1_gluoncv.params'.
        Pretrained model path.
    gpus: str, default=''.
        Training with GPUs, you can specify 1,3 for example.
    image_short: int, default is 256.
        Resize the image short edge to 256, 320 and so on. 
    confidence_thresh: float, default is 0.5.
        Threshold of object score when visualize the bboxes.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    pretrained: str, default is 'True'.
        Load weights from previously saved parameters.    
    """
    def __init__(self, model, gpus, image_short=256, 
                 confidence_thresh=0.5, nms_thresh=0.45, nms_topk=200, 
                 pretrained=True, **kwargs):
        super(MobileFaceDetection, self).__init__(**kwargs)
        self._model = model
        self._gpus = gpus
        self._image_short = image_short
        self._confidence_thresh = confidence_thresh
        self._nms_thresh = nms_thresh
        self._nms_topk = nms_topk
        self._pretrained = pretrained

        ctx = [mx.gpu(int(i)) for i in self._gpus.split(',') if i.strip()]
        self.ctx = [mx.cpu()] if not ctx else ctx 

        self.net = mobilefacedetnet_v1(self._model)  
        self.net.set_nms(self._nms_thresh, self._nms_topk)
        self.net.collect_params().reset_ctx(ctx = self.ctx)     
    
    def mobileface_detector(self, image_dir, image_mat): 
        """Face bounding box detection.
        Parameters
        ----------
        image_dir: str, default=''.
            The path of test images, use comma to split multiple.
        image_mat : Mat 
            The Mat data format of reading from the original image using opencv.
        Returns
        -------
        type: list
            Results of Face Detection:
            [[int(xmin),int(ymin),int(xmax),int(ymax),float(confidence),str(class)],
            [...],...].
        """        
        result_list = [] 
        x, img = presets.yolo.load_test(image_dir, short=self._image_short)
        x = x.as_in_context(self.ctx[0])
        # ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        tic = time.time()
        result =  self.net(x)
        toc = time.time() - tic
        print('Inference time: %fms' % (toc*1000))
        ids, scores, bboxes = [xx[0].asnumpy() for xx in result]

        h, w, c = image_mat.shape
        scale = float(self._image_short) / float(min(h, w))
        for i, bbox in enumerate(bboxes):
            if scores[i]< self._confidence_thresh:
                continue
            xmin, ymin, xmax, ymax = [int(x/scale) for x in bbox]
            result_list.append([xmin, ymin, xmax, ymax, float(scores[i]), self.net.classes[int(ids[i])]])
        return result_list