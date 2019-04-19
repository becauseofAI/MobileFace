"""MobileFaceAttribute Demo script based on SE-ResNetV2-18."""
from __future__ import absolute_import
from __future__ import division
import time
import mxnet as mx
from mxnet import nd, image
from gluoncv.data.transforms.presets.imagenet import transform_eval
from mobileface_attr_net import get_mf_attr_net


class MobileFaceAttribute():
    """MobileFaceAttrNet V1 attribute predict network.
    Parameters
    ----------
    model: str, default is '../MobileFace_Attrirbute/model/mobileface_attr_v1_gluoncv.params'.
        Pretrained model path.
    gpus: str, default='' is for CPU.
        Training with GPUs, you can specify 1,3 for example.
    version: int, default is 1.
        Version of MobileFace attribute network.
    num_layers: int, default is 18.
        Numbers of ResNet layer.
    image_size: int, default is 96.
        Image size of network inpute.
    pretrained: str, default is 'True'.
        Load weights from previously saved parameters.    
    """
    def __init__(self, model, gpus, version=1, num_layers=18, image_size=96, 
                 pretrained=True, **kwargs):
        super(MobileFaceAttribute, self).__init__(**kwargs)
        self._version = version
        self._num_layers = num_layers
        self._model = model
        self._gpus = gpus
        self._image_size = image_size
        self._pretrained = pretrained

        ctx = [mx.gpu(int(i)) for i in self._gpus.split(',') if i.strip()]
        self.ctx = [mx.cpu()] if not ctx else ctx 

        self.net = get_mf_attr_net(self._version, self._num_layers, self._model, self.ctx, pretrained=True)  

        # self.attribute_map1 = [{0: 'female', 1: 'male'},
        #                       {0: 5, 1: 15, 2: 25, 3: 35, 4: 50, 5: 70},
        #                       {0: 'laugh', 1: 'smile', 2: 'calm', 3: 'sad', 
        #                        4: 'amazed', 5: 'anger', 6: 'despised', 7: 'complicated'}] 

        self.attribute_map2 = [{0: 'female', 1: 'male'},
                              {0: 5, 1: 15, 2: 25, 3: 35, 4: 50, 5: 70},
                              {0: 'laugh', 1: 'smile', 2: 'smile', 3: 'calm', 
                               4: 'laugh', 5: 'anger', 6: 'despised', 7: 'complicated'}] 

        # self.attribute_map3 = [{0: 'female', 1: 'male'},
        #                       {0: 5, 1: 15, 2: 25, 3: 35, 4: 50, 5: 70},
        #                       {0: 'exaggerated', 1: 'positive', 2: 'calm', 3: 'negtive', 
        #                        4: 'exaggerated', 5: 'exaggerated', 6: 'negtive', 7: 'complicated'}] 

        # self.attribute_map4 = [{0: 'female', 1: 'male'},
        #                       {0: 5, 1: 15, 2: 25, 3: 35, 4: 50, 5: 70},
        #                       {0: 'exaggerated', 1: 'positive', 2: 'calm', 3: 'calm', 
        #                        4: 'exaggerated', 5: 'exaggerated', 6: 'negtive', 7: 'complicated'}]

    def get_attribute(self, image): 
        """Face attribute predictor.
        Parameters
        ----------
        image: NDArray.
            The NDArray data format for MXNet to process, such as (H, W, C).
        Returns
        -------
        type: tuple
            Results of Face Attribute Predict:
            (str(gender), int(age), str(expression)).
        """     
        img = transform_eval(image, resize_short=self._image_size, crop_size=self._image_size)
        img = img.as_in_context(self.ctx[0])   
        tic = time.time()
        pred = self.net(img)
        toc = time.time() - tic
        print('Attribute inference time: %fms' % (toc*1000))

        topK = 1
        topK_age = 6
        topK_exp = 2
        age = 0
        ind_1 = nd.topk(pred[0], k=topK)[0].astype('int')
        ind_2 = nd.topk(pred[1], k=topK_age)[0].astype('int')
        ind_3 = nd.topk(pred[2], k=topK_exp)[0].astype('int')
        for i in range(topK_age):
            age += int(nd.softmax(pred[1])[0][ind_2[i]].asscalar() * self.attribute_map2[1][ind_2[i].asscalar()])
        gender = self.attribute_map2[0][ind_1[0].asscalar()]
        if  nd.softmax(pred[2])[0][ind_3[0]].asscalar() < 0.45:
            expression = self.attribute_map2[2][7]
        else:
            expression_1 = self.attribute_map2[2][ind_3[0].asscalar()]
            expression_2 = self.attribute_map2[2][ind_3[1].asscalar()]  

        return (gender, age, (expression_1, expression_2))

