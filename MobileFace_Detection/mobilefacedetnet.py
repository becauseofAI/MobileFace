"""Mobilefacedetnet as YOLO3-like network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from utils.yolo3 import YOLOV3

__all__ = ['MFDetV1', 'get_mfdet', 'mfdet']

def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class MFDetBasicBlockV1(gluon.HybridBlock):
    """Mobilefacedet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv.

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    """
    def __init__(self, channel, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MFDetBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        # 3x3 conv expand
        self.body.add(_conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


class MFDetV1(gluon.HybridBlock):
    """Mobilefacedet v1 backbone.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    classes : int, default is 1000
        Number of classes, which determines the dense layer output channels.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Attributes
    ----------
    features : mxnet.gluon.nn.HybridSequential
        Feature extraction layers.
    output : mxnet.gluon.nn.Dense
        A classes(1000)-way Fully-Connected Layer.

    """
    def __init__(self, layers, channels, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MFDetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        with self.name_scope():
            self.features = nn.HybridSequential()
            # first 3x3 conv!!!
            self.features.add(_conv2d(channels[0], 3, 1, 1,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(MFDetBasicBlockV1(channel // 2,
                                                          norm_layer=BatchNorm,
                                                          norm_kwargs=None))
            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)

# default configurations
mfdet_versions = {'v1': MFDetV1}
mfdet_spec = {
    'v1': {24: ([1, 2, 2, 2, 2], [16, 32, 64, 128, 256, 256]),}
}

def get_mfdet_base(mfdet_version, num_layers, pretrained=True, ctx=mx.cpu(), **kwargs):
    """Get mobilefacedet backbone by `version` and `num_layers` info.

    Parameters
    ----------
    mobilefacedet_version : str
        Mobilefacedet version, choices are ['v1'].
    num_layers : int
        Number of layers.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Mobilefacedet network.

    Examples
    --------
    >>> model = get_mfdet('v1', 24, pretrained=True)
    >>> print(model)

    """
    assert mfdet_version in mfdet_versions and mfdet_version in mfdet_spec, (
        "Invalid mfdet version: {}. Options are {}".format(
            mfdet_version, str(mfdet_versions.keys())))
    specs = mfdet_spec[mfdet_version]
    assert num_layers in specs, (
        "Invalid number of layers: {}. Options are {}".format(num_layers, str(specs.keys())))
    layers, channels = specs[num_layers]
    mfdet_class = mfdet_versions[mfdet_version]
    net = mfdet_class(layers, channels, **kwargs)
    return net

def mfdet24(**kwargs):
    """Mobilefacedet v1 backbone, 24 layer network.

    Parameters
    ----------
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Mobilefacedet backbone network.

    """
    return get_mfdet_base('v1', 24, **kwargs)

def get_mfdetv1(stages, filters, anchors, strides, classes, model_path, 
                pretrained=True, ctx=mx.cpu(), **kwargs):
    """Get mobilefacedet as YOLO3-like models.
    Parameters
    ----------
    stages : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    anchors : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    strides : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    model_path : str
        Model weights storing path.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    HybridBlock
        A mobilefacedet as YOLO3-like face detection network.
    """
    net = YOLOV3(stages, filters, anchors, strides, classes=classes, **kwargs)
    net.load_parameters(model_path, ctx=ctx)
    return net

def mobilefacedetnet_v1(model_path, pretrained_base=False, pretrained=True,
                     norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """Mobilefacedet: A YOLO3-like multi-scale with mfdet24 base network for fast face detection.
    Parameters
    ----------
    model_path : str
        Model weights storing path.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid mobilefacedet network.
    """
    # pretrained_base = False if pretrained else pretrained_base
    base_net = mfdet24(
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:9], base_net.features[9:12], base_net.features[12:]]
    anchors = [[10, 12, 16, 20, 23, 29], [43, 54, 60, 75, 80, 106], [118, 157, 186, 248, 285, 379]]
    strides = [8, 16, 32]
    classes =  ('face',)
    return get_mfdetv1(
        stages, [256, 128, 64], anchors, strides, classes, model_path,
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
