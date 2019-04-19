# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring
"""SE_ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['MobileFace_AttributeV1','SE_BasicBlockV2']

import os
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


# Blocks
class SE_BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for SE_ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(SE_BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, 1, channels)

        self.se = nn.HybridSequential(prefix='')
        self.se.add(nn.Dense(channels//16, use_bias=False))
        self.se.add(nn.Activation('relu'))
        self.se.add(nn.Dense(channels, use_bias=False))
        self.se.add(nn.Activation('sigmoid'))

        # downsample = True
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.se(w)
        x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        return x + residual


# Nets
class MobileFace_AttributeV1(HybridBlock):
    r"""MobileFace Attribute model based SE_ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are SE_BasicBlockV1, SE_BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=True,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MobileFace_AttributeV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                # stride = 1 if i == 0 else 2
                stride = 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]

            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))

            self.branch1 = nn.HybridSequential(prefix='')
            self.branch1.add(nn.Conv2D(64, 1, 1, 0, use_bias=False))
            self.branch1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch1.add(nn.Activation('relu'))
            self.branch1.add(nn.Conv2D(128, 3, 1, 1, use_bias=False))
            self.branch1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch1.add(nn.Activation('relu'))
            self.branch1.add(nn.Conv2D(64, 1, 1, 0, use_bias=False))
            self.branch1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch1.add(nn.Activation('relu'))
            self.branch1.add(nn.Conv2D(128, 3, 1, 1, use_bias=False))
            self.branch1.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch1.add(nn.Activation('relu'))
            self.branch1.add(nn.GlobalAvgPool2D())
            self.branch1.add(nn.Flatten())
            self.output1 = nn.Dense(2, in_units=128)

            self.branch2 = nn.HybridSequential(prefix='')
            self.branch2.add(nn.Conv2D(128, 1, 1, 0, use_bias=False))
            self.branch2.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch2.add(nn.Activation('relu'))
            self.branch2.add(nn.Conv2D(256, 3, 1, 1, use_bias=False))
            self.branch2.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch2.add(nn.Activation('relu'))
            self.branch2.add(nn.Conv2D(128, 1, 1, 0, use_bias=False))
            self.branch2.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch2.add(nn.Activation('relu'))
            self.branch2.add(nn.Conv2D(256, 3, 1, 1, use_bias=False))
            self.branch2.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch2.add(nn.Activation('relu'))
            self.branch2.add(nn.GlobalAvgPool2D())
            self.branch2.add(nn.Flatten())
            self.output2 = nn.Dense(6, in_units=256)

            self.branch3 = nn.HybridSequential(prefix='')
            self.branch3.add(nn.Conv2D(128, 1, 1, 0, use_bias=False))
            self.branch3.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch3.add(nn.Activation('relu'))
            self.branch3.add(nn.Conv2D(256, 3, 1, 1, use_bias=False))
            self.branch3.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch3.add(nn.Activation('relu'))
            self.branch3.add(nn.Conv2D(128, 1, 1, 0, use_bias=False))
            self.branch3.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch3.add(nn.Activation('relu'))
            self.branch3.add(nn.Conv2D(256, 3, 1, 1, use_bias=False))
            self.branch3.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.branch3.add(nn.Activation('relu'))
            self.branch3.add(nn.GlobalAvgPool2D())
            self.branch3.add(nn.Flatten())
            self.output3 = nn.Dense(8, in_units=256)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x1 = self.branch1(x)
        x1 = self.output1(x1)
        x2 = self.branch2(x)
        x2 = self.output2(x2)
        x3 = self.branch3(x)
        x3 = self.output3(x3)
        return (x1, x2, x3)

# Specification
mobileface_attr_net_versions = [MobileFace_AttributeV1]
resnet_block_versions = [{'basic_block': SE_BasicBlockV2}]
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [16, 32, 64, 128, 256])}


# Constructor
def get_mf_attr_net(version, num_layers, model_path, ctx=cpu(), pretrained=True, **kwargs):
    r"""MobileFace Attribute model based SE_ResNet:
    SE_ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    SE_ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2, 3.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    model_path : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 3, \
        "Invalid resnet version: %d. Options are 1, 2 and 3."%version
    resnet_class = mobileface_attr_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    net.load_parameters(model_path, ctx=ctx)
    return net