# -*- coding:utf-8 -*-

import mxnet as mx
import numpy as np

'''
MobileFace for face Identification with ResNet and MobileNetV2, implemented in MXNet.
Reference:
Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381
'''
__author__ = 'becauseofAI'
__version__ = 'v2.0'
__date__ = '2018/07/01'


def ConvDepthwise_ResUnit(data, num_filter, num_group, stride, dim_match, name, bottle_neck=True, workspace=256, memonger=False):
    conv0_0 = mx.sym.Convolution(data=data, num_filter=num_group, kernel=(1, 1), num_group=1, stride=(1,1), pad=(0, 0),
                                  no_bias=True, workspace=workspace, name=name + '_conv0_0')
    conv0_1 = mx.sym.Convolution(data=conv0_0, num_filter=num_group, kernel=(3, 3), num_group=num_group, stride=(2,2), pad=(1, 1),
                                  no_bias=True, workspace=workspace, name=name + '_conv0_1')
    act0 = mx.sym.LeakyReLU(data=conv0_1, act_type='prelu', name=name + '_prelu0')

    conv1 = mx.sym.Convolution(data=act0, num_filter=num_group, kernel=(3, 3), num_group=num_group, stride=stride, pad=(1, 1),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    act1 = mx.sym.LeakyReLU(data=conv1, act_type='prelu', name=name + '_prelu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_group, kernel=(3,3), num_group=num_group, stride=stride, pad=(1, 1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    act2 = mx.sym.LeakyReLU(data=conv2, act_type='prelu', name=name + '_prelu2')
    shortcut = act0

    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return act2 + shortcut


def ResNet_MobileNet(input_data, units, num_stages, filter_list, group_list, bottle_neck=True, workspace=256, memonger=False):
    num_unit = len(units)
    assert(num_unit == num_stages)
    body = mx.sym.Convolution(data=input_data, num_filter=32, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                              no_bias=True, name="conv1", workspace=workspace)
    body = mx.sym.LeakyReLU(data=body, act_type='prelu', name='prelu1')
    for i in range(num_stages):
        for j in range(units[i]):
            body = ConvDepthwise_ResUnit(body, filter_list[i+1], group_list[i+1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 1),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    body = mx.sym.Flatten(data=body)
    fc5 = mx.sym.FullyConnected(data=body, num_hidden=256, no_bias=True, name='fc5')
    fc5_bn = mx.sym.BatchNorm(fc5, fix_gamma=False, use_global_stats=False)
    return fc5_bn

def get_feature_symbol_mobileface_v3():
    in_data = mx.symbol.Variable(name='data')
    # in_data = in_data-127.5
    # in_data = in_data*0.0078125
    fc5_bn = ResNet_MobileNet(
        input_data=in_data,
        units=[1, 1, 1, 1],
        num_stages=4,
        filter_list=[32, 32, 32, 64, 128],
        group_list=[32, 32, 32, 64, 128],
        bottle_neck=False,
        workspace=128)
    feature_net = mx.symbol.L2Normalization(data=fc5_bn)
    return feature_net

def get_model_mobileface_v3():
    model = get_feature_symbol_mobileface_v3()
    shape = {'data': (32, 3, 112, 112)}
    print(mx.viz.print_summary(model, shape = shape))
    digraph = mx.viz.plot_network(model, shape = shape)
    digraph.view()
    model.save('MobileFace_Identification_V3.json')


if __name__ == '__main__':
    # get_feature_symbol_mobileface_v3()
    get_model_mobileface_v3()
