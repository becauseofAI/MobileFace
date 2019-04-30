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
__version__ = 'v1.1'
__date__ = '2018/06/01'

def Relu6(data):
    return mx.sym.minimum(mx.sym.maximum(data, 0), 6)


def ConvNolinear(data, num_filter, kernel, stride, pad, use_global_stats):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats)
    relu6 = Relu6(bn)
    return relu6


def ConvDepthwise(data, num_filter, kernel, stride, pad, use_global_stats):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_filter, no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats)
    relu6 = Relu6(bn)
    return relu6


def ConvLinear(data, num_filter, kernel, stride, pad, use_global_stats):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats)
    return bn


def Bottleneck(data, in_c, out_c, t, s, use_global_stats):
    if s == 1:
        conv_nolinear = ConvNolinear(data, num_filter=in_c*t, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats)
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=in_c*t, kernel=(3,3), stride=(1,1), pad=(1,1), use_global_stats=use_global_stats)
        conv_linear = ConvLinear(conv_dw, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats)

        if in_c == out_c:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True)
            shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, use_global_stats=use_global_stats)
            shortcut = Relu6(shortcut)

        if True:
            shortcut._set_attr(mirror_stage='True')
        return shortcut + conv_linear

    if s == 2:
        conv_nolinear = ConvNolinear(data, num_filter=in_c*t, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats)
        conv_dw = ConvDepthwise(conv_nolinear, num_filter=in_c*t, kernel=(3,3), stride=(2,2), pad=(1,1), use_global_stats=use_global_stats)
        conv_linear = ConvLinear(conv_dw, num_filter=out_c, kernel=(1,1), stride=(1,1), pad=(0,0), use_global_stats=use_global_stats)
        return conv_linear


def get_symbol_mobilenet2(in_data, **kwargs):
    T = 3
    use_global_stats = False
    conv1 = mx.sym.Convolution(data=in_data, num_filter=32, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, use_global_stats=use_global_stats, name='bn1')
    relu1 = Relu6(bn1)

    bottleneck1_0 = Bottleneck(relu1, in_c=32, out_c=32, t=T, s=2, use_global_stats=use_global_stats)
    bottleneck1_1 = Bottleneck(bottleneck1_0, in_c=32, out_c=32, t=T, s=1, use_global_stats=use_global_stats)
    bottleneck1_2 = Bottleneck(bottleneck1_1, in_c=32, out_c=32, t=T, s=1, use_global_stats=use_global_stats)

    bottleneck2_0 = Bottleneck(bottleneck1_2, in_c=32, out_c=32, t=T, s=2, use_global_stats=use_global_stats)
    bottleneck2_1 = Bottleneck(bottleneck2_0, in_c=32, out_c=32, t=T, s=1, use_global_stats=use_global_stats)
    bottleneck2_2 = Bottleneck(bottleneck2_1, in_c=32, out_c=32, t=T, s=1, use_global_stats=use_global_stats)

    bottleneck3_0 = Bottleneck(bottleneck2_2, in_c=32, out_c=64, t=T, s=2, use_global_stats=use_global_stats)
    bottleneck3_1 = Bottleneck(bottleneck3_0, in_c=64, out_c=64, t=T, s=1, use_global_stats=use_global_stats)
    bottleneck3_2 = Bottleneck(bottleneck3_1, in_c=64, out_c=64, t=T, s=1, use_global_stats=use_global_stats)

    bottleneck4_0 = Bottleneck(bottleneck3_2, in_c=64, out_c=128, t=T, s=2, use_global_stats=use_global_stats)
    bottleneck4_1 = Bottleneck(bottleneck4_0, in_c=128, out_c=128, t=T, s=1, use_global_stats=use_global_stats)
    bottleneck4_2 = Bottleneck(bottleneck4_1, in_c=128, out_c=128, t=T, s=1, use_global_stats=use_global_stats)

    flatten = mx.sym.Flatten(data=bottleneck4_2)
    fc5 = mx.sym.FullyConnected(data=flatten, num_hidden=256, no_bias=True, name='fc5')
    fc5_bn = mx.sym.BatchNorm(fc5, fix_gamma=False, use_global_stats=use_global_stats, name='fc5_bn')

    return fc5_bn

def get_feature_symbol_mobileface_v2():
    in_data = mx.symbol.Variable(name='data')
    # in_data = in_data-127.5
    # in_data = in_data*0.0078125
    # feature_net = get_symbol_mobilenet2(in_data)
    fc5_bn = get_symbol_mobilenet2(in_data)
    feature_net = mx.symbol.L2Normalization(data=fc5_bn)
    return feature_net

def get_model_mobileface_v2():
    in_data = mx.symbol.Variable(name='data')
    model = get_symbol_mobilenet2(in_data)
    shape = {'data': (1, 3, 112, 112)}
    print(mx.viz.print_summary(model, shape = shape))
    digraph = mx.viz.plot_network(model, shape=shape)
    digraph.view()
    model.save('MobileFace_Identification_V2.json')


if __name__ == '__main__':
    # get_feature_symbol_mobileface_v2()
    get_model_mobileface_v2()


