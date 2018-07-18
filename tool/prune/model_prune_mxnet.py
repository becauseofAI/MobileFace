from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx

parser = argparse.ArgumentParser(description='MXNet model prune')
# general
parser.add_argument('--model', default='./MobileFace_Identification_V3,150', help='path to load model.')
args = parser.parse_args()

_vec = args.model.split(',')
assert len(_vec)==2
prefix = _vec[0]
epoch = int(_vec[1])

print('loading',prefix, epoch)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers['batchnorm0_output']

dellist = []
for k,v in arg_params.iteritems():
  if k.startswith('fc7'):
    dellist.append(k)
for d in dellist:
  del arg_params[d]

resave_epoch = 0
resave_prefix = './MobileFace_Identification_V3'
print('saving', resave_prefix, resave_epoch)
mx.model.save_checkpoint(resave_prefix, resave_epoch, sym, arg_params, aux_params)
print('prune done!')


