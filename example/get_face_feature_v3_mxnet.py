import os
import time
import sys
import cv2
import numpy
import mxnet
from glob import glob
from collections import namedtuple

__author__ = 'becauseofAI'
__version__ = 'v1.0'
__date__ = '2018/05/01'

sys.path.append('../MobileFace_Identification')
from Symbol_MobileFace_Identification_V3 import *


class MobileFaceFeatureExtractor(object):
    def __init__(self, model_file, epoch, batch_size, context = mxnet.cpu(), gpu_id = 0):
        self.model_file = model_file
        self.epoch = epoch
        self.batch_size = batch_size
        self.context = context

        network = get_feature_symbol_mobileface_v3() 
        self.model = mxnet.mod.Module(symbol = network, context = context)
        self.model.bind(for_training = False, data_shapes=[('data', (self.batch_size, 3, 112, 112))])
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(self.model_file, self.epoch)
        self.model.set_params(arg_params, aux_params)

    def get_face_feature_batch(self, face_batch):
        Batch = namedtuple('Batch', ['data'])
        batch_data = numpy.zeros((self.batch_size, 3, 112, 112))
        face_batch = face_batch.astype(numpy.float32, copy=False)
        face_batch = (face_batch - 127.5)/127.5
        face_num = len(face_batch)
        # batch_data[:face_num, 0, :, :] = face_batch
        batch_data = face_batch.transpose(0, 3, 1, 2)
        self.model.forward(Batch([mxnet.nd.array(batch_data)]))
        feature = self.model.get_outputs()[0].asnumpy().copy()
        return feature[:face_num, ...]


if __name__ == "__main__":
    model_file = '../MobileFace_Identification/MobileFace_Identification_V3'
    epoch = 0
    gpu_id = 0
    batch_size = 6
    # context = mxnet.gpu(gpu_id)
    context = mxnet.cpu()
    face_feature_extractor = MobileFaceFeatureExtractor(model_file, epoch, batch_size, context, gpu_id)

    # root_path = "../data/LFW-Aligned-100Pair/Aaron_Peirsol/"
    root_path = "../data/test/"
    file_names = glob(root_path + '*.*')
    count = 0
    face_batch = []
    for face_one in file_names:
        img = cv2.imread(face_one, 1)
        face_batch.append(img)
        count += 1
        if count % batch_size == 0:
            feature = face_feature_extractor.get_face_feature_batch(numpy.array(face_batch))
            face_batch = []
            print('count:', count)
            print('feature:', feature)
            print('feature.shape:', feature.shape)


