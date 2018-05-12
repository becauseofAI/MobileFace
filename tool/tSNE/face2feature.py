# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy
import mxnet
from glob import glob
from compiler.ast import flatten
sys.path.append('../../example')
sys.path.append('../../MobileFace_Identification')
from get_face_feature_mxnet import *

feature_txt = open('feature_LFW-Aligned.txt', 'w')
lable_txt = open('lable_LFW-Aligned.txt', 'w')
model_file = '../../MobileFace_Identification/MobileFace_Identification_V1'
epoch = 118
gpu_id = 0
batch_size = 1
# context = mxnet.gpu(gpu_id)
context = mxnet.cpu()
face_feature_extractor = MobileFaceFeatureExtractor(model_file, epoch, batch_size, context, gpu_id)
# root_dir = "../../data/LFW-Aligned-100Pair/"
root_dir = "../../data/LFW-Aligned/"
dir_list = os.listdir(root_dir)
dir_list.sort()
class_count = -1
# feature_batch = []
# label_batch = []
for dir in dir_list:
    file_names = glob(os.path.join(root_dir, dir) + '/*.*')
    class_count += 1
    for face_one in file_names:
        img = cv2.imread(face_one, 0)
        feature = face_feature_extractor.get_face_feature_batch(numpy.array(img))
        # feature_batch.append(feature)
        # label_batch.append(class_count)       
        # feature = feature.tolist()
        # feature = flatten(feature)
        for i in range(0, len(feature[0])):
            feature_txt.write(str(feature[0][i]) + ' ')
        feature_txt.write('\n')
        lable_txt.write(str(class_count) + '\n')  
        print class_count

# numpy.savetxt("feature_LFW-Aligned-100Pair.txt", numpy.array(feature_batch));
# numpy.savetxt("lable_LFW-Aligned-100Pair.txt", numpy.array(label_batch));      
# print feature_batch
# print label_batch
# numpy.savetxt("feature_LFW-Aligned-100Pair.txt", feature_batch);
# numpy.savetxt("lable_LFW-Aligned-100Pair.txt", label_batch);  
feature_txt.close()
lable_txt.close()

print 'Done!'