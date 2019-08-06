# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy
import mxnet
import scipy.spatial
import seaborn as sns
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
# from compiler.ast import flatten
sys.path.append('../../example')
sys.path.append('../../MobileFace_Identification')
from get_face_feature_v1_mxnet import *


model_file = '../../MobileFace_Identification/MobileFace_Identification_V1'
epoch = 0
gpu_id = 0
batch_size = 2
# context = mxnet.gpu(gpu_id)
context = mxnet.cpu()
face_feature_extractor = MobileFaceFeatureExtractor(model_file, epoch, batch_size, context, gpu_id)
root_dir = "../../data/LFW-Aligned-100Pair/"
dir_list = os.listdir(root_dir)
dir_list.sort()
class_count = -1
feature_row = []
feature_col = []
face_batch = []
count = 0
for dir in dir_list:
    file_names = glob(os.path.join(root_dir, dir) + '/*.*')
    class_count += 1
    for face_one in file_names:
        img = cv2.imread(face_one, 0)
        face_batch.append(img)
        count += 1
        if count % batch_size == 0:
            feature = face_feature_extractor.get_face_feature_batch(numpy.array(face_batch))
            face_batch = []
    feature_row.append(feature[0])
    feature_col.append(feature[1])
    print('class_count: %d' % class_count)

sns.set()
numpy.random.seed(0)
sim_matrix = numpy.random.rand(100, 100)
for i in range(0, len(feature_row)):
    for j in range(0, len(feature_col)):
        sim_matrix[i][j] = 1 - scipy.spatial.distance.cosine(feature_row[i], feature_col[j])
        print(i, j, sim_matrix[i][j])

ax = sns.heatmap(sim_matrix)
ax.xaxis.set_ticks_position('top')
ax.set_title('LFW-100Pair MobileFace_V1 ConfusionMatrix Similarity Heatmap', fontsize=14, fontweight='bold', x=0.5, y=1.08)
plt.show()
plt.savefig("examples.png")