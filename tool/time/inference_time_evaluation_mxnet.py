import sys
import argparse
import datetime
import mxnet
import numpy 
from mxnet import ndarray as nd
# path_mxnet = '/path/to/incubator-mxnet/python'
# sys.path.append(path_mxnet)

class InferenceEvaluation(object):
    def __init__(self, symbol_file, input_shape, iteration, input_name='data', context=mxnet.cpu()):
        self.symbol_file = symbol_file
        self.input_shape = input_shape
        self.iteration = iteration
        self.input_name = input_name
        self.context = context

    def init_module(self):
        symbol = mxnet.symbol.load(self.symbol_file)

        module = mxnet.mod.Module(symbol=symbol, label_names=None, context=self.context)
        module.bind(data_shapes=[(self.input_name, self.input_shape)], for_training=False)
        module.init_params(initializer=mxnet.initializer.Xavier(magnitude=2.0))
        module.init_optimizer()

        return module

    def get_time(self, module):
        data = nd.zeros(self.input_shape)
        batch = mxnet.io.DataBatch(data=(data,))
        
        all_time = []

        symbol_name = self.symbol_file.split('/')[-1]
        print 'Start to evaluate: %s' % (symbol_name)
        for i in xrange(self.iteration):
            time_start = datetime.datetime.now()

            module.forward(batch, is_train=False)
            net_out = module.get_outputs()[0].asnumpy()

            time_end = datetime.datetime.now()
            one_time = time_end - time_start
            all_time.append(one_time.total_seconds())

        print 'Finish %d iterations in %f ms. Average infer time is [%f ms].' % (
        self.iteration, numpy.sum(all_time)*1000, numpy.mean(all_time)*1000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do inference evaluation')
    # general
    parser.add_argument('--symbol_version', help='select the symbol file version: V1, V2, V3')
    parser.add_argument('--input_shape', default=(1, 1, 100, 100), type=int, help='inpute shape: (N, C, H, W).')
    # parser.add_argument('--symbol_file', default='../../MobileFace_Identification/MobileFace_Identification_V2-symbol.json', help='path to symbol file')
    # parser.add_argument('--input_shape', default=(1, 3, 112, 112), type=int, help='inpute shape: (N, C, H, W).')
    # parser.add_argument('--symbol_file', default='../../MobileFace_Identification/MobileFace_Identification_V3-symbol.json', help='path to symbol file')
    # parser.add_argument('--input_shape', default=(1, 3, 112, 112), type=int, help='inpute shape: (N, C, H, W).')
    parser.add_argument('--input_name', default='data', type=str, help='input data name')
    parser.add_argument('--iteration', default=100, type=int, help='')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    args = parser.parse_args()

    if args.symbol_version == 'V1':
        symbol_file = '../../MobileFace_Identification/MobileFace_Identification_V1-symbol.json'
    elif args.symbol_version == 'V2':
        symbol_file = '../../MobileFace_Identification/MobileFace_Identification_V2-symbol.json'
    elif args.symbol_version == 'V3':
        symbol_file = '../../MobileFace_Identification/MobileFace_Identification_V3-symbol.json'
    else:
        print 'Cannot find the inpute version, please check the inpute symbol version file'

    # symbol_file = args.symbol_file
    input_shape = args.input_shape
    input_name = args.input_name
    iteration = args.iteration
    gpu_id = args.gpu_id
    # context = mxnet.gpu(args.gpu_id)
    context = mxnet.cpu()

    infer = InferenceEvaluation(symbol_file, input_shape, iteration, input_name, context)

    model = infer.init_module()
    infer.get_time(model)