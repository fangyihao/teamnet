'''
Created on Jun 21, 2018

@author: fangy5
'''
'''
Created on Jun 21, 2018

@author: fangy5
'''


import argparse
import moe
import tensorflow as tf
from tensor2tensor.utils import t2t_model

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="mnist")
    
    parser.add_argument(
        '--batch_size', type=int, default = 512, 
        help="batch size")
    '''
    parser.add_argument(
        '--max_steps', type=int, default = 10000, 
        help="number of batches to run")
    '''
    parser.add_argument(
        '--max_epochs', type=int, default = 150, 
        help="maximum number of epochs to run")
    
    parser.add_argument(
        '--train_dataset_size', type=int, default = 60000, 
        help="train dataset size")
        
    parser.add_argument(
        '--log_frequency', type=int, default = 10, 
        help="how often to log results to the console")
    
    parser.add_argument(
        '--num_experts', type=int, default = 4, 
        help="number of partitions")
    
    parser.add_argument(
        '--model_dir', default = "mnist/ckpt", 
        help="train directory")
     
    parser.add_argument(
        '--data_dir', default = "mnist/data", 
        help="data directory")
    
    parser.add_argument(
        '--tmp_dir', default = "mnist/tmp", 
        help="temporary directory")
    
    parser.add_argument(
        '--log_dir', default = "mnist/log", 
        help="log directory")    

    parser.add_argument(
        '--log_device_placement', default = False, 
        help="whether to log device placement")
    
    parser.add_argument(
        '--stale_interval', type=int, default = 50, 
        help="stale interval")
    
    parser.add_argument(
        '--mc_steps', type=int, default = 1, 
        help="number of mc steps")

    parser.add_argument(
        '--input_shape', default = (None, 28, 28, 1), 
        help="input shape")

    parser.add_argument(
        '--model', default = "basic_fc_relu", 
        help="model")
    
    parser.add_argument(
        '--model_cls', default = None, 
        help="model class")
    
    parser.add_argument(
        '--problem', default = "image_mnist", 
        help="problem")
    
    parser.add_argument(
        '--hparams', default = "basic_fc_small", 
        help="hparams")
    
    parser.add_argument(
        '--generate_data', default = False, 
        help="generate_data")
    
    parser.add_argument(
        '--reshuffle_each_epoch', default = True, 
        help="reshuffle for each epoch")
        
    parser.add_argument(
        '--mode', default = "train", 
        help="mode")
    
    parser.add_argument(
        '--logger', default = None, 
        help="logger")
    
    parser.add_argument(
        '--device', default = None, 
        help="device")
    
    parser.add_argument(
        '--gate_device', default = None, 
        help="gate device")
    
    parser.add_argument(
        '--num_gates', type=int, default = 8, 
        help="number of gates")
    
    parser.add_argument(
        '--verbose', default = False, 
        help="verbose")
    
    parser.add_argument(
        '--socket_nodes', default = None, 
        help="socket nodes")
    return parser

parser = create_parser()
params = parser.parse_args()

'''
class VanillaCNN_3(t2t_model.T2TModel):

    def body(self, features):
    
        inputs = features["inputs"]
        filters = self.hparams.hidden_size
        x = inputs
        for _ in range(3):
            x = tf.layers.conv2d(x, filters,
                          kernel_size=(3, 3), strides=(2, 2))
            print("x:",x.get_shape())
        return x
class VanillaCNN_6(t2t_model.T2TModel):

    def body(self, features):
    
        inputs = features["inputs"]
        filters = self.hparams.hidden_size
        x = inputs
        for _ in range(6):
            x = tf.layers.conv2d(x, filters,
                          kernel_size=(3, 3), strides=(2, 2))
            print("x:",x.get_shape())
        return x
class VanillaCNN_12(t2t_model.T2TModel):

    def body(self, features):
    
        inputs = features["inputs"]
        filters = self.hparams.hidden_size
        x = inputs
        for _ in range(12):
            x = tf.layers.conv2d(x, filters,
                          kernel_size=(3, 3), strides=(2, 2))
            print("x:",x.get_shape())
        return x

model_clses = [VanillaCNN_3, VanillaCNN_6, VanillaCNN_12]
'''
hparams_col = ["basic_fc_4", "basic_fc_8", "basic_fc_16"][::-1]
batch_size_col = [512,256,128][::-1]
num_experts_col = [4,2,1][::-1]
range_col = [16, 8, 4][::-1]
for i, num_experts in enumerate(num_experts_col):
    params.device= [0,1]
    if num_experts == 1:
        for j in range(range_col[i]):
            params.max_epochs = 10*(j+1)
            
            params.num_experts = num_experts
            params.model_dir = "mnist/ckpt_%s_e%d"%(hparams_col[i],num_experts)
            params.hparams = hparams_col[i]
            params.batch_size = batch_size_col[i]
            #params.model_cls = model_clses[i]
            params.mode = "train"
            params.logger = None
            moe.main(params)
            params.mode = "predict"
            params.logger = None
            moe.main(params)

