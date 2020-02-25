'''
Created on Jun 21, 2018

@author: fangy5
'''


import argparse
import router

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="cifar10")
    
    parser.add_argument(
        '--batch_size', type=int, default = 512, 
        help="batch size")
    '''
    parser.add_argument(
        '--max_steps', type=int, default = 50000, 
        help="number of batches to run")
    '''
    parser.add_argument(
        '--max_epochs', type=int, default = 300, 
        help="maximum number of epochs to run")
    
    parser.add_argument(
        '--train_dataset_size', type=int, default = 50000, 
        help="train dataset size")
    
    parser.add_argument(
        '--log_frequency', type=int, default = 10, 
        help="how often to log results to the console")
    
    parser.add_argument(
        '--num_experts', type=int, default = 4, 
        help="number of partitions")
    
    parser.add_argument(
        '--model_dir', default = "cifar10/ckpt", 
        help="train directory")
     
    parser.add_argument(
        '--data_dir', default = "cifar10/data", 
        help="data directory")
    
    parser.add_argument(
        '--tmp_dir', default = "cifar10/tmp", 
        help="temporary directory")
    
    parser.add_argument(
        '--log_dir', default = "cifar10/log", 
        help="train directory")
    
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
        '--input_shape', default = (None, 32, 32, 3), 
        help="input shape")

    parser.add_argument(
        '--model', default = "shake_shake", 
        help="model")
    
    parser.add_argument(
        '--model_cls', default = None, 
        help="model class")
    
    parser.add_argument(
        '--problem', default = "image_cifar10", 
        help="problem")
    
    parser.add_argument(
        '--hparams', default = "shakeshake_big", 
        help="hparams")
    
    parser.add_argument(
        '--generate_data', default = False, 
        help="generate_data")
    
    parser.add_argument(
        '--reshuffle_each_epoch', default = False, 
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
        '--num_gates', type=int, default = 4, 
        help="number of gates")
    
    parser.add_argument(
        '--verbose', default = False, 
        help="verbose")
    return parser

parser = create_parser()
params = parser.parse_args()

#model_dir_col = ["cifar10/ckpt_ssbig_l8_e4", "cifar10/ckpt_ssbig_l14_e2", "cifar10/ckpt_ssbig_l26_e1"]
hparams_col = ["shakeshake_big_quick_l8", "shakeshake_big_quick_l14", "shakeshake_big_quick_l26"]
batch_size_col = [128,128,128]
num_experts_col = [4,2,1]
for i, num_experts in enumerate(num_experts_col):
    params.device= [0,1]
    params.gate_device= [2,3]
    params.num_gates = 8
    if num_experts == 2:
        for j in range(24):
            params.max_epochs = 20*(j+1)
            
            params.num_experts = num_experts
            params.model_dir = "cifar10/ckpt_%s_%s_e%d"%(hparams_col[i],'rs' if params.reshuffle_each_epoch else 'nrs',num_experts)
            params.hparams = hparams_col[i]
            params.batch_size = batch_size_col[i]
            params.mode = "train"
            params.logger = None
            router.main(params)
            params.mode = "predict"
            params.logger = None
            router.main(params)
        

