'''
Created on Sep 11, 2018

@author: Yihao Fang
'''
from multiprocessing import Process, Manager
from work_node import WorkNode
from moe import StaleObserver
from moe import Expert
import numpy as np
import os
from os import listdir
from os.path import isfile, isdir, join
import re
import logging
from datetime import datetime

def get_ckpt_iters(params):
    ds = [d for d in listdir(params.model_dir) if isdir(join(params.model_dir, d))]
    all_num_iters = []
    for d in ds:
        if re.search('_rtr', d):
            d_full = join(params.model_dir, d)
            fs=[f for f in listdir(d_full) if isfile(join(d_full, f))] 
            num_iters = []
            for f in fs:
                m = re.search('\-([0-9]+).meta', f)
                if m:
                    num_iters.append(int(m.group(1)))
            num_iter = np.max(num_iters) if len(num_iters) > 0 else 0
            all_num_iters.append(num_iter)
    all_num_iter = np.max(all_num_iters) if len(all_num_iters) > 0 else 0
    return all_num_iter



class DataReader (Process):
    def __init__(self, name, device, params):
        Process.__init__(self)
        
        self.name = name
        self.device = device
        self.params = params
        self.manager = Manager()
        self.r_queue = self.manager.Queue(1)
        
    def run(self):
        self.params.logger.info("begin DataReader")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id
        
        import tensorflow as tf
        from tensor2tensor import problems
        
        # Enable TF Eager execution
        tfe = tf.contrib.eager
        tfe.enable_eager_execution()
        
        # Setup the training data
        
        # Fetch the MNIST problem
        problem = problems.problem(self.params.problem)
        # The generate_data method of a problem will download data and process it into
        # a standard format ready for training and evaluation.
        if self.params.generate_data == True:
            problem.generate_data(self.params.data_dir, self.params.tmp_dir)

        Modes = tf.estimator.ModeKeys
        
        if self.params.mode == "train":
            mode = Modes.TRAIN
            max_epochs = self.params.max_epochs
            start_epoch = get_ckpt_iters(self.params)*self.params.batch_size//self.params.train_dataset_size
            num_repeats = max_epochs-start_epoch

        elif self.params.mode == "predict":
            mode = Modes.EVAL
            max_epochs = self.params.max_epochs
            start_epoch = get_ckpt_iters(self.params)*self.params.batch_size//self.params.train_dataset_size
            num_repeats = 1
            #self.params.logger.info("epoch #%d"%self.params.max_epochs)
            
        
        if num_repeats > 0:
            dataset = problem.dataset(mode, self.params.data_dir)
            
            dataset = dataset.shuffle(buffer_size = 256, reshuffle_each_iteration=self.params.reshuffle_each_epoch)
            dataset = dataset.repeat(num_repeats).batch(self.params.batch_size)
            
            pre_r = -1
            for count, example in enumerate(tfe.Iterator(dataset)):
                if self.params.mode == "train":
                    r = start_epoch + count*self.params.batch_size//self.params.train_dataset_size
                elif self.params.mode == "predict":
                    r = start_epoch
                    
                if r > pre_r:
                    self.params.logger.info("epoch #%d"%(r+1))
                    pre_r = r
                
                inputs, targets = example["inputs"], example["targets"]
    
                self.r_queue.put([inputs.numpy(), targets.numpy()])
                
        self.r_queue.put(None)     
        
        self.params.logger.info("end DataReader")
  

class RouterTrainer(WorkNode):

    def run(self):
        self.params.logger.info("begin RouterTrainer")
        self.worknodes = []
        
        data_reader = DataReader("reader", None, self.params)
        self.worknodes.append(data_reader)
        
        stale_observer = StaleObserver(self.params, c_queue = data_reader.r_queue)
        self.worknodes.append(stale_observer)
        
        router_caller = RouterCaller(self.params, c_queue = stale_observer.r_queue)
        self.worknodes.append(router_caller)
        
        
        for worknode in self.worknodes:
            worknode.start()
        
        for worknode in self.worknodes:
            worknode.join()

        self.params.logger.info("end RouterTrainer")
    
class RouterCaller(WorkNode):

    def run(self):
        self.params.logger.info("begin RouterCaller")
        model_name = self.params.model

        router = Expert(model_name+"_rtr", "/device:GPU:0", self.params)
        router.start()
        router.r_queue.get()
        
        while True:   
            self.params.logger.debug("begin RouterTrainer c_queue.get")
            batch = self.c_queue.get()
            self.params.logger.debug("end RouterTrainer c_queue.get")
            if batch is None:
                break
            
            [batch_xs, _], uncs = batch
            
            batch_ys = np.argmin(uncs, axis = 1).reshape(-1)
            
            #batch_ys = np.eye(self.params.num_experts)[batch_ys]
            
            router.c_queue.put(["train",[batch_xs, batch_ys]])
            
            router.r_queue.get()
                        
        
        router.c_queue.put(["exit"])
        router.r_queue.get()
        router.join()   
            
        self.params.logger.info("end RouterTrainer") 

def get_logger(params):
    logger = logging.getLogger("%s_%s_%s_e%d_rtr"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.num_experts))
    logger.setLevel(logging.DEBUG if params.verbose else logging.INFO)
    if not logger.hasHandlers():
        # create file handler which logs even debug messages
        fh = logging.FileHandler(join(params.log_dir, "%s_%s_%s_e%d_rtr.log"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.num_experts)))
        fh.setLevel(logging.DEBUG if params.verbose else logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d/%(threadName)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger



def train(params):
    
    router_trainer = RouterTrainer(params)
    router_trainer.start()
    router_trainer.join()


def predict(params):
    worknodes = []
        
    data_reader = DataReader("reader", None, params)
    worknodes.append(data_reader)
    
    stale_observer = StaleObserver(params, c_queue = data_reader.r_queue)
    worknodes.append(stale_observer)
    
    for worknode in worknodes:
        worknode.start()
    
    
    
    params.logger.info("creating model sessions")
    model_name = params.model
    
    router = Expert(model_name+"_rtr", None, params)
    router.start()
    router.r_queue.get()

    accs = []
    ts = []
    
    while True:
        batch = stale_observer.r_queue.get()
        
        if batch is None:
            break
        [batch_xs, _], uncs = batch
        
        batch_ys = np.argmin(uncs, axis = 1).reshape(-1)
        #batch_ys = np.eye(params.num_experts)[batch_ys]

        router.c_queue.put(["predict",[batch_xs, batch_ys]])

        acc, _, t = router.r_queue.get()
        accs.append(acc)
        
        
        ts.append(t)
    
    
    accs = np.concatenate(accs, axis=0)
    

    router.c_queue.put(["exit"])
    router.r_queue.get()
    router.join()

    params.logger.info('%s: precision: %.3f , elapsed time: %.3f ms' % (datetime.now(), np.mean(accs), 
                                                               1e3*np.mean(ts)))
    
    
    for worknode in worknodes:
        worknode.join()

def main(params):   
    
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    if not os.path.exists(params.tmp_dir):
        os.makedirs(params.tmp_dir)
    if not os.path.exists(params.data_dir):
        os.makedirs(params.data_dir)
    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
        
        
    max_epochs = params.max_epochs
    start_epoch = get_ckpt_iters(params)*params.batch_size//params.train_dataset_size
    
        
    #logging.basicConfig(filename=join(params.log_dir, "%s_%s_e%d.log"%(params.mode, params.hparams, params.num_experts)),level=logging.INFO)
    if params.logger is None:
        params.logger = get_logger(params)

    if params.mode == "train":
        if max_epochs-start_epoch > 0: 
            train(params)
    if params.mode == "predict":
        if max_epochs-start_epoch >= 0: 
            predict(params)

if __name__ == '__main__':
    main()