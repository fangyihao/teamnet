'''
Created on Jul 18, 2018

@author: yfang
'''

from multiprocessing import Process, Manager
from gate import Gate, Meta
import time
import os
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, isdir, join
import re
import logging
from work_node import WorkNode
import socket
import pickle
from utility import SysUtilLogger
from struct import *
'''
model.set_mode(Modes.EVAL)
cifar10_eval_dataset = cifar10_problem.dataset(Modes.EVAL, data_dir)

# Create eval metric accumulators for accuracy (ACC) and accuracy in
# top 5 (ACC_TOP5)
metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5])

for count, example in enumerate(tfe.Iterator(cifar10_eval_dataset)):
  
  if count >= 200:
    break
  
  # Make the inputs and targets 4D
  example["inputs"] = tf.reshape(example["inputs"], [1, 28, 28, 1])
  example["targets"] = tf.reshape(example["targets"], [1, 1, 1, 1])

  # Call the model
  predictions, _ = model(example)

  # Compute and accumulate metrics
  metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
  print("%s: %.2f" % (name, val))
'''  
class GPUtil (Process):
    def __init__(self):
        Process.__init__(self)
        self.manager = Manager()
        self.r_queue = self.manager.Queue(1)
        
    def run(self):
        self.r_queue.put(len(self.get_available_gpus()))
        
    def get_available_gpus(self):
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

gputil = GPUtil()
gputil.start()
NUM_GPUS = gputil.r_queue.get()
gputil.join()


def get_ckpt_iters(params):
    ds = [d for d in listdir(params.model_dir) if isdir(join(params.model_dir, d))]
    all_num_iters = []
    for d in ds:
        if re.search('_e([0-9]+)',d):
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
  

class Expert (Process):
    def __init__(self, name, device, params):
        Process.__init__(self)
        
        self.name = name
        self.device = device
        self.params = params
        
        self.manager = Manager()
        self.c_queue = self.manager.Queue(1)
        self.r_queue = self.manager.Queue(1)


    def run(self):
        self.params.logger.info("begin Expert")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id
        import tensorflow as tf
        from tensor2tensor.utils import trainer_lib
        from tensor2tensor.utils import registry
        from tensor2tensor import problems
        from tensor2tensor.utils import t2t_model
        from tensorflow.python.client import session
        from tensorflow.python.training import saver as saver_mod
        
        def get_predictive_entropy(probs):
            probs = np.maximum(probs, 1e-37)
            entropy = -np.sum(probs * np.log(probs), axis = 1)
            return entropy  
        """Train CIFAR-10 for a number of steps."""
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
    
            inputs = tf.placeholder(tf.float32, shape=self.params.input_shape)
            targets = tf.placeholder(tf.int32, shape=(None))

            problem = problems.problem(self.params.problem)
            # Create your own model
            hparams = trainer_lib.create_hparams(self.params.hparams, data_dir=self.params.data_dir, problem_name=self.params.problem)

            Modes = tf.estimator.ModeKeys
           
            if self.params.model_cls is not None:
                model = self.params.model_cls(hparams, Modes.TRAIN)
            else:
                model = registry.model(self.params.model)(hparams, Modes.TRAIN)

            example = {}
            example["inputs"] = inputs
            example["targets"] = targets
            example["targets"] = tf.reshape(example["targets"], [-1, 1, 1, 1])
            
            logits, losses_dict = model(example)
            logits = tf.reshape(logits, (-1, logits.get_shape()[-1]))
            
            
            
            # Accumulate losses
            loss = tf.add_n([losses_dict[key] for key in sorted(losses_dict.keys())])
            
            if "shakeshake" in self.params.hparams:
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(loss, global_step)
            else:
                train_op = model.optimize(loss)
    
            probs = tf.nn.softmax(logits, axis=1)
            

            params = self.params
    
            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""
    
                def begin(self):
                    self._start_time = time.time()
    
                def before_run(self, run_context):
                    
                    return tf.train.SessionRunArgs([loss, global_step])    # Asks for loss value.
    
                def after_run(self, run_context, run_values):
                    loss_value, global_step_value = run_values.results
                    if global_step_value % params.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
    
                        examples_per_sec = params.log_frequency * params.batch_size / duration
                        sec_per_batch = float(duration / params.log_frequency)
    
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                                    'sec/batch)')
                        params.logger.info (format_str % (datetime.now(), global_step_value, loss_value,
                                                                 examples_per_sec, sec_per_batch))
            
            
            
            saver = tf.train.Saver()

            scaffold = tf.train.Scaffold(saver=saver)
            
            checkpoint_dir = self.params.model_dir + "/" + self.name
            saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir, save_steps=self.params.stale_interval, scaffold=scaffold)
                   
            '''
            if self.device is None:
                config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, 
                        allow_soft_placement=True, device_count = {'CPU': 1},
                        log_device_placement=self.params.log_device_placement)
            else:
            '''
            config = tf.ConfigProto(log_device_placement=self.params.log_device_placement, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction=0.42
            #tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            
            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=checkpoint_dir,
                    scaffold=scaffold,
                    hooks=[saver_hook, 
                           tf.train.NanTensorHook(loss),
                                 _LoggerHook()
                                 ],
                    config=config,
                    save_checkpoint_secs=None,
                    save_summaries_secs=None, 
                    save_summaries_steps=None, 
                    ) as mon_sess:

                self.r_queue.put("ready")
                sess_exit = False
                while not sess_exit:
                    command = self.c_queue.get()
                    if command[0] == "train":
                        if command[1] is not None:
                            batch_xs, batch_ys = command[1]
                            mon_sess.run(train_op, {inputs: batch_xs, targets: batch_ys})

                            self.r_queue.put(None)
                            
                    if command[0] == "predict":
                        if command[1] is not None:       
                            bt = time.time()
                            
                            batch_xs, batch_ys = command[1] 
                            
                            ps_sum = None
                            for _ in range(self.params.mc_steps):
                                ps = mon_sess.run(probs, {inputs: batch_xs, targets: batch_ys})
                                ps_sum = (ps if ps_sum is None else ps_sum + ps)
                        
                            ps = ps_sum/self.params.mc_steps
                            
                            unc = get_predictive_entropy(ps)
                            
                            acc = np.equal(np.argmax(ps, axis=1), np.reshape(batch_ys, -1))
                            
                            et = time.time()
                            t = et - bt
                            
                            self.r_queue.put([acc, unc, t])
                    
                    elif command[0] == "restore":
                        '''
                        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            self.params.logger.info("Restores from checkpoint: %s" % ckpt.model_checkpoint_path)
                            saver.restore(mon_sess, ckpt.model_checkpoint_path) 
                        '''
                        
                        #mon_sess._coordinated_creator._session_creator._get_session_manager()._restore_checkpoint(mon_sess._coordinated_creator._session_creator._master, mon_sess._coordinated_creator._session_creator._scaffold._saver, checkpoint_dir)    
                        
                        session_creator = mon_sess._coordinated_creator._session_creator
                        session_manager = session_creator._get_session_manager()
                        
                        session_manager._target = session_creator._master
                        sess = session.Session(session_manager._target, graph=session_manager._graph, config=config)
                        
                        ckpt = saver_mod.get_checkpoint_state(checkpoint_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            self.params.logger.info("Restores from checkpoint: %s" % ckpt.model_checkpoint_path)
                            # Loads the checkpoint.
                            session_creator._scaffold._saver.restore(sess, ckpt.model_checkpoint_path)    
                            
                        self.r_queue.put(None)     
                    elif command[0] == "exit":
                        sess_exit = True
                        self.r_queue.put(None)  
        
        self.params.logger.info("end Expert")
                        
    
    
def get_device_id(device, i):
    if device is None:
        device_id = i%NUM_GPUS
    else:
        device_id = device[i%len(device)]
    return device_id



class MoETrainer(WorkNode):
    '''
    def __init__(self, params):
        Process.__init__(self)
        self.params = params
    '''    

    def run(self):
        self.params.logger.info("begin MoETrainer")
        self.worknodes = []
        
        data_reader = DataReader("reader", None, self.params)
        self.worknodes.append(data_reader)
        stale_observer = StaleObserver(self.params, c_queue = data_reader.r_queue)
        self.worknodes.append(stale_observer)
        
        gate_committee = GateCommittee(self.params, c_queue = stale_observer.r_queue, num_gates=self.params.num_gates)
        self.worknodes.append(gate_committee)
        
        trainer = ExpertTrainer(self.params, c_queue = gate_committee.r_queue)
        self.worknodes.append(trainer)
        
        for worknode in self.worknodes:
            worknode.start()
        
        for worknode in self.worknodes:
            worknode.join()

        self.params.logger.info("end MoETrainer")

class ExpertTrainer(WorkNode):

    def run(self):
        self.params.logger.info("begin ExpertTrainer")
        model_name = self.params.model
        
        experts = []
        for i in range(self.params.num_experts):
            expert = Expert(model_name + "_e%d" % i, "/device:GPU:%d"%get_device_id(self.params.device, i), self.params)
            experts.append(expert)
            expert.start()
            expert.r_queue.get()
         
        while True:   
            self.params.logger.debug("begin ExpertTrainer c_queue.get")
            batch = self.c_queue.get()
            self.params.logger.debug("end ExpertTrainer c_queue.get")
            if batch is None:
                break
            
            [batch_xs, batch_ys], inds = batch
            
            for i, expert in enumerate(experts): 
                if len(inds[i]) > 0:
                    expert.c_queue.put(["train",[batch_xs[inds[i]], batch_ys[inds[i]]]])
                    
    
            for i, expert in enumerate(experts): 
                if len(inds[i]) > 0:
                    expert.r_queue.get()
                        
        for expert in experts: 
            expert.c_queue.put(["exit"])
            expert.r_queue.get()
            expert.join()   
            
        self.params.logger.info("end ExpertTrainer") 





        
class GateCommittee(WorkNode):
    def __init__(self, params, c_queue = None, r_queue = None, num_gates = 4):
        super().__init__(params, c_queue, r_queue)
        self.num_gates = num_gates
        
        
    def run(self):
        self.params.logger.info("begin GateCommittee")
        
        self.worknodes = []
        gates = []
        self.manager = Manager()
        for i in range(self.num_gates):
            gate_caller = GateCaller(self.params, c_queue = self.manager.Queue(self.params.stale_interval//self.num_gates), r_queue = self.r_queue, id = i)
            self.worknodes.append(gate_caller)
            gates.append(gate_caller)
            
        gate_dispatcher = GateDispatcher(self.params, c_queue = self.c_queue, gates = gates)
        self.worknodes.append(gate_dispatcher)

        for worknode in self.worknodes:
            worknode.start()
        
        for worknode in self.worknodes:
            worknode.join()
            
        self.r_queue.put(None)
        self.params.logger.info("end GateCommittee")

class GateDispatcher(WorkNode):
    def __init__(self, params, c_queue = None, r_queue = None, gates = None):
        super().__init__(params, c_queue, r_queue)
        self.gates = gates
    def run(self):
        self.params.logger.info("begin GateDispatcher")
        
        
        qsizes = []
        while True:
            batch = self.c_queue.get()
            
            if batch is None:
                self.params.logger.info("queue sizes: %s"%str(qsizes))
                for gate in self.gates:
                    self.params.logger.debug("begin gate.c_queue.put")
                    gate.c_queue.put(None)
                    self.params.logger.debug("end gate.c_queue.put")
                break
            
            qsizes = []
            for gate in self.gates:
                qsizes.append(gate.c_queue.qsize())
            
            self.gates[np.argmin(qsizes)].c_queue.put(batch)
        
        self.params.logger.info("#queue sizes: %s"%str(qsizes))
        self.params.logger.info("end GateDispatcher")

class GateCaller(WorkNode):
    def __init__(self, params, c_queue = None, r_queue = None, id = 0):
        super().__init__(params, c_queue, r_queue)
        self.id = id
        
    def run(self):
        self.params.logger.info("begin GateCaller %d" % self.id)
        #gate = None
        
        meta = Meta("meta_%d" % self.id, "/device:GPU:%d"%get_device_id(self.params.gate_device, self.id), self.params)
        meta.start()
        
        gate = Gate("/device:GPU:%d"%get_device_id(self.params.gate_device, self.id), self.params, meta)
        gate.start()
        self.params.logger.debug("begin GateCaller %d ready" % self.id)
        gate.r_queue.get() 
        self.params.logger.debug("end GateCaller %d ready" % self.id)
        
        bi = 0
        while True:
            self.params.logger.debug("begin GateCaller %d c_queue.get" % self.id)
            batch = self.c_queue.get()
            self.params.logger.debug("end GateCaller %d c_queue.get" % self.id)
        
            if batch is None:
                #self.r_queue.put(None)
                break
            [batch_xs, batch_ys], uncs = batch
            
            #if bi % (10*self.params.stale_interval) == 0:   
            if gate is not None:
                print("GateCaller - gate failure: %d"%gate.failure.value)
             
            if gate is not None and gate.failure.value >= 3: 
                if gate is not None:
                    gate.c_queue.put(["exit"])
                    gate.r_queue.get()
                    gate.join()
                gate = Gate("/device:GPU:%d"%get_device_id(self.params.gate_device, self.id), self.params, meta)
                gate.start()
                self.params.logger.debug("begin GateCaller %d ready" % self.id)
                gate.r_queue.get() 
                self.params.logger.debug("end GateCaller %d ready" % self.id)
            
            
            self.params.logger.debug("begin GateCaller %d train" % self.id)
            gate.c_queue.put(["train",uncs])
            inds = gate.r_queue.get() 
            self.params.logger.debug("end GateCaller %d train" % self.id)
             
            self.params.logger.debug("begin GateCaller %d r_queue.put" % self.id)
            batch = self.r_queue.put([[batch_xs, batch_ys],inds])
            self.params.logger.debug("end GateCaller %d r_queue.put" % self.id)
            bi += 1
            
        if gate is not None:    
            gate.c_queue.put(["exit"])
            gate.r_queue.get()
            gate.join()
            
        meta.c_queue.put(["exit"])
        meta.r_queue.get()
        meta.join()
        
        self.params.logger.info("end GateCaller %d" % self.id)

class StaleObserver(WorkNode):
    
    def run(self):
        self.params.logger.info("begin StaleObserver")
        model_name = self.params.model
        
        stale_experts = []
        #gate = None
        bi = 0
        while True:
            batch = self.c_queue.get()
        
            if batch is None:
                self.r_queue.put(None)
                break
            batch_xs, batch_ys = batch
            
            if bi % (10 *self.params.stale_interval) == 0:
                self.params.logger.info("creating stale experts")
                for stale_expert in stale_experts: 
                    stale_expert.c_queue.put(["exit"])
                    stale_expert.r_queue.get()
                    stale_expert.join()
                stale_experts = []
                for i in range(self.params.num_experts):
                    stale_expert = Expert(model_name + "_e%d" % i, "/device:GPU:%d"%get_device_id(self.params.device, i), self.params)
                    stale_experts.append(stale_expert)
                    stale_expert.start()
                    stale_expert.r_queue.get()
                    
            if bi % self.params.stale_interval == 0 and bi % (10 *self.params.stale_interval) != 0:
                for stale_expert in stale_experts: 
                    stale_expert.c_queue.put(["restore"])
                for stale_expert in stale_experts: 
                    stale_expert.r_queue.get()
            
            for stale_expert in stale_experts: 
                stale_expert.c_queue.put(["predict",[batch_xs, batch_ys]])
            
            accs = []
            uncs = []
            for expert in stale_experts: 
                acc, unc,_ = expert.r_queue.get()
                accs.append(acc)
                uncs.append(unc)
            if bi % self.params.log_frequency == 0:
                self.params.logger.info("accuracy: %f" % np.mean(accs))  
            uncs = np.transpose(uncs)
          
            self.r_queue.put([batch,uncs])
            
            bi += 1

        for stale_expert in stale_experts: 
            stale_expert.c_queue.put(["exit"])
            stale_expert.r_queue.get()
            stale_expert.join()
       
        self.params.logger.info("end StaleObserver")
'''
num_experts must be 1
'''
def fast_train(params):
    reader = DataReader("reader", None, params)
    reader.start()
    
    params.logger.info("creating model sessions")
    model_name = params.model

    expert = Expert(model_name + "_e%d" % 0, "/device:GPU:%d"%get_device_id(params.device, 0), params)
    expert.start()
    expert.r_queue.get()
    
    while True:
        values = reader.r_queue.get()
        
        if values is None:
            break
        batch_xs, batch_ys = values
        expert.c_queue.put(["train",[batch_xs, batch_ys]])
        expert.r_queue.get()
           
    expert.c_queue.put(["exit"])
    expert.r_queue.get()
    expert.join()
  

def train(params):
    
    moe_trainer = MoETrainer(params)
    moe_trainer.start()
    moe_trainer.join()
           
    
'''
num_experts must be 1
'''
def fast_predict(params):
    reader = DataReader("reader", None, params)
    reader.start()
    
    params.logger.info("creating model sessions")
    model_name = params.model
    
    expert = Expert(model_name + "_e%d" % 0, None, params)
    
    expert.start()
    expert.r_queue.get()

    all_accs = []
    all_uncs = []
    all_ts = []
    while True:
        values = reader.r_queue.get()
        
        if values is None:
            break
        batch_xs, batch_ys = values
        
        expert.c_queue.put(["predict",[batch_xs, batch_ys]])

        acc, unc, t = expert.r_queue.get()
        all_accs.append(acc)
        all_uncs.append(unc)
        all_ts.append(t)
        params.logger.info('elapsed time: %.3f ms' % t)
    
    expert.c_queue.put(["exit"])
    expert.r_queue.get()
    expert.join()
    
    all_accs = np.concatenate(all_accs, axis=0)
    all_uncs = np.concatenate(all_uncs, axis=0)

    
    params.logger.info('%s: precision: %.3f , elapsed time: %.3f ms' % (datetime.now(), np.mean(all_accs), 
                                                               1e3*np.sum(all_ts)/len(all_accs)))
        

def predict(params):
    reader = DataReader("reader", None, params)
    reader.start()
    
    params.logger.info("creating model sessions")
    model_name = params.model
    
    experts = []
    for i in range(params.num_experts):
        expert = Expert(model_name + "_e%d" % i, None, params)
        experts.append(expert)
        expert.start()
        expert.r_queue.get()

    
    all_accs = []
    all_uncs = []
    all_ts = []
    all_labels = []
    
    i = 0
    while True:
        values = reader.r_queue.get()
        
        if values is None:
            break
        batch_xs, batch_ys = values
        
        # warm up
        if i == 0: 
            for expert in experts: 
                expert.c_queue.put(["predict",[batch_xs, batch_ys]])
                expert.r_queue.get()
        
        accs = []
        uncs = []
        ts = []
        labels = []
        for expert in experts: 
            expert.c_queue.put(["predict",[batch_xs, batch_ys]])

            acc, unc, t = expert.r_queue.get()
            accs.append(acc)
            uncs.append(unc)
            ts.append(t)
            labels.append(np.reshape(batch_ys, -1))

        all_accs.append(np.transpose(accs))
        all_uncs.append(np.transpose(uncs))
        all_ts.append(ts)
        all_labels.append(np.transpose(labels))
        
        i += 1
        
    for expert in experts: 
        expert.c_queue.put(["exit"])
        expert.r_queue.get()
        expert.join()
    

    all_accs = np.concatenate(all_accs, axis=0)
    all_uncs = np.concatenate(all_uncs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    for i in range(params.num_experts):
        params.logger.info('%s: precision (expert %d): %.3f' % (datetime.now(), i+1, np.mean(all_accs, axis=0)[i]))
    
    '''
    final_accs = []
    for i in range(len(all_uncs)):
        j = np.argmin(all_uncs[i])
        final_accs.append(all_accs[i][j])
    '''
    final_accs = cal_accuracy(all_uncs,all_accs)
        
    params.logger.info('%s: precision: %.3f , elapsed time: %.3f ms' % (datetime.now(), np.mean(final_accs), 
                                                               1e3*np.max(np.sum(all_ts, axis=0))/len(final_accs)))
    
    # calculate drop one, two, ... accuracies
    import itertools
    for r in range(1,params.num_experts):
        drop_accs = []
        for t in itertools.combinations(range(params.num_experts), r):
            tmp_uncs = all_uncs[:,np.array(t)]
            tmp_accs = all_accs[:,np.array(t)]
            drop_accs.append(cal_accuracy(tmp_uncs,tmp_accs))
        drop_accs = np.transpose(drop_accs)
        mean_drop_acc = np.mean(drop_accs,axis=1)
        std_drop_acc = np.std(drop_accs, axis=1)
        params.logger.info('%s: drop %d precision: mean: %.3f std: %.3f' %(datetime.now(), params.num_experts - r, np.mean(mean_drop_acc), np.mean(std_drop_acc)) )   
    
    if params.num_experts >= 2:
        analyze(all_accs, all_uncs, all_labels, params)


def split_and_send(conn, data, socket_buffer_size):
    if data is None:
        data_size = 0
        conn.send(pack("I", data_size))
    else:
        raw_data = pickle.dumps(data)
        data_size = len(raw_data)
        conn.send(pack("I", data_size))
        packets = [raw_data[i * socket_buffer_size: (i + 1)* socket_buffer_size] for i in range(data_size // socket_buffer_size +1)]
        for packet in packets:
            conn.send(packet)
        
def recv_and_concat(conn, socket_buffer_size):
    raw_data = b''
    data_size = unpack("I",(conn.recv(calcsize("I"))))[0]
    
    while len(raw_data) < data_size:
        buf = conn.recv(socket_buffer_size)
        raw_data += buf
                
    if len(raw_data) == 0:
        data = None
    else:
        data = pickle.loads(raw_data)
    return data

    

class ServerSocketHandler(WorkNode):
    def __init__(self, params, addr):
        super().__init__(params)
        self.addr = addr
    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(self.addr)
        server_socket.listen()
        try:
            conn, addr = server_socket.accept()
            print('Connection address:', addr)
            
            while True:
                Command = self.c_queue.get()
                if Command is None: 
                    break
                if Command[0] == "send":
                    data = Command[1]
                    split_and_send(conn, data, self.params.socket_buffer_size)
                    self.r_queue.put(None)
                    
                elif Command[0] == "recv":
                    data = recv_and_concat(conn, self.params.socket_buffer_size)
                    
                    self.r_queue.put(data)
                    
                
        finally:
            conn.close()
            server_socket.close()

class ClientSocketHandler(WorkNode):
    def __init__(self, params, addr):
        super().__init__(params)
        self.addr = addr
    def run(self):
        
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            conn.connect(self.addr)
            
            while True:
                Command = self.c_queue.get()
                if Command is None: 
                    break
                if Command[0] == "send":
                    data = Command[1]
                    split_and_send(conn, data, self.params.socket_buffer_size)
                    self.r_queue.put(None)
                    
                elif Command[0] == "recv":
                    
                    data = recv_and_concat(conn, self.params.socket_buffer_size)
                    
                    self.r_queue.put(data)
                    
        finally:
            conn.close()

def co_predict(params):
    '''  
    def split_and_broadcast(conns, raw_data):
        data_size = len(raw_data)
        packets = [raw_data[i * params.socket_buffer_size: (i + 1)* params.socket_buffer_size] for i in range(data_size // params.socket_buffer_size +1)]
        for conn in conns:
            conn.send(pickle.dumps(data_size))
            for packet in packets:
                conn.send(packet)
    '''        
    
    
    params.logger.info("creating model sessions")
    model_name = params.model
    
    

    if params.socket_node_index == 0:
        
        expert = Expert(model_name + "_e%d" % params.socket_node_index, None, params)
        expert.start()
        expert.r_queue.get()
        
        try:
            
            client_handlers = []
            for k in range(params.num_experts):
                if k == 0:
                    client_handlers.append(None)
                else:
                    ip, port = params.socket_nodes[k].split(':')
                    addr = (ip, int(port))
                    client_handler = ClientSocketHandler(params, addr)
                    client_handler.start()
                    client_handlers.append(client_handler)
        
            reader = DataReader("reader", None, params)
            reader.start()

        
            all_accs = []
            all_uncs = []
            all_tts = []
            all_pts = []
            all_cts = []
            all_labels = []

            i = 0
            while True:
                values = reader.r_queue.get()
                
                if values is None:
                    break
                batch_xs, batch_ys = values

                # warm up
                if i == 0: 
                    expert.c_queue.put(["predict",[batch_xs, batch_ys]])
                    
                    for client_handler in client_handlers[1:]:
                        client_handler.c_queue.put(["send",["predict",[batch_xs, batch_ys]]])    
                    
                    for client_handler in client_handlers[1:]:
                        client_handler.r_queue.get()   
                        
                    expert.r_queue.get()   
                        
                    for client_handler in client_handlers[1:]:
                        client_handler.c_queue.put(["recv"]) 
                        
                    for client_handler in client_handlers[1:]:
                        client_handler.r_queue.get()
                        
                
                accs = []
                uncs = []
                tts = []
                pts = []
                cts = []
                labels = []
                
                start_time = time.time()
                
                expert.c_queue.put(["predict",[batch_xs, batch_ys]])
                
                for client_handler in client_handlers[1:]:
                    client_handler.c_queue.put(["send",["predict",[batch_xs, batch_ys]]])
                for client_handler in client_handlers[1:]:
                    client_handler.r_queue.get()
                     
                acc, unc, t = expert.r_queue.get()
                accs.append(acc)
                uncs.append(unc)
                pts.append(t)        
                for client_handler in client_handlers[1:]:
                    client_handler.c_queue.put(["recv"])
                for client_handler in client_handlers[1:]:
                    acc, unc, t = client_handler.r_queue.get()
                    accs.append(acc)
                    uncs.append(unc)
                    pts.append(t)
                        
                for k in range(params.num_experts):
                    labels.append(np.reshape(batch_ys, -1))

        
                end_time = time.time()
                
                for k in range(params.num_experts):
                    tts.append(end_time-start_time)
                    cts.append(end_time-start_time-pts[k])
        
                all_accs.append(np.transpose(accs))
                all_uncs.append(np.transpose(uncs))
                all_tts.append(tts)
                all_pts.append(pts)
                all_cts.append(cts)
                all_labels.append(np.transpose(labels))
                
                i += 1
                
        finally:
            for client_handler in client_handlers:
                if client_handler is not None:
                    client_handler.c_queue.put(["send", None])
                    client_handler.r_queue.get()
                    client_handler.c_queue.put(None)
                    client_handler.join()
 
     
        all_accs = np.concatenate(all_accs, axis=0)
        all_uncs = np.concatenate(all_uncs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        for i in range(params.num_experts):
            params.logger.info('%s: precision (expert %d): %.3f' % (datetime.now(), i+1, np.mean(all_accs, axis=0)[i]))
        
        
        final_accs = cal_accuracy(all_uncs,all_accs)
            
        params.logger.info('%s: precision: %.3f , total time: %.3f ms, computation time: %.3f ms, communication time: %.3f ms' % (datetime.now(), np.mean(final_accs), 
                                                                   1e3*np.max(np.sum(all_tts, axis=0))/len(final_accs), 
                                                                   1e3*np.max(np.sum(all_pts, axis=0))/len(final_accs),
                                                                   1e3*np.max(np.sum(all_cts, axis=0))/len(final_accs)))
        
        # calculate drop one, two, ... accuracies
        import itertools
        for r in range(1,params.num_experts):
            drop_accs = []
            for t in itertools.combinations(range(params.num_experts), r):
                tmp_uncs = all_uncs[:,np.array(t)]
                tmp_accs = all_accs[:,np.array(t)]
                drop_accs.append(cal_accuracy(tmp_uncs,tmp_accs))
            drop_accs = np.transpose(drop_accs)
            mean_drop_acc = np.mean(drop_accs,axis=1)
            std_drop_acc = np.std(drop_accs, axis=1)
            params.logger.info('%s: drop %d precision: mean: %.3f std: %.3f' %(datetime.now(), params.num_experts - r, np.mean(mean_drop_acc), np.mean(std_drop_acc)) )   
        
        if params.num_experts >= 2:
            analyze(all_accs, all_uncs, all_labels, params)
            
            
        expert.c_queue.put(["exit"])
        expert.r_queue.get()
        expert.join()
            
    else:
        addr = ('0.0.0.0', int(params.socket_nodes[params.socket_node_index].split(":")[1]))
        server_handler = ServerSocketHandler(params, addr)
        server_handler.start()
        
        expert = Expert(model_name + "_e%d" % params.socket_node_index, None, params)
        expert.start()
        expert.r_queue.get() 
        
        try:
            
            
            while True:
                
                server_handler.c_queue.put(["recv"])
                
                data = server_handler.r_queue.get()

                if data is None:
                    break
                
                expert.c_queue.put(data)
                acc, unc, t = expert.r_queue.get()
                
                server_handler.c_queue.put(["send",[acc, unc, t]])
                
                server_handler.r_queue.get()

        finally:
            
            server_handler.c_queue.put(None)
            server_handler.join()

        
        expert.c_queue.put(["exit"])
        expert.r_queue.get()
        expert.join()

    

'''
def co_predict(params):

    params.logger.info("creating model sessions")
    model_name = params.model

    experts = []
    all_accs = []
    all_uncs = []
    all_ts = []
    all_labels = []
    for i in range(params.num_experts):
        reader = DataReader("reader", None, params)
        reader.start()

        #expert = Expert(model_name + "_e%d" % i, "/device:GPU:0", params)
        expert = Expert(model_name + "_e%d" % i, None, params)
        experts.append(expert)
        expert.start()
        expert.r_queue.get()

        accs = []
        uncs = []
        ts = []
        labels = []

        i = 0
        while True:
            values = reader.r_queue.get()

            if values is None:
                break
            batch_xs, batch_ys = values

            # warm up
            if i == 0:
                expert.c_queue.put(["predict",[batch_xs, batch_ys]])
                expert.r_queue.get()

            expert.c_queue.put(["predict",[batch_xs, batch_ys]])

            acc, unc, t = expert.r_queue.get()
            accs.append(acc)
            uncs.append(unc)
            ts.append(t)
            labels.append(np.reshape(batch_ys, -1))

            i += 1

        expert.c_queue.put(["exit"])
        expert.r_queue.get()
        expert.join()

        all_accs.append(np.concatenate(accs, axis=0))
        all_uncs.append(np.concatenate(uncs, axis=0))
        all_ts.append(ts)
        all_labels.append(np.concatenate(labels, axis=0))

    all_accs = np.transpose(all_accs)
    all_uncs = np.transpose(all_uncs)
    all_ts = np.transpose(all_ts)
    all_labels = np.transpose(all_labels)

    for i in range(params.num_experts):
        params.logger.info('%s: precision (expert %d): %.3f' % (datetime.now(), i+1, np.mean(all_accs, axis=0)[i]))

    
    final_accs = cal_accuracy(all_uncs,all_accs)

    params.logger.info('%s: precision: %.3f , elapsed time: %.3f ms' % (datetime.now(), np.mean(final_accs),
                                                               1e3*np.max(np.sum(all_ts, axis=0))/len(final_accs)))

    # calculate drop one, two, ... accuracies
    import itertools
    for r in range(1,params.num_experts):
        drop_accs = []
        for t in itertools.combinations(range(params.num_experts), r):
            tmp_uncs = all_uncs[:,np.array(t)]
            tmp_accs = all_accs[:,np.array(t)]
            drop_accs.append(cal_accuracy(tmp_uncs,tmp_accs))
        drop_accs = np.transpose(drop_accs)
        mean_drop_acc = np.mean(drop_accs,axis=1)
        std_drop_acc = np.std(drop_accs, axis=1)
        params.logger.info('%s: drop %d precision: mean: %.3f std: %.3f' %(datetime.now(), params.num_experts - r, np.mean(mean_drop_acc), np.mean(std_drop_acc)) )

    if params.num_experts >= 2:
        analyze(all_accs, all_uncs, all_labels, params)
'''
    
def cal_accuracy(uncs,accs):
    final_accs = []
    for i in range(len(uncs)):
        j = np.argmin(uncs[i])
        final_accs.append(accs[i][j])
    return np.array(final_accs)
        
def analyze(all_accs, all_uncs, all_labels, params):  

    all_bins = np.zeros([10])
    
    for i, lbl in enumerate(all_labels):
        all_bins[lbl] += 1
        
    params.logger.info("#examples:\t%s"% str(all_bins))
    
    if params.num_experts == 2:

        both = np.zeros([10])
        neither = np.zeros([10])
        e1 = np.zeros([10])
        e2 = np.zeros([10])
        
        for i, lbl in enumerate(all_labels):
            if all_accs[i][0]==1 and all_accs[i][1]==1:
                both[lbl] += 1
            elif all_accs[i][0]==0 and all_accs[i][1]==0:
                neither[lbl] += 1
            elif all_accs[i][0]==1 and all_accs[i][1]==0:
                e1[lbl] += 1
            elif all_accs[i][0]==0 and all_accs[i][1]==1:
                e2[lbl] += 1
        params.logger.info("accurate prediction:")        
        params.logger.info("both:\t%s" % str(both))
        params.logger.info("neither:\t%s" % str(neither))
        params.logger.info("expert 1:\t%s" % str(e1))
        params.logger.info("expert 2:\t%s" % str(e2))
    
    
    least_uncertain = np.zeros([params.num_experts, 10])
        
    for i, lbl in enumerate(all_labels):
        j = np.argmin(all_uncs[i])
        least_uncertain[j][lbl] += 1
    
    params.logger.info("which expert is least uncertain?")    
    for i in range(params.num_experts):
        params.logger.info("expert %d:\t%s" % ((i+1), str(least_uncertain[i])))
    
def get_logger(params):
    logger = logging.getLogger("%s_%s_%s_e%d"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.num_experts))
    logger.setLevel(logging.DEBUG if params.verbose else logging.INFO)
    if not logger.hasHandlers():
        # create file handler which logs even debug messages
        fh = logging.FileHandler(join(params.log_dir, "%s_%s_%s_e%d.log"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.num_experts)))
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
            if params.num_experts == 1:
                fast_train(params)
            else:
                train(params)
    if params.mode == "predict":
        if max_epochs-start_epoch >= 0: 
            
            util_Logger = SysUtilLogger(params = params, task = "%s_%s_%s_e%d_sys"%(params.mode, params.hparams, 'rs' if params.reshuffle_each_epoch else 'nrs', params.num_experts) )
            util_Logger.start()
            util_Logger.r_queue.get()
            
            if params.num_experts == 1:
                predict(params)
            else:
                if params.socket_nodes is not None:
                    co_predict(params)
                else:
                    predict(params)
                    
            util_Logger.c_queue.put(["exit"])
            util_Logger.join()
            

if __name__ == '__main__':
    main()