'''
Created on May 25, 2018

@author: fangy5
'''
import numpy as np
import os
from multiprocessing import Process, Manager
#from scipy.constants.constants import golden


class Gate(Process):
    def __init__(self, device, params, meta):
        Process.__init__(self)
        self.manager = Manager()
        self.c_queue = self.manager.Queue(1)
        self.r_queue = self.manager.Queue(1)
        self.device = device
        self.params = params
        
        self.meta = meta
        
        self.failure = self.manager.Value('i', 0)
        
    def run(self):
        self.params.logger.info("begin Gate")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id
        
        #self.meta = Meta(self.device, self.params)
        #self.meta.start()
        
        self.train()
        self.params.logger.info("end Gate")
    
    
    def train(self):

        import tensorflow as tf
        
        
        def softargmin(logits, axis, b = 3e2): #3e1

            return tf.reduce_sum((tf.cumsum(tf.ones_like(-logits),axis)-1)*tf.nn.softmax(b*(-logits), axis), axis)
    
        
        def softround(x, stop = 32):
            s = 0
            for k in range(1,stop + 1):
                s = s + tf.sin(2*np.pi*k*(x+0.5))/k
            return x+(1/np.pi)*s   
        
        
        k = self.params.num_experts
        unc = tf.placeholder(dtype=tf.float64,shape=(None, k))
        b = tf.placeholder(dtype=tf.float64, shape=())
        
        G_beta = tf.argmin(unc, axis=1)
        def f_G():
            # count n_1, n_2, ..., n_k
            n_G = tf.reduce_sum(tf.one_hot(G_beta,k,  dtype=tf.float64), axis=0)

            N_G = tf.reduce_sum(n_G)
            # calculate f_1, f_2, ..., f_k
            f_G = n_G / N_G
            return f_G
        
        def momentum(f_G, p = 0.1):  #1/golden
            p = tf.constant(p, dtype=tf.float64) 
            lb = tf.constant(-1/k, dtype=tf.float64)
            ub = tf.constant(1/k, dtype=tf.float64)
            momentum = tf.minimum(tf.maximum(f_G - 1/k, lb), ub) * p
            return momentum
        
        
        '''
        
        num_noise = k * 32
        #noise = tf.random_normal(shape=(1, num_noise), mean=0.0, stddev=0.5, dtype=tf.float64)
        noise = tf.random_uniform((1, num_noise),minval=-1/tf.sqrt(tf.cast(num_noise,tf.float64)),maxval=1/tf.sqrt(tf.cast(num_noise,tf.float64)), dtype=tf.float64)
        w_1 = tf.get_variable("w_1", shape = (num_noise,num_noise), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.sqrt(tf.cast(num_noise,tf.float64)),maxval=1/tf.sqrt(tf.cast(num_noise,tf.float64)),dtype=tf.float64))
        #w_1 = tf.get_variable("w_1", shape = (num_noise,num_noise), dtype=tf.float64, initializer = tf.random_normal_initializer(mean=0.0, stddev=1/tf.sqrt(tf.cast(num_noise,tf.float64))))
        b_1 = tf.get_variable("b_1", shape = (num_noise), dtype=tf.float64, initializer = tf.constant_initializer(0,dtype=tf.float64))
        a_1 = tf.nn.tanh(tf.matmul(noise,w_1)+b_1)
        
        w_out = tf.get_variable("w_out", shape = (num_noise,k), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.sqrt(tf.cast(num_noise,tf.float64)), maxval=1/tf.sqrt(tf.cast(num_noise,tf.float64)),dtype=tf.float64))
        #w_out = tf.get_variable("w_out", shape = (num_noise,k), dtype=tf.float64, initializer = tf.random_normal_initializer(mean=0.0, stddev=1/tf.sqrt(tf.cast(num_noise,tf.float64))))
        b_out = tf.get_variable("b_out", shape = (k), dtype=tf.float64, initializer = tf.constant_initializer(0,dtype=tf.float64))
        Phi = tf.matmul(a_1,w_out)+b_out
        '''
        
        num_noise = k * 32
        noise = tf.random_uniform((1, num_noise),minval=-1,maxval=1, dtype=tf.float64)
        x = noise
        #e2
        #num_units = [num_noise, num_noise*4]
        #e4
        num_units = [num_noise] + [num_noise]*3
        for i in range(len(num_units)):
            if i < len(num_units) - 1:
                w_hid = tf.get_variable("w_%d"%i, shape = (num_units[i],num_units[i+1]), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.cast(num_units[i],tf.float64),maxval=1/tf.cast(num_units[i],tf.float64),dtype=tf.float64))
                b_hid = tf.get_variable("b_%d"%i, shape = (num_units[i+1]), dtype=tf.float64, initializer = tf.constant_initializer(0,dtype=tf.float64))
                x = tf.matmul(x,w_hid)+b_hid
                x = tf.nn.tanh(x)
        w_out = tf.get_variable("w_out", shape = (num_units[-1],k), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.cast(num_units[-1],tf.float64), maxval=1/tf.cast(num_units[-1],tf.float64),dtype=tf.float64))
        b_out = tf.get_variable("b_out", shape = (k), dtype=tf.float64, initializer = tf.constant_initializer(0,dtype=tf.float64))
        Phi = tf.matmul(x,w_out)+b_out
        
        #e4
        #epsilon = tf.random_uniform((1,k),minval=-0.1,maxval=0.1, dtype=tf.float64)
        #Phi = (1 + epsilon) * Phi
        
        # mean
        E_H = tf.reduce_mean(unc, axis=1)
        E_H = tf.reshape(E_H, (-1,1))
        # mean aboslute deviation
        D_H = tf.reduce_mean(tf.abs(unc - E_H), axis=1)
        D_H = tf.reshape(D_H, (-1,1))
        # relative mean aboslute deviation
        Delta_H_bar = D_H / E_H
        
        Delta_H = tf.reduce_mean(Delta_H_bar, axis = 0)
        Delta_H = tf.reshape(Delta_H, (1,1))
        
        # counter-bias ratio
        delta = 1 + Phi * Delta_H
        
        adj_unc = delta * unc

        G_bar_beta = softargmin(adj_unc, axis=1, b=b)
        #G_bar_beta = softround(G_bar_beta)
        
        # count n_1, n_2, ..., n_k
        n = []
        for i in range(k):
            #n.append(tf.reduce_sum(tf.nn.relu(-(tf.abs(G_bar_beta - i)-1))))
            #n.append(tf.reduce_sum(2*(tf.nn.relu(-(tf.abs(G_bar_beta - i)-0.5))))) 
            n.append(tf.reduce_sum(tf.tanh(10*tf.nn.relu(-(tf.abs(G_bar_beta - i)-0.5))))) 
            #n.append(tf.reduce_sum(tf.clip_by_value(tf.nn.relu(-(tf.abs(G_bar_beta - i)-0.5)),0,0.1)*10)) 
              
        n = tf.stack(n)
        
        N = tf.reduce_sum(n)
        # calculate f_1, f_2, ..., f_k
        f = n / N
        
        # mean absolute deviation of the frequency
        D_f = tf.reduce_mean(tf.abs(f+momentum(f_G()) - 1/k))
        
        loss = D_f
        
        #e4
        #epsilon = tf.random_uniform((),minval=-0.1,maxval=0.1, dtype=tf.float64)
        #loss = (1+epsilon) * loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        gvs = optimizer.compute_gradients(loss)
        # noise_gvs = [(grad + tf.random_uniform(tf.shape(grad),minval=-1e-2,maxval=1e-2, dtype=tf.float64), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(gvs)
      
      
        config = tf.ConfigProto(log_device_placement=self.params.log_device_placement, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction=0.1
        with tf.Session(config = config) as sess:
            self.params.logger.debug("begin Gate ready")
            self.r_queue.put("ready")
            self.params.logger.debug("end Gate ready")
            
            sess.run(tf.global_variables_initializer())
            
            sess_exit = False
            while not sess_exit:
                self.params.logger.debug("begin Gate c_queue.get")
                command = self.c_queue.get()
                self.params.logger.debug("end Gate c_queue.get")
                if command[0] == "train":
                    if command[1] is not None:
                        unc_value = command[1]
                        
                        unc_value = self.normalize(unc_value, axis=1)
                        
                        #b_value = self.cal_argmin_b(unc_value, axis=1)
                        
                        self.meta.c_queue.put(["train",unc_value])
                        b_value = self.meta.r_queue.get()
                        
                        #sess.run(tf.global_variables_initializer())
                        G_beta_value, f_G_value, D_f_value, delta_value = sess.run([G_beta, f_G(), D_f, delta], {unc: unc_value, b:b_value})
                        self.params.logger.info("*f_G: %s" % str(f_G_value))
                        self.params.logger.info("*D_f: %f" % D_f_value)
                        self.params.logger.debug("*G_beta: %s" % str(G_beta_value))
                        self.params.logger.info("*delta: %s" % str(delta_value))
                        
                        for i in range(int(1e5)):
                            _, G_bar_beta_value, f_value, D_f_value, delta_value = sess.run([train_op, G_bar_beta, f, D_f, delta], {unc: unc_value, b:b_value})
                            if i % int(1e3) == 0:
                                self.params.logger.info("steps: %d" % i)
                                self.params.logger.info("f: %s"% str(f_value))
                                self.params.logger.info("D_f: %f" % D_f_value)
                                self.params.logger.debug("G_bar_beta: %s"% str(G_bar_beta_value))
                                self.params.logger.info("delta: %s" % str(delta_value))
                                
                            if D_f_value < 1/(200*k):
                                self.params.logger.info("#steps: %d" % i)
                                self.params.logger.info("#f: %s"% str(f_value))
                                self.params.logger.info("#D_f: %f" % D_f_value)
                                self.params.logger.debug("#G_bar_beta: %s"% str(G_bar_beta_value))
                                self.params.logger.info("#delta: %s" % str(delta_value))
                                
                                
                                self.failure.value -= 1 
                                self.failure.value = max(0,self.failure.value)   
                                break
                        
                        if i >= int(1e5) - 1 and D_f_value >= 1/(50*k):
                            self.failure.value += 1   
                            print("gate failure: %d"% self.failure.value)
                        
                        G_bar_beta_value = np.round(G_bar_beta_value)    
                        G_bar_beta_value = G_bar_beta_value.astype(int)
                        
                        
                        inds = []
                        for i in range(k):
                            ind = np.reshape(np.argwhere(G_bar_beta_value == i), (-1))
                            inds.append(ind)
                            self.params.logger.info("expert %d data examples: %d"%(i, len(ind)))
                        self.r_queue.put(inds)
                        
                elif command[0] == "exit":
                    
                    #self.meta.c_queue.put(["exit"])
                    #self.meta.r_queue.get()
                    #self.meta.join()
                    
                    sess_exit = True
                    self.r_queue.put(None)  
    
    def normalize(self, logits, axis):
        logits = np.array(logits).astype(dtype=np.float64)
        
        E_arg = np.expand_dims(np.mean(logits, axis),axis)
        MD_arg = np.mean(np.abs(logits - E_arg),axis)
        self.params.logger.info("zero deviations: %d" % (len(MD_arg) - np.count_nonzero(MD_arg)))
        minval = np.min(MD_arg[np.nonzero(MD_arg)])
        
        noise = np.random.uniform(0,minval/100,logits.shape)
        logits = logits + noise
        
        E_arg = np.expand_dims(np.mean(logits, axis),axis)
        MD_arg = np.mean(np.abs(logits - E_arg),axis)
        self.params.logger.info("zero deviations (2nd time): %d" % (len(MD_arg) - np.count_nonzero(MD_arg)))
        minval = np.min(MD_arg[np.nonzero(MD_arg)])
        
        MD_arg = np.maximum(MD_arg,minval)
        MD_arg = np.expand_dims(MD_arg,axis)
        logits = logits/MD_arg
        self.params.logger.info("deviation: %f" % np.mean(MD_arg))
        self.params.logger.debug("deviation: %s" % str(np.reshape(MD_arg, (-1))))
        self.params.logger.info("mean: %f" % np.mean(E_arg))
        self.params.logger.debug("mean: %s" % str(np.reshape(E_arg, (-1))))
        return logits
    
class Meta(Process):
    def __init__(self, name, device, params):
        Process.__init__(self)
        self.manager = Manager()
        self.c_queue = self.manager.Queue(1)
        self.r_queue = self.manager.Queue(1)
        self.name = name
        self.device = device
        self.params = params       
        
    def run(self):
        self.params.logger.info("begin Meta")
        if self.device is None:
            device_id = ""
        else:
            device_id = self.device[len(self.device)-1]
        os.environ["CUDA_VISIBLE_DEVICES"]=device_id
        
        import tensorflow as tf
        
        k = self.params.num_experts
        unc = tf.placeholder(dtype=tf.float64,shape=(None, k))
        
        global_step = tf.train.get_or_create_global_step()
        
        #e4
        num_noise = 4
        noise = tf.random_uniform((1, num_noise),minval=-1,maxval=1, dtype=tf.float64)
        x = noise
        # cifar10
        # num_units = [num_noise] + [num_noise] * 2
        # mlp
        num_units = [num_noise] + [num_noise] * 1
        for i in range(len(num_units)):
            if i < len(num_units) - 1:
                w_hid = tf.get_variable("w_%d"%i, shape = (num_units[i],num_units[i+1]), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.cast(num_units[i],tf.float64),maxval=1/tf.cast(num_units[i],tf.float64),dtype=tf.float64))
                b_hid = tf.get_variable("b_%d"%i, shape = (num_units[i+1]), dtype=tf.float64, initializer = tf.constant_initializer(0,dtype=tf.float64))
                x = tf.matmul(x,w_hid)+b_hid
                x = tf.nn.tanh(x)
        w_out = tf.get_variable("w_out", shape = (num_units[-1],1), dtype=tf.float64, initializer = tf.random_uniform_initializer(minval=-1/tf.cast(num_units[-1],tf.float64), maxval=1/tf.cast(num_units[-1],tf.float64),dtype=tf.float64))
        b_out = tf.get_variable("b_out", shape = (1), dtype=tf.float64, initializer = tf.constant_initializer(1.0,dtype=tf.float64))
        b = tf.matmul(x,w_out)+b_out
        b = tf.reshape(b, ())
        
        #e2
        #b = tf.get_variable("b", shape=(),dtype=tf.float64, initializer = tf.constant_initializer(1.0))
        
        #e4
        epsilon = tf.random_uniform((),minval=-0.1,maxval=0.1, dtype=tf.float64)
        b = (1 + epsilon) * b
        
        axis = 1
        ref = tf.cumsum(tf.ones_like(-unc),axis)-1
    
        ind = tf.reduce_sum(ref*tf.nn.softmax(b*(-unc), axis), axis)
        ind = tf.reshape(ind, (-1,1))
        smoothness = tf.reduce_mean(tf.reduce_min(tf.abs(ind - ref),axis))
        
        loss = tf.abs(smoothness - 8e-3)
        
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step) #e2: SGD 1e-2
        
        
        saver = tf.train.Saver()
        scaffold = tf.train.Scaffold(saver=saver)
        checkpoint_dir = self.params.model_dir + "/__meta__/" + self.name
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir, save_steps=self.params.stale_interval*int(1e2), scaffold=scaffold)
        
        config = tf.ConfigProto(log_device_placement=self.params.log_device_placement, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction=0.05
        
        with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=checkpoint_dir,
                    scaffold=scaffold,
                    hooks=[saver_hook],
                    config=config,
                    save_checkpoint_secs=None,
                    save_summaries_secs=None, 
                    save_summaries_steps=None, 
                    ) as mon_sess:
        
        #with tf.Session(config = config) as sess:
            #mon_sess.run(tf.global_variables_initializer())
            sess_exit = False
            sess_init = True
            num_steps = int(1e7)
            while not sess_exit:
                command = self.c_queue.get()
                if command[0] == "train":
                    if command[1] is not None:
                        unc_value = command[1]
                        
                        for _ in range(num_steps):
                            _, loss_value = mon_sess.run([train_op, loss], {unc: unc_value})
                            self.params.logger.debug("meta loss: %f"% loss_value)
                            if loss_value < 1e-3: #e2: 1e-6
                                break
                        self.params.logger.info("meta loss: %f"% loss_value)
                        
                        b_value = mon_sess.run(b, {unc: unc_value})
                        self.params.logger.info("b: %f"% b_value)
                            
                        self.r_queue.put(b_value)
                        
                        if sess_init == True:
                            num_steps = int(1e5)
                            sess_init = False
                        
                elif command[0] == "exit":
                    sess_exit = True
                    self.r_queue.put(None)  
                    
        self.params.logger.info("end Meta")