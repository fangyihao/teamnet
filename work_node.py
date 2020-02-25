'''
Created on Sep 11, 2018

@author: Yihao Fang
'''
mode = "predict"
if mode == "train":
    from multiprocessing import Process, Manager
    class WorkNode(Process):
        def __init__(self, params, c_queue = None, r_queue = None):
            Process.__init__(self)
            self.params = params
            
            self.manager = Manager()
            if c_queue is None:
                self.c_queue = self.manager.Queue(params.stale_interval)
            else:
                self.c_queue = c_queue
                
            if r_queue is None:
                self.r_queue = self.manager.Queue(params.stale_interval)
            else:
                self.r_queue = r_queue
        def run(self):
            pass
else:
    from threading import Thread
    from queue import Queue
    
    class WorkNode(Thread):
        def __init__(self, params, c_queue = None, r_queue = None):
            Thread.__init__(self)
            self.params = params
            
            if c_queue is None:
                self.c_queue = Queue(params.stale_interval)
            else:
                self.c_queue = c_queue
                
            if r_queue is None:
                self.r_queue = Queue(params.stale_interval)
            else:
                self.r_queue = r_queue
        def run(self):
            pass