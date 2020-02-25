import re
import os
from datetime import datetime
import subprocess
import numpy as np
from os.path import join
import psutil
from work_node import WorkNode
from queue import Empty
import time
import copy
BIN_PATH = '~/tegrastats'

class SysUtilLogger(WorkNode):
	
	def __init__(self, params, c_queue = None, r_queue = None, pid = os.getpid(), task = "", LOG_FILE_PATH = "dtl", STATS_FILE_PATH = "smmr"):
		super().__init__(params, c_queue, r_queue)
		
		print("pid:", pid)
		
		self.RAM_CPU_GPU = re.compile(r"^RAM (\d+/\d+)MB .+ CPU \[(\d+)%@\d+,off,off,(\d+)%@\d+,(\d+)%@\d+,(\d+)%@\d+\] .+ GR3D_FREQ (\d+)%@.+")
		self.pid = pid
		
		self.task = task
		self.edge_hardware = params.edge_hardware
		self.LOG_FILE_PATH = LOG_FILE_PATH
		self.STATS_FILE_PATH = STATS_FILE_PATH


	def run(self):
		
		records = {}
		records["memory"] = []
		records["cpu"] = []
		records["cpu1"] = []
		records["cpu2"] = []
		records["cpu3"] = []
		records["cpu4"] = []
		records["gpu"]  = []
		
		reference = copy.deepcopy(psutil.virtual_memory().used)
		self.r_queue.put("ready")
		#main_proc = psutil.Process(self.pid)
		if self.edge_hardware == 'tx2':
			cmds = ["echo 'nvidia' |sudo -S " + BIN_PATH +' --interval 200']
			tegra_proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True)
		#p_mem = subprocess.Popen("top -b -d 0.2 -p %d | grep %d" % (self.pid,self.pid), stdout=subprocess.PIPE, shell=True)
		with open(join(self.params.log_dir, self.task+"_"+self.LOG_FILE_PATH+"_"+self.edge_hardware+".txt"),"a") as dtlf:
			while True:
				if self.edge_hardware == 'tx2':
					current_stat = tegra_proc.stdout.readline().decode().strip()
				#mem_stat = p_mem.stdout.readline().decode().strip()
				#mem_splited = mem_stat.split()
	
				#mem_stat_1 = mem_splited[-3]
				#top_cpu = mem_splited[-4]
				
				if self.edge_hardware == 'tx2':
					if current_stat == '':
						print("tegrastats error")
						break
				if self.edge_hardware == 'tx2':
					_, cpu1, cpu2, cpu3, cpu4, gpu = self.RAM_CPU_GPU.sub(r'\1,\2,\3,\4,\5,\6',current_stat).split(",")
					
					
				timestamp = datetime.now()
				
				#memory = main_proc.memory_percent()
				mem_stats = psutil.virtual_memory()
				memory = (mem_stats.used - reference)/mem_stats.total * 100
				cpu = psutil.cpu_percent()
				
				if self.edge_hardware == 'tx2':
					text = "%s,%s%%,%s%%,%s%%,%s%%,%s%%,%s%%,%s%%" % (str(timestamp), memory, cpu, cpu1, cpu2, cpu3, cpu4, gpu)
				elif self.edge_hardware == 'pi':
					text = "%s,%s%%,%s%%" % (str(timestamp), memory, cpu)
				
				dtlf.write(text+"\n")
	
				
				try:
					records["memory"].append(float(memory))
					records["cpu"].append(float(cpu))
					if self.edge_hardware == 'tx2':
						records["cpu1"].append(int(cpu1))
						records["cpu2"].append(int(cpu2))
						records["cpu3"].append(int(cpu3))
						records["cpu4"].append(int(cpu4))
						records["gpu"].append(int(gpu))
				except ValueError:
					print(text)
					pass
				
				
				try:
					command = self.c_queue.get_nowait()
					if command[0] == "exit":
						break
				except Empty:
					pass

				if self.edge_hardware == 'pi':
					time.sleep(0.2)
		
		self.write_summeries(str(timestamp), records)
		
		

	def write_summeries(self, timestamp, records):
		print("Calculating system utility statistics...")
		np_memory = np.array(records["memory"])
		np_cpu = np.array(records["cpu"])
		if self.edge_hardware == 'tx2':
			np_cpu1 = np.array(records["cpu1"])
			np_cpu2 = np.array(records["cpu2"])
			np_cpu3 = np.array(records["cpu3"])
			np_cpu4 = np.array(records["cpu4"])
			np_gpu  = np.array(records["gpu"])

		with open(join(self.params.log_dir, self.task+"_"+self.STATS_FILE_PATH+"_"+self.edge_hardware+".txt"),"a") as smmrf:
			if self.edge_hardware == 'tx2':
				smmrf.write("Complete Time:%s, Avg Memory Usage (%%):%.1f, Std Memory Usage:%.1f, Max Memory Usage(%%):%.1f, Avg CPU Usage (%%):%.1f, Std CPU Usage:%.1f, Avg CPU1 Usage (%%):%.1f, Std CPU1 Usage:%.1f, Avg CPU2 Usage (%%):%.1f, Std CPU2 Usage:%.1f, Avg CPU3 Usage (%%):%.1f, Std CPU3 Usage:%.1f, Avg CPU4 Usage (%%):%.1f, Std CPU4 Usage:%.1f, Avg GPU Usage (%%):%.1f, Std GPU Usage:%.1f\n" 
					% ( timestamp,
						np.mean(np_memory),
						np.std(np_memory),
						np.max(np_memory),
						np.mean(np_cpu),
						np.std(np_cpu),
						np.mean(np_cpu1),
						np.std(np_cpu1),
						np.mean(np_cpu2),
						np.std(np_cpu2),
						np.mean(np_cpu3),
						np.std(np_cpu3),
						np.mean(np_cpu4),
						np.std(np_cpu4),
						np.mean(np_gpu),
						np.std(np_gpu)
						)
					)
			elif self.edge_hardware == 'pi':
				smmrf.write("Complete Time:%s, Avg Memory Usage (%%):%.1f, Std Memory Usage:%.1f, Max Memory Usage(%%):%.1f, Avg CPU Usage (%%):%.1f, Std CPU Usage:%.1f\n" 
					% ( timestamp,
						np.mean(np_memory),
						np.std(np_memory),
						np.max(np_memory),
						np.mean(np_cpu),
						np.std(np_cpu)
						)
					)
