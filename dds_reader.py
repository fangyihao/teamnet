###############################################################################
# (c) 2005-2015 Copyright, Real-Time Innovations.  All rights reserved.       #
# No duplications, whole or partial, manual or electronic, may be made        #
# without express written permission.  Any such copies, or revisions thereof, #
# must display this notice unaltered.                                         #
# This code contains trade secrets of Real-Time Innovations, Inc.             #
###############################################################################

"""Samples's reader."""

from __future__ import print_function
from sys import path as sysPath
from os import path as osPath
from time import sleep
from work_node import WorkNode
from struct import *
import pickle
import socket
filepath = osPath.dirname(osPath.realpath(__file__))

import rticonnextdds_connector as rti
import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="mnist")

    parser.add_argument(
        '--stale_interval', type=int, default = 50, 
        help="stale interval")
    parser.add_argument(
        '--socket_buffer_size', default = 32768, 
        help="socket buffer size")

    return parser

parser = create_parser()
params = parser.parse_args()



connector = rti.Connector("MyParticipantLibrary::Zero",
                          "ShapeExample.xml")
inputDDS = connector.getInput("MySubscriber::MySquareReader")


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

addr = ("127.0.0.1", int("8000"))
client_handler = ClientSocketHandler(params, addr)
client_handler.start()
client_handler.c_queue.put(["send","ready"])
client_handler.r_queue.get()   
for i in range(1, 500):
    print(i, "1")
    inputDDS.take()
    numOfSamples = inputDDS.samples.getLength()
    print("numOfSamples:",numOfSamples)
    for j in range(1, numOfSamples+1):
        if inputDDS.infos.isValid(j):
            # This gives you a dictionary
            sample = inputDDS.samples.getDictionary(j)
            x = sample['x']
            y = sample['y']

            # Or you can just access the field directly
            size = inputDDS.samples.getNumber(j, "shapesize")
            color = inputDDS.samples.getString(j, "color")
            toPrint = "Received x: " + repr(x) + " y: " + repr(y) + \
                      " size: " + repr(size) + " color: " + repr(color)

            print(toPrint)
    print(i, "2")
    sleep(1)
client_handler.c_queue.put(None)    
client_handler.join()

    