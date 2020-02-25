###############################################################################
# (c) 2005-2015 Copyright, Real-Time Innovations.  All rights reserved.       #
# No duplications, whole or partial, manual or electronic, may be made        #
# without express written permission.  Any such copies, or revisions thereof, #
# must display this notice unaltered.                                         #
# This code contains trade secrets of Real-Time Innovations, Inc.             #
###############################################################################

"""Samples's writer."""

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
outputDDS = connector.getOutput("MyPublisher::MySquareWriter")

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

addr = ('0.0.0.0', int("8000"))
server_handler = ServerSocketHandler(params, addr)
server_handler.start()
server_handler.c_queue.put(["recv"])
print(server_handler.r_queue.get())
for i in range(1, 500):
    print(i,"1")
    outputDDS.instance.setNumber("x", i)
    outputDDS.instance.setNumber("y", i*2)
    outputDDS.instance.setNumber("shapesize", 30)
    outputDDS.instance.setString("color", "BLUE")
    print(i,"2")
    outputDDS.write()
    print(i,"3")
    sleep(1)
    
server_handler.c_queue.put(None)
server_handler.join()
    