#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
from threading import Thread
import json
import socket

from dhps.server_agent.server_config import PORT
from dhps.server_agent.server_impl import _impl_run_command, _impl_start_train, _impl_get_result

class TrainThread(Thread):
    """
    A threading train procedure
    """

    def __init__(self, script_name, config_string, config_name, dataset_name):
        Thread.__init__(self)
        self.script_name = script_name
        self.config_string = config_string
        self.config_name = config_name
        self.dataset_name = dataset_name

    def run(self):
        _impl_start_train(self.script_name, self.config_string, self.config_name, self.dataset_name)

class Worker(object):

    def __init__(self):
        self.thread = None
        self.cur_script_name = None
        self.cur_config_string = None
        self.cur_config_name = None
        self.cur_dataset_name = None

    def run_command(self, command):
        """
        Run any shell command
        :param command:
        :return:
        """

        print("-----------------------run_command-----------------------")
        print("command = {}".format(command))
        result =  _impl_run_command(command)
        print("---------------------------------------------------------")
        return result

    def is_busy(self):
        """
        Check busy service or not
        :return:
        """

        print("---------------------------is_busy-----------------------")
        is_busy = False
        if self.thread:
            if self.thread.is_alive():
                is_busy = True

        print ("is_busy = {}".format(is_busy))
        print("---------------------------------------------------------")
        return is_busy

    def start_train(self, script_name, config_string, config_name, dataset_name):
        """
        Start train procedure

        :param script_name:  example: dc.py
        :param config_string:
        :param config_name:  example: dc_1.json
        :param dataset_name: example: data_s
        :return:
        """

        print("-----------------------start_train-----------------------")
        print("script_name, config_name, dataset_name = {}, {}, {}".format(script_name, config_name, dataset_name))
        if self.thread:
            if self.thread.is_alive():
                raise Exception("ERROR: Call start_train, but train thread is_alive.")
            else:
                self.thread = TrainThread(script_name, config_string, config_name, dataset_name)
                self.thread.start()
                self.cur_script_name = script_name
                self.cur_config_string = config_string
                self.cur_config_name = config_name
                self.cur_dataset_name = dataset_name
        else:
            self.thread = TrainThread(script_name, config_string, config_name, dataset_name)
            self.thread.start()
            self.cur_script_name = script_name
            self.cur_config_string = config_string
            self.cur_config_name = config_name
            self.cur_dataset_name = dataset_name
        print("---------------------------------------------------------")
        return True

    def get_result(self):
        """
        Return current result train procedur
        :return:
        """
        print("-----------------------get_result------------------------")
        config_dict = json.loads(self.cur_config_string)
        result = _impl_get_result(self.cur_dataset_name, config_dict)
        print("---------------------------------------------------------")
        return result

class RPCThreading(ThreadingMixIn, SimpleXMLRPCServer):
    def handle_error(self, request, client_address):
        SimpleXMLRPCServer.handle_error(self, request, client_address)
        global g_service_state
        g_service_state = False

def get_ip_address():
    """
    https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-of-eth0-in-python
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def run():
    HOST = get_ip_address()
    server_address = (HOST, PORT)
    server = RPCThreading(server_address)
    print ("Host {} listening on port {}...".format(HOST, PORT))

    server.register_introspection_functions()
    server.register_multicall_functions()

    w = Worker()

    server.register_function(w.run_command,          "run_command")
    server.register_function(w.is_busy,              "is_busy")
    server.register_function(w.start_train,          "start_train")
    server.register_function(w.get_result,           "get_result")

    server.serve_forever()


if __name__ == "__main__":
    run()

