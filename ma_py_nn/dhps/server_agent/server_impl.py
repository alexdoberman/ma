#!/usr/bin/env python
# -*- coding: utf-8 -*-


import subprocess
import os
import sys

from dhps.server_agent.server_config import DATASETS_PATH, TEMP_PATH
from mic_py_nn.utils.file_op_utils import load_obj

def _impl_run_command(command):
    """

    :param command:
    :return:
    """

    print ('run_shell_command: {}'.format(command))
    PIPE = subprocess.PIPE
    p = subprocess.Popen(command, shell=True, stdin=PIPE, stdout=PIPE,
            stderr=subprocess.STDOUT, close_fds=False)
    while True:
        s = p.stdout.readline()
        if not s: break
        print (s)
    return True

def _impl_start_train_detach(script_name, config_string, config_name, dataset_name):
    """
    Start detach train procedure

    :param script_name:   training script
    :param config_string: string represent config file
    :param config_name:   name config file
    :param dataset_name:  dataset name
    :return:
    """

    config_path = os.path.join(TEMP_PATH, config_name)
    with open(config_path, "w") as text_file:
        text_file.write(config_string)

    log_path = os.path.join(TEMP_PATH, 'log_train.log')
    f_log = open(log_path, 'a')

    scpript_path = os.path.join(r'./mic_py_nn/mains/', script_name)
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)

    # my_env = os.environ.copy()
    # my_env["PYTHONPATH"] = "."

    cmd = '{} {} --config {} --root_path {} --stage train'.format('python', scpript_path, config_path, dataset_path)
    print ('Run: {}'.format(cmd))

    p = subprocess.Popen(cmd, shell=True, stdin=None, stdout=f_log,
                         stderr=f_log, close_fds=True)
    return p.pid

def _impl_start_train(script_name, config_string, config_name, dataset_name):
    """
    Start detach train procedure

    :param script_name:   training script
    :param config_string: string represent config file
    :param config_name:   name config file
    :param dataset_name:  dataset name
    :return:
    """

    config_path = os.path.join(TEMP_PATH, config_name)
    with open(config_path, "w") as text_file:
        text_file.write(config_string)

    scpript_path = os.path.join(r'./mic_py_nn/mains/', script_name)
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)

    # my_env = os.environ.copy()
    # my_env["PYTHONPATH"] = "."

    cmd = '{} {} --config {} --root_path {} --stage train'.format('python', scpript_path, config_path, dataset_path)
    print ('Run: {}'.format(cmd))

    PIPE = subprocess.PIPE
    p = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
            stderr=subprocess.STDOUT, close_fds=False)
    while True:
        s = p.stdout.readline()
        if not s: break
        print (s)

    # Delete tmp file
    if os.path.isfile(config_path):
        os.remove(config_path)

    return True

def _impl_get_result(dataset_name, config_dict):
    """

    :param config_dict:
    :return:
    """

    result_file_path = os.path.join(DATASETS_PATH, dataset_name, '_experiments' ,config_dict['exp_name'], r'results/result.json')
    print('_impl_get_result from: {}'.format(result_file_path))
    return load_obj(result_file_path)





