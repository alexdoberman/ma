# -*- coding: utf-8 -*-
import json
from bunch import Bunch, bunchify
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    #TODO do bunch depth 1
    config.batcher = Bunch(config_dict['batcher'])
    config.trainer = Bunch(config_dict['trainer'])
    config.model = Bunch(config_dict['model'])

    if 'predictor' in config_dict:
        config.predictor = Bunch(config_dict['predictor'])

    return config, config_dict


def process_config(json_file, root_path):
    config, _ = get_config_from_json(json_file)

    config.experiments_dir = os.path.join(root_path, "_experiments")
    config.summary_dir     = os.path.join(config.experiments_dir, config.exp_name, "summary/")
    config.checkpoint_dir  = os.path.join(config.experiments_dir, config.exp_name, "checkpoint/")
    config.results_dir     = os.path.join(config.experiments_dir, config.exp_name, "results/")
    config.to_separate_dir = os.path.join(config.experiments_dir, config.exp_name, "to_separate/")
    config.batcher.rir_dir = os.path.join(root_path, "rir_store")

    return config
