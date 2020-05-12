# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np


from mic_py_nn.models.chimera_model import ChimeraModel
from mic_py_nn.models.unet_model import UNetModel
from mic_py_nn.utils.config import process_config
from mic_py_nn.utils.utils import get_args
import freeze_graph

def store_model_(config_path, in_model_path, out_model_path):
    """

    :param config_path:
    :param in_model_path:
    :param out_model_path:
    :return:
    """

    #################################################################
    # 0 - Capture the config path from the run arguments then process the json configuration file
    config = process_config(config_path, '.')
    config.checkpoint_dir = in_model_path

    #################################################################
    # 1 - Create tensorflow session and load model if exists
    sess = tf.Session()
    model = ChimeraModel(config)
    model.load(sess)

    #################################################################
    # 2 - Create tensorflow session and load model if exists
    tf.train.write_graph(sess.graph_def, out_model_path + '/models_pb/', 'graph.pb', as_text=True)

    #tf.train.Saver(tf.trainable_variables()).save(sess, out_model_path + '/models_pb2')

    #X_in = np.arange(3*257, dtype=np.float32).reshape((1,3,257))
    #current_prediction  = model.get_masks(sess, X_in)
    #print (current_prediction.shape)
    #print (current_prediction.dtype)
    #np.save('mi_out', current_prediction)

def convert_model():

    # We save out the graph to disk, and then call the const conversion
    # routine.
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    input_graph_path      = "/home/stc/MA_ALG/datasets/test_ma/chimera_8/checkpoint/models_pb/graph.pb"
    input_saver_def_path  = ""
    input_binary          = False
    input_checkpoint_path = "/home/stc/MA_ALG/datasets/test_ma/chimera_8/checkpoint/-99000"

    # Note that we this normally should be only "output_node"!!!
    output_node_names     = "network/Softmax:0" 
    restore_op_name       = "save/restore_all"
    filename_tensor_name  = "save/Const:0"
    output_graph_path     = "/home/stc/MA_ALG/datasets/test_ma/chimera_8/checkpoint/models_pb/output_graph.pb"
    clear_devices         = False

    
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices)


def main():

    config_path    = r'/home/stc/MA_ALG/datasets/test_ma/chimera_8/8_chimera_r09_em_30_a05_ctx_100_tanh_snr_3_size_4x500.json'
    in_model_path  = r'/home/stc/MA_ALG/datasets/test_ma/chimera_8/checkpoint/'
    out_model_path = r'/home/stc/MA_ALG/datasets/test_ma/chimera_8/checkpoint/'

    store_model_(config_path, in_model_path, out_model_path)


if __name__ == '__main__':
    #main()
    convert_model()
