# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import fnmatch
import numpy as np
import random
from shutil import copyfile
import soundfile as sf


from mic_py_nn.models.chimera_model import ChimeraModel
from mic_py_nn.trainers.clustering_predict import ChimeraClusteringPredict, ChimeraMaskInferencePredict

from mic_py_nn.models.unet_model import UNetModel
from mic_py_nn.trainers.unet_predict import UNetPredict

# from mic_py_nn.models.crnn_dcase import CRNNMaskFRAttentionModel as CRNN
from mic_py_nn.models.td_cnn_model import TDCNNModel as TDCNN
from mic_py_nn.models.crnn_dcase import CRNNMaskModel as CRNN
from mic_py_nn.trainers.crnn_dcase_predict import CRNNMaskPredict

from mic_py_nn.models.alex_net_model import AlexNetModel as MadAlex

from mic_py_nn.utils.config import process_config
from mic_py_nn.utils.utils import get_args
from mic_py_nn.utils.file_op_utils import find_files

from mic_py_nn.features.feats import stft, istft
from mic_py_nn.features.preprocessing import dc_preprocess, dcce_preprocess, dan_preprocess, normalize_signal, \
    chimera_preprocess


# def main():
#
#     #################################################################
#     # 0 - Capture the config path from the run arguments then process the json configuration file
#     args = get_args()
#     config = process_config(args.config, args.root_path)
#     predict_path = os.path.join(config.experiments_dir, config.exp_name, "sample_predict/")
#
#     #################################################################
#     # 1 - Create tensorflow session and load model if exists
#     sess = tf.Session()
#     model = ChimeraModel(config)
#     model.load(sess)
#
#     #################################################################
#     # 3 - Load files to predict
#     lst_mix_files = list(find_files(predict_path, '*.wav'))
#
#     #################################################################
#     # 4 - Predict result
#     predictor = ChimeraMaskInferencePredict(sess, model, config)
#     predictor.predict(lst_mix_files, predict_path)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ChimeraPredict(metaclass=Singleton):

    def __init__(self, config_path, in_model_path):
        self.config = process_config(config_path, '.')
        self.config.checkpoint_dir = in_model_path
        self.sess = tf.Session()
        self.model = ChimeraModel(self.config)
        self.model.load(self.sess)

    def predict_mask(self, in_wav_path, predict_path):
        lst_mix_files = [in_wav_path]

        predictor = ChimeraMaskInferencePredict(self.sess, self.model, self.config)
        predictor.predict(lst_mix_files, predict_path, index_naming=False, save_masks=True)

        mask_path = os.path.join(predict_path, "{}_mask.npy".format(os.path.splitext(os.path.basename(in_wav_path))[0]))
        masks = np.load(mask_path)
        print('masks.shape : ', masks.shape)

        return masks


class ChimeraPredictFrozen(metaclass=Singleton):

    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.graph = self.load_graph(self.path_to_model)
        self.fftsize = 512
        self.overlap = 2

        # We can verify that we can access the list of operations in the graph
        # for op in self.graph.get_operations():
        #    print(op.name)

    def predict_mask(self, in_wav_path):

        # We access the input and output nodes
        x = self.graph.get_tensor_by_name('prefix/Placeholder_1:0')
        y = self.graph.get_tensor_by_name('prefix/network/Softmax:0')

        # We launch a Session
        with tf.Session(graph=self.graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants

            x_in = self.prepare_file(in_wav_path)
            _, frames, bins = x_in.shape
            y_out = sess.run(y, feed_dict={x: x_in})

            y_out = y_out.reshape((frames, 257, 2))

        return y_out

    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def prepare_file(self, in_wav_path):

        raw_wav, rate = sf.read(in_wav_path)
        signal = normalize_signal(raw_wav)
        stft_sig = stft(signal, fftsize=self.fftsize, overlap=self.overlap)
        x_in = chimera_preprocess(stft_sig)
        x_in = np.reshape(x_in, (1, x_in.shape[0], x_in.shape[1]))

        return x_in


'''
def chimera_predict(config_path, in_model_path, in_wav_path, predict_path):
    """

    :param config_path:
    :param in_model_path:
    :param in_wav_path:
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
    # 2 - Load files to predict
    lst_mix_files = [in_wav_path]

    #################################################################
    # 3 - Predict result
    predictor = ChimeraMaskInferencePredict(sess, model, config)
    predictor.predict(lst_mix_files, predict_path, index_naming=False, save_masks=True)


    #################################################################
    # 4 - Load mask from file and return

    mask_path = os.path.join(predict_path, "{}_mask.npy".format(os.path.splitext(os.path.basename(in_wav_path))[0]))
    masks = np.load(mask_path)
    print ('masks.shape : ', masks.shape)

    sess.close()

    return masks
'''


class UnetPredict(metaclass=Singleton):

    def __init__(self, config_path, in_model_path):
        self.config = process_config(config_path, '.')
        self.config.checkpoint_dir = in_model_path
        self.sess = tf.Session()
        self.model = UNetModel(self.config)
        self.model.init_model()
        self.model.load(self.sess)

        self.predictor = UNetPredict(self.sess, self.model, self.config)

    def predict_mask(self, in_wav_path):
        speech_mask, noise_mask = self.predictor.ss_predict_mask(in_wav_path)

        return noise_mask


class MADPredict(metaclass=Singleton):

    def __init__(self, config_path, in_model_path, model_name):
        self.config = process_config(config_path, '.')
        self.config.checkpoint_dir = in_model_path
        self.sess = tf.Session()

        no_content_pred = False
        self.model = CRNN(self.config)
        # self.model.init_model(sr_fr_pr=no_content_pred)
        # self.model.build_model()
        # self.model.init_saver()
        self.model.init_model(enable_rnn=True)
        if model_name is not None:
            self.model.simple_load(self.sess, model_name)
        else:
            self.model.load(self.sess)

        self.predictor = CRNNMaskPredict(self.sess, self.model, self.config, mel_feats=False,
                                         no_context_pred=no_content_pred)

    def predict_mask(self, in_wav_path):

        vad_mask = self.predictor.ss_predict_mask(in_wav_path)

        return vad_mask

'''
def unet_predict(config_path, in_model_path, in_wav_path, predict_path):
    """

    :param config_path:
    :param in_model_path:
    :param in_wav_path:
    :return:
    """

    #################################################################
    # 0 - Capture the config path from the run arguments then process the json configuration file
    config = process_config(config_path, '.')
    config.checkpoint_dir = in_model_path

    #################################################################
    # 1 - Create tensorflow session and load model if exists
    tf.reset_default_graph()
    sess = tf.Session()
    model = UNetModel(config)
    model.init_model()
    model.load(sess)

    #################################################################
    # 2 - Predict result
    predictor = UNetPredict(sess, model, config)
    speech_mask, noise_mask = predictor.ss_predict_mask(in_wav_path)

    print('masks.shape : ', noise_mask.shape)

    sess.close()

    return noise_mask
'''


def unet_predict_main():
    config_path = r'/home/stc/MA_ALG/datasets/test_ma/unet/unet.json'
    in_model_path = r'/home/stc/MA_ALG/datasets/test_ma/unet/checkpoint/'
    predict_path = r'/home/stc/MA_ALG/datasets/data_dc_v11/to_separate'

    out_dir = r'/home/stc/MA_ALG/datasets/test_ma/result_unet_mask_v11/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    files = fnmatch.filter(os.listdir(predict_path), '*.wav')
    files = [os.path.join(predict_path, f) for f in files]

    config = process_config(config_path, '.')
    config.checkpoint_dir = in_model_path

    tf.reset_default_graph()
    sess = tf.Session()
    model = UNetModel(config)
    model.init_model()
    model.load(sess)

    predictor = UNetPredict(sess, model, config)

    predictor.predict(files, out_dir)




'''
def main():

    config_path    = r'/home/stc/MA_ALG/datasets/test_ma/8_chimera_r09_em_30_a05_ctx_100_tanh_snr_3_size_4x500.json'
    in_model_path  = r'/home/stc/MA_ALG/datasets/test_ma/checkpoint/'
    in_wav_path    = r'/home/stc/MA_ALG/datasets/test_ma/to_separate/result_out_mus2_spk1_snr_-15_DS.wav'
    predict_path   = r'/home/stc/MA_ALG/datasets/test_ma/to_separate'

    mask = chimera_predict(config_path, in_model_path, in_wav_path, predict_path)
'''

if __name__ == '__main__':
    # main()
    unet_predict_main()
