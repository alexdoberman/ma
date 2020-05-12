# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
from shutil import copyfile

from mic_py_nn.data_generator.dataset import Dataset
from mic_py_nn.data_generator.batchers.mad_batcher import MADBatcher

# from mic_py_nn.models.crnn_dcase import CRNNMaskFRAttentionModel as CRNN
from mic_py_nn.models.crnn_dcase import CRNNMaskModel as CRNN

from mic_py_nn.trainers.crnn_dcase_trainer import CRNNMaskTrainer
from mic_py_nn.trainers.crnn_dcase_predict import CRNNMaskPredict

from mic_py_nn.utils.config import process_config
from mic_py_nn.utils.logger import Logger
from mic_py_nn.utils.utils import get_args
from mic_py_nn.utils.file_op_utils import create_dirs, save_obj


from mic_py_nn.utils import f1_util
from mic_py_nn.utils import log_util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    #################################################################
    # 0 - Capture the config path from the run arguments then process the json configuration file
    args = get_args()
    config = process_config(args.config, args.root_path)

    #################################################################
    # 1 - Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.results_dir, config.to_separate_dir])

    #################################################################
    # 1.1 - log file

    log_dir = config.experiments_dir
    log_name = config.log_file_name

    log_path = os.path.join(log_dir, log_name)

    pic_dir = config.experiments_dir

    #################################################################
    # 2 - Create tensorflow session and load model if exists

    # sess = tf.Session()
    '''
        allow_soft_placement=True: place summary op on cpu
    '''
    no_context_pred = False

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)

    model = CRNN(config)
    model.build_model(enable_rnn=True)
    model.init_saver()

    model.load(sess)

    #################################################################
    # 3 - Load dataset
    dataset = Dataset(root_path=args.root_path)

    train_spk_files, train_noise_files = dataset.get_train_files()
    valid_spk_files, valid_noise_files = dataset.get_valid_files()
    dict_test_mix_files = dataset.get_test_files()
    to_separate_files = dataset.get_to_separate_files()

    #################################################################
    # 4 - Create batchers
    train_data = MADBatcher(train_spk_files, train_noise_files, config, mel_feats=False,
                            no_context_pred=no_context_pred,
                            one_hot_mask=True, simple_mix=True)
    valid_data = MADBatcher(valid_spk_files, valid_noise_files, config, mel_feats=False,
                            no_context_pred=no_context_pred,
                            one_hot_mask=True, simple_mix=True)

    #################################################################
    # 4 - Train model
    use_phase = False
    logger = Logger(sess, config)
    trainer = CRNNMaskTrainer(sess, model, train_data, valid_data, config, logger,
                              mel_feats=False, use_phase=use_phase,
                              enable_validation_on_real_data=True)
    trainer.train()

    create_dirs([os.path.join(config.results_dir, folder_name) for folder_name, _ in dict_test_mix_files.items()])

    #################################################################
    # 7 - Predict result
    predictor = CRNNMaskPredict(sess, model, config, mel_feats=False, use_phase=use_phase,
                                no_context_pred=no_context_pred)
    for folder_name, lst_mix_files in dict_test_mix_files.items():
        test_mix_files = [item[0] for item in lst_mix_files]
        predictor.predict(test_mix_files, os.path.join(config.results_dir, folder_name))
    thrs = np.linspace(0.1, 1, num=9)

    av_f1 = []
    n_bases = len(dict_test_mix_files.keys())

    pic = False
    for thr in thrs:
        print('-------------------------------------------------------')
        print('!! {} !!'.format(thr))
        av_f1_curr = 0

        for folder_name, _ in dict_test_mix_files.items():
            true_files_path = os.path.join(args.root_path, 'audio_test', folder_name)
            predicted_mask_path = os.path.join(config.results_dir, folder_name)
            (f1_score, mse, av_TN, av_FR), (bins, vals) = f1_util.process_files(true_files_path, predicted_mask_path,
                                                                                bin_thr=thr,
                                                                                return_mse=False, return_all_info=True,
                                                                                return_out_vec_dist=True)
            if not pic:
                plt.bar(x=vals[:-1].tolist(), height=bins.tolist(), width=0.02)
                plt.grid(True)

                pic_name = 'out_hist_{}_{}.png'.format(folder_name, thr)
                pic_path = os.path.join(pic_dir, pic_name)
                plt.savefig(pic_path)
                plt.clf()
                pic = True

            print('base_name: {}, f1_score: {}, mse: {}, av_TN: {}, av_FR: {}'.format(folder_name, f1_score, mse,
                                                                                      av_TN, av_FR))
            av_f1_curr += f1_score

        av_f1.append(av_f1_curr / n_bases)

    best_thr = thrs[np.argmax(av_f1)]
    _, dc_name = os.path.split(args.root_path)

    log_util.write_log(log_path, config, **{'av_f1': np.max(av_f1), 'best_thr': best_thr, 'dc': dc_name})


if __name__ == '__main__':
    main()
