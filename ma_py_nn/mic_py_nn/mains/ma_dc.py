# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import random
from shutil import copyfile

from mic_py_nn.data_generator.ma_dataset import Dataset

from mic_py_nn.models.ma_dc_model import DCModel
from mic_py_nn.data_generator.batchers.multi_channel_dc_batcher import WithoutSilenceBatcherWrapper as Batcher
from mic_py_nn.trainers.clustering_train import DCTrainer
from mic_py_nn.trainers.clustering_predict import MADCPredict

from mic_py_nn.utils.config import process_config
from mic_py_nn.utils.file_op_utils import create_dirs, save_obj
from mic_py_nn.utils.logger import Logger
from mic_py_nn.utils.utils import get_args


def main():

    #################################################################
    # 0 - Capture the config path from the run arguments then process the json configuration file
    args = get_args()
    config = process_config(args.config, args.root_path)

    #################################################################
    # 1 - Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.results_dir, config.to_separate_dir])

    #################################################################
    # 2 - Create tensorflow session and load model if exists
    sess = tf.Session()
    model = DCModel(config)
    model.load(sess)

    #################################################################
    # 3 - Load dataset
    dataset = Dataset(root_path=args.root_path)

    train_spk_files = dataset.get_train_files()
    valid_spk_files = dataset.get_valid_files()
    dict_test_mix_files = dataset.get_test_files()
    to_separate_files = dataset.get_to_separate_files()

    #################################################################
    # 4 - Create subfolders in the result dir
    create_dirs([os.path.join(config.results_dir, folder_name) for folder_name,_ in dict_test_mix_files.items()])

    #################################################################
    # 5 - Create batchers
    train_data = Batcher(train_spk_files, config)
    valid_data = Batcher(valid_spk_files, config)

    #################################################################
    # 6 - Train model
    if args.stage == 'train':
        logger = Logger(sess, config)
        trainer = DCTrainer(sess, model, train_data, valid_data, config, logger)
        trainer.train()

    #################################################################
    # 7 - Predict result
    predictor = MADCPredict(sess, model, config)

    if args.stage == 'train' or args.stage == 'eval':
        for folder_name, lst_mix_files in dict_test_mix_files.items():
            test_mix_files = [item[0] for item in lst_mix_files]
            predictor.predict(test_mix_files, os.path.join(config.results_dir, folder_name))

    # 7.1 Sample predict for real data
    if args.stage == 'train' or args.stage == 'eval' or args.stage == 'predict':
        if len(to_separate_files) > 0:
            predictor.predict(to_separate_files, config.to_separate_dir, index_naming=False)

    if args.stage == 'train' or args.stage == 'eval':
        #################################################################
        # 8 - Calc metric
        SDRi_dict = {}
        for folder_name, _ in dict_test_mix_files.items():
            true_separated_path = os.path.join(args.root_path, 'audio_test', folder_name)
            separated_path = os.path.join(config.results_dir, folder_name)
            SDR_impr_aver, SIR_impr_aver, SAR_impr_aver = dataset.evaluate_sdr_metric(true_separated_path,
                                                                                      separated_path)

            SDRi_dict[folder_name + '_SDRi'] = SDR_impr_aver[0]

        #################################################################
        # 9 - Save result to json obj
        save_obj(data=SDRi_dict, filename=os.path.join(config.results_dir, 'result.json'))

        #################################################################
        # 10 - Copy config scripts to result dir
        copyfile(args.config, os.path.join(config.results_dir, os.path.basename(args.config)))


if __name__ == '__main__':
    main()
