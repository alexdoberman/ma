# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from shutil import copyfile

from mic_py_nn.data_generator.dataset import Dataset
# from mic_py_nn.data_generator.batchers.base_batchers import MaskBatcher
from mic_py_nn.data_generator.batchers.cocktail_noise_batcher import MaskBatcher

from mic_py_nn.models.unet_model import UNetModel
from mic_py_nn.trainers.unet_trainer import UNetTrainer
from mic_py_nn.trainers.unet_predict import UNetPredict

from mic_py_nn.utils.config import process_config
from mic_py_nn.utils.logger import Logger
from mic_py_nn.utils.utils import get_args
from mic_py_nn.utils.file_op_utils import create_dirs, save_obj


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

    '''
    tf_config = tf.ConfigProto(allow_soft_placement=True)

    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)
    '''

    sess = tf.Session()
    model = UNetModel(config)
    model.build_model()
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
    train_data = MaskBatcher(train_spk_files, train_noise_files, config)
    valid_data = MaskBatcher(valid_spk_files, valid_noise_files, config)

    #################################################################
    # 4 - Train model
    logger  = Logger(sess, config)
    trainer = UNetTrainer(sess, model, train_data, valid_data, config, logger)
    trainer.train()

    create_dirs([os.path.join(config.results_dir, folder_name) for folder_name,_ in dict_test_mix_files.items()])

    #################################################################
    # 7 - Predict result
    predictor = UNetPredict(sess, model, config)

    for folder_name, lst_mix_files in dict_test_mix_files.items():
        test_mix_files = [item[0] for item in lst_mix_files]
        predictor.predict(test_mix_files, os.path.join(config.results_dir, folder_name))

    '''
    # 7.1 Sample predict for real data
    if len(to_separate_files) > 0:
        predictor.predict(to_separate_files, config.to_separate_dir)
    '''

    #################################################################
    # 8 - Calc metric
    SDRi_dict = {}
    for folder_name, _ in dict_test_mix_files.items():
        true_separated_path = os.path.join(args.root_path, 'audio_test', folder_name)
        separated_path = os.path.join(config.results_dir, folder_name)
        SDR_impr_aver, SIR_impr_aver, SAR_impr_aver = dataset.evaluate_sdr_metric(true_separated_path, separated_path)

        SDRi_dict[folder_name + '_SDRi'] = SDR_impr_aver[0]

    #################################################################
    # 9 - Save result to json obj
    save_obj(data=SDRi_dict, filename=os.path.join(config.results_dir, 'result.json'))

    #################################################################
    # 10 - Copy config scripts to result dir
    copyfile(args.config, os.path.join(config.results_dir, os.path.basename(args.config)))


if __name__ == '__main__':
    main()
