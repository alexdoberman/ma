import logging

import sys
import numpy as np
import soundfile as sf
sys.path.append('../')
from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import propagation_vector_free_field


class BaseFilter:

    levels = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET
    }

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        self.mic_config = mic_config
        self.enable_logging = logger_flag

        try:
            logging.basicConfig(level=self.levels[logger_level])
        except:
            logging.basicConfig(level=logging.DEBUG)
            logging.warning('\n Can\'t configure logger with given settings. Set logger level to DEBUG.')

    def get_raw_wav(self, path_to_record):
        x_all_arr, sr = read_mic_wav_from_folder(path_to_record, self.mic_config.vert_mic_count,
                                                 self.mic_config.hor_mic_count,
                                                 max_len_sec=self.mic_config.max_len_sec)

        (n_channels, n_samples) = x_all_arr.shape
        logging.debug('\n Reading of input wav done!\n\t n_channels  = {}\n\t n_samples   = {}\n\t freq        = {}'
                      .format(n_channels, n_samples, sr))

        return x_all_arr, sr

    def get_stft(self, arr):
        stft_res = stft_arr(arr, fftsize=self.mic_config.n_fft)
        n_bins, n_sensors, n_frames = stft_res.shape
        logging.debug('\n STFT calc done! \n\t n_bins     = {} \n\t n_sensors  = {} \n\t n_frames   = {}'
                      .format(n_bins, n_sensors, n_frames))

        return stft_res

    def get_istft(self, arr, overlap=2):
        return istft(arr.T, overlap=overlap)

    def get_steering(self, angle_h, angle_v, sr, radius=6):
        sensor_positions = get_sensor_positions(self.mic_config.hor_mic_count, self.mic_config.vert_mic_count,
                                                dHor=self.mic_config.dHor, dVert=self.mic_config.dVert)
        source_position = get_source_position(angle_h, angle_v, radius=radius)
        d_arr = propagation_vector_free_field(sensor_positions, source_position, N_fft=self.mic_config.n_fft,
                                              F_s=sr)

        logging.debug('\n Calc  steering vector done!')
        logging.debug('\n\t(angle_h, angle_v) = {}, {}'.format(angle_h, angle_v))

        return d_arr

    def write_result(self, out_wav_path, signal, sr):
        sf.write(out_wav_path, signal, sr)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, **kwargs):
        """

        :param in_wav_path:
        :param out_wav_path:
        :param filter_cfg:
        :param kwargs:  - parameters for filter
        :return:
        """
        pass
