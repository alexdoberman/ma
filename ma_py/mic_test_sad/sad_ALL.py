# -*- coding: utf-8 -*-
import sys
import logging
sys.path.append('../')

import numpy as np
from base_sad import BaseSAD


class ALL_SAD(BaseSAD):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(ALL_SAD, self).__init__(mic_config, logger_flag, logger_level)


    def do_sad(self, in_wav_path, filter_cfg, **kwargs):

        threshold = kwargs.get('threshold', 0.0)
        logging.debug('\n threshold = {}'.format(threshold))

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        (n_bins, n_sensors, n_frames) = stft_all.shape
        frame_segm = np.ones(n_frames)

        return frame_segm



