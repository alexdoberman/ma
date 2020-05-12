# -*- coding: utf-8 -*-
import numpy as np
from mic_py import mic_ds_beamforming
from mic_py import mic_adaptfilt


def eq_filter(stft_arr_one_ch, gain):
    """
    EQ filter

    :stft_arr_one_ch: - specter for sensors - shape (bins, frames)
    :gain:            - gain                - shape (bins)

    :return:
        eq_spec        - result spectral  - shape (bins, frames)
    """

    eq_spec = stft_arr_one_ch*gain
    return eq_spec
