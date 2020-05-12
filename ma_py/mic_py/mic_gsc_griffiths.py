# -*- coding: utf-8 -*-
import numpy as np
from mic_py import mic_ds_beamforming
from mic_py import mic_adaptfilt


def gsc_griffiths_filter(stft_arr, d_arr, mic_pos='closed'):
    """
    GSC filter

    :stft_arr: - specter for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)
    :mic_pos:  - sensors: 'closed' - [0, 1]
                          'diam' - [0, 65]

    :return:
        ds_spec - result spectral  - shape (bins, frames)
    """

    speech_dir_beamforming = mic_ds_beamforming.ds_beamforming(stft_arr, d_arr)
    sp_d = np.einsum('ijk,ij->ijk', stft_arr, d_arr.conj())
    _, num_sensors = d_arr.shape

    if mic_pos == 'closed':
        new_ref = sp_d[:, 0, :] - sp_d[:, 1, :]
    elif mic_pos == 'diam':
        new_ref = sp_d[:, 0, :] - sp_d[:, num_sensors-1, :]
    else:
        print('such sensors position is not supported yet!')
        return

    output = mic_adaptfilt.spectral_substract_filter(stft_main=speech_dir_beamforming,
                                                     stft_ref=new_ref, alfa_PX=0.01, alfa_PN=0.99)

    return output


