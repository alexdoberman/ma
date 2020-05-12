# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import copy
import math
import sys
import os

sys.path.append('../')

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector

from mic_py.mic_cov_marix_taper import cov_matrix_tapper_bandwidth

from mic_py.mic_zelin import zelin_filter

from mic_py_nn.mains.sample_predict import ChimeraPredict, ChimeraPredictFrozen

EXP_AV_COEF = 0.99
REG_COEF = 0.001
MVDR_TIME_STEP = 5
TIME_STEP = 5


def time_to_frame(time, sr, n_fft, overlap):
    hop_size = n_fft // overlap
    return math.floor(time * sr / hop_size)


if __name__ == '__main__':

    #################################################################
    # 1.0 - _du_hast PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 2 * 60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_du_hast/'
    out_wav_path = r'./data/out/'

    _noise_start = 8
    _noise_end = 17

    _mix_start = 17
    _mix_end = 84

    _sp_start = 84
    _sp_end = 102
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_mix = x_all_arr[:,(np.int32)(_noise_start*sr):(np.int32)(_mix_end*sr)]

    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

    stft_mix = stft_arr(x_mix, fftsize=n_fft)

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)

    # 4 - DS filter
    result_spec = ds_beamforming(stft_mix, d_arr.T)

    # 5 - inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    out_ds_path = r'{}/tmp_ds.wav'.format(os.path.dirname(out_wav_path))
    sf.write(out_ds_path, sig_out, sr)

    # 6 - get noise mask
    print('Get noise mask form neural net!')

    # shape - (time, freq, 2)
    config_path = r'/home/stc/MA_ALG/datasets/test_ma/' \
                  r'chimera_v12/9_chimera.json'
    in_model_path = r'/home/stc/MA_ALG/datasets/test_ma/chimera_v12/checkpoint/'

    frozen_model_path = r'/home/stc/MA_ALG/datasets/test_ma/chimera_frozen/model_chimera_v11.pb'

    mask = ChimeraPredictFrozen(frozen_model_path).predict_mask(out_ds_path)

    # mask = ChimeraPredict(config_path, in_model_path).predict_mask(out_ds_path, os.path.dirname(out_wav_path))

    # 7  - Calc psd matrix
    # bin x frames x mask
    mask = np.transpose(mask, (1, 0, 2))

    print('Calc psd matrix!')
    print('     mask.shape      = {}'.format(mask.shape))
    print('     stft_mix.shape  = {}'.format(stft_mix.shape))

    # TODO in deep clustering need choice mask
    actual_mask = mask[:, :, 1]

    stft_mix_noise = copy.deepcopy(stft_mix)
    # stft_mix_noise = copy.deepcopy(stft_mix[:,:,0:-2])

    (n_bins, n_sensors, n_frames) = stft_all.shape
    (n_bins_mix, n_sensors_mix, n_frames_mix) = stft_mix.shape

    for i in range(0, n_sensors):
        stft_mix_noise[:, i, 0:-2] *= actual_mask

    sig_out = istft(stft_mix_noise[:, 1, :], overlap=2)

    sf.write(r"out/noise.wav", sig_out, sr)

    frame_step = time_to_frame(TIME_STEP, sr, n_fft, 2)

    EXP_AV_COEF = 1 - (1 / frame_step)

    print('EXP_AV_COEF: {}'.format(EXP_AV_COEF))

    # res_spec = ds_beamforming(stft_mix[:, :, :frame_step], d_arr.T)
    res_spec = None
    psd = np.zeros(shape=(257, 66, 66), dtype=np.complex)

    taper = np.ones(shape=(257, 66, 66))

    mvdr_step = time_to_frame(MVDR_TIME_STEP, sr, n_fft, 2)

    for i in range(0, n_frames_mix, mvdr_step):

        for j in range(min(mvdr_step, n_frames_mix - i - 1)):

            psd_curr = np.zeros((257, 66, 66), dtype=np.complex)
            for k in range(n_bins):
                psd_curr[k] = np.outer(stft_mix_noise[k, :, i + j], stft_mix_noise[k, :, i + j].conj())

            psd_curr *= taper
            psd = EXP_AV_COEF * psd + (1 - EXP_AV_COEF) * psd_curr

        psd_reg = psd + REG_COEF * np.identity(psd.shape[-1])

        w = get_mvdr_vector(d_arr.T, psd_reg)

        if i == 0:
            taper, _ = cov_matrix_tapper_bandwidth(stft_mix[:, :, 0:mvdr_step], hor_mic_count,
                                                   vert_mic_count, dHor=dHor,
                                                   dVert=dVert, angle_v=angle_v,
                                                   angle_h=angle_h, sr=sr)
            w = np.zeros((n_bins, n_sensors))
            pp = apply_beamforming_vector(w, stft_mix[:, :, i:i + mvdr_step])
            res_spec = pp
            w = np.zeros((n_bins, n_sensors))

        pp = apply_beamforming_vector(w, stft_mix[:, :, i + mvdr_step:i + 2 * mvdr_step])

        if res_spec is None:
            res_spec = pp
        else:
            res_spec = np.hstack((res_spec, pp))

    print('Result shape: {}'.format(res_spec.shape))

    align_stft_arr = ds_align(stft_mix, d_arr.T)

    _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)

    res_spec = res_spec * H

    # 5 - inverse STFT and save
    sig_out = istft(res_spec.T, overlap=2)

    sf.write(r"out/AD_NN_du_hast.wav", sig_out, sr)

