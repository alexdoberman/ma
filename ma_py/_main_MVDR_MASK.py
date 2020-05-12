# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import copy

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector


if __name__ == '__main__':

    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_sol/'
    out_wav_path   = r'./data/out/'


    _mix_start     = 28
    _mix_end       = 98

    #################################################################



    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

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


    #################################################################
    # 4  - Calc psd matrix
    mask = np.load(r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\result_sol\mask_ds.npy')

    # bin x frames x mask
    mask = np.transpose(mask, (1,0,2))
    actual_mask = mask[:, 0:n_frames, 0]

    stft_all_noise = copy.deepcopy(stft_all)
    for i in range(0, n_sensors):
        stft_all_noise[:, i, :] *=  actual_mask

    psd_noise_matrix = get_power_spectral_density_matrix(stft_all_noise)

    #################################################################
    # 5 -Regularisation psd matrix
    psd_noise_matrix = psd_noise_matrix + 0.01*np.identity(psd_noise_matrix.shape[-1])

    #################################################################
    # 6 - Apply MVDR
    w = get_mvdr_vector(d_arr.T,  psd_noise_matrix)
    result_spec = apply_beamforming_vector(w, stft_all)

    #################################################################
    # 6 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_mvdr_mask.wav", sig_out, sr)


    #################################################################
    # 7.1 - Do align
    align_stft_arr = ds_align(stft_all, d_arr.T)

    #################################################################
    # 7.2 save ds output
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)



