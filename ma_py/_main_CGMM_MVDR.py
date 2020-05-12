import numpy as np
import soundfile as sf
import copy

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import *
from mic_py.mic_cgmm import est_cgmm, permute_mask, est_cgmm_ex
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector

import matplotlib.pyplot as plt



if __name__ == '__main__':


    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log)         = (13.8845, 6.60824)
    (angle_inf_hor_log, angle_inf_vert_log) = (-15.06, -0.31)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    angle_inf_h = -angle_inf_hor_log
    angle_inf_v = -angle_inf_vert_log


    in_wav_path    = r'./data/_sol/'
    out_wav_path   = r'./data/out/'

    _mix_start     = 28
    _mix_end       = 60
    #################################################################

    # #################################################################
    # # 1.0 - _nastya_alex PROFILE
    #
    # vert_mic_count = 1
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 256
    #
    # in_wav_path = r'./data/_nastya_alex/'
    # out_wav_path = r'./data/out/'
    #
    # # mus MicGridProcessor::SetDirectionAngles, angleFiHorz = 24.7455, angleFiVert = 18.2763
    # # sp MicGridProcessor::SetDirectionAngles, angleFiHorz = -30.9141, angleFiVert = 12.483
    #
    # (angle_hor_log, angle_vert_log)         = (-30.9141, 12.483)
    # (angle_inf_hor_log, angle_inf_vert_log) = (24.7455, 18.2763)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    #
    # _mix_start     = 30
    # _mix_end       = 38


    #################################################################
    # 1 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec=max_len_sec)
    x_all_arr = x_all_arr[:, (np.int32)(_mix_start * sr):(np.int32)(_mix_end * sr)]
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all = stft_arr(x_all_arr, fftsize=n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

    print("STFT calc done!")
    print("    n_bins     = ", n_bins)
    print("    n_sensors  = ", n_sensors)
    print("    n_frames   = ", n_frames)


    #################################################################
    # 3 - Calc  steering vector and PSD matrix
    print('Calc  steering vector!')

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)

    source_position  = get_source_position(angle_Hor=angle_h, angle_Vert=angle_v, radius=6.0)
    d_arr_sp = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    source_position = get_source_position(angle_Hor=angle_inf_h, angle_Vert=angle_inf_v, radius=6.0)
    d_arr_noise = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    psd_sp = np.einsum('i...,j...->...ij', d_arr_sp, d_arr_sp.conj())
    psd_noise = np.einsum('i...,j...->...ij', d_arr_noise, d_arr_noise.conj())

    #################################################################
    # 4 - Calc cgmm mask
    mask, R = est_cgmm_ex(stft_all, psd_sp, psd_noise, num_iters=10, allow_cov_update=False)


    # #################################################################
    # # 4.1 - Show results
    #
    # show_mask = mask[:,::-1,:]
    # spec_ch_0_0 = np.log(np.abs(stft_all[::-1,0,:]) + 0.0001)
    #
    # plt.subplot(311)
    # plt.imshow(show_mask[0,:,:])
    # plt.xlabel('Time')
    # plt.ylabel('Freq')
    #
    # plt.subplot(312)
    # plt.imshow(show_mask[1,:,:])
    # plt.xlabel('Time')
    # plt.ylabel('Freq')
    #
    # plt.subplot(313)
    # plt.imshow(spec_ch_0_0)
    # plt.xlabel('Time')
    # plt.ylabel('Freq')
    #
    # plt.show()

    #################################################################
    stft_all_noise = copy.deepcopy(stft_all)
    for i in range(0, n_sensors):
        stft_all_noise[:, i, :] *=  mask[0,:,:]

    #np.save('mask', mask)
    #np.save('R', R)
    #
    # mask = np.load('mask.npy')
    # R = np.load('R.npy')

    psd_noise_matrix = get_power_spectral_density_matrix(stft_all_noise)

    #################################################################
    # 5 -Regularisation psd matrix
    psd_noise_matrix = psd_noise_matrix + 0.01*np.identity(psd_noise_matrix.shape[-1])

    #################################################################
    # 6 - Apply MVDR
    w = get_mvdr_vector(d_arr_sp.T,  psd_noise_matrix)
    result_spec = apply_beamforming_vector(w, stft_all)

    #################################################################
    # 6 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_mvdr_cgmm_mask.wav", sig_out, sr)



