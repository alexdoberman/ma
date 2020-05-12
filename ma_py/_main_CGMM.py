import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import *
from mic_py.mic_cgmm import est_cgmm, permute_mask, est_cgmm_ex

import matplotlib.pyplot as plt


if __name__ == '__main__':

    #################################################################
    # 1.0 - _nastya_alex PROFILE

    vert_mic_count = 1
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 45
    n_fft = 256

    #in_wav_path = r'./data/_nastya_alex_cut/'
    in_wav_path = r'./data/_nastya_alex/'

    out_wav_path = r'./data/out/'

    _mix_start     = 30
    _mix_end       = 38
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

    # mus MicGridProcessor::SetDirectionAngles, angleFiHorz = 24.7455, angleFiVert = 18.2763
    # sp MicGridProcessor::SetDirectionAngles, angleFiHorz = -30.9141, angleFiVert = 12.483

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = get_source_position(angle_Hor=30.9, angle_Vert=-12.48, radius=6.0)
    d_arr_1 = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    source_position = get_source_position(angle_Hor=-24.7, angle_Vert=-18.27, radius=6.0)
    d_arr_2 = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    psd_1 = np.einsum('i...,j...->...ij', d_arr_1, d_arr_1.conj())
    psd_2 = np.einsum('i...,j...->...ij', d_arr_2, d_arr_2.conj())

    #################################################################
    #4 - Calc mask
    mask, R = est_cgmm(stft_all, num_iters=10)

    np.save('mask', mask)
    np.save('R', R)
    #
    # mask = np.load('mask.npy')
    # R = np.load('R.npy')
    #
    perm_mask = permute_mask(mask=mask, R=R)


    #################################################################
    # 4 - Calc mask ex
    # mask, R = est_cgmm_ex(stft_all, psd_1, psd_2, num_iters=10, allow_cov_update=False)
    # np.save('mask', mask)
    # np.save('R', R)

    #
    # mask = np.load('mask.npy')
    # R = np.load('R.npy')


    #################################################################
    # 4.1 - Show results

    show_mask = mask[:,::-1,:]
    spec_ch_0_0 = np.log(np.abs(stft_all[::-1,0,:]) + 0.0001)

    plt.subplot(311)
    plt.imshow(show_mask[0,:,:])
    plt.xlabel('Time')
    plt.ylabel('Freq')

    plt.subplot(312)
    plt.imshow(show_mask[1,:,:])
    plt.xlabel('Time')
    plt.ylabel('Freq')

    plt.subplot(313)
    plt.imshow(spec_ch_0_0)
    plt.xlabel('Time')
    plt.ylabel('Freq')

    plt.show()

    #
    # ##############################################################
    # show_mask = perm_mask[:,::-1,:]
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
    # 4 - Save result

    result_spec = np.zeros((2, n_bins, n_frames), dtype=np.complex)

    for k in range(2):
        for f in range(n_bins):
            for t in range(n_frames):
                result_spec[k, f, t] = stft_all[f, 0, t] * perm_mask[k, f, t]
                #result_spec[k, f, t] = stft_all[f, 0, t] * mask[k, f, t]

    for k in range(2):
        sig_out = istft(result_spec[k, :, :].T, overlap=2)
        sf.write(r"out/out_CGMM_" + str(k) + ".wav", sig_out, sr)

