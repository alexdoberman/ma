# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from mic_py.feats import istft, stft
from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field

import pyroomacoustics


def main():

    # #################################################################
    # # 1.0 - LANDA PROFILE
    # vert_mic_count = 1
    # hor_mic_count  = 8
    # dHor           = 0.05
    # dVert          = 0.05
    # max_len_sec    = 60
    # n_fft          = 512
    # n_overlap      = 2
    #
    # (angle_hor_log, angle_vert_log) = (16.0, 0.0)
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # _mix_start     = 17
    # _mix_end       = 40
    #
    # in_wav_path    = r'./data/_landa/'
    # out_wav_path   = r'./data/out/'
    # #################################################################

    #################################################################
    # 1.0 - _du_hast PROFILE

    #vert_mic_count = 6
    #hor_mic_count  = 11

    vert_mic_count = 1
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45

    n_fft          = 512
    n_overlap      = 2

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)
    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    _mix_start     = 17
    _mix_end       = 40

    in_wav_path    = r'./data/_du_hast/'
    out_wav_path   = r'./data/out/'
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
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft, overlap=n_overlap)
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
    # Reshape to (nframes, nfrequencies, nchannels)
    M_spec = stft_all.transpose((2, 0, 1))

    def callback_fn(X):
        print ('    invoke callback_fn')


    # #(nfrequencies, nchannels, nchannels)
    # W = np.array([np.eye(n_sensors, n_sensors) for f in range(n_bins)], dtype=M_spec.dtype)
    # for i in range(n_bins):
    #     W[i, : , 0 ] = d_arr[:, i]/8.0
    # Y = np.zeros((n_frames, n_bins, n_sensors), dtype=M_spec.dtype)
    # def demix(Y, X, W):
    #     for f in range(n_bins):
    #         Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))
    # demix(Y, M_spec, W)


    # # IVA Demix
    Y = pyroomacoustics.bss.auxiva(M_spec, n_iter=20, proj_back=True, callback=callback_fn)

    # Save result
    for i in range(Y.shape[2]):
        sig_out = istft(Y[:, :, i], overlap=n_overlap)
        sf.write(r'out/iva_res_{}.wav'.format(i), sig_out, sr)




if __name__ == '__main__':

    main()