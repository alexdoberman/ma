# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from mic_py.mic_adaptfilt import *
from mic_py.feats import *

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_seg_io import read_seg_file_pause, convert_time_segm_to_frame_segm
import pyroomacoustics
from mic_py.mic_adaptfilt import *
from mic_py.mic_adaptfilt_time_domain import affine_projection_filter, lms_filter
import time


def do_ds():
    #################################################################
    # 1.0 - out_mus1_spk1_snr_-10 PROFILE

    vert_mic_count = 6
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 60
    n_fft = 512

    # (angle_hor_log, angle_vert_log) = (7.1539, 7.39515)
    (angle_hor_log, angle_vert_log) = (-16.0721, -0.163439)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    _mix_start = 10
    _mix_end = 50

    in_wav_path = r'./data/_sdr_test/out_mus1_spk1_snr_-15/mix/'
    out_wav_path = r'./data/out/'
    #################################################################

    #################################################################
    # 1.0 - Read signal
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
    # 3 - Calc  steering vector
    print('Calc  steering vector!')
    print('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = get_source_position(angle_h, angle_v, radius=6.0)
    d_arr = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    #################################################################
    # 4 - Calc DS filter output
    result_spec = ds_beamforming(stft_all, d_arr.T)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/out_DS.wav", sig_out, sr)


# def spectral_substract_filter(stft_main, stft_ref, alfa_PX=0.01, alfa_PN=0.99):
#     """
#     spectral subtraction filter
#
#     :stft_main: - spectr  main signal  - shape (bins, frames)
#     :stft_ref:  - spectr  ref signal   - shape (bins, frames)
#     :alfa_PX:   - smooth factor, range: 0 .. 1
#     :alfa_PN:   - smooth factor, range: 0 .. 1
#
#     :return:
#         output - spectral subtraction compensate  - shape (bins, frames)
#     """
#
#     X_mag = np.absolute(stft_main)
#     N_mag = np.absolute(stft_ref)
#
#     PX = X_mag ** 2
#     PN = N_mag ** 2
#
#     def exp_average(X, Alpha):
#         nLen = X.shape[0]
#
#         Y = np.zeros(X.shape)
#         for i in range(0, nLen - 1, 1):
#             Y[i + 1, :] = Alpha * Y[i, :] + (1 - Alpha) * X[i + 1]
#         return Y
#
#     PX = exp_average(PX, alfa_PX)  # 0   .. 0.5
#     PN = exp_average(PN, alfa_PN)  # 0.5 .. 1
#
#     # Power subtraction
#     alfa = 0.5
#     beta = 1.0
#     gamma = 1.0
#     #
#     # # Magnitude subtraction
#     # alfa  = 1.0
#     # beta  = 0.5
#     # gamma = 1.0
#
#     # # Wiener gain
#     # alfa  = 1.0
#     # beta  = 1.0
#     # gamma = 1.0
#
#
#     Gain = np.maximum(1.0 - (PN / (PX + eps) * gamma) ** beta, 0.01) ** alfa
#
#     result = stft_main * Gain
#     return result


def main2():
    X_wav_path = r'./out/ADAPT_SAD/SNR_-8/out_DS_mix_sp.wav'
    N_ideal_wav_path = r'./out/ADAPT_SAD/SNR_-8/out_DS_mus_sp.wav'
    N_wav_path = r'./out/ADAPT_SAD/SNR_-8/out_DS_mix_mus.wav'

    # N_wav_path  = r'./out/ADAPT_SAD/SNR_-8/out_DS_mus_sp.wav'
    # N_wav_path = r'./out/ADAPT_SAD/SNR_-8/out_DS_mix_mus.wav'
    OUT_wav_path = r'./out/ADAPT_SAD/SNR_-8/_result.wav'

    n_fft = 512
    overlap = 2

    ####################################################
    # 1 - Load signal
    X_sig, rate = sf.read(X_wav_path)
    N_ideal_sig, rate = sf.read(N_ideal_wav_path)
    N_sig, rate = sf.read(N_wav_path)

    # X_sig = np.roll(X_sig, -10)

    X_spec = stft(X_sig, fftsize=n_fft, overlap=overlap)
    N_spec = stft(N_sig, fftsize=n_fft, overlap=overlap)
    N_ideal_spec = stft(N_ideal_sig, fftsize=n_fft, overlap=overlap)

    n_frames, b_bins = X_spec.shape

    start = time.time()
    #OUT = spectral_substract_filter(stft_main=X_spec, stft_ref=N_spec, alfa_PX=0.001, alfa_PN=0.00199199)
    #OUT  = compensate_ref_ch_filter(stft_main=X_spec, stft_ref=N_spec, alfa=0.75)
    #OUT = compensate_ref_ch_filter_ex(stft_main=X_spec, stft_ref=N_spec, alfa=.0075, beta=0.95)
    OUT = smb_filter(stft_main=X_spec, stft_ref=N_spec, gain_max=18)


    # 6.1 - AP filter + cyclic parameters
    M = 200
    step = 0.05
    L = 5
    leak = 0.1
    delay = -5

    # AP filter
    #sig_result = affine_projection_filter(main=X_sig, ref=N_sig, M=M, step=step, L=L, leak=leak)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    ################################################

    start = time.time()
    for g in range (1,28):
        OUT = smb_filter(stft_main=X_spec, stft_ref=N_spec, gain_max=g)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    ################################################








    # NLMS
    #sig_result = lms_filter(main=X_sig, ref=N_sig, M=M, step=step, leak=leak, norm=True)

    # LMS
    #sig_result = lms_filter(main=X_sig, ref=N_sig, M=M, step=step, leak=leak, norm=False)


    # ####################################################
    # # 4 - Plot
    # n_bin = 50
    # T = np.arange(0, n_frames) * (n_fft/overlap) / rate
    # E1 = np.abs(X_spec)[:, n_bin]
    # E2 = np.abs(N_ideal_spec)[:, n_bin]
    # E3 = np.abs(est_noise)[:, n_bin]
    #
    # plt.plot(T, E1,'r--', T, E2,'b', T, E3,'g')
    # plt.show()
    # ####################################################
    #
    # X_sig
    # plt.plot(T, E1,'r--', T, E2,'b', T, E3,'g')
    # plt.show()


    sig_out = istft(OUT, overlap=overlap)
    #sig_out = sig_result
    sf.write(OUT_wav_path, sig_out, rate)


def main():
    seg_path = r'./out/ADAPT_SAD/ds_spk.seg'
    X_wav_path = r'./out/ADAPT_SAD/out_DS_sp.wav'
    N_wav_path = r'./out/ADAPT_SAD/out_DS_inf.wav'
    S_wav_path = r'./out/ADAPT_SAD/ds_spk.wav'

    n_fft = 512
    overlap = 256

    ####################################################
    # 1 - Load signal
    X_sig, rate = sf.read(X_wav_path)
    N_sig, rate = sf.read(N_wav_path)
    S_sig, rate = sf.read(S_wav_path)

    ####################################################
    # 2 - SFFT signal
    X_spec = stft(X_sig, fftsize=n_fft, overlap=2)
    N_spec = stft(N_sig, fftsize=n_fft, overlap=2)
    S_spec = stft(S_sig, fftsize=n_fft, overlap=2)

    (n_frames, n_bins) = X_spec.shape
    assert X_spec.shape == N_spec.shape and X_spec.shape == S_spec.shape, 'X_spec.shape == N_spec.shape and X_spec.shape == S_spec.shape'
    print('X_spec.shape = ', X_spec.shape)
    print('N_spec.shape = ', N_spec.shape)
    print('S_spec.shape = ', S_spec.shape)

    ####################################################
    # 3 - Read true time segmentation
    count_frames = X_spec.shape[0]
    true_time_segm, freq = read_seg_file_pause(seg_file=seg_path)
    true_frame_segm = convert_time_segm_to_frame_segm(time_seg=true_time_segm,
                                                      count_frames=count_frames,
                                                      fs=freq, overlap=overlap)
    true_frame_segm = 1 - true_frame_segm

    ####################################################
    # 4 - Plot
    T = np.arange(0, n_frames) * overlap / rate
    E = np.sum(np.abs(X_spec), axis=1)
    E = E / np.max(E)

    plt.plot(T, E, T, true_frame_segm)
    plt.show()

    ####################################################
    # Filter signal
    # S_spec =  spectral_substract_filter(stft_main= X_spec , stft_ref= N_spec, alfa_PX = 0.01, alfa_PN = 0.99)
    # S_spec = smb_filter(stft_main=X_spec, stft_ref=N_spec, gain_max=18)

    ####################################################
    # ISFFT signal and write result
    sig_out = istft(S_spec, overlap=2)
    sf.write(r'./out/ADAPT_SAD/_result.wav', sig_out, rate)

    ####################################################
    # initialize the filter
    rls = pyroomacoustics.adaptive.RLS(30)

    # run the filter on a stream of samples
    for i in range(100):
        rls.update(X_sig[i], N_sig[i])

    # the reconstructed filter is available
    print('Reconstructed filter:', rls.w)


# def main(X_wav_path, N_wav_path, OUT_wav_path):
#
#     # Load signal
#     X_sig, rate = sf.read(X_wav_path)
#     N_sig, rate = sf.read(N_wav_path)
#
#     # SFFT signal
#     X_spec    = stft(X_sig)
#     N_spec    = stft(N_sig)
#
#     # Filter signal
#     #S_spec =  compensate_ref_ch_filter(stft_main = X_spec, stft_ref = N_spec, alfa = 0.7)
#     S_spec =  spectral_substract_filter(stft_main= X_spec , stft_ref= N_spec, alfa_PX = 0.01, alfa_PN = 0.99)
#
#
#     # ISFFT signal
#     sig_out = istft(S_spec)
#
#     # ISFFT signal
#     sf.write(OUT_wav_path, sig_out, rate)


if __name__ == '__main__':
    #     #X_wav_path = r'.\data\_adapt_simple\spk_out_DS.wav'
    #     #N_wav_path  = r'.\data\_adapt_simple\mus_out_DS.wav'
    #     #OUT_wav_path = r'.\out\ss.wav'
    #
    #     X_wav_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Y.wav'
    #     N_wav_path  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Z.wav'
    #     OUT_wav_path  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_OUT_spec_subs.wav'
    #
    #
    # #    X_wav_path = r'.\data\_adapt_simple\x.wav'
    # #    N_wav_path  = r'.\data\_adapt_simple\n.wav'
    # #    OUT_wav_path = r'.\out\ss.wav'
    #
    #     main(X_wav_path, N_wav_path, OUT_wav_path)
    # do_ds()
    main2()
