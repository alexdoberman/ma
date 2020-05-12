# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from scipy import signal
import os

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_gsc import *
from mic_py.mic_gannot import rtf_filter
from mic_py.mic_geometry import get_sensor_positions_circular, get_source_position_circular
import matplotlib.pyplot as plt

def read_mic_wav_from_path(in_wav_path, mic_count):
    """
    Returns array observation.

    :return:
        shape (sensors, samples)
        sr  - sample rate
    """

    res = []
    for i in range(mic_count):
        file_name = os.path.join(in_wav_path, 'EN2001b.Array1-0{}.wav'.format(i+1))
        y, sr = sf.read(file_name, dtype=np.float64)
        res.append(y.T)

    return  np.vstack(res), sr

def ds_respk():

    # #################################################################
    # 1.0 - AMI dataset small session
    n_fft             = 512
    n_overlap         = 2

    r_circular_array   = 0.1
    mic_count          = 8
    in_wav_path        = r'./data/_ami_data'
    out_wav_path       = r'./data/out/'

    _mix_start          = 15.0
    _mix_end            = 22.0
    angle_azimuth       = 64.0


    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_path(in_wav_path, mic_count)
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)

    x_mix_arr = x_all_arr[:, (np.int32)(_mix_start * sr):(np.int32)(_mix_end * sr)]

    #################################################################
    # 2 - Do STFT
    stft_mix = stft_arr(x_mix_arr, fftsize=n_fft, overlap=n_overlap)

    (n_bins, n_sensors, n_frames) = stft_mix.shape
    print("STFT calc done!")
    print("    n_bins     = ", n_bins)
    print("    n_sensors  = ", n_sensors)
    print("    n_frames   = ", n_frames)


    #################################################################
    # 3 - Plot DN

    angle_step = 1
    print ("Begin steering calc ...")
    arr_angle = range(-180, 180, angle_step)
    arr_d_arr   = np.zeros((len(arr_angle), n_sensors, n_bins), dtype=np.complex)
    print ('arr_d_arr = ' , arr_d_arr.shape)

    for i , angle in enumerate (arr_angle):
        sensor_positions = get_sensor_positions_circular(r_circular_array, mic_count)
        source_position  = get_source_position_circular(azimuth = angle, polar = 90.0)
        arr_d_arr[i,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")

    #################################################################
    # 4 - Calc  map
    POW   = np.zeros(    (len(arr_angle))      )

    print ("Begin calc map ...")

    for i , angle_h in enumerate (arr_angle):
        print ("    process angle =  {}".format(angle_h))
        d_arr = arr_d_arr[i,:,:]

        # DS beamforming
        result_spec = ds_beamforming(stft_mix, d_arr.T)

        POW[i]      = np.real(np.sum(result_spec *np.conjugate(result_spec)) / n_frames)
    print ("Calc map done!")

    #################################################################
    # 5 - Scale to power ch_0_0
    P0  = np.sum(stft_mix[:,0,:] *np.conjugate(stft_mix[:,0,:])) / n_frames
    POW = POW/P0

    #################################################################
    # 6 - Plot DN
    plt.plot(arr_angle, POW)
    plt.xlabel('angle (s)')
    plt.ylabel('pow_res/pow_s0')
    plt.title('DS alg')
    plt.grid(True)
    plt.savefig(r".\out\DS_mic{}.png".format(mic_count))
    plt.show()

    #################################################################
    # 7 - Do DS beamforming
    print ('Calc  DS beamforming!')
    print ('    (angle_azimuth) = ', angle_azimuth)
    sensor_positions         = get_sensor_positions_circular(r_circular_array, mic_count)
    source_position          = get_source_position_circular(azimuth=angle_azimuth, polar=90.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    #################################################################
    # 4 - Calc DS filter output
    result_spec = ds_beamforming(stft_mix, d_arr.T)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_DS_{}.wav".format(angle_azimuth), sig_out, sr)


def plot_coherence():

    in_wav_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\data\_du_hast\32ch_output.wav'
    mic_count         = 32
    _mix_start        = 8
    _mix_end          = 17

    # in_wav_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\data\_no_echo_dist_3_angle_0_0\output.wav'
    # mic_count         = 32
    # _mix_start        = 0
    # _mix_end          = 60

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_file(in_wav_path, mic_count)
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)
    x_mix_arr = x_all_arr[:, (np.int32)(_mix_start * sr):(np.int32)(_mix_end * sr)]

    #################################################################
    # 1.0 - Plot coherence
    lst_mic_id_coh = [1,2,3,4,5,6,7]
    for i in lst_mic_id_coh:
        f, Cxy = signal.coherence(x = x_all_arr[0,:], y = x_all_arr[i,:], fs = sr, nperseg=1024)
        plt.semilogy(f, Cxy,  label='coh: {} -> {}'.format(0, i))

    plt.legend()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.savefig(r".\out\coh_mic.png".format(mic_count))
    plt.show()


    # #################################################################
    # # 2.0 - determine lag
    # def determine_lag(x, y, max_lag):
    #     lags = []
    #     for i in range(-max_lag, max_lag + 1, 1):
    #         corr = np.sum(x * np.roll(y, i))
    #         lags.append((i, corr))
    #
    #     m = max(lags, key=lambda item: item[1])
    #     #print(m)
    #     #    shift_y = np.roll(y, m[0])
    #     return m[0]
    #
    #
    # for i in lst_mic_id_coh:
    #     lag = determine_lag(x_all_arr[0, :], x_all_arr[i, :], max_lag =50)
    #     print ('lag: {} -> {}  =  {}'.format(0, i, lag))


    # #################################################################
    # # 3.0 - plot xcorr
    # for i in lst_mic_id_coh:
    #     plt.xcorr(x_mix_arr[0, :], x_mix_arr[i, :], maxlags =50)
    #     plt.savefig(r".\out\xcorr_{}.png".format(i))
    # plt.show()

if __name__ == '__main__':
    ds_respk()
    #plot_coherence()

