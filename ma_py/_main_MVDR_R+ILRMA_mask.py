# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector
from mic_test.calc_metric_simulation import calc_metric
from mic_py.mic_double_exp_averaging import double_exp_average
from mic_py.mic_make_mask import make_mask, mask_to_frames
from mic_py.mic_ilrma import ilrma
from scipy.stats import entropy


if __name__ == '__main__':

    # 0 - Define params
    #################################################################
    # 1.0 - _du_hast PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (7.1539, 7.39515)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_sdr_test/out_mus2_spk1_snr_-20/mix'
    out_wav_path   = r'./data/out/'



    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_all_arr = x_all_arr[:, 10*sr:]
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)




    # 2 - Make mask

    stft_all_arr = stft_arr(x_all_arr, fftsize=n_fft)

    stft_all_arr_ilrma = np.concatenate(
        (np.transpose(stft_all_arr, (2, 0, 1))[:, :, :7], np.transpose(stft_all_arr, (2, 0, 1))[:, :, 11:18]),
        axis=-1)

    weights = np.load(r'./mic_utils/room_simulation\room_simulation_Gleb\weights14.npy')
    res = ilrma(stft_all_arr_ilrma, n_iter=15, n_components=2, W0=weights, seed=0)

    entr = np.zeros(res.shape[-1])
    for i in range(res.shape[-1]):
        for j in range(res.shape[1]):
            entr[i] += entropy(np.real(res[:, j, i] * np.conj(res[:, j, i])))


    resulting_sig = istft(res[:, :, np.argmin(entr)], overlap=2)

    sf.write(r'./out/simulation/res_sig.wav', resulting_sig, sr)

    average_sig = double_exp_average(resulting_sig, sr)
    average_sig[-300:] = average_sig[-300]
    average_sig[:300] = average_sig[300]

    mask = make_mask(average_sig, percent_threshold=100)
    mask_frames = mask_to_frames(mask, int(n_fft), int(n_fft / 2))

    # plt.plot(mask_frames)
    # plt.show()

    print('Mask is ready!')

    #################################################################
    # 2 - Do STFT

    indicies = mask_frames == 1
    stft_mix_arr   =  stft_all_arr
    stft_noise_arr = stft_all_arr[:,:, indicies]

    # stft_noise_arr = stft_noise_arr[:,:, :623]




    (n_bins, n_sensors, n_frames) = stft_noise_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_noise_arr.shape = ", stft_noise_arr.shape)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)



    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position    = get_source_position(angle_h, angle_v)
    d_arr              = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4  - Calc psd matrix
    psd_noise_matrix = get_power_spectral_density_matrix(stft_noise_arr)
    print ('Calc psd matrix done!')
    print ('    psd_noise_matrix.shape = ', psd_noise_matrix.shape)

    np.save(r'./out/psd_noise_matrix', psd_noise_matrix)


    #################################################################
    # 5 -Regularisation psd matrix
    psd_noise_matrix = psd_noise_matrix + 0.01*np.identity(psd_noise_matrix.shape[-1])

    #################################################################
    # 6 - Apply MVDR
    w = get_mvdr_vector(d_arr.T,  psd_noise_matrix)
    result_spec = apply_beamforming_vector(w, stft_mix_arr)

    #################################################################
    # 6 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)

    out_path = r"out/Simulation/MVDR+ILRMA/mvdr+ilrma.wav"
    sf.write(out_path, sig_out, sr)

    # 7 Calculate metric

    sp_path = r'./data/_sdr_test/out_mus2_spk1_snr_-20/ds_spk.wav'

    ds_speaker_path = r'./data/_sdr_test/out_mus2_spk1_snr_-20/ds_mix.wav'

    print(np.round(calc_metric(ds_speaker_path, sp_path, sp_path, out_path)[0][0],2))





    #################################################################
    # # 7.1 - Do align
    # align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    #
    # #################################################################
    # # 7.2 save ds output
    # result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    # sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    # sf.write(r"out/ds.wav", sig_out, sr)



