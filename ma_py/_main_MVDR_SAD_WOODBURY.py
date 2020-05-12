# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from numba import jit

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector, get_mvdr_vector_svd
from mic_py.mic_phase_sad import phase_sad
from mic_py.mic_ilrma_sad import ilrma_sad
from mic_py.mic_cov_marix_taper import get_taper
from mic_py.mic_zelin import zelin_filter
from mic_py.mic_ds_beamforming import ds_align
from numpy.linalg import solve

# def test_inv_woodbury():
#
#
#     # def inv_mat(R_inv, X, alfa):
#     #     T = np.dot(np.transpose(np.conj(X)), R_inv)
#     #     R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
#     #     return R_inv_new
#
#     def inv_mat(R_inv, X, alfa):
#         X =  np.expand_dims(X,axis=-1)
#         T = np.dot(np.transpose(np.conj(X)), R_inv)
#         R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
#         return R_inv_new
#
#     N = 66
#     #M = np.random.rand(N, N) + 1j * np.random.rand(N, N)
#     M = np.random.rand(N,N) + 1j * np.random.rand(N, N)
#     M_inv = np.linalg.inv(M)
#
#     E = np.dot(M,M_inv)
#     err = np.sum(E)
#     print('M err = ', err)
#
#     X = np.random.rand(N) + 1j * np.random.rand(N)
#     alfa = 0.7
#
#     # we need calc
#     # (alfa*M + (1-alfa)*np.outer(X,X.conj())) ^ -1
#
#     R = alfa * M + (1 - alfa) * np.outer(X, X.conj())
#     R_inv = np.linalg.inv(R)
#     E = np.dot(R,R_inv)
#     err = np.sum(E)
#     print('R err = ', err)
#
#     RR_inv = inv_mat(R_inv=M_inv, X=X, alfa=alfa)
#
#     E = R_inv-RR_inv
#     err = np.sum(E)
#     print('RR err = ', err)



@jit(nopython=True)
def mvdr_woodbury(stft_mix, d_arr, mask, alfa = 0.997):
    """

    :param stft_mix: - spectr for each sensors - shape (n_bins, n_sensors, n_frames)
    :param d_arr: - steering vector    - shape (n_bins, n_sensors)
    :param mask: - speech absence mask - shape (n_bins, n_frames)
    :param alfa: - const average psd matrix
    :return:
    """

    (n_bins, n_sensors, n_frames) = stft_mix_arr.shape

    result = np.zeros((n_bins, n_frames), dtype=np.complex128)
    R = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex128)
    for i in range(n_bins):
        R[i, :, :] = np.identity(n_sensors)

    R_inv_lst = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex128)
    for i in range(n_bins):
        R_inv_lst[i, :, :] = np.identity(n_sensors)

    w = np.zeros((n_bins, n_sensors), dtype=np.complex128)


    # def inv_mat(R_inv, X, alfa):
    #     T = np.dot(np.transpose(np.conj(X)), R_inv)
    #     R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
    #     return R_inv_new


    # def inv_mat(R_inv, X, alfa):
    #     X =  np.expand_dims(X,axis=-1)
    #     T = np.dot(np.transpose(np.conj(X)), R_inv)
    #     R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
    #     return R_inv_new


    for frame in range(n_frames):
        print('frame = ', frame)

        for bin in range(n_bins):

            x = stft_mix[bin, :, frame]

            R[bin] = alfa*R[bin] + (1.0-alfa)*np.outer(x, np.transpose(np.conj(x)))
            RR_inv = np.linalg.inv(R[bin] + 0.001 * np.identity(n_sensors))

            # + 0.001 * np.identity(n_sensors))
            #R_inv_lst[bin] += 0.001 * np.identity(n_sensors)

            #R_inv = inv_mat(R_inv_lst[bin], x, alfa=alfa)
            R_inv = RR_inv
            R_inv_lst[bin] = R_inv.copy()

            # MVDR
            num = np.dot(R_inv, d_arr[bin,:])
            denom = np.dot(np.conj(d_arr[bin, :]), num)
            w[bin] = num/denom

            if bin in [20]:
                print('     NORM R_inv = ', np.linalg.norm(R_inv))
                print('     NORM w     = ', np.linalg.norm(w[bin]))
                #print('     diff', np.linalg.norm(RR_inv-R_inv))


            # Apply beamforming
            result[bin, frame] = np.sum(np.conj(w[bin]) * x)

    return result

# @jit(nopython=True)
# def mvdr_woodbury(stft_mix, d_arr, mask, alfa = 0.97):
#     """
#
#     :param stft_mix: - spectr for each sensors - shape (n_bins, n_sensors, n_frames)
#     :param d_arr: - steering vector    - shape (n_bins, n_sensors)
#     :param mask: - speech absence mask - shape (n_bins, n_frames)
#     :param alfa: - const average psd matrix
#     :return:
#     """
#
#     (n_bins, n_sensors, n_frames) = stft_mix_arr.shape
#
#     result = np.zeros((n_bins, n_frames), dtype=np.complex128)
#     R_inv = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex128)
#     for i in range(n_bins):
#         R_inv[i, :, :] = 1000*np.identity(n_sensors)
#     w = np.zeros((n_bins, n_sensors), dtype=np.complex128)
#
#
#     # def inv_mat(R_inv, X, alfa):
#     #     #T = np.dot(np.transpose(X.conj()),R_inv)
#     #     T = np.dot(np.transpose(np.conj(X)), R_inv)
#     #     R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
#     #     return R_inv_new
#
#     def inv_mat(R_inv, X, alfa):
#         #T = np.dot(np.transpose(X.conj()),R_inv)
#         T = np.dot(np.transpose(np.conj(X)), R_inv)
#         R_inv_new = 1.0/alfa*R_inv - (1-alfa)/alfa*np.dot(np.dot(R_inv, X), T)/(alfa + (1-alfa)* np.dot(T, X))
#         return R_inv_new
#
#     for frame in range(n_frames):
#         print('frame = ', frame)
#         for bin in range(n_bins):
#             x = stft_mix[bin, :, frame]
#
#             #Recurent inverse PSD
#             # R_inv[bin] = 1.0/alfa*R_inv[bin] - (1-alfa)/alfa* \
#             #                                    np.dot(np.dot(R_inv[bin], x), np.dot(x.conj(), R_inv[bin])) \
#             #                                    / (alfa + (1 - alfa) * np.dot(np.dot(x.conj(), R_inv[bin]), x))
#             R_inv[bin] = inv_mat(R_inv[bin], x, alfa)
#
#             # MVDR
#             num = np.dot(R_inv[bin], d_arr[bin,:])
#             #denom = np.dot(d_arr[bin,:].conj(), num)
#             denom = np.dot(np.conj(d_arr[bin, :]), num)
#             w[bin] = num/denom
#             if bin in [20]:
#                 #print('     bin = {}, norm(w) = {}'.format(bin, w[bin]))
#                 #print(' norm(w)', w[bin])
#                 print(' NORM = ', np.linalg.norm(R_inv[bin]))
#
#             # Apply beamforming
#             result[bin, frame] = np.dot(np.conj(w[bin]), x)
#
#     return result

if __name__ == '__main__':
    # test_inv_woodbury()
    # exit()

    #################################################################
    # 1.0 - _du_hast PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512
    n_overlap      = 2

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_du_hast/'
    out_wav_path   = r'./data/out/'


    # _noise_start   = 8
    # _noise_end     = 17

    # _noise_start   = 17
    # _noise_end     = 84

    # _mix_start     = 17
    # _mix_end       = 84

    _mix_start     = 8
    _mix_end       = 40


    # _mix_start     = 0
    # _mix_end       = 20


    # _sp_start      = 84
    # _sp_end        = 102
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)


    x_mix_arr   = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    print ("Array data read done!")
    print ("    x_mix_arr.shape    = ", x_mix_arr.shape)


    #################################################################
    # 2 - Do STFT
    stft_mix_arr   =  stft_arr(x_mix_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_mix_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)


    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position    = get_source_position(angle_h, angle_v)
    d_arr              = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)

    # #################################################################
    # # 4 - Do SAD
    # fft_hop_size = n_fft / n_overlap
    #
    # type_sad = 'phase'
    # use_cmt = False
    # use_zelin = False
    #
    # if type_sad == 'phase':
    #
    #     kwargs = {'threshold_type': 'hist',
    #               'bias_for_base_level': 0.0}
    #
    #     # kwargs = {'threshold_type': 'gmm',
    #     #           'delta': 3.0}
    #
    #     frame_segm = phase_sad(stft_mix=stft_mix_arr, d_arr_sp=d_arr.T, sr=sr, fft_hop_size=fft_hop_size,
    #                            **kwargs)
    # elif type_sad == 'ilrma':
    #     frame_segm = ilrma_sad(stft_mix_arr, sr, n_fft, n_overlap)
    # elif type_sad == 'all':
    #     frame_segm = np.ones((n_frames))
    # else:
    #     assert False, 'type_sad = {} unsupported'.format(type_sad)
    #
    # noise_time = (np.sum((frame_segm)) * fft_hop_size) / sr
    # print('MVDR_SAD_FILTER type = {} , detect noise only period = {} sec.'.format(type_sad, noise_time))

    #################################################################
    # 5 - Create mask by SAD segm
    #mask = np.ones((n_bins, n_frames)) * (frame_segm)
    mask = np.ones((n_bins, n_frames))

    #################################################################
    # 6 - Filter output
    result_spec = mvdr_woodbury(stft_mix=stft_mix_arr, d_arr=d_arr.T, mask=mask)

    # #################################################################
    # # 6 - Filter output
    # psd = get_power_spectral_density_matrix(stft_mix_arr, mask=mask)
    # if use_cmt:
    #     T_matrix = get_taper(hor_mic_count=hor_mic_count,
    #                          vert_mic_count=vert_mic_count,
    #                          dHor=dHor,
    #                          dVert=dVert,
    #                          angle_h=angle_h,
    #                          angle_v=angle_v,
    #                          sr=sr,
    #                          fft_size=n_fft,
    #                          bandwidth=0.5)
    #     for i in range(n_bins):
    #         psd[i, :, :] = np.multiply(psd[i, :, :], T_matrix[i])
    #
    # # Regularisation
    # psd = psd + 0.001 * np.identity(psd.shape[-1])
    # w = get_mvdr_vector(d_arr.T, psd)
    # result_spec = apply_beamforming_vector(w, stft_mix_arr)
    #
    # if use_zelin:
    #     # 7 - Do align
    #     align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    #
    #     # 8 - Calc zelin filter output
    #     _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
    #     print('Calc zelin filter output done!')
    #
    #     # 9 - Calc MVDR + Zelin filter output
    #     result_spec = result_spec * H

    #################################################################
    # 7 - Inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_mvdr_woodbury.wav", 50*sig_out, sr)


    #################################################################
    # 8 - Do align and save DS output
    align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)






