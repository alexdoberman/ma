# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

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
from mic_py.mic_mcra import mcra_filter


if __name__ == '__main__':


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

    _mix_start     = 17
    _mix_end       = 84

    # _sp_start      = 84
    # _sp_end        = 102
    #################################################################


    # #################################################################
    # # 1.0 - _rameses PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    # n_overlap      = 2
    #
    # (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_rameses/'
    # out_wav_path   = r'./data/out/'
    #
    #
    # _noise_start   = 8
    # _noise_end     = 26
    #
    # _mix_start     = 26
    # _mix_end       = 104
    #
    # _sp_start      = 104
    # _sp_end        = 128
    # #################################################################


    '''
    #back 0-7
    #sp [2017-12-01 13:46:23] [0x00000344] :  [INF] MicGridProcessor::SetDirectionAngles, angleFiHorz = 2.37685,  angleFiVert = -4.85129
    #################################################################
    # 1.0 - _5 PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 2*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (2.37685, -4.85129)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_5/'
    out_wav_path   = r'./data/out/'


#    _noise_start   = 0
#    _noise_end     = 7

    _noise_start   = 7
    _noise_end     = 88

    _mix_start     = 7
    _mix_end       = 88

    _sp_start      = 88
    _sp_end        = 90
    #################################################################

    '''

    '''

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


    _noise_start   = 9
    _noise_end     = 28

    _mix_start     = 28
    _mix_end       = 98

    _sp_start      = 98
    _sp_end        = 112
    #################################################################
    '''

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

    #################################################################
    # 4 - Do SAD
    fft_hop_size = n_fft / n_overlap

    type_sad = 'phase'
    use_cmt = True
    use_zelin = True
    use_mcra = True

    if type_sad == 'phase':

        kwargs = {'threshold_type': 'hist',
                  'bias_for_base_level': 0.0}

        # kwargs = {'threshold_type': 'gmm',
        #           'delta': 3.0}

        frame_segm = phase_sad(stft_mix=stft_mix_arr, d_arr_sp=d_arr.T, sr=sr, fft_hop_size=fft_hop_size,
                               **kwargs)
    elif type_sad == 'ilrma':
        frame_segm = ilrma_sad(stft_mix_arr, sr, n_fft, n_overlap)
    elif type_sad == 'all':
        frame_segm = np.ones((n_frames))
    else:
        assert False, 'type_sad = {} unsupported'.format(type_sad)

    noise_time = (np.sum((frame_segm)) * fft_hop_size) / sr
    print('MVDR_SAD_FILTER type = {} , detect noise only period = {} sec.'.format(type_sad, noise_time))

    #################################################################
    # 5 - Create mask by SAD segm
    mask = np.ones((n_bins, n_frames)) * (frame_segm)

    #################################################################
    # 6 - Filter output
    psd = get_power_spectral_density_matrix(stft_mix_arr, mask=mask)
    if use_cmt:
        T_matrix = get_taper(hor_mic_count=hor_mic_count,
                             vert_mic_count=vert_mic_count,
                             dHor=dHor,
                             dVert=dVert,
                             angle_h=angle_h,
                             angle_v=angle_v,
                             sr=sr,
                             fft_size=n_fft,
                             bandwidth=0.5)
        for i in range(n_bins):
            psd[i, :, :] = np.multiply(psd[i, :, :], T_matrix[i])

    # Regularisation
    psd = psd + 0.001 * np.identity(psd.shape[-1])
    w = get_mvdr_vector(d_arr.T, psd)
    result_spec = apply_beamforming_vector(w, stft_mix_arr)

    if use_mcra:
        result_spec = mcra_filter(stft_arr=result_spec)

    if use_zelin:
        # 7 - Do align
        align_stft_arr = ds_align(stft_mix_arr, d_arr.T)

        # 8 - Calc zelin filter output
        _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
        print('Calc zelin filter output done!')

        # 9 - Calc MVDR + Zelin filter output
        result_spec = result_spec * H

    #################################################################
    # 7 - Inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_mvdr_sad_{}_use_cmt_{}_zelin_{}_mcra_{}.wav".format(type_sad, use_cmt, use_zelin, use_mcra), 50.0*sig_out, sr)


    #################################################################
    # 8 - Do align and save DS output
    align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)



