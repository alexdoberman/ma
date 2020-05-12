# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import configparser as cfg
import os
import mir_eval


from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py.mic_adaptfilt_time_domain import affine_projection_filter
from mic_py.mic_null import null_filter, null_filter_ex
import matplotlib.pyplot as plt

def calc_sdr_impr_metric(ref_sp, ref_mus, mix, est):
    """

    :param ref_sp:
    :param ref_mus:
    :param mix:
    :param est:
    :return:
    """

    ref_sp = ref_sp[:len(est)]
    ref_mus = ref_mus[:len(est)]
    mix = mix[:len(est)]

    if ((len(ref_sp) != len(ref_mus)) or (len(ref_mus) != len(mix)) or  (len(ref_mus) != len(est)) ):
        raise ValueError(' len ref_sp, ref_mus, mix, est = {} {} {} {}'.format(len(ref_sp), len(ref_mus), len(mix), len(est)))


    def determine_lag(x, y, max_lag = 8000):
        lags = []
        for i in range(-max_lag, max_lag + 1, 1):
            corr = np.sum(x * np.roll(y, i))
            lags.append((i, corr))
        m = max(lags, key=lambda item: item[1])
        print('determine_lag = ', m[0])
        return m[0]

    lag = determine_lag(ref_sp[0:16000], est[0:16000], max_lag = 2500)
    est = np.roll(est, lag)


    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources=ref_sp,
                                                                 estimated_sources=est, compute_permutation=False)

    (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources=ref_sp,
                                                                                     estimated_sources=mix,
                                                                                     compute_permutation=False)

    sdr_impr = sdr - sdr_base
    sir_impr = sir - sir_base
    sar_impr = sar - sar_base

    return sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base


def run_once():


    # #################################################################
    # # 1.0 - _wav_wbn45_dict0 PROFILE
    #
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (0.0, 0.0)
    # (angle_inf_hor_log, angle_inf_vert_log) = (45, 0)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # _mix_start     = 0.0
    # _mix_end       = 20.0
    #
    # in_wav_path    = r'./data/_wav_wbn45_dict0/'
    # out_wav_path   = r'./data/out/'
    # #################################################################


    # #################################################################
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
    _mix_end       = 48
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
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    source_position_inf      = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf                = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - Calc DS beamforming in desired and noise direction
    result_spec = ds_beamforming(stft_all, d_arr.T)
    result_spec_inf = ds_beamforming(stft_all, d_arr_inf.T)

    #################################################################
    # 5 - ISTFT
    sig_sp = istft(result_spec.transpose((1,0)), overlap = 2)
    sig_inf = istft(result_spec_inf.transpose((1,0)), overlap = 2)

    sf.write(r"out/out_ds_sp.wav", sig_sp, sr)
    sf.write(r"out/out_ds_inf.wav", sig_inf, sr)


    #################################################################
    # 4 - NULL filter output
    result_spec, _  = null_filter(stft_all, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_null.wav", sig_out, sr)

    #result_spec, _ = null_filter_ex(stft_mix_arr, d_arr_sp=d_arr.T, lst_d_arr_inf=[d_arr_inf.T, d_arr_inf_d1.T, d_arr_inf_d2.T])




    # #################################################################
    # # 6 - AP filter
    # M = 200
    # step = 0.05
    # L = 5
    # sig_result =  affine_projection_filter(main=sig_sp, ref=sig_inf, M=M,step=step,L=5)

    #################################################################
    # 6.1 - AP filter + cyclic parameters
    # M = 200
    # step = 0.05
    # L = 5
    # leak = 0.0
    # delay = -5

    #params = [(200, 0.005, 5), (200, 0.05, 5), (200, 0.1, 5), (200, 0.5, 5)]
    #params = [(200, 0.005, 5),(200, 0.005, 11), (200, 0.005, 25) , (200, 0.005, 35)]

    params = [(100, 0.01, 5, 0.01, -5)]
    for (M,step,L, leak, delay) in params:
        print("process     M = {}, step = {}, L = {}, leak = {}, dealy = {}".format(M,step,L, leak, delay))
        sig_inf = np.roll(sig_inf, delay)
        sig_result =  affine_projection_filter(main=sig_sp, ref=sig_inf, M=M,step=step,L=L, leak=leak)
        sf.write(r"out/out_ap_null_M_{}_step_{}_L_{}_leak_{}_delay_{}.wav".format(M,step,L,leak, delay), sig_result, sr)



        sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_sdr_impr_metric(ref_sp, ref_mus, mix, sig_result)


        print(base_name)
        print("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr,
                                                                                                                   sir_impr,
                                                                                                                   sar_impr,
                                                                                                                   sdr_base,
                                                                                                                   sir_base,
                                                                                                                   sar_base))


    # #################################################################
    # # 7 Save result
    # sf.write(r"out/out_ap_null.wav", sig_result, sr)

def params_iterate():

    base_name = 'out_mus1_spk1_snr_-10'
    #base_name = 'out_wgn_spk_snr_-10'



    config = cfg.ConfigParser()
    config.read(os.path.join(r'./data/_sdr_test', base_name, 'config.cfg'))

    in_Main     = r'./data/_sdr_test/' + base_name + r'/ds_mix.wav'
    in_Main_Sp  = r'./data/_sdr_test/' + base_name + r'/ds_spk.wav'
    in_Main_Mus = r'./data/_sdr_test/' + base_name + r'/ds_mus.wav'


    # #################################################################
    # 1.0 - read PROFILE
    vert_mic_count = int(config['MIC_CFG']['vert_mic_count'])
    hor_mic_count  = int(config['MIC_CFG']['hor_mic_count'])
    dHor           = float(config['MIC_CFG']['dhor'])
    dVert          = float(config['MIC_CFG']['dvert'])
    max_len_sec    = int(config['MIC_CFG']['max_len_sec'])
    n_fft          = int(config['MIC_CFG']['fft_size'])

    angle_h = -float(config['FILTER_CFG']['angle_h'])
    angle_v = -float(config['FILTER_CFG']['angle_v'])
    angle_inf_h = -float(config['FILTER_CFG']['angle_inf_h'])
    angle_inf_v = -float(config['FILTER_CFG']['angle_inf_v'])

    in_wav_path    = r'./data/_sdr_test/' + base_name + r'/mix'
    out_wav_path   = r'./data/out/'

    _mix_start     = float(config['FILTER_CFG']['start_mix_time'])
    _mix_end       = 30
    #################################################################


    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)

    ref_sp,  _ = sf.read(in_Main_Sp)
    ref_mus, _ = sf.read(in_Main_Mus)
    mix, _     = sf.read(in_Main)

    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    ref_sp = ref_sp[:(np.int32)((_mix_end - _mix_start)*sr)]
    ref_mus = ref_mus[:(np.int32)((_mix_end - _mix_start)*sr)]
    mix = mix[(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]


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
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    source_position_inf      = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf                = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - Calc DS beamforming in desired and noise direction
    result_spec = ds_beamforming(stft_all, d_arr.T)
    result_spec_inf = ds_beamforming(stft_all, d_arr_inf.T)

    #################################################################
    # 5 - ISTFT
    sig_sp = istft(result_spec.transpose((1,0)), overlap = 2)
    sig_inf = istft(result_spec_inf.transpose((1,0)), overlap = 2)

    sf.write(r"out/out_ds_sp.wav", sig_sp, sr)
    sf.write(r"out/out_ds_inf.wav", sig_inf, sr)


    #################################################################
    # 6.1 - AP filter + cyclic parameters
    # M = 200
    # step = 0.05
    # L = 5
    # leak = 0.0
    # delay = -5


    params = [
        #(150, 0.005, 5, 0.001, 0),
        (150, 0.005, 5, 0.001, 0),
        (200, 0.05, 5, 0.001, 0),
              ]

    for (M,step,L, leak, delay) in params:
        print("process     M = {}, step = {}, L = {}, leak = {}, delay = {}".format(M,step,L, leak, delay))
        sig_inf = np.roll(sig_inf, delay)

        sig_result =  affine_projection_filter(main=sig_sp, ref=sig_inf, M=M,step=step,L=L, leak=leak)
        sf.write(r"out/out_ap_null_M_{}_step_{}_L_{}_leak_{}_delay_{}.wav".format(M,step,L,leak, delay), sig_result, sr)


        sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_sdr_impr_metric(ref_sp, ref_mus, mix, sig_result)


        print(base_name)
        print("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr,
                                                                                                                   sir_impr,
                                                                                                                   sar_impr,
                                                                                                                   sdr_base,
                                                                                                                   sir_base,
                                                                                                                   sar_base))
        print('---------------------------------------------')

def params_iterate_null():

    #base_name = 'out_mus1_spk1_snr_-10'
    base_name = 'out_wgn_spk_snr_-10'

    config = cfg.ConfigParser()
    config.read(os.path.join(r'./data/_sdr_test', base_name, 'config.cfg'))

    in_Main     = r'./data/_sdr_test/' + base_name + r'/ds_mix.wav'
    in_Main_Sp  = r'./data/_sdr_test/' + base_name + r'/ds_spk.wav'
    in_Main_Mus = r'./data/_sdr_test/' + base_name + r'/ds_mus.wav'


    # #################################################################
    # 1.0 - read PROFILE
    vert_mic_count = int(config['MIC_CFG']['vert_mic_count'])
    hor_mic_count  = int(config['MIC_CFG']['hor_mic_count'])
    dHor           = float(config['MIC_CFG']['dhor'])
    dVert          = float(config['MIC_CFG']['dvert'])
    max_len_sec    = int(config['MIC_CFG']['max_len_sec'])
    n_fft          = int(config['MIC_CFG']['fft_size'])

    angle_h = -float(config['FILTER_CFG']['angle_h'])
    angle_v = -float(config['FILTER_CFG']['angle_v'])
    angle_inf_h = -float(config['FILTER_CFG']['angle_inf_h'])
    angle_inf_v = -float(config['FILTER_CFG']['angle_inf_v'])

    in_wav_path    = r'./data/_sdr_test/' + base_name + r'/mix'
    out_wav_path   = r'./data/out/'

    _mix_start     = float(config['FILTER_CFG']['start_mix_time'])
    _mix_end       = 30
    #################################################################


    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)

    ref_sp,  _ = sf.read(in_Main_Sp)
    ref_mus, _ = sf.read(in_Main_Mus)
    mix, _     = sf.read(in_Main)

    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    ref_sp = ref_sp[:(np.int32)((_mix_end - _mix_start)*sr)]
    ref_mus = ref_mus[:(np.int32)((_mix_end - _mix_start)*sr)]
    mix = mix[(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]


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
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    source_position_inf      = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf                = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - Calc DS beamforming in desired and noise direction
    result_spec = ds_beamforming(stft_all, d_arr.T)
    result_spec_inf = ds_beamforming(stft_all, d_arr_inf.T)

    #################################################################
    # 5 - ISTFT
    sig_sp = istft(result_spec.transpose((1,0)), overlap = 2)
    sig_inf = istft(result_spec_inf.transpose((1,0)), overlap = 2)

    sf.write(r"out/out_ds_sp.wav", sig_sp, sr)
    sf.write(r"out/out_ds_inf.wav", sig_inf, sr)


    #################################################################
    # 5.1 - NULL filter + cyclic parameters
    out_spec, _  = null_filter(stft_all, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)
    sig_out = istft(out_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_null.wav", sig_out, sr)

    #result_spec, _ = null_filter_ex(stft_mix_arr, d_arr_sp=d_arr.T, lst_d_arr_inf=[d_arr_inf.T, d_arr_inf_d1.T, d_arr_inf_d2.T])



    sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_sdr_impr_metric(ref_sp, ref_mus, mix, sig_out)

    print(base_name)
    print("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr,
                                                                                                               sir_impr,
                                                                                                               sar_impr,
                                                                                                               sdr_base,
                                                                                                               sir_base,
                                                                                                               sar_base))
    print('---------------------------------------------')




if __name__ == '__main__':
    params_iterate_null()
