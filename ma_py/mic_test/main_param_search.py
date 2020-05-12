# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import filter_DS
import filter_GSC_ALL_SUBS
import filter_GSC_RLS
import filter_ZELIN
import filter_MVDR
import filter_MVDR_ZELIN
import filter_COV_MATRIX_TRACKING
import filter_MVDR_MIX_ZELIN
import filter_MVDR_AD_MIX
import filter_HARD_NULL
import filter_MN
import filter_MVDR_IBM_ZELIN
import filter_MVDR_MIX
import filter_MVDR_MIX_ZELIN
import filter_CGMM_MVDR
import filter_LCMV
import filter_CMT
import filter_EXP_STEERING
import filter_TEST_EVAD
import configparser as cfg
import os

from calc_metric import *
from mic_config import MicConfig
from filter_config import FilterConfig

'''
def process_one_path(base_name, filter_name, param):

    wav_path         = r'..\data\_sdr_test/' + base_name + r'\mix'

    config = cfg.ConfigParser()
    config.read(os.path.join('..\data\_sdr_test', base_name, 'config.cfg'))

    mic_cfg = MicConfig(dict(config.items('MIC_CFG')))
    filter_cfg = FilterConfig(dict(config.items('FILTER_CFG')))

    out_path        = r'.\temp\result.wav'
    # out_path_2 = r'.\temp' + base_name + '_' + filter_name + '.wav'

    module_name    = 'filter_' + filter_name
    function_name  = 'do_filter'
    class_name = filter_name + '_FILTER'
    filter_class = getattr(globals()[module_name], class_name)
    filter_inst = filter_class(mic_cfg)
    filter_method = getattr(filter_inst, function_name)

    filter_method(wav_path, out_path, filter_cfg, mic_cfg, param)

    in_Main     = r'..\data\_sdr_test/' + base_name + r'\ds_mix.wav'
    in_Main_Sp  = r'..\data\_sdr_test/' + base_name + r'\ds_spk.wav'
    in_Main_Mus = r'..\data\_sdr_test/' + base_name + r'\ds_mus.wav'
    Est_Sp      = out_path

    sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp)

    print ('---------------------------------------------')
    print (base_name)
    print ("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n"
           .format(sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base))
    print ('---------------------------------------------')
    return sdr_impr


if __name__ == '__main__':

    lst_filter_names = ['DS', 'GSC_GRIFFITITHS', 'COV_MATRIX_TRACKING', 'GSC_GSC_ALL_SUBS', 'GSC_RLS', 'ZELIN', 'MVDR',
                        'MVDR_ZELIN']

    lst_bases = [
                 'out_wgn_spk_snr_5', 'out_wgn_spk_snr_0', 'out_wgn_spk_snr_-5', 'out_wgn_spk_snr_-10',
                 'out_wgn_spk_snr_-15',
                 # 'out_bn1_spk1_snr_5', 'out_bn1_spk1_snr_0', 'out_bn1_spk1_snr_-5', 'out_bn1_spk1_snr_-10',
                 'out_mus1_spk1_snr_-5', 'out_mus1_spk1_snr_-10', 'out_mus1_spk1_snr_-15', 'out_mus1_spk1_snr_-20',
                 # 'out_mus2_spk1_snr_-5', 'out_mus2_spk1_snr_-10', 'out_mus2_spk1_snr_-15', 'out_mus2_spk1_snr_-20'
    ]

    filter_name = 'GSC_GRIFFITITHS_PHASE_COMPENSATE'

    num_it = 50
    params = np.linspace(0.1, 5, num=num_it)

    lst_sdr_impr = []

    for i in range(num_it):
        for base in lst_bases:
            sdr_impr = process_one_path(base, filter_name, params[i])
            lst_sdr_impr.append(sdr_impr)

    print('Result: {}'.format(filter_name))
    for i in range(num_it):
        print('---------------------------------------------')
        print('FOR PARAMETER: {}'.format(params[i]))
        print('---------------------------------------------')
        for j, base in enumerate(lst_bases):
            print('{} : {}'.format(base, lst_sdr_impr[i + j]))
'''


def process_one_path(base_name, filter_name, params):

    wav_path         = '../data/_sdr_test/' + base_name + '/mix'

    config = cfg.ConfigParser()

    config_path = '../data/_sdr_test/' + base_name + '/config.cfg'
    config.read(config_path)

    mic_cfg = MicConfig(dict(config.items('MIC_CFG')))
    filter_cfg = FilterConfig(dict(config.items('FILTER_CFG')))

    out_path        = './temp/result.wav'
    out_path_2 = './temp/{}_result.wav'.format(base_name)

    module_name    = 'filter_' + filter_name
    function_name  = 'do_filter'
    class_name = filter_name + '_FILTER'
    filter_class = getattr(globals()[module_name], class_name)
    filter_inst = filter_class(mic_cfg)
    filter_method = getattr(filter_inst, function_name)

    filter_method(wav_path, out_path, filter_cfg, mic_cfg, '../data/_sdr_test/' + base_name + '/ds_spk.wav', params)

    in_Main     = '../data/_sdr_test/' + base_name + '/ds_mix.wav'
    in_Main_Sp  = '../data/_sdr_test/' + base_name + '/ds_spk.wav'
    in_Main_Mus = '../data/_sdr_test/' + base_name + '/ds_mus.wav'
    Est_Sp      = out_path

    sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp)

    print ('---------------------------------------------')
    print (base_name)
    print ("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n"
           .format(sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base))
    print ('---------------------------------------------')
    return sdr_impr


if __name__ == '__main__':

    lst_filter_names = ['DS', 'GSC_GRIFFITITHS', 'COV_MATRIX_TRACKING', 'GSC_GSC_ALL_SUBS', 'GSC_RLS', 'ZELIN', 'MVDR',
                        'MVDR_ZELIN']

    lst_bases = [
                 # 'out_wgn_spk_snr_5', 'out_wgn_spk_snr_0', 'out_wgn_spk_snr_-5', 'out_wgn_spk_snr_-10',
                 # 'out_wgn_spk_snr_-15',
                 # 'out_bn1_spk1_snr_5', 'out_bn1_spk1_snr_0', 'out_bn1_spk1_snr_-5', 'out_bn1_spk1_snr_-10',
                 'out_mus1_spk1_snr_-5', 'out_mus1_spk1_snr_-10', 'out_mus1_spk1_snr_-15', 'out_mus1_spk1_snr_-20',
                 'out_mus2_spk1_snr_-5', 'out_mus2_spk1_snr_-10', 'out_mus2_spk1_snr_-15', 'out_mus2_spk1_snr_-20'
    ]

    filter_name = 'TEST_EVAD'

    param1 = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    param2 = np.array([0, 1, 2, 3, 4, 5, 6])

    lst_sdr_impr = []

    for i in param1:
        for j in param2:
            lst_sdr_imp_cur = []
            for base in lst_bases:
                sdr_impr = process_one_path(base, filter_name, (i, j))
                lst_sdr_imp_cur.append(sdr_impr)
            lst_sdr_impr.append(lst_sdr_imp_cur)

    print('Result: {}'.format(filter_name))
    for idx_i, i in enumerate(param1):
        for idx_j, j in enumerate(param2):
            print('---------------------------------------------')
            print('FOR PARAMETERS: {}, {}'.format(i, j))
            print('---------------------------------------------')
            for k, base in enumerate(lst_bases):
                print('{} : {}'.format(base, lst_sdr_impr[idx_i*param2.shape[0] + idx_j][k]))
