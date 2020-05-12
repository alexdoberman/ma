# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import filter_TEST_EVAD
import configparser as cfg
import os

from calc_metric import *
from mic_config import MicConfig
from filter_config import FilterConfig


def process_one_path(base_name, filter_name):

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

    filter_method(wav_path, out_path, filter_cfg, mic_cfg, '../data/_sdr_test/' + base_name + '/ds_spk.wav',
                  out_wav_path_2=out_path_2)

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

    lst_sdr_impr = []
    for base in lst_bases:
        sdr_impr = process_one_path(base, filter_name)
        lst_sdr_impr.append(sdr_impr)

    print('Result: {}'.format(filter_name))
    print('---------------------------------------------')
    for i, base in enumerate(lst_bases):
        print('{} : {}'.format(base, lst_sdr_impr[i]))