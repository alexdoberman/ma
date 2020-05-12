# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import filter_DS
import filter_GSC_GRIFFITITHS
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
import filter_MVDR_MIX_SVD
import filter_DS_CHEBYSHEV
import filter_CHEBYSHEV_NULLS
import filter_MVDR_SAD
import filter_CMT
import filter_MVDR_HENDRIKS

# import filter_UNET_MVDR_MIX
# import filter_MVDR_MIX_MAD_MASK
import configparser as cfg
import os

from calc_metric import *
from mic_config import MicConfig
from filter_config import FilterConfig


def process_one_path(base_name, filter_name, **kwargs):

    wav_path         = '../data/_sdr_test/' + base_name + '/mix'
    config_path = '../data/_sdr_test/' + base_name + '/config.cfg'
         
    config = cfg.ConfigParser()
    config.read(config_path)

    mic_cfg = MicConfig(dict(config.items('MIC_CFG')))
    filter_cfg = FilterConfig(dict(config.items('FILTER_CFG')))
    out_path        = r'./temp/result_{}_{}.wav'.format(base_name, filter_name)

    module_name    = 'filter_' + filter_name
    function_name  = 'do_filter'
    class_name = filter_name + '_FILTER'
    filter_class = getattr(globals()[module_name], class_name)
    filter_inst = filter_class(mic_cfg)
    filter_method = getattr(filter_inst, function_name)

    filter_method(wav_path, out_path, filter_cfg, mic_cfg, **kwargs)

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
                        'MVDR_ZELIN', 'MVDR_MIX_SVD', 'MVDR_MIX', 'DS_CHEBYSHEV', 'CHEBYSHEV_NULLS', 'MVDR_SAD', 'CMT', 'HARD_NULL',
                        'MVDR_HENDRIKS']

    lst_bases = [
                 # 'out_wgn_spk_snr_5', 'out_wgn_spk_snr_0', 'out_wgn_spk_snr_-5', 'out_wgn_spk_snr_-10',
                 # 'out_wgn_spk_snr_-15',
                 # 'out_bn1_spk1_snr_5', 'out_bn1_spk1_snr_0', 'out_bn1_spk1_snr_-5', 'out_bn1_spk1_snr_-10',
                 'out_mus1_spk1_snr_-5', 'out_mus1_spk1_snr_-10', 'out_mus1_spk1_snr_-15', 'out_mus1_spk1_snr_-20',
                 'out_mus2_spk1_snr_-5', 'out_mus2_spk1_snr_-10', 'out_mus2_spk1_snr_-15', 'out_mus2_spk1_snr_-20'
    ]


    #################################################################
    # DS
    # kwargs = {}
    # filter_name = 'DS'

    #################################################################
    # MVDR_MIX
    # kwargs = {}
    # filter_name = 'MVDR_MIX'

    #################################################################
    # MVDR_SAD phase + hist
    # kwargs = {'type_sad': 'phase',
    #           'threshold_type': 'hist',
    #           'bias_for_base_level': 0.0,
    #           'use_cmt':True,
    #           'use_zelin':True}
    # filter_name = 'MVDR_SAD'

    #################################################################
    # MVDR_SAD phase + 'gmm'
    # kwargs = {'type_sad': 'phase',
    #           'threshold_type': 'gmm',
    #           'delta': 3.0}
    # filter_name = 'MVDR_SAD'

    #################################################################
    # MVDR_SAD phase + 'all'
    # kwargs = {'type_sad': 'all'}
    # filter_name = 'MVDR_SAD'

    #################################################################
    # MVDR_MIX_MAD_MASK
    # kwargs = {'out_wav_path_2': './temp/result_{}.wav'.format(base)}
    # filter_name = 'MVDR_MIX_MAD_MASK'

    #################################################################
    # HARD_NULL
    # kwargs = {'alg_type': 1,
    #           'time_domain_filter':False}
    # filter_name = 'HARD_NULL'

    #################################################################
    # MVDR_HENDRIKS
    # kwargs = {'reg_const_hendriks': 0.1,
    #           'reg_const_mvdr': 0.01,
    #           'use_cmt': False,
    #           'use_zelin': False}
    # filter_name = 'MVDR_HENDRIKS'

    #################################################################
    # MVDR_SAD + CMT + MCRA + ZELIN
    # kwargs = {'type_sad': 'phase',
    #           'threshold_type': 'hist',
    #           'bias_for_base_level': 0.0,
    #           'use_cmt':True,
    #           'use_zelin':True,
    #           'use_mcra':True}
    # filter_name = 'MVDR_SAD'



    # kwargs = {'type_sad': 'phase',
    #           'threshold_type': 'hist',
    #           'bias_for_base_level': 0.0,
    #           'use_cmt':True,
    #           'use_zelin':True,
    #           'use_mcra':True}
    # filter_name = 'MVDR_SAD'


    kwargs = {}
    filter_name = 'MVDR'



    lst_sdr_impr = []
    for base in lst_bases:
        #kwargs = {'out_wav_path_2': './temp/result_{}.wav'.format(base)}
        sdr_impr = process_one_path(base, filter_name, **kwargs)
        lst_sdr_impr.append(sdr_impr)

    print('Result: {}'.format(filter_name))
    print('---------------------------------------------')
    for i, base in enumerate(lst_bases):
        print('{} : {}'.format(base, lst_sdr_impr[i]))
