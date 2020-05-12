# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
import os
import mir_eval


def calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp, Est_Sp_lag = 256):
    """
    Calc SDR impr 

    :in_Main:     - path to mus+sp wav file
    :in_Main_Sp:  - path to sp wav file
    :in_Main_Mus: - path to mus wav file
    :Est_Sp:      - path to estimate sp wav file
    :Est_Sp_lag:  - lag Est_Sp

    return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base
    """


    ref, rate = sf.read(in_Main_Sp)
    mix, rate = sf.read(in_Main)
    est, rate = sf.read(Est_Sp)


#    # align est by ref 
#    lag = determine_lag(ref[0:16000], est[0:16000], max_lag = 2500)
#     lag = Est_Sp_lag
#     est = np.roll(est, lag)

    min_len = min(len(ref), len(mix), len(est))

    ref = ref [0:min_len]
    mix = mix [0:min_len]
    est = est [0:min_len]

    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources = ref, 
                                                    estimated_sources = est, compute_permutation=False)

    (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources = ref, 
                                                    estimated_sources = mix, compute_permutation=False)


    sdr_impr = sdr - sdr_base
    sir_impr = sir - sir_base
    sar_impr = sar - sar_base


    return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base



def determine_lag(x, y, max_lag):

    lags = []
    for i in range(-max_lag, max_lag+1, 1):
        corr = np.sum(x*np.roll(y, i))
        lags.append((i, corr))

    m = max(lags, key=lambda item:item[1])
    print (m)
#    shift_y = np.roll(y, m[0])
    return m[0]


if __name__ == "__main__":

    in_Main     = r'./in/mus_spk.wav'
    in_Main_Sp  = r'./in/ref_spk.wav'
    in_Main_Mus = r'./in/ref_mus.wav'
    Est_Sp      = r'./in/result.wav'


    sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp)
    print ("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base))


