# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
import os
import mir_eval


# def calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp):
#     """
#     Calc SDR impr
#
#     :in_Main:     - path to mus+sp wav file
#     :in_Main_Sp:  - path to sp wav file
#     :in_Main_Mus: - path to mus wav file
#     :Est_Sp:      - path to estimate sp wav file
#
#     return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base
#     """
#
#
#     ref_sp,  rate = sf.read(in_Main_Sp)
#     ref_mus, rate = sf.read(in_Main_Mus)
#     mix, rate     = sf.read(in_Main)
#     est, rate     = sf.read(Est_Sp)
#
#
#     # align
#     begin_skip = len(mix) - len(ref_sp)
#     mix = mix[begin_skip:]
#
#     if len(mix) != len(est):
#         est = est[begin_skip:]
#
#     if  len(mix) - len(est) < 1024 and  len(mix) - len(est) > 0:
#         est = np.append(est, np.zeros(len(mix) - len(est)))
#
#     if ((len(ref_sp) != len(ref_mus)) or (len(mix) < len(ref_sp))  or (len(mix) != len(est))):
#         raise ValueError(' len ref_sp, ref_mus, mix, est = {} {} {} {}'.format(len(ref_sp), len(ref_mus), len(mix), len(est)))
#
#     # begin_skip = len(mix) - len(ref_sp)
#     # mix = mix[begin_skip:]
#     # est = est[begin_skip:]
#
#
#     (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources = ref_sp,
#                                                     estimated_sources = est, compute_permutation=False)
#
#     (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources = ref_sp,
#                                                     estimated_sources = mix, compute_permutation=False)
#
#
#     sdr_impr = sdr - sdr_base
#     sir_impr = sir - sir_base
#     sar_impr = sar - sar_base
#
#
#     return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base


def si_snr(x, s):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2normp(x):
        return (np.linalg.norm(x, 2)**2)

    # zero mean
    x_zm = x - np.mean(x)
    s_zm = s - np.mean(s)
    t = np.inner(x_zm, s_zm) * s_zm / vec_l2normp(s_zm)
    return 10 * np.log10(vec_l2normp(t) / vec_l2normp(x_zm - t))


def calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp):
    """
    Calc SDR impr

    :in_Main:     - path to mus+sp wav file
    :in_Main_Sp:  - path to sp wav file
    :in_Main_Mus: - path to mus wav file
    :Est_Sp:      - path to estimate sp wav file

    return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base
    """


    ref_sp,  rate = sf.read(in_Main_Sp)
    ref_mus, rate = sf.read(in_Main_Mus)
    mix, rate     = sf.read(in_Main)
    est, rate     = sf.read(Est_Sp)


    # align
    begin_skip = len(mix) - len(ref_sp)
    mix = mix[begin_skip:]

    if len(mix) != len(est):
        est = est[begin_skip:]

    if  len(mix) - len(est) < 1024 and  len(mix) - len(est) > 0:
        est = np.append(est, np.zeros(len(mix) - len(est)))

    if ((len(ref_sp) != len(ref_mus)) or (len(mix) < len(ref_sp))  or (len(mix) != len(est))):
        raise ValueError(' len ref_sp, ref_mus, mix, est = {} {} {} {}'.format(len(ref_sp), len(ref_mus), len(mix), len(est)))

    # begin_skip = len(mix) - len(ref_sp)
    # mix = mix[begin_skip:]
    # est = est[begin_skip:]


    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources = ref_sp,
                                                    estimated_sources = est, compute_permutation=False)

    (sdr_base, sir_base, sar_base, perm_base) = mir_eval.separation.bss_eval_sources(reference_sources = ref_sp,
                                                    estimated_sources = mix, compute_permutation=False)

    sdr_base = si_snr(mix, ref_sp)
    sdr = si_snr(est, ref_sp)

    sdr_impr = sdr - sdr_base
    sir_impr = 0.0
    sar_impr = 0.0


    return sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base

