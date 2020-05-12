# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.calc_metric import calc_metric

if __name__ == '__main__':

    db_str = '-20'
    lag    = 256

    in_Main     = r'/mnt/sda1/shuranov/city_16000/F001_WN/ch_0_0.wav'
    in_Main_Sp  = r'/mnt/sda1/shuranov/city_16000/F001_16000.wav'
    in_Main_Mus = r'/mnt/sda1/shuranov/city_16000/F001_16000.wav'
    Est_Sp      = r'/mnt/sda1/shuranov/city_16000/F001_WN/ch_0_0.wav'

    # in_Main     = r'./data/_mus1_spk1_snr_' + db_str + '/mus_spk.wav'
    # in_Main_Sp  = r'./data/_mus1_spk1_snr_' + db_str + '/ref_spk.wav'
    # in_Main_Mus = r'./data/_mus1_spk1_snr_' + db_str + '/ref_mus.wav'
    # Est_Sp      = r'./out/ds.wav'

    sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp, lag)
    print ("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base))



    in_Main     = r'/mnt/sda1/shuranov/city_16000/F001_WN/ch_0_0.wav'
    in_Main_Sp  = r'/mnt/sda1/shuranov/city_16000/F001_16000.wav'
    in_Main_Mus = r'/mnt/sda1/shuranov/city_16000/F001_16000.wav'
    Est_Sp      = r'/mnt/sda1/shuranov/city_16000/F001_WN_out/out_DS.wav'
    # in_Main     = r'./data/_mus1_spk1_snr_' + db_str + '/mus_spk.wav'
    # in_Main_Sp  = r'./data/_mus1_spk1_snr_' + db_str + '/ref_spk.wav'
    # in_Main_Mus = r'./data/_mus1_spk1_snr_' + db_str + '/ref_mus.wav'
    # Est_Sp      = r'./out/out_superdirectivity.wav'

    sdr_impr,  sir_impr, sar_impr, sdr_base, sir_base, sar_base = calc_metric(in_Main, in_Main_Sp, in_Main_Mus, Est_Sp, lag)
    print ("sdr_impr, sir_impr, sar_impr  =  {}, {}, {},  sdr_base, sir_base, sar_base  =  {}, {}, {}\n".format(sdr_impr, sir_impr, sar_impr, sdr_base, sir_base, sar_base))




