# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from mic_py.feats import stft, istft
from mic_py.mic_adaptfilt_kurt  import maximize_kutrosis_filter, maximize_kutrosis_filter_dbg
from mic_py.mic_adaptfilt_negentropy  import maximize_negentropy_filter


if __name__ == '__main__':

    f_main = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Y.wav'
    f_ref  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Z.wav'
    f_out  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_OUTMK.wav'

    # f_main = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\data\_adapt_simple\s_n\mix.wav'
    # f_ref  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\data\_adapt_simple\s_n\nn.wav'
    # f_out  = r'out\out_MN.wav'

    # f_main = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Y.wav'
    # f_ref  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Z.wav'
    # f_out  = r'out\out_MN_GSC_1.wav'

    speech_distribution_coeff_path = r'mic_utils\alg_data\gg_params_freq_f_scale.npy'


#################################################################
    # 1.0 - Read signal
    y_main, sr  = sf.read(f_main, dtype = np.float64)
    y_ref, sr  = sf.read(f_ref, dtype = np.float64)

    #################################################################
    # 2.0 - STFT signal
    stft_main = stft(y_main, fftsize = 512, overlap = 2)
    stft_ref  = stft(y_ref, fftsize = 512, overlap = 2)

    #################################################################
    # 3.0 - Filter signal
    #stft_out = maximize_kutrosis_filter(stft_main, stft_ref)
    stft_out = maximize_negentropy_filter(stft_main, stft_ref, speech_distribution_coeff_path=speech_distribution_coeff_path)

    #################################################################
    # 4.0 - ISFFT signal
    sig_out = istft(stft_out, overlap = 2)

    #################################################################
    # 5.0 - Write output
    sf.write(f_out, sig_out, sr)
