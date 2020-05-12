# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from mic_py.mic_adaptfilt import *
from mic_py.feats import *

def main(X_wav_path, N_wav_path, OUT_wav_path):

    # Load signal 
    X_sig, rate = sf.read(X_wav_path)
    N_sig, rate = sf.read(N_wav_path)

    # SFFT signal
    X_spec    = stft(X_sig)
    N_spec    = stft(N_sig)

    # Filter signal
    #S_spec =  compensate_ref_ch_filter(stft_main = X_spec, stft_ref = N_spec, alfa = 0.7)
    S_spec =  spectral_substract_filter(stft_main= X_spec , stft_ref= N_spec, alfa_PX = 0.01, alfa_PN = 0.99)


    # ISFFT signal
    sig_out = istft(S_spec)

    # ISFFT signal
    sf.write(OUT_wav_path, sig_out, rate)


if __name__ == '__main__':

    #X_wav_path = r'.\data\_adapt_simple\spk_out_DS.wav'
    #N_wav_path  = r'.\data\_adapt_simple\mus_out_DS.wav'
    #OUT_wav_path = r'.\out\ss.wav'

    X_wav_path = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Y.wav'
    N_wav_path  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_Z.wav'
    OUT_wav_path  = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\out\out_GSC_OUT_spec_subs.wav'


#    X_wav_path = r'.\data\_adapt_simple\x.wav'
#    N_wav_path  = r'.\data\_adapt_simple\n.wav'
#    OUT_wav_path = r'.\out\ss.wav'

    main(X_wav_path, N_wav_path, OUT_wav_path)
